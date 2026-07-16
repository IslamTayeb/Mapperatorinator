"""Size FlashDecode-style adaptive split-KV occupancy vs accepted split_kv_8."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from utils.profile_native_prefix_dtype_scout import (  # noqa: E402
    CapturedGraph,
    _accepted_main_session_run,
    _all_cache_snapshots,
    _assert_scout_args,
    _cache_tensor,
    _capture_cuda_graph,
    _capture_representative_layer,
    _load_args,
    _max_abs,
    _observe_prefix_graph,
    _reciprocal_graph_rounds,
    _restore_all_cache,
)


DECODER_LAYERS = 12
MAX_ABS_DRIFT = 1e-3
# Scout gate for residual headroom after compiled-cross (~0.2s+ class).
SAVING_TARGET_SECONDS = 0.1
FIXED_MAIN_GENERATED_TOKENS = 8_294
FIXED_MAIN_DECODE_REPLAYS = 8_207
FIXED_MAIN_WINDOWS = 87
BASE_TIP_CHOICE = {
    "selected": "codex/500tps-coalesced-split-kv@e84e7e10",
    "alternative": "codex/500tps-arena-compiled-cross-last-mile@0dbab9e5",
    "reason": (
        "attention-only FlashDecode split-count delta on existing split-KV "
        "kernels/harness; compiled-cross owns selected-stack cross BMM "
        "orthogonally and is not mixed into this component gate"
    ),
}


def _git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_live_graph_cache(
    graph_cache: dict[Any, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    if not isinstance(graph_cache, dict) or not graph_cache:
        raise ValueError("live graph cache must be a non-empty mapping")
    entries: dict[int, dict[str, Any]] = {}
    for entry in graph_cache.values():
        if not isinstance(entry, dict):
            raise TypeError("live graph cache entries must be mappings")
        prefix = entry.get("active_prefix_length")
        count = entry.get("decode_replays")
        if (
            isinstance(prefix, bool)
            or not isinstance(prefix, int)
            or prefix <= 0
            or prefix in entries
        ):
            raise ValueError(f"invalid or duplicate live prefix {prefix!r}")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"live prefix {prefix} has invalid replay count {count!r}")
        missing = [
            name for name in ("graph", "outputs", "static_inputs") if name not in entry
        ]
        if missing:
            raise ValueError(f"live prefix {prefix} is missing {missing}")
        entries[prefix] = entry
    return dict(sorted(entries.items()))


def validate_fixed_main_work(
    run: dict[str, Any],
    entries: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    processor = run.get("processor")
    stats = getattr(processor, "last_generation_stats", None)
    if not isinstance(stats, dict):
        raise RuntimeError("main Processor did not expose aggregate generation stats")
    generated_tokens = stats.get("generated_tokens")
    if isinstance(generated_tokens, bool) or not isinstance(generated_tokens, int):
        raise RuntimeError("main generated-token count is missing or invalid")
    decode_replays = sum(int(entry["decode_replays"]) for entry in entries.values())
    windows = generated_tokens - decode_replays
    expected = (
        FIXED_MAIN_GENERATED_TOKENS,
        FIXED_MAIN_DECODE_REPLAYS,
        FIXED_MAIN_WINDOWS,
    )
    actual = (generated_tokens, decode_replays, windows)
    if actual != expected:
        raise RuntimeError(
            "fixed SALVALAI work changed: "
            f"expected tokens/replays/windows={expected}, got {actual}"
        )
    return {
        "generated_tokens": generated_tokens,
        "decode_replays": decode_replays,
        "prefill_windows": windows,
        "bucket_counts": {
            str(prefix): int(entry["decode_replays"])
            for prefix, entry in entries.items()
        },
    }


def summarize_component(
    buckets: dict[str, dict[str, Any]],
    *,
    total_replays: int,
) -> dict[str, Any]:
    if not buckets or total_replays <= 0:
        raise ValueError("component summary requires buckets and positive total replays")
    measured = 0
    accepted_weighted_ms = 0.0
    candidate_weighted_ms = 0.0
    failures: dict[str, list[str]] = {}
    for prefix, entry in sorted(buckets.items(), key=lambda item: int(item[0])):
        count = int(entry["decode_replays"])
        accepted_ms = float(entry["accepted_ms_per_call"])
        candidate_ms = float(entry["candidate_ms_per_call"])
        if count <= 0 or min(accepted_ms, candidate_ms) <= 0:
            raise ValueError(f"prefix {prefix} has invalid count or timing")
        if not all(math.isfinite(value) for value in (accepted_ms, candidate_ms)):
            raise ValueError(f"prefix {prefix} has non-finite timing")
        measured += count
        accepted_weighted_ms += count * accepted_ms
        candidate_weighted_ms += count * candidate_ms
        accepted_verifier = entry.get("accepted_verifier")
        if not isinstance(accepted_verifier, dict) or not bool(
            accepted_verifier.get("pass")
        ):
            raise ValueError(f"prefix {prefix} accepted cache verifier failed")
        if not bool(entry.get("accepted_memory_stable")):
            raise ValueError(f"prefix {prefix} accepted timing allocated memory")
        bucket_failures = [
            name
            for name, passed in entry["candidate_verifier"].items()
            if isinstance(passed, bool) and not passed
        ]
        if not bool(entry["candidate_memory_stable"]):
            bucket_failures.append("memory_stable")
        for name, value in entry["drift"].items():
            if not math.isfinite(float(value)) or float(value) < 0:
                bucket_failures.append(f"{name}_invalid")
            elif float(value) > MAX_ABS_DRIFT:
                bucket_failures.append(f"{name}_above_{MAX_ABS_DRIFT}")
        if bucket_failures:
            failures[prefix] = bucket_failures
    if measured > total_replays:
        raise ValueError("measured replay count exceeds live total")
    accepted_seconds = DECODER_LAYERS * accepted_weighted_ms / 1000.0
    candidate_seconds = DECODER_LAYERS * candidate_weighted_ms / 1000.0
    saving = accepted_seconds - candidate_seconds
    coverage_pass = measured == total_replays
    correctness_pass = not failures
    sizing_pass = saving >= SAVING_TARGET_SECONDS
    return {
        "measured_replays": measured,
        "total_replays": total_replays,
        "coverage_fraction": measured / total_replays,
        "coverage_pass": coverage_pass,
        "weighted_accepted_seconds_12_layers": accepted_seconds,
        "weighted_candidate_seconds_12_layers": candidate_seconds,
        "projected_main_saving_seconds": saving,
        "saving_target_seconds": SAVING_TARGET_SECONDS,
        "weighted_local_speedup": (
            accepted_weighted_ms / candidate_weighted_ms
        ),
        "correctness_failures": failures,
        "correctness_pass": correctness_pass,
        "sizing_pass": sizing_pass,
        "promotion_pass": coverage_pass and correctness_pass and sizing_pass,
    }


def _attention_inputs(capture):
    from osuT5.osuT5.inference.optimized.scout.native_prefix import _trim_mask

    module = capture.module
    self_attn = module.self_attn
    normalized = module.self_attn_layer_norm(capture.hidden_states)
    qkv = self_attn.Wqkv(normalized).view(
        1, 1, 3, self_attn.num_heads, self_attn.head_dim
    )
    cos, sin = self_attn.rotary_emb(qkv, position_ids=capture.position_ids)
    keys = _cache_tensor(capture.past_key_value, "self", capture.layer_idx, "keys")
    values = _cache_tensor(capture.past_key_value, "self", capture.layer_idx, "values")
    mask = _trim_mask(capture.attention_mask, capture.active_prefix_length)
    return qkv, keys, values, cos, sin, mask


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


@torch.no_grad()
def profile_component(
    args,
    *,
    output_path: Path,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("FlashDecode occupancy profiler requires CUDA")
    if torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError("FlashDecode occupancy profiler requires SM75")
    if warmup < 1 or iters < 1:
        raise ValueError("warmup and iters must be positive")
    _assert_scout_args(args)
    output_path.mkdir(parents=True, exist_ok=True)
    run = _accepted_main_session_run(args, output_path=output_path)
    model = run["model"]
    model.eval()
    entries = validate_live_graph_cache(run["session"].graph_cache)
    work = validate_fixed_main_work(run, entries)

    from osuT5.osuT5.inference.optimized.kernels.q1_attention import (
        flashdecode_split_count,
        native_q1_rope_cache_attention,
        native_q1_rope_cache_attention_flashdecode,
        preload_native_q1_attention,
    )
    from osuT5.osuT5.inference.optimized.scout.verifier import (
        verify_candidate_cache_behavior,
    )

    torch.cuda.synchronize()
    started = time.perf_counter()
    preload_native_q1_attention()
    torch.cuda.synchronize()
    preload_seconds = time.perf_counter() - started

    buckets: dict[str, Any] = {}
    for prefix, accepted in entries.items():
        capture = _capture_representative_layer(
            model, accepted["static_inputs"], prefix=prefix
        )
        qkv, keys, values, cos, sin, mask = _attention_inputs(capture)
        snapshots = _all_cache_snapshots(capture.past_key_value, capture.cache_position)
        splits = flashdecode_split_count(int(prefix))

        def accepted_call() -> torch.Tensor:
            return native_q1_rope_cache_attention(
                qkv, keys, values, cos, sin, capture.cache_position, mask, prefix
            )

        def candidate_call() -> torch.Tensor:
            return native_q1_rope_cache_attention_flashdecode(
                qkv, keys, values, cos, sin, capture.cache_position, mask, prefix
            )

        calls = {"accepted": accepted_call, "flashdecode": candidate_call}
        verifiers = {
            name: verify_candidate_cache_behavior(
                capture.past_key_value,
                layer_idx=capture.layer_idx,
                cache_position=capture.cache_position,
                candidate=call,
                repeats=2,
            )
            for name, call in calls.items()
        }
        graphs: dict[str, CapturedGraph] = {}
        for name, call in calls.items():
            _restore_all_cache(capture.past_key_value, snapshots)
            graphs[name] = _capture_cuda_graph(
                call, context=lambda: _null_context(), warmup=0
            )
        timings, rounds, memory = _reciprocal_graph_rounds(
            {name: graph.graph for name, graph in graphs.items()},
            restore=lambda: _restore_all_cache(capture.past_key_value, snapshots),
            warmup=warmup,
            iters=iters,
        )
        observations = {
            name: _observe_prefix_graph(graph, capture=capture)
            for name, graph in graphs.items()
        }
        reference = observations["accepted"]
        candidate = observations["flashdecode"]
        buckets[str(prefix)] = {
            "decode_replays": int(accepted["decode_replays"]),
            "flashdecode_split_count": splits,
            "accepted_ms_per_call": timings["accepted"],
            "candidate_ms_per_call": timings["flashdecode"],
            "accepted_memory_stable": bool(memory["accepted"]),
            "candidate_memory_stable": bool(memory["flashdecode"]),
            "accepted_verifier": verifiers["accepted"],
            "candidate_verifier": verifiers["flashdecode"],
            "drift": {
                "output_max_abs": _max_abs(reference.output, candidate.output),
                "cache_key_slot_max_abs": _max_abs(
                    reference.key_slot, candidate.key_slot
                ),
                "cache_value_slot_max_abs": _max_abs(
                    reference.value_slot, candidate.value_slot
                ),
            },
            "rounds": rounds,
        }
        _restore_all_cache(capture.past_key_value, snapshots)

    return {
        "schema_version": 1,
        "metadata": {
            "candidate": "sm75_split_kv_flashdecode_occupancy",
            "hypothesis": (
                "long-prefix decode occupancy improves when split-KV grid uses "
                "FlashDecode-style adaptive {8,16,32} counts vs fixed 8"
            ),
            "exactness_claim": False,
            "base_tip_choice": BASE_TIP_CHOICE,
            "commit": _git_head(),
            "precision": "fp32",
            "bucket_mode": "all",
            "decoder_layers": DECODER_LAYERS,
            "warmup": warmup,
            "iters": iters,
            "extension_preload_seconds": preload_seconds,
            "device_name": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
        "fixed_main_work": work,
        "summary": summarize_component(
            buckets, total_replays=int(work["decode_replays"])
        ),
        "buckets": buckets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("overrides", nargs="*")
    cli = parser.parse_args()
    overrides = list(cli.overrides)
    if not any(value.startswith("output_path=") for value in overrides):
        overrides.append(f"output_path={cli.output_path}")
    args = _load_args(cli.config_name, overrides)
    report = profile_component(
        args,
        output_path=cli.output_path,
        warmup=cli.warmup,
        iters=cli.iters,
    )
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if not report["summary"]["promotion_pass"]:
        raise SystemExit("STOP_FLASHDECODE_OCCUPANCY_COMPONENT")


if __name__ == "__main__":
    main()
