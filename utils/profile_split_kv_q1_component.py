"""Profile the opt-in SM75 split-KV q1 kernel on real SALVALAI tensors."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from utils.profile_native_prefix_dtype_scout import (  # noqa: E402
    CapturedGraph,
    SENTINEL_BUCKETS,
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
    select_buckets,
    validate_accepted_graph_cache,
)


DECODER_LAYERS = 12
MAX_ABS_DRIFT = 1e-3
REQUIRED_LOCAL_SPEEDUP = 2.0
REQUIRED_MAIN_SAVING_SECONDS = 1.503


def _git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def split_counts_for_prefix(prefix: int) -> tuple[int, ...]:
    if prefix <= 0:
        raise ValueError("prefix must be positive")
    return (1,) if prefix == 128 else (2, 4, 8)


def _candidate_pass(entry: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    verifier = entry.get("verifier")
    if not isinstance(verifier, dict) or not verifier.get("pass"):
        failures.append("cache_verifier")
    if not bool(entry.get("memory_stable")):
        failures.append("memory_stable")
    drift = entry.get("drift")
    if not isinstance(drift, dict):
        failures.append("drift_schema")
    else:
        for name in ("output_max_abs", "cache_key_slot_max_abs", "cache_value_slot_max_abs"):
            value = drift.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                failures.append(f"{name}_invalid")
            elif float(value) > MAX_ABS_DRIFT:
                failures.append(f"{name}_above_{MAX_ABS_DRIFT}")
    return not failures, failures


def summarize_split_kv(
    buckets: dict[str, dict[str, Any]],
    *,
    total_replays: int,
) -> dict[str, Any]:
    if not buckets or total_replays <= 0:
        raise ValueError("summary requires buckets and a positive total replay count")
    measured_replays = 0
    accepted_weighted_ms = 0.0
    candidate_weighted_ms = 0.0
    selected: dict[str, Any] = {}
    failures: dict[str, Any] = {}
    for raw_prefix, bucket in sorted(buckets.items(), key=lambda item: int(item[0])):
        count = int(bucket["decode_replays"])
        accepted_ms = float(bucket["accepted_ms_per_call"])
        if count <= 0 or accepted_ms <= 0 or not math.isfinite(accepted_ms):
            raise ValueError(f"bucket {raw_prefix} has invalid count/timing")
        measured_replays += count
        accepted_weighted_ms += count * accepted_ms
        passing: list[tuple[float, str, dict[str, Any]]] = []
        bucket_failures: dict[str, list[str]] = {}
        candidates = bucket.get("candidates")
        if not isinstance(candidates, dict) or not candidates:
            raise ValueError(f"bucket {raw_prefix} has no candidates")
        for name, entry in candidates.items():
            candidate_ms = float(entry["ms_per_call"])
            if candidate_ms <= 0 or not math.isfinite(candidate_ms):
                raise ValueError(f"bucket {raw_prefix} candidate {name} has invalid timing")
            passed, candidate_failures = _candidate_pass(entry)
            if passed:
                passing.append((candidate_ms, name, entry))
            else:
                bucket_failures[name] = candidate_failures
        if not passing:
            failures[raw_prefix] = bucket_failures
            candidate_weighted_ms += count * accepted_ms
            continue
        best_ms, best_name, _ = min(passing, key=lambda item: item[0])
        candidate_weighted_ms += count * best_ms
        selected[raw_prefix] = {
            "candidate": best_name,
            "accepted_ms_per_call": accepted_ms,
            "candidate_ms_per_call": best_ms,
            "local_speedup": accepted_ms / best_ms,
            "decode_replays": count,
        }
        if bucket_failures:
            failures[raw_prefix] = bucket_failures

    accepted_seconds = DECODER_LAYERS * accepted_weighted_ms / 1000.0
    candidate_seconds = DECODER_LAYERS * candidate_weighted_ms / 1000.0
    saving_seconds = accepted_seconds - candidate_seconds
    weighted_speedup = (
        accepted_weighted_ms / candidate_weighted_ms
        if candidate_weighted_ms > 0
        else math.inf
    )
    correctness_pass = len(selected) == len(buckets)
    return {
        "measured_replays": measured_replays,
        "total_replays": int(total_replays),
        "coverage_fraction": measured_replays / total_replays,
        "weighted_accepted_seconds_12_layers": accepted_seconds,
        "weighted_candidate_seconds_12_layers": candidate_seconds,
        "projected_main_saving_seconds": saving_seconds,
        "weighted_local_speedup": weighted_speedup,
        "required_local_speedup": REQUIRED_LOCAL_SPEEDUP,
        "required_main_saving_seconds": REQUIRED_MAIN_SAVING_SECONDS,
        "correctness_pass": correctness_pass,
        "sizing_pass": bool(
            weighted_speedup >= REQUIRED_LOCAL_SPEEDUP
            and saving_seconds >= REQUIRED_MAIN_SAVING_SECONDS
        ),
        "promotion_pass": bool(
            correctness_pass
            and weighted_speedup >= REQUIRED_LOCAL_SPEEDUP
            and saving_seconds >= REQUIRED_MAIN_SAVING_SECONDS
        ),
        "selected": selected,
        "candidate_failures": failures,
        "unmeasured_buckets_assigned_zero_delta": True,
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


def _observe(
    graph: CapturedGraph,
    *,
    capture,
):
    return _observe_prefix_graph(graph, capture=capture)


@torch.no_grad()
def profile_split_kv_component(
    args,
    *,
    output_path: Path,
    bucket_mode: str,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("split-KV q1 profiler requires CUDA")
    if torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError("split-KV q1 profiler requires an SM75 GPU")
    if warmup < 1 or iters < 1:
        raise ValueError("warmup and iters must be positive")
    _assert_scout_args(args)
    output_path.mkdir(parents=True, exist_ok=True)
    run = _accepted_main_session_run(args, output_path=output_path)
    model = run["model"]
    model.eval()
    accepted_entries = validate_accepted_graph_cache(run["session"].graph_cache)
    selected_entries = select_buckets(accepted_entries, bucket_mode)
    total_replays = sum(int(entry["decode_replays"]) for entry in accepted_entries.values())

    from osuT5.osuT5.inference.optimized.kernels.q1_attention import (
        native_q1_rope_cache_attention,
        preload_native_q1_attention,
    )
    from osuT5.osuT5.inference.optimized.scout.split_kv_q1 import (
        preload_split_kv_q1,
        split_kv_q1_rope_cache_attention,
    )
    from osuT5.osuT5.inference.optimized.scout.verifier import (
        verify_candidate_cache_behavior,
    )

    torch.cuda.synchronize()
    preload_started = time.perf_counter()
    preload_native_q1_attention()
    preload_split_kv_q1()
    torch.cuda.synchronize()
    preload_seconds = time.perf_counter() - preload_started

    buckets: dict[str, Any] = {}
    for prefix, accepted in selected_entries.items():
        capture = _capture_representative_layer(model, accepted["static_inputs"], prefix=prefix)
        qkv, keys, values, cos, sin, mask = _attention_inputs(capture)
        snapshots = _all_cache_snapshots(capture.past_key_value, capture.cache_position)

        def accepted_call() -> torch.Tensor:
            return native_q1_rope_cache_attention(
                qkv, keys, values, cos, sin, capture.cache_position, mask, prefix
            )

        calls: dict[str, Callable[[], torch.Tensor]] = {"accepted": accepted_call}
        for split_count in split_counts_for_prefix(prefix):
            calls[f"split_{split_count}"] = (
                lambda split_count=split_count: split_kv_q1_rope_cache_attention(
                    qkv,
                    keys,
                    values,
                    cos,
                    sin,
                    capture.cache_position,
                    mask,
                    prefix,
                    split_count,
                )
            )

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
            graphs[name] = _capture_cuda_graph(call, context=lambda: _null_context(), warmup=0)
        timings, rounds, memory = _reciprocal_graph_rounds(
            {name: graph.graph for name, graph in graphs.items()},
            restore=lambda: _restore_all_cache(capture.past_key_value, snapshots),
            warmup=warmup,
            iters=iters,
        )
        observations = {
            name: _observe(graph, capture=capture) for name, graph in graphs.items()
        }
        reference = observations["accepted"]
        candidates: dict[str, Any] = {}
        for name, observation in observations.items():
            if name == "accepted":
                continue
            candidates[name] = {
                "ms_per_call": timings[name],
                "capture_seconds": graphs[name].setup_seconds,
                "peak_vram_bytes": graphs[name].peak_vram_bytes,
                "memory_stable": bool(memory[name]),
                "verifier": verifiers[name],
                "drift": {
                    "output_max_abs": _max_abs(reference.output, observation.output),
                    "cache_key_slot_max_abs": _max_abs(reference.key_slot, observation.key_slot),
                    "cache_value_slot_max_abs": _max_abs(reference.value_slot, observation.value_slot),
                },
            }
        buckets[str(prefix)] = {
            "decode_replays": int(accepted["decode_replays"]),
            "cache_position": int(capture.cache_position.item()),
            "accepted_ms_per_call": timings["accepted"],
            "accepted_memory_stable": bool(memory["accepted"]),
            "accepted_verifier": verifiers["accepted"],
            "candidates": candidates,
            "rounds": rounds,
        }
        _restore_all_cache(capture.past_key_value, snapshots)

    return {
        "schema_version": 1,
        "metadata": {
            "candidate": "sm75_split_kv_q1_rope_cache_attention",
            "commit": _git_head(),
            "precision": "fp32",
            "bucket_mode": bucket_mode,
            "sentinel_buckets": list(SENTINEL_BUCKETS),
            "decoder_layers": DECODER_LAYERS,
            "warmup": warmup,
            "iters": iters,
            "extension_preload_seconds": preload_seconds,
            "device_name": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
        "summary": summarize_split_kv(buckets, total_replays=total_replays),
        "buckets": buckets,
    }


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--bucket-mode", choices=("sentinel", "all"), default="sentinel")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("overrides", nargs="*")
    cli = parser.parse_args()
    overrides = list(cli.overrides)
    if not any(value.startswith("output_path=") for value in overrides):
        overrides.append(f"output_path={cli.output_path}")
    args = _load_args(cli.config_name, overrides)
    report = profile_split_kv_component(
        args,
        output_path=cli.output_path,
        bucket_mode=cli.bucket_mode,
        warmup=cli.warmup,
        iters=cli.iters,
    )
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if not report["summary"]["promotion_pass"]:
        raise SystemExit("STOP_SPLIT_KV_COMPONENT")


if __name__ == "__main__":
    main()
