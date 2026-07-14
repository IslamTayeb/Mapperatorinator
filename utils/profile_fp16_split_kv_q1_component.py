"""Profile FP16-storage SM75 split-KV q1 self-attention on live tensors."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import statistics
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
    _accepted_main_session_run,
    _all_cache_snapshots,
    _cache_tensor,
    _capture_cuda_graph,
    _capture_representative_layer,
    _load_args,
    _max_abs,
    _observe_prefix_graph,
    _restore_all_cache,
    _time_graph,
)


SENTINEL_BUCKETS = (128, 576, 640)
DECODER_LAYERS = 12
MAX_ABS_DRIFT = 1e-3
DEFAULT_TIMING_CYCLES = 5


def _git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_fp16_args(args: Any) -> None:
    requirements = {
        "inference_engine": "optimized",
        "precision": "fp16",
        "attn_implementation": "sdpa",
        "device": "cuda",
        "use_server": False,
        "parallel": False,
        "cfg_scale": 1.0,
        "num_beams": 1,
    }
    failures = {
        name: {"expected": expected, "actual": getattr(args, name, None)}
        for name, expected in requirements.items()
        if getattr(args, name, None) != expected
    }
    if failures:
        raise ValueError(f"FP16 split-KV component requires {requirements}; got {failures}")


def validate_live_graph_cache(
    graph_cache: dict[Any, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    if not isinstance(graph_cache, dict) or not graph_cache:
        raise ValueError("live graph cache must be a non-empty mapping")
    entries: dict[int, dict[str, Any]] = {}
    for raw in graph_cache.values():
        if not isinstance(raw, dict):
            raise TypeError("live graph cache entries must be mappings")
        prefix = raw.get("active_prefix_length")
        count = raw.get("decode_replays")
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
            name for name in ("graph", "outputs", "static_inputs") if name not in raw
        ]
        if missing:
            raise ValueError(f"live prefix {prefix} is missing {missing}")
        entries[prefix] = raw
    missing_sentinels = sorted(set(SENTINEL_BUCKETS) - set(entries))
    if missing_sentinels:
        raise ValueError(f"live graph cache is missing sentinel buckets {missing_sentinels}")
    return dict(sorted(entries.items()))


def _attention_inputs(capture: Any) -> tuple[torch.Tensor, ...]:
    from osuT5.osuT5.inference.optimized.scout.native_prefix import _trim_mask

    module = capture.module
    self_attention = module.self_attn
    normalized = module.self_attn_layer_norm(capture.hidden_states)
    qkv = self_attention.Wqkv(normalized).view(
        1,
        1,
        3,
        self_attention.num_heads,
        self_attention.head_dim,
    )
    cos, sin = self_attention.rotary_emb(qkv, position_ids=capture.position_ids)
    keys = _cache_tensor(capture.past_key_value, "self", capture.layer_idx, "keys")
    values = _cache_tensor(capture.past_key_value, "self", capture.layer_idx, "values")
    mask = _trim_mask(capture.attention_mask, capture.active_prefix_length)
    tensors = (qkv, keys, values, cos, sin)
    if any(tensor.dtype != torch.float16 for tensor in tensors):
        raise TypeError(
            "FP16 split-KV component requires FP16 qkv/cache/cos/sin storage"
        )
    return qkv, keys, values, cos, sin, mask


def reciprocal_timings(
    graphs: dict[str, torch.cuda.CUDAGraph],
    *,
    restore: Callable[[], None],
    warmup: int,
    iters: int,
    cycles: int,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, bool]]:
    if set(graphs) != {"accepted_fp16", "split_kv_8_fp16"}:
        raise ValueError("reciprocal timing requires accepted and split FP16 graphs")
    if min(warmup, iters, cycles) <= 0:
        raise ValueError("warmup, iters, and cycles must be positive")
    rows: list[dict[str, Any]] = []
    memory_stable = {name: True for name in graphs}
    for cycle in range(cycles):
        order = (
            ("accepted_fp16", "split_kv_8_fp16")
            if cycle % 2 == 0
            else ("split_kv_8_fp16", "accepted_fp16")
        )
        for variant in order:
            restore()
            result = _time_graph(graphs[variant], warmup=warmup, iters=iters)
            memory_stable[variant] &= bool(result["memory_stable"])
            rows.append(
                {
                    "cycle": cycle,
                    "order": list(order),
                    "variant": variant,
                    **result,
                }
            )
    medians = {
        name: float(
            statistics.median(
                row["ms_per_call"] for row in rows if row["variant"] == name
            )
        )
        for name in graphs
    }
    return medians, rows, memory_stable


def summarize(
    buckets: dict[str, dict[str, Any]],
    *,
    live_counts: dict[int, int],
) -> dict[str, Any]:
    if set(map(int, buckets)) != set(SENTINEL_BUCKETS):
        raise ValueError("component report must cover exactly the sentinel buckets")
    total_replays = sum(live_counts.values())
    measured_replays = sum(live_counts[prefix] for prefix in SENTINEL_BUCKETS)
    accepted_ms = 0.0
    selected_ms = 0.0
    failures: dict[str, list[str]] = {}
    selected: dict[str, str] = {}
    for raw_prefix, entry in buckets.items():
        prefix = int(raw_prefix)
        count = live_counts[prefix]
        accepted = float(entry["accepted_fp16"]["ms_per_call"])
        if not math.isfinite(accepted) or accepted <= 0:
            raise ValueError(f"prefix {prefix} has invalid accepted timing")
        accepted_ms += count * accepted
        split = entry.get("split_kv_8_fp16")
        if split is None:
            if prefix != 128 or entry.get("selector") != "accepted_fallback":
                raise ValueError(f"prefix {prefix} is missing its split candidate")
            selected_ms += count * accepted
            selected[raw_prefix] = "accepted_fallback"
            continue
        candidate = float(split["ms_per_call"])
        drift = split.get("drift")
        candidate_failures: list[str] = []
        if not math.isfinite(candidate) or candidate <= 0:
            raise ValueError(f"prefix {prefix} has invalid split timing")
        if not split.get("cache_verifier", {}).get("pass"):
            candidate_failures.append("cache_verifier")
        if not split.get("memory_stable"):
            candidate_failures.append("memory_stable")
        if not isinstance(drift, dict):
            candidate_failures.append("drift_schema")
        else:
            for name in (
                "output_max_abs",
                "cache_key_slot_max_abs",
                "cache_value_slot_max_abs",
            ):
                value = drift.get(name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0
                ):
                    candidate_failures.append(f"{name}_invalid")
                elif float(value) > MAX_ABS_DRIFT:
                    candidate_failures.append(f"{name}>{MAX_ABS_DRIFT}")
        if candidate >= accepted:
            candidate_failures.append("not_faster")
        if candidate_failures:
            failures[raw_prefix] = candidate_failures
            selected_ms += count * accepted
            selected[raw_prefix] = "accepted_fallback"
        else:
            selected_ms += count * candidate
            selected[raw_prefix] = "split_kv_8_fp16"
    accepted_seconds = DECODER_LAYERS * accepted_ms / 1000.0
    selected_seconds = DECODER_LAYERS * selected_ms / 1000.0
    saving_seconds = accepted_seconds - selected_seconds
    return {
        "total_live_replays": total_replays,
        "measured_replays": measured_replays,
        "coverage_fraction": measured_replays / total_replays,
        "weighted_accepted_fp16_seconds_12_layers": accepted_seconds,
        "weighted_selected_fp16_seconds_12_layers": selected_seconds,
        "projected_main_saving_seconds": saving_seconds,
        "selected": selected,
        "candidate_failures": failures,
        "unmeasured_buckets_assigned_zero_delta": True,
        "correctness_pass": not failures,
        "promotion_pass": not failures and saving_seconds > 0,
    }


@torch.no_grad()
def profile_fp16_split_kv_component(
    args: Any,
    *,
    output_path: Path,
    warmup: int,
    iters: int,
    timing_cycles: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("FP16 split-KV component requires CUDA")
    if torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError("FP16 split-KV component requires SM75")
    if min(warmup, iters, timing_cycles) <= 0:
        raise ValueError("warmup, iters, and timing cycles must be positive")
    validate_fp16_args(args)
    output_path.mkdir(parents=True, exist_ok=True)
    run = _accepted_main_session_run(args, output_path=output_path)
    model = run["model"]
    model.eval()
    entries = validate_live_graph_cache(run["session"].graph_cache)
    live_counts = {
        prefix: int(entry["decode_replays"]) for prefix, entry in entries.items()
    }

    from osuT5.osuT5.inference.optimized.kernels.q1_attention import (
        accepted_q1_rope_cache_attention,
        native_q1_rope_cache_attention_variant,
        preload_native_q1_attention,
        split_kv_q1_rope_cache_attention,
    )
    from osuT5.osuT5.inference.optimized.scout.verifier import (
        verify_candidate_cache_behavior,
    )

    torch.cuda.synchronize()
    preload_started = time.perf_counter()
    preload_native_q1_attention()
    torch.cuda.synchronize()
    preload_seconds = time.perf_counter() - preload_started

    buckets: dict[str, dict[str, Any]] = {}
    for prefix in SENTINEL_BUCKETS:
        accepted_entry = entries[prefix]
        capture = _capture_representative_layer(
            model,
            accepted_entry["static_inputs"],
            prefix=prefix,
        )
        qkv, keys, values, cos, sin, mask = _attention_inputs(capture)
        snapshots = _all_cache_snapshots(
            capture.past_key_value,
            capture.cache_position,
        )

        def accepted_call() -> torch.Tensor:
            return accepted_q1_rope_cache_attention(
                qkv,
                keys,
                values,
                cos,
                sin,
                capture.cache_position,
                mask,
                prefix,
            )

        selector = native_q1_rope_cache_attention_variant(qkv, prefix)
        calls: dict[str, Callable[[], torch.Tensor]] = {
            "accepted_fp16": accepted_call,
        }
        if selector == "split_kv_8":
            calls["split_kv_8_fp16"] = lambda: split_kv_q1_rope_cache_attention(
                qkv,
                keys,
                values,
                cos,
                sin,
                capture.cache_position,
                mask,
                prefix,
            )
        elif prefix != 128:
            raise RuntimeError(
                f"FP16 selector unexpectedly rejected measured prefix {prefix}"
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
        graphs = {}
        for name, call in calls.items():
            _restore_all_cache(capture.past_key_value, snapshots)
            graphs[name] = _capture_cuda_graph(call, context=nullcontext, warmup=0)

        if "split_kv_8_fp16" in graphs:
            timings, rounds, memory = reciprocal_timings(
                {name: graph.graph for name, graph in graphs.items()},
                restore=lambda: _restore_all_cache(
                    capture.past_key_value,
                    snapshots,
                ),
                warmup=warmup,
                iters=iters,
                cycles=timing_cycles,
            )
        else:
            _restore_all_cache(capture.past_key_value, snapshots)
            accepted_timing = _time_graph(
                graphs["accepted_fp16"].graph,
                warmup=warmup,
                iters=iters,
            )
            timings = {"accepted_fp16": float(accepted_timing["ms_per_call"])}
            rounds = [{"variant": "accepted_fp16", **accepted_timing}]
            memory = {
                "accepted_fp16": bool(accepted_timing["memory_stable"]),
            }

        observations = {
            name: _observe_prefix_graph(graph, capture=capture)
            for name, graph in graphs.items()
        }
        reference = observations["accepted_fp16"]
        bucket: dict[str, Any] = {
            "decode_replays": live_counts[prefix],
            "selector": selector if selector == "split_kv_8" else "accepted_fallback",
            "accepted_fp16": {
                "ms_per_call": timings["accepted_fp16"],
                "capture_seconds": graphs["accepted_fp16"].setup_seconds,
                "peak_vram_bytes": graphs["accepted_fp16"].peak_vram_bytes,
                "memory_stable": memory["accepted_fp16"],
                "cache_verifier": verifiers["accepted_fp16"],
            },
            "rounds": rounds,
        }
        if "split_kv_8_fp16" in graphs:
            candidate = observations["split_kv_8_fp16"]
            bucket["split_kv_8_fp16"] = {
                "ms_per_call": timings["split_kv_8_fp16"],
                "capture_seconds": graphs["split_kv_8_fp16"].setup_seconds,
                "peak_vram_bytes": graphs["split_kv_8_fp16"].peak_vram_bytes,
                "memory_stable": memory["split_kv_8_fp16"],
                "cache_verifier": verifiers["split_kv_8_fp16"],
                "drift": {
                    "output_max_abs": _max_abs(reference.output, candidate.output),
                    "cache_key_slot_max_abs": _max_abs(
                        reference.key_slot,
                        candidate.key_slot,
                    ),
                    "cache_value_slot_max_abs": _max_abs(
                        reference.value_slot,
                        candidate.value_slot,
                    ),
                },
            }
        buckets[str(prefix)] = bucket
        _restore_all_cache(capture.past_key_value, snapshots)

    return {
        "schema_version": 1,
        "metadata": {
            "candidate": "sm75_split_kv_8_fp16_storage_fp32_accumulation",
            "control": "accepted_unsplit_fp16_storage_fp32_accumulation",
            "commit": _git_head(),
            "precision": "fp16",
            "sentinel_buckets": list(SENTINEL_BUCKETS),
            "max_abs_drift": MAX_ABS_DRIFT,
            "warmup": warmup,
            "iters": iters,
            "timing_cycles": timing_cycles,
            "extension_preload_seconds": preload_seconds,
            "device_name": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
        "live_bucket_replay_counts": {
            str(prefix): count for prefix, count in live_counts.items()
        },
        "buckets": buckets,
        "summary": summarize(buckets, live_counts=live_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--timing-cycles", type=int, default=DEFAULT_TIMING_CYCLES)
    parser.add_argument("overrides", nargs="*")
    cli = parser.parse_args()
    overrides = list(cli.overrides)
    if not any(value.startswith("output_path=") for value in overrides):
        overrides.append(f"output_path={cli.output_path}")
    args = _load_args(cli.config_name, overrides)
    report = profile_fp16_split_kv_component(
        args,
        output_path=cli.output_path,
        warmup=cli.warmup,
        iters=cli.iters,
        timing_cycles=cli.timing_cycles,
    )
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if not report["summary"]["promotion_pass"]:
        raise SystemExit("STOP_FP16_SPLIT_KV_COMPONENT")


if __name__ == "__main__":
    main()
