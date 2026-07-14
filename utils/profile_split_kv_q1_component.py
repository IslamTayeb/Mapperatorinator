"""Profile the opt-in SM75 split-KV q1 kernel on real SALVALAI tensors."""

from __future__ import annotations

import argparse
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
    _restore_all_cache,
    _time_graph,
)


DECODER_LAYERS = 12
MAX_ABS_DRIFT = 1e-3
SENTINEL_BUCKETS = (128, 576, 640)
FIXED_MAIN_GENERATED_TOKENS = 8_294
FIXED_MAIN_DECODE_REPLAYS = 8_207
FIXED_MAIN_WINDOWS = 87
DEFAULT_TIMING_CYCLES = 5
TIMING_MAD_SCALE = 1.4826
TIMING_MAD_MULTIPLIER = 3.0
TIMING_NOISE_ABSOLUTE_MS = 0.00025
TIMING_NOISE_RELATIVE_FRACTION = 0.01


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


def validate_live_graph_cache(
    graph_cache: dict[Any, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Validate the graph manifest without imposing a stale bucket schedule."""

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


def select_live_buckets(
    entries: dict[int, dict[str, Any]],
    mode: str,
) -> dict[int, dict[str, Any]]:
    if mode == "all":
        return dict(entries)
    if mode != "sentinel":
        raise ValueError("bucket mode must be 'sentinel' or 'all'")
    missing = sorted(set(SENTINEL_BUCKETS) - set(entries))
    if missing:
        raise ValueError(f"live graph cache is missing sentinel buckets {missing}")
    return {prefix: entries[prefix] for prefix in SENTINEL_BUCKETS}


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
    if generated_tokens != FIXED_MAIN_GENERATED_TOKENS:
        raise RuntimeError(
            "fixed SALVALAI main token count changed: "
            f"expected {FIXED_MAIN_GENERATED_TOKENS}, got {generated_tokens}"
        )
    if decode_replays != FIXED_MAIN_DECODE_REPLAYS:
        raise RuntimeError(
            "fixed SALVALAI decode replay count changed: "
            f"expected {FIXED_MAIN_DECODE_REPLAYS}, got {decode_replays}"
        )
    if windows != FIXED_MAIN_WINDOWS:
        raise RuntimeError(
            "fixed SALVALAI prefill/window reconciliation changed: "
            f"expected {FIXED_MAIN_WINDOWS}, got {windows}"
        )
    return {
        "generated_tokens": generated_tokens,
        "decode_replays": decode_replays,
        "prefill_windows": windows,
        "reconciled": generated_tokens == decode_replays + windows,
        "bucket_counts": {
            str(prefix): int(entry["decode_replays"])
            for prefix, entry in entries.items()
        },
    }


def validate_split_mask(
    mask: torch.Tensor | None,
    *,
    prefix: int,
    split_counts: tuple[int, ...],
) -> None:
    """Reject a measured split whose complete mask would have zero mass."""

    if mask is None:
        return
    flattened = mask.reshape(-1)
    if flattened.numel() != prefix:
        raise ValueError("attention mask must match the active prefix")
    for split_count in split_counts:
        for split in range(split_count):
            begin = prefix * split // split_count
            end = prefix * (split + 1) // split_count
            if bool(torch.isneginf(flattened[begin:end]).all().item()):
                raise RuntimeError(
                    "split-KV profiler rejects an all-masked partition "
                    f"(prefix={prefix}, split_count={split_count}, split={split})"
                )


def _median(values: list[float], *, name: str) -> float:
    if not values or not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError(f"{name} must contain positive finite timings")
    return float(statistics.median(values))


def _paired_graph_rounds(
    graphs: dict[str, torch.cuda.CUDAGraph],
    *,
    restore: Callable[[], None],
    warmup: int,
    iters: int,
    cycles: int,
) -> tuple[
    dict[str, float],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    dict[str, bool],
]:
    """Measure every candidate against accepted in repeated symmetric pairs."""

    if "accepted" not in graphs or len(graphs) < 2:
        raise ValueError("paired timing requires accepted and at least one candidate")
    if cycles < 1:
        raise ValueError("paired timing cycles must be positive")
    candidates = [name for name in graphs if name != "accepted"]
    raw_rounds: list[dict[str, Any]] = []
    paired: dict[str, dict[str, Any]] = {}
    memory_stable = {name: True for name in graphs}
    timings_by_variant: dict[str, list[float]] = {name: [] for name in graphs}

    for candidate in candidates:
        samples: list[dict[str, Any]] = []
        for cycle in range(cycles):
            patterns = (
                ("ABBA", ("accepted", candidate, candidate, "accepted")),
                ("BAAB", (candidate, "accepted", "accepted", candidate)),
            )
            for pattern_name, order in patterns:
                pattern_rows: list[dict[str, Any]] = []
                for position, variant in enumerate(order):
                    restore()
                    result = _time_graph(
                        graphs[variant],
                        warmup=warmup,
                        iters=iters,
                    )
                    memory_stable[variant] &= bool(result["memory_stable"])
                    timing = float(result["ms_per_call"])
                    timings_by_variant[variant].append(timing)
                    row = {
                        "candidate_pair": candidate,
                        "cycle": cycle,
                        "pattern": pattern_name,
                        "order": list(order),
                        "position": position,
                        "variant": variant,
                        **result,
                    }
                    pattern_rows.append(row)
                    raw_rounds.append(row)
                accepted_ms = statistics.mean(
                    float(row["ms_per_call"])
                    for row in pattern_rows
                    if row["variant"] == "accepted"
                )
                candidate_ms = statistics.mean(
                    float(row["ms_per_call"])
                    for row in pattern_rows
                    if row["variant"] == candidate
                )
                samples.append(
                    {
                        "cycle": cycle,
                        "pattern": pattern_name,
                        "accepted_ms_per_call": accepted_ms,
                        "candidate_ms_per_call": candidate_ms,
                        "delta_ms_per_call": accepted_ms - candidate_ms,
                        "speedup": accepted_ms / candidate_ms,
                    }
                )

        deltas = [float(sample["delta_ms_per_call"]) for sample in samples]
        median_delta = float(statistics.median(deltas))
        mad = float(statistics.median(abs(delta - median_delta) for delta in deltas))
        accepted_median = _median(
            [float(sample["accepted_ms_per_call"]) for sample in samples],
            name=f"{candidate} accepted pair timings",
        )
        candidate_median = _median(
            [float(sample["candidate_ms_per_call"]) for sample in samples],
            name=f"{candidate} candidate pair timings",
        )
        noise_floor = max(
            TIMING_NOISE_ABSOLUTE_MS,
            accepted_median * TIMING_NOISE_RELATIVE_FRACTION,
        )
        robust_lower = median_delta - (TIMING_MAD_MULTIPLIER * TIMING_MAD_SCALE * mad)
        conservative_lower = robust_lower - noise_floor
        paired[candidate] = {
            "samples": samples,
            "accepted_median_ms_per_call": accepted_median,
            "candidate_median_ms_per_call": candidate_median,
            "median_paired_delta_ms_per_call": median_delta,
            "paired_delta_mad_ms_per_call": mad,
            "robust_delta_lower_bound_ms_per_call": robust_lower,
            "noise_floor_ms_per_call": noise_floor,
            "conservative_delta_lower_bound_ms_per_call": conservative_lower,
            "faster_than_noise": conservative_lower > 0,
        }

    timings = {
        name: _median(values, name=f"{name} raw timings")
        for name, values in timings_by_variant.items()
    }
    return timings, paired, raw_rounds, memory_stable


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
                failures.append(f"{name}_invalid")
            elif float(value) > MAX_ABS_DRIFT:
                failures.append(f"{name}_above_{MAX_ABS_DRIFT}")
    return not failures, failures


def summarize_split_kv(
    buckets: dict[str, dict[str, Any]],
    *,
    total_replays: int,
    require_full_coverage: bool = False,
) -> dict[str, Any]:
    if not buckets or total_replays <= 0:
        raise ValueError("summary requires buckets and a positive total replay count")
    measured_replays = 0
    accepted_weighted_ms = 0.0
    candidate_weighted_ms = 0.0
    selected: dict[str, Any] = {}
    failures: dict[str, Any] = {}
    for raw_prefix, bucket in sorted(buckets.items(), key=lambda item: int(item[0])):
        accepted_verifier = bucket.get("accepted_verifier")
        if not isinstance(accepted_verifier, dict) or not bool(
            accepted_verifier.get("pass")
        ):
            raise ValueError(
                f"bucket {raw_prefix} accepted cache verifier did not pass"
            )
        if not bool(bucket.get("accepted_memory_stable")):
            raise ValueError(
                f"bucket {raw_prefix} accepted control was not memory stable"
            )
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
                raise ValueError(
                    f"bucket {raw_prefix} candidate {name} has invalid timing"
                )
            passed, candidate_failures = _candidate_pass(entry)
            paired_timing = entry.get("paired_timing")
            if not isinstance(paired_timing, dict):
                raise ValueError(
                    f"bucket {raw_prefix} candidate {name} is missing paired timing"
                )
            conservative_delta = paired_timing.get(
                "conservative_delta_lower_bound_ms_per_call"
            )
            if (
                isinstance(conservative_delta, bool)
                or not isinstance(conservative_delta, (int, float))
                or not math.isfinite(float(conservative_delta))
            ):
                raise ValueError(
                    f"bucket {raw_prefix} candidate {name} has invalid paired delta"
                )
            faster_than_noise = bool(paired_timing.get("faster_than_noise"))
            if not faster_than_noise:
                candidate_failures.append("not_faster_than_noise")
            if passed and faster_than_noise and float(conservative_delta) > 0:
                passing.append((float(conservative_delta), name, entry))
            if candidate_failures:
                bucket_failures[name] = candidate_failures
        if not passing:
            candidate_weighted_ms += count * accepted_ms
            selected[raw_prefix] = {
                "candidate": "accepted",
                "accepted_ms_per_call": accepted_ms,
                "candidate_ms_per_call": accepted_ms,
                "measured_candidate_ms_per_call": None,
                "local_speedup": 1.0,
                "conservative_delta_lower_bound_ms_per_call": 0.0,
                "decode_replays": count,
                "fallback": True,
            }
            if bucket_failures:
                failures[raw_prefix] = bucket_failures
            continue
        best_delta, best_name, best_entry = max(passing, key=lambda item: item[0])
        best_ms = float(best_entry["ms_per_call"])
        effective_candidate_ms = accepted_ms - best_delta
        if effective_candidate_ms <= 0:
            raise ValueError(
                f"bucket {raw_prefix} conservative delta exceeds accepted timing"
            )
        candidate_weighted_ms += count * effective_candidate_ms
        selected[raw_prefix] = {
            "candidate": best_name,
            "accepted_ms_per_call": accepted_ms,
            "candidate_ms_per_call": effective_candidate_ms,
            "measured_candidate_ms_per_call": best_ms,
            "local_speedup": accepted_ms / best_ms,
            "conservative_local_speedup": accepted_ms / effective_candidate_ms,
            "conservative_delta_lower_bound_ms_per_call": best_delta,
            "decode_replays": count,
            "fallback": False,
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
    coverage_pass = measured_replays == total_replays
    if measured_replays > total_replays:
        raise ValueError("measured replay count exceeds the live total")
    correctness_pass = len(selected) == len(buckets)
    component_pass = bool(
        correctness_pass
        and (coverage_pass or not require_full_coverage)
        and saving_seconds > 0
    )
    return {
        "measured_replays": measured_replays,
        "total_replays": int(total_replays),
        "coverage_fraction": measured_replays / total_replays,
        "weighted_accepted_seconds_12_layers": accepted_seconds,
        "weighted_candidate_seconds_12_layers": candidate_seconds,
        "projected_main_saving_seconds": saving_seconds,
        "weighted_local_speedup": weighted_speedup,
        "coverage_pass": coverage_pass,
        "correctness_pass": correctness_pass,
        "component_pass": component_pass,
        "sizing_pass": saving_seconds > 0,
        "promotion_pass": component_pass,
        "selected": selected,
        "candidate_failures": failures,
        "unmeasured_buckets_assigned_zero_delta": not coverage_pass,
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
    timing_cycles: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("split-KV q1 profiler requires CUDA")
    if torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError("split-KV q1 profiler requires an SM75 GPU")
    if warmup < 1 or iters < 1 or timing_cycles < 1:
        raise ValueError("warmup, iters, and timing cycles must be positive")
    _assert_scout_args(args)
    output_path.mkdir(parents=True, exist_ok=True)
    run = _accepted_main_session_run(args, output_path=output_path)
    model = run["model"]
    model.eval()
    accepted_entries = validate_live_graph_cache(run["session"].graph_cache)
    work_manifest = validate_fixed_main_work(run, accepted_entries)
    selected_entries = select_live_buckets(accepted_entries, bucket_mode)
    total_replays = int(work_manifest["decode_replays"])

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
        capture = _capture_representative_layer(
            model, accepted["static_inputs"], prefix=prefix
        )
        qkv, keys, values, cos, sin, mask = _attention_inputs(capture)
        split_counts = split_counts_for_prefix(prefix)
        validate_split_mask(
            mask,
            prefix=prefix,
            split_counts=split_counts,
        )
        snapshots = _all_cache_snapshots(capture.past_key_value, capture.cache_position)

        def accepted_call() -> torch.Tensor:
            return native_q1_rope_cache_attention(
                qkv, keys, values, cos, sin, capture.cache_position, mask, prefix
            )

        calls: dict[str, Callable[[], torch.Tensor]] = {"accepted": accepted_call}
        for split_count in split_counts:
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
            graphs[name] = _capture_cuda_graph(
                call, context=lambda: _null_context(), warmup=0
            )
        timings, paired_timings, rounds, memory = _paired_graph_rounds(
            {name: graph.graph for name, graph in graphs.items()},
            restore=lambda: _restore_all_cache(capture.past_key_value, snapshots),
            warmup=warmup,
            iters=iters,
            cycles=timing_cycles,
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
                "paired_timing": paired_timings[name],
                "capture_seconds": graphs[name].setup_seconds,
                "peak_vram_bytes": graphs[name].peak_vram_bytes,
                "memory_stable": bool(memory[name]),
                "verifier": verifiers[name],
                "drift": {
                    "output_max_abs": _max_abs(reference.output, observation.output),
                    "cache_key_slot_max_abs": _max_abs(
                        reference.key_slot, observation.key_slot
                    ),
                    "cache_value_slot_max_abs": _max_abs(
                        reference.value_slot, observation.value_slot
                    ),
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
        "schema_version": 2,
        "metadata": {
            "candidate": "sm75_split_kv_q1_rope_cache_attention",
            "commit": _git_head(),
            "precision": "fp32",
            "bucket_mode": bucket_mode,
            "sentinel_buckets": list(SENTINEL_BUCKETS),
            "decoder_layers": DECODER_LAYERS,
            "warmup": warmup,
            "iters": iters,
            "timing_cycles": timing_cycles,
            "timing_patterns": ["ABBA", "BAAB"],
            "timing_noise_absolute_ms": TIMING_NOISE_ABSOLUTE_MS,
            "timing_noise_relative_fraction": TIMING_NOISE_RELATIVE_FRACTION,
            "timing_mad_scale": TIMING_MAD_SCALE,
            "timing_mad_multiplier": TIMING_MAD_MULTIPLIER,
            "selection_policy": "maximum_positive_conservative_paired_delta_or_accepted",
            "extension_preload_seconds": preload_seconds,
            "device_name": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
        "fixed_main_work": work_manifest,
        "summary": summarize_split_kv(
            buckets,
            total_replays=total_replays,
            require_full_coverage=bucket_mode == "all",
        ),
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
    parser.add_argument("--bucket-mode", choices=("sentinel", "all"), default="all")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--timing-cycles", type=int, default=DEFAULT_TIMING_CYCLES)
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
        timing_cycles=cli.timing_cycles,
    )
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if not report["summary"]["promotion_pass"]:
        raise SystemExit("STOP_SPLIT_KV_COMPONENT")


if __name__ == "__main__":
    main()
