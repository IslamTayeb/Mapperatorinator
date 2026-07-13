"""Profile accepted FP32 decoder regions on real SALVALAI graph inputs.

The harness first runs the unchanged accepted optimized engine to retain its
real one-token static graph inputs.  It then replays the original decoder
eagerly, with the accepted FP32 specialized attention dispatch enabled, under
the existing detail ranges and ``torch.profiler``.  This is a sizing tool: it
does not install a replacement decoder or candidate fusion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from utils.profile_native_prefix_dtype_scout import (  # noqa: E402
    SENTINEL_BUCKETS,
    _accepted_main_session_run,
    _all_cache_snapshots,
    _assert_scout_args,
    _cache_from_static_inputs,
    _load_args,
    _restore_all_cache,
    _time_graph,
    validate_accepted_graph_cache,
)


SCHEMA_VERSION = 1
REQUIRED_HEADROOM_SECONDS = 1.412
GRAPH_WARMUP = 100
GRAPH_ITERATIONS = 1_000
REGIONS = (
    "self_norm_qkv",
    "fused_self_attention",
    "self_out_residual",
    "cross_norm_q",
    "q1_bmm_cross",
    "cross_out_residual",
    "mlp",
    "final_norm_logits",
)

_ATTENTION_SUFFIXES = {
    ".self.qkv_proj": "self_norm_qkv",
    ".self.rope": "fused_self_attention",
    ".self.rope_cache_native_q1": "fused_self_attention",
    ".self.out_proj": "self_out_residual",
    ".cross.q_proj": "cross_norm_q",
    ".cross.sdpa": "q1_bmm_cross",
    ".cross.out_proj": "cross_out_residual",
}
_DECODER_LAYER_SUFFIXES = {
    ".self_attn_norm": "self_norm_qkv",
    ".cross_attn_norm": "cross_norm_q",
    ".mlp_norm": "mlp",
    ".mlp.fc1": "mlp",
    ".mlp.activation": "mlp",
    ".mlp.fc2": "mlp",
}
_FINAL_MARKERS = {
    "decoder.final_norm": "final_norm_logits",
    "decoder.output_projection": "final_norm_logits",
}


def classify_detail_range(name: str) -> str | None:
    """Map one non-overlapping leaf detail range to its decoder region."""

    normalized = name.removeprefix("mapperatorinator.")
    final = _FINAL_MARKERS.get(normalized)
    if final is not None:
        return final
    if normalized.startswith("attention.layer"):
        for suffix, region in _ATTENTION_SUFFIXES.items():
            if normalized.endswith(suffix):
                return region
    if normalized.startswith("decoder.layer"):
        for suffix, region in _DECODER_LAYER_SUFFIXES.items():
            if normalized.endswith(suffix):
                return region
    return None


def _device_time_total_us(event: Any) -> float:
    for attribute in ("device_time_total", "cuda_time_total"):
        value = getattr(event, attribute, None)
        if value is not None:
            parsed = float(value)
            if not math.isfinite(parsed) or parsed < 0:
                raise ValueError(
                    f"profiler event {getattr(event, 'key', '<unknown>')!r} has "
                    f"invalid {attribute}={value!r}"
                )
            return parsed
    raise RuntimeError(
        f"profiler event {getattr(event, 'key', '<unknown>')!r} exposes no "
        "device_time_total or cuda_time_total"
    )


def aggregate_region_times(
    events: Iterable[Any],
    *,
    iterations: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return region milliseconds/call and raw marker device microseconds."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    region_us: dict[str, float] = defaultdict(float)
    marker_us: dict[str, float] = defaultdict(float)
    for event in events:
        key = getattr(event, "key", None)
        if not isinstance(key, str):
            continue
        region = classify_detail_range(key)
        if region is None:
            continue
        elapsed_us = _device_time_total_us(event)
        region_us[region] += elapsed_us
        marker_us[key] += elapsed_us
    missing = [region for region in REGIONS if region_us[region] <= 0]
    if missing:
        raise RuntimeError(
            f"profiler did not record positive device time for {missing}"
        )
    return (
        {region: region_us[region] / iterations / 1_000.0 for region in REGIONS},
        dict(sorted(marker_us.items())),
    )


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _accepted_context(prefix: int, *, detail_ranges: bool):
    from osuT5.osuT5.runtime_profiling import generation_profile_context

    return generation_profile_context(
        detail_ranges=detail_ranges,
        active_prefix_self_attention_length=prefix,
        q1_bmm_cross_attention=True,
        native_q1_self_attention=True,
        native_q1_rope_cache_self_attention=True,
    )


def _restore_and_synchronize(cache: Any, snapshots: list[Any]) -> None:
    _restore_all_cache(cache, snapshots)
    torch.cuda.synchronize()


@torch.no_grad()
def _profile_bucket(
    model: torch.nn.Module,
    *,
    prefix: int,
    accepted_graph: torch.cuda.CUDAGraph,
    static_inputs: dict[str, Any],
    warmup: int,
    iterations: int,
    trace_path: Path | None,
) -> dict[str, Any]:
    if warmup < 1 or iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    cache = _cache_from_static_inputs(static_inputs)
    cache_position = static_inputs.get("cache_position")
    if not isinstance(cache_position, torch.Tensor):
        raise TypeError("real static inputs do not expose tensor cache_position")
    snapshots = _all_cache_snapshots(cache, cache_position)

    _restore_and_synchronize(cache, snapshots)
    production_graph = _time_graph(
        accepted_graph,
        warmup=GRAPH_WARMUP,
        iters=GRAPH_ITERATIONS,
    )
    _restore_and_synchronize(cache, snapshots)

    with _accepted_context(prefix, detail_ranges=False):
        for _ in range(warmup):
            _restore_and_synchronize(cache, snapshots)
            model(**static_inputs, return_dict=True)
            torch.cuda.synchronize()

    elapsed_ms: list[float] = []
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with _accepted_context(prefix, detail_ranges=True):
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        ) as profiler:
            for _ in range(iterations):
                _restore_and_synchronize(cache, snapshots)
                started = torch.cuda.Event(enable_timing=True)
                finished = torch.cuda.Event(enable_timing=True)
                started.record()
                model(**static_inputs, return_dict=True)
                finished.record()
                finished.synchronize()
                elapsed_ms.append(float(started.elapsed_time(finished)))
                profiler.step()

    _restore_and_synchronize(cache, snapshots)
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_path))
    regions_ms, raw_marker_us = aggregate_region_times(
        profiler.key_averages(),
        iterations=iterations,
    )
    event_mean = sum(elapsed_ms) / len(elapsed_ms)
    assigned = sum(regions_ms.values())
    production_graph_ms = float(production_graph["ms_per_call"])
    calibration_scale = production_graph_ms / event_mean
    calibrated_regions = {
        region: value * calibration_scale for region, value in regions_ms.items()
    }
    return {
        "production_graph_ms_per_call": production_graph_ms,
        "production_graph_timing": production_graph,
        "cuda_event_ms_per_call": {
            "mean": event_mean,
            "minimum": min(elapsed_ms),
            "maximum": max(elapsed_ms),
        },
        "regions_ms_per_call": regions_ms,
        "production_calibrated_regions_ms_per_call": calibrated_regions,
        "eager_to_production_graph_scale": calibration_scale,
        "assigned_region_ms_per_call": assigned,
        "unattributed_ms_per_call": max(0.0, event_mean - assigned),
        "over_attributed_ms_per_call": max(0.0, assigned - event_mean),
        "raw_detail_range_device_us_total": raw_marker_us,
        "range_coverage": {
            "self_out_residual": (
                "named range covers self out projection; residual add is in unattributed time"
            ),
            "cross_out_residual": (
                "named range covers cross out projection; residual add is in unattributed time"
            ),
            "mlp": (
                "named ranges cover norm/fc1/activation/fc2; dropout and residual add are "
                "in unattributed time"
            ),
        },
    }


def summarize_weighted_ceiling(
    buckets: dict[str, Any],
    *,
    live_counts: dict[int, int],
) -> dict[str, Any]:
    measured = tuple(sorted(int(prefix) for prefix in buckets))
    if measured != tuple(SENTINEL_BUCKETS):
        raise ValueError(
            f"region ceiling requires sentinel buckets {SENTINEL_BUCKETS}, got {measured}"
        )
    if any(
        prefix not in live_counts or live_counts[prefix] <= 0 for prefix in measured
    ):
        raise ValueError("live counts must contain positive sentinel replay counts")
    weighted_regions = {
        region: sum(
            live_counts[prefix]
            * float(
                buckets[str(prefix)]["production_calibrated_regions_ms_per_call"][
                    region
                ]
            )
            / 1_000.0
            for prefix in measured
        )
        for region in REGIONS
    }
    weighted_total = sum(
        live_counts[prefix]
        * float(buckets[str(prefix)]["production_graph_ms_per_call"])
        / 1_000.0
        for prefix in measured
    )
    weighted_unattributed = sum(
        live_counts[prefix]
        * float(buckets[str(prefix)]["unattributed_ms_per_call"])
        * float(buckets[str(prefix)]["eager_to_production_graph_scale"])
        / 1_000.0
        for prefix in measured
    )
    weighted_over_attributed = sum(
        live_counts[prefix]
        * float(buckets[str(prefix)]["over_attributed_ms_per_call"])
        * float(buckets[str(prefix)]["eager_to_production_graph_scale"])
        / 1_000.0
        for prefix in measured
    )
    removable = sum(weighted_regions.values())
    largest_region = max(weighted_regions, key=weighted_regions.__getitem__)
    largest_region_seconds = weighted_regions[largest_region]
    return {
        "regions_seconds": weighted_regions,
        "region_decisions": {
            region: {
                "seconds": seconds,
                "clears_1_412s": seconds >= REQUIRED_HEADROOM_SECONDS,
            }
            for region, seconds in weighted_regions.items()
        },
        "measured_production_graph_seconds": weighted_total,
        "weighted_unattributed_seconds": weighted_unattributed,
        "weighted_over_attributed_seconds": weighted_over_attributed,
        "optimistic_removable_seconds": removable,
        "largest_region": largest_region,
        "largest_region_seconds": largest_region_seconds,
        "required_headroom_seconds": REQUIRED_HEADROOM_SECONDS,
        "clears_required_headroom": (
            largest_region_seconds >= REQUIRED_HEADROOM_SECONDS
        ),
        "unmeasured_buckets_assumed_seconds": 0.0,
        "interpretation": (
            "eager detail-range shares are scaled to matched production CUDA-graph replay; "
            "the gate is based on the largest concrete calibrated region and unmeasured "
            "buckets contribute zero savings"
        ),
    }


def _validate_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected region-profiler schema version")
    metadata = report.get("metadata")
    buckets = report.get("buckets")
    weighted = report.get("weighted")
    if (
        not isinstance(metadata, dict)
        or not isinstance(buckets, dict)
        or not isinstance(weighted, dict)
    ):
        raise TypeError(
            "report metadata, buckets, and weighted sections must be objects"
        )
    if tuple(metadata.get("measured_buckets", ())) != SENTINEL_BUCKETS:
        raise ValueError("report must cover the exact sentinel bucket set")
    if set(buckets) != {str(prefix) for prefix in SENTINEL_BUCKETS}:
        raise ValueError("bucket payload does not match sentinel metadata")
    for prefix in SENTINEL_BUCKETS:
        entry = buckets[str(prefix)]
        regions = entry.get("regions_ms_per_call")
        calibrated = entry.get("production_calibrated_regions_ms_per_call")
        if (
            not isinstance(regions, dict)
            or tuple(regions) != REGIONS
            or not isinstance(calibrated, dict)
            or tuple(calibrated) != REGIONS
        ):
            raise ValueError(f"bucket {prefix} does not expose the exact region order")
        numbers = list(regions.values()) + list(calibrated.values()) + [
            entry["cuda_event_ms_per_call"][name]
            for name in ("mean", "minimum", "maximum")
        ] + [entry["production_graph_ms_per_call"]]
        if not all(
            math.isfinite(float(value)) and float(value) >= 0 for value in numbers
        ):
            raise ValueError(f"bucket {prefix} contains invalid timings")
        if any(float(value) <= 0 for value in (*regions.values(), *calibrated.values())):
            raise ValueError(f"bucket {prefix} contains an empty mapped region")
    required = float(weighted["required_headroom_seconds"])
    removable = float(weighted["optimistic_removable_seconds"])
    if required != REQUIRED_HEADROOM_SECONDS:
        raise ValueError("required headroom changed")
    region_decisions = weighted.get("region_decisions")
    if not isinstance(region_decisions, dict) or tuple(region_decisions) != REGIONS:
        raise ValueError("weighted report does not expose exact per-region decisions")
    for region in REGIONS:
        seconds = float(weighted["regions_seconds"][region])
        decision = region_decisions[region]
        if float(decision["seconds"]) != seconds:
            raise ValueError(f"weighted decision seconds changed for {region}")
        if decision["clears_1_412s"] is not (seconds >= required):
            raise ValueError(f"weighted decision is inconsistent for {region}")
    largest_region = max(
        weighted["regions_seconds"],
        key=weighted["regions_seconds"].__getitem__,
    )
    largest_seconds = float(weighted["regions_seconds"][largest_region])
    if weighted["largest_region"] != largest_region or not math.isclose(
        float(weighted["largest_region_seconds"]),
        largest_seconds,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("largest-region evidence is inconsistent")
    expected_pass = largest_seconds >= required
    if weighted["clears_required_headroom"] is not expected_pass:
        raise ValueError("weighted headroom decision is inconsistent")
    region_sum = sum(float(value) for value in weighted["regions_seconds"].values())
    if not math.isclose(region_sum, removable, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("weighted removable ceiling does not equal its regions")


@torch.no_grad()
def profile_accepted_decoder_regions(
    args,
    *,
    capture_output_path: Path,
    warmup: int,
    iterations: int,
    trace_dir: Path | None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("accepted decoder region profiler requires CUDA")
    if warmup < 1 or iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    _assert_scout_args(args)
    capture_output_path.mkdir(parents=True, exist_ok=True)
    run = _accepted_main_session_run(args, output_path=capture_output_path)
    model = run["model"]
    model.eval()
    accepted_entries = validate_accepted_graph_cache(run["session"].graph_cache)
    live_counts = {
        prefix: int(entry["decode_replays"])
        for prefix, entry in accepted_entries.items()
    }
    buckets: dict[str, Any] = {}
    for prefix in SENTINEL_BUCKETS:
        entry = accepted_entries[prefix]
        result = _profile_bucket(
            model,
            prefix=prefix,
            accepted_graph=entry["graph"],
            static_inputs=entry["static_inputs"],
            warmup=warmup,
            iterations=iterations,
            trace_path=(trace_dir / f"decoder-regions-prefix-{prefix}.json")
            if trace_dir is not None
            else None,
        )
        result["decode_replays"] = live_counts[prefix]
        buckets[str(prefix)] = result
    measured_replays = sum(live_counts[prefix] for prefix in SENTINEL_BUCKETS)
    total_replays = sum(live_counts.values())
    report = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "result_class": "accepted-decoder-region-ceiling",
            "production_wiring": False,
            "candidate_fusion": False,
            "original_decoder_layer": True,
            "dispatch": {
                "precision": "fp32",
                "q1_bmm_cross_attention": True,
                "native_q1_self_attention": True,
                "native_q1_rope_cache_self_attention": True,
            },
            "metric": "eager real-tensor CUDA time under torch.profiler detail ranges",
            "measured_buckets": list(SENTINEL_BUCKETS),
            "accepted_bucket_replay_counts": {
                str(prefix): count for prefix, count in live_counts.items()
            },
            "measured_replays": measured_replays,
            "total_replays": total_replays,
            "coverage_fraction": measured_replays / total_replays,
            "warmup": warmup,
            "iterations": iterations,
            "full_salvalai_runs": 1,
            "commit": _git_head(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "manual"),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(),
            "result_path": run["result_path"],
        },
        "buckets": buckets,
        "weighted": summarize_weighted_ceiling(buckets, live_counts=live_counts),
    }
    _validate_report(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--capture-output-path", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("overrides", nargs="*")
    cli = parser.parse_args()
    overrides = list(cli.overrides)
    if not any(value.startswith("output_path=") for value in overrides):
        overrides.append(f"output_path={cli.capture_output_path}")
    args = _load_args(cli.config_name, overrides)
    report = profile_accepted_decoder_regions(
        args,
        capture_output_path=cli.capture_output_path,
        warmup=cli.warmup,
        iterations=cli.iterations,
        trace_dir=cli.trace_dir,
    )
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["metadata"], indent=2))
    print(json.dumps(report["weighted"], indent=2))


if __name__ == "__main__":
    main()
