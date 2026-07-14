"""Profile decoder-region ceilings inside the winning batched timing graph."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from utils.profile_accepted_decoder_regions import (  # noqa: E402
    REGIONS,
    _profile_bucket,
)
from utils.profile_batched_super_timing_graph import (  # noqa: E402
    _load_args,
    profile,
)


SCHEMA_VERSION = 1
SENTINEL_PREFIXES = (128, 576, 640)
PROMOTION_THRESHOLD_FRACTION = 0.05
NOMINAL_BATCH_SIZE = 16
SALVALAI_AUDIO_SHA256 = "f16f06549e732ef5fb0fceb4dcec95efe5beb6167efe13f6998491b5a16826fe"
CANDIDATE_FAMILIES = {
    "q1_bmm_cross_attention": ("q1_bmm_cross",),
    "shared_traits_self_attention": ("fused_self_attention",),
    "native_cross_mlp_tail": (
        "cross_norm_q",
        "q1_bmm_cross",
        "cross_out_residual",
        "mlp",
    ),
}
DISPATCH_FAMILIES = (
    "native_q1_rope_cache_self_attention",
    "native_q1_self_attention",
    "q1_bmm_cross_attention",
    "native_cross_mlp_tail",
)


def _git_head() -> str:
    return subprocess.run(
        ("git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _framework_context(prefix: int, *, detail_ranges: bool):
    from osuT5.osuT5.runtime_profiling import generation_profile_context

    return generation_profile_context(
        detail_ranges=detail_ranges,
        active_prefix_self_attention_length=prefix,
        q1_bmm_cross_attention=False,
        native_q1_self_attention=False,
        native_q1_rope_cache_self_attention=False,
        native_cross_mlp_tail=False,
        optimized_expected_dtype=torch.float32,
    )


def _entry_batch_size(entry: dict[str, Any]) -> int:
    static_inputs = entry.get("static_inputs")
    if not isinstance(static_inputs, dict):
        raise TypeError("graph entry static_inputs must be an object")
    for key in ("decoder_input_ids", "input_ids", "inputs_embeds"):
        value = static_inputs.get(key)
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            batch = int(value.shape[0])
            if batch <= 0:
                raise ValueError("graph entry has an empty batch dimension")
            return batch
    for value in static_inputs.values():
        if isinstance(value, torch.Tensor) and value.ndim > 1:
            batch = int(value.shape[0])
            if batch <= 0:
                raise ValueError("graph entry has an empty batch dimension")
            return batch
    raise RuntimeError("graph entry exposes no batch-shaped tensor")


def select_live_graph_entries(
    graph_cache: dict[Any, dict[str, Any]],
    *,
    actual_batch_sizes: set[int],
    sentinel_prefixes: tuple[int, ...] = SENTINEL_PREFIXES,
) -> dict[tuple[int, int], dict[str, Any]]:
    if not isinstance(graph_cache, dict) or not graph_cache:
        raise ValueError("batched region profiling requires a nonempty graph cache")
    if not actual_batch_sizes or any(batch <= 1 for batch in actual_batch_sizes):
        raise ValueError("actual batched graph sizes must all be greater than one")
    selected: dict[tuple[int, int], dict[str, Any]] = {}
    for entry in graph_cache.values():
        if not isinstance(entry, dict):
            raise TypeError("graph cache entries must be objects")
        prefix = entry.get("active_prefix_length")
        if prefix not in sentinel_prefixes:
            continue
        batch = _entry_batch_size(entry)
        if batch not in actual_batch_sizes:
            continue
        identity = (batch, int(prefix))
        if identity in selected:
            raise RuntimeError(f"graph cache repeats live signature {identity}")
        required = ("graph", "outputs", "static_inputs", "decode_replays")
        missing = [field for field in required if field not in entry]
        if missing:
            raise RuntimeError(f"graph signature {identity} is missing {missing}")
        replays = entry["decode_replays"]
        if isinstance(replays, bool) or not isinstance(replays, int) or replays <= 0:
            raise ValueError(f"graph signature {identity} has invalid replay count")
        selected[identity] = entry

    expected = {
        (batch, prefix)
        for batch in actual_batch_sizes
        for prefix in sentinel_prefixes
    }
    missing = sorted(expected - set(selected))
    unexpected = sorted(set(selected) - expected)
    if missing or unexpected:
        raise RuntimeError(
            "batched region signatures do not match live batch/prefix coverage: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return dict(sorted(selected.items()))


def summarize_batched_region_ceiling(
    profiles: dict[str, dict[str, Any]],
    *,
    complete_super_timing_seconds: float,
    total_graph_replays: int,
) -> dict[str, Any]:
    if not math.isfinite(complete_super_timing_seconds) or complete_super_timing_seconds <= 0:
        raise ValueError("complete super-timing wall must be positive and finite")
    if isinstance(total_graph_replays, bool) or total_graph_replays <= 0:
        raise ValueError("total graph replays must be positive")
    if not profiles:
        raise ValueError("at least one batched region profile is required")

    identities: set[tuple[int, int]] = set()
    measured_replays = 0
    regions_seconds = {region: 0.0 for region in REGIONS}
    for name, row in profiles.items():
        batch = row.get("batch_size")
        prefix = row.get("active_prefix_length")
        replays = row.get("decode_replays")
        if (
            isinstance(batch, bool)
            or not isinstance(batch, int)
            or batch <= 1
            or isinstance(prefix, bool)
            or not isinstance(prefix, int)
            or prefix not in SENTINEL_PREFIXES
            or isinstance(replays, bool)
            or not isinstance(replays, int)
            or replays <= 0
        ):
            raise ValueError(f"profile {name!r} has invalid live-signature metadata")
        identity = (batch, prefix)
        if identity in identities:
            raise ValueError(f"profiles repeat live signature {identity}")
        identities.add(identity)
        calibrated = row.get("production_calibrated_regions_ms_per_call")
        if not isinstance(calibrated, dict) or tuple(calibrated) != REGIONS:
            raise ValueError(f"profile {name!r} has an invalid region schema")
        for region, value in calibrated.items():
            parsed = float(value)
            if not math.isfinite(parsed) or parsed <= 0:
                raise ValueError(f"profile {name!r} has invalid timing for {region}")
            regions_seconds[region] += replays * parsed / 1_000.0
        measured_replays += replays

    if measured_replays > total_graph_replays:
        raise ValueError("measured sentinel replays exceed the full graph replay count")
    threshold_seconds = complete_super_timing_seconds * PROMOTION_THRESHOLD_FRACTION
    family_seconds = {
        family: sum(regions_seconds[region] for region in regions)
        for family, regions in CANDIDATE_FAMILIES.items()
    }
    region_decisions = {
        region: {
            "seconds": seconds,
            "fraction_of_complete_wall": seconds / complete_super_timing_seconds,
            "clears_five_percent": seconds >= threshold_seconds,
        }
        for region, seconds in regions_seconds.items()
    }
    family_decisions = {
        family: {
            "regions": list(CANDIDATE_FAMILIES[family]),
            "optimistic_seconds": seconds,
            "fraction_of_complete_wall": seconds / complete_super_timing_seconds,
            "clears_five_percent": seconds >= threshold_seconds,
        }
        for family, seconds in family_seconds.items()
    }
    passing = [
        family
        for family in CANDIDATE_FAMILIES
        if family_decisions[family]["clears_five_percent"]
    ]
    return {
        "complete_super_timing_seconds": complete_super_timing_seconds,
        "promotion_threshold_fraction": PROMOTION_THRESHOLD_FRACTION,
        "promotion_threshold_seconds": threshold_seconds,
        "measured_graph_replays": measured_replays,
        "total_graph_replays": total_graph_replays,
        "replay_coverage_fraction": measured_replays / total_graph_replays,
        "regions_seconds": regions_seconds,
        "region_decisions": region_decisions,
        "candidate_family_decisions": family_decisions,
        "passing_candidate_families": passing,
        "promotion_pass": bool(passing),
        "unmeasured_prefixes_assumed_savings_seconds": 0.0,
    }


def _validate_workload_report(report: dict[str, Any]) -> set[int]:
    metadata = report.get("metadata")
    gates = report.get("gates")
    dispatch = report.get("dispatch")
    graphs = report.get("graphs")
    workload = report.get("workload")
    if not all(
        isinstance(value, dict)
        for value in (metadata, gates, dispatch, graphs, workload)
    ):
        raise TypeError("batched workload report is missing required sections")
    required_metadata = {
        "mode": "graph",
        "batch_size": NOMINAL_BATCH_SIZE,
        "precision": "fp32",
        "seed": 12345,
        "timer_iterations": 20,
        "timer_num_beams": 1,
        "timer_cfg_scale": 1.0,
        "public_wiring": False,
        "audio_sha256": SALVALAI_AUDIO_SHA256,
    }
    mismatches = {
        key: {"actual": metadata.get(key), "required": value}
        for key, value in required_metadata.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"batched workload contract changed: {mismatches}")
    if gates.get("pass") is not True:
        raise RuntimeError("batched graph workload did not pass its gates")
    capture_hits = dispatch.get("capture_hits")
    if (
        dispatch.get("modes") != ["framework_batch"]
        or not isinstance(capture_hits, dict)
        or tuple(capture_hits) != DISPATCH_FAMILIES
        or any(int(capture_hits[family]) != 0 for family in DISPATCH_FAMILIES)
    ):
        raise RuntimeError("batched workload did not preserve framework-only dispatch")
    if graphs.get("cache_ownership_pass") is not True:
        raise RuntimeError("batched workload failed cache ownership")
    batches = workload.get("batches_by_iteration")
    if not isinstance(batches, dict) or not batches:
        raise ValueError("batched workload exposes no live batch manifest")
    actual = {
        int(batch)
        for row in batches.values()
        if isinstance(row, list)
        for batch in row
    }
    if not actual or any(batch <= 1 or batch > NOMINAL_BATCH_SIZE for batch in actual):
        raise ValueError(f"batched workload has invalid live batches {sorted(actual)}")
    observed = set(graphs.get("observed_cache_batch_sizes", ()))
    if observed != actual:
        raise RuntimeError(
            f"cache batches {sorted(observed)} do not match workload {sorted(actual)}"
        )
    return actual


@torch.no_grad()
def profile_batched_decoder_regions(
    args,
    *,
    output_path: Path,
    trace_dir: Path | None,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("batched decoder region profiling requires CUDA")
    if args.precision != "fp32":
        raise ValueError("batched decoder region profiling is gated on FP32 first")
    if warmup <= 0 or iterations <= 0:
        raise ValueError("warmup and iterations must be positive")

    artifacts: dict[str, Any] = {}
    workload_report = profile(
        args,
        mode="graph",
        batch_size=NOMINAL_BATCH_SIZE,
        seed=12345,
        artifacts=artifacts,
    )
    actual_batch_sizes = _validate_workload_report(workload_report)
    runtime = artifacts.get("runtime")
    model = artifacts.get("model")
    if runtime is None or not isinstance(model, torch.nn.Module):
        raise RuntimeError("batched profiler did not retain model/runtime artifacts")
    if model.training:
        raise RuntimeError("batched decoder region profiling requires eval mode")
    session = runtime.new_context_state()
    selected = select_live_graph_entries(
        session.graph_cache,
        actual_batch_sizes=actual_batch_sizes,
    )

    profiles: dict[str, Any] = {}
    for (batch, prefix), entry in selected.items():
        name = f"b{batch}-p{prefix}"
        result = _profile_bucket(
            model,
            prefix=prefix,
            accepted_graph=entry["graph"],
            static_inputs=entry["static_inputs"],
            warmup=warmup,
            iterations=iterations,
            trace_path=(trace_dir / f"batched-regions-{name}.json")
            if trace_dir is not None
            else None,
            context_factory=_framework_context,
        )
        result.update(
            batch_size=batch,
            active_prefix_length=prefix,
            decode_replays=int(entry["decode_replays"]),
        )
        profiles[name] = result

    complete_seconds = float(
        workload_report["timing"]["complete_super_timing_seconds"]
    )
    total_replays = int(workload_report["graphs"]["decode_replays"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "result_class": "batched-decoder-region-ceiling",
            "production_wiring": False,
            "candidate_kernel_wiring": False,
            "precision": "fp32",
            "nominal_batch_size": NOMINAL_BATCH_SIZE,
            "actual_batch_sizes": sorted(actual_batch_sizes),
            "sentinel_prefixes": list(SENTINEL_PREFIXES),
            "framework_decoder_operations": True,
            "specialized_dispatch_hits": workload_report["dispatch"]["capture_hits"],
            "warmup": warmup,
            "iterations": iterations,
            "seed": 12345,
            "commit": _git_head(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "manual"),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(),
        },
        "workload": workload_report,
        "profiles": profiles,
        "weighted": summarize_batched_region_ceiling(
            profiles,
            complete_super_timing_seconds=complete_seconds,
            total_graph_replays=total_replays,
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("overrides", nargs="*")
    cli = parser.parse_args()
    args = _load_args(cli.config_name, list(cli.overrides))
    report = profile_batched_decoder_regions(
        args,
        output_path=cli.report_path,
        trace_dir=cli.trace_dir,
        warmup=cli.warmup,
        iterations=cli.iterations,
    )
    print(json.dumps({
        "metadata": report["metadata"],
        "weighted": report["weighted"],
    }, indent=2))
    if not report["weighted"]["promotion_pass"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
