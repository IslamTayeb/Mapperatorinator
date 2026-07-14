"""Aggregate fixed-work, natural-wall, and five-song confirmation evidence."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from utils.analyze_reciprocal_full_song_candidate import (
    CandidateAnalysisError,
    _material_differences,
    _parse_run,
    _structure_divergence,
    _token_divergence,
)
from utils.run_final_confirmation import SCHEMA_VERSION as RUN_SCHEMA_VERSION
from utils.run_serial_final_confirmation import SCHEMA_VERSION as SUITE_SCHEMA_VERSION


ANALYSIS_SCHEMA_VERSION = "mapperatorinator.final-confirmation-analysis.v2"
EXPECTED_MAIN_STEPS = 8294


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateAnalysisError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CandidateAnalysisError(f"JSON root must be an object: {path}")
    return payload


def _evidence_files(root: Path, variant: str) -> list[Path]:
    paths = sorted(root.glob(f"{variant}-*/evidence.json"))
    if not paths:
        raise CandidateAnalysisError(f"no {variant} evidence under {root}")
    return paths


def _runtime_identity(evidence: dict[str, Any], *, variant: str) -> dict[str, Any]:
    runtime = evidence.get("runtime")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("name"), str):
        raise CandidateAnalysisError(f"{variant} evidence lacks runtime identity")
    spec_hash = evidence.get("runtime_spec_sha256")
    if variant == "baseline" and spec_hash is not None:
        raise CandidateAnalysisError("baseline must not have a candidate runtime spec")
    if variant == "candidate" and not isinstance(spec_hash, str):
        raise CandidateAnalysisError("candidate runtime spec hash is missing")
    return {
        "name": runtime["name"],
        "factory": runtime.get("factory"),
        "runtime_spec_sha256": spec_hash,
    }


def _rng_material(run: Any) -> dict[str, Any]:
    return {
        key: value
        for key, value in sorted(run.metadata.items())
        if key == "reciprocal_seed_policy"
        or key.startswith("reciprocal_seed_")
        or key.startswith("reciprocal_rng_")
    }


def _run_metrics(run: Any) -> dict[str, float]:
    keys = (
        "timing_model_seconds",
        "timing_tokens",
        "timing_tps",
        "timing_outer_stage_wall_seconds",
        "timing_postprocess_seconds",
        "main_model_seconds",
        "main_tokens",
        "main_tps",
        "main_outer_stage_wall_seconds",
        "final_postprocess_write_seconds",
        "complete_request_wall_seconds",
        "cold_process_outer_wall_seconds",
        "setup_stage_sum_seconds",
        "graph_capture_seconds",
        "setup_plus_capture_seconds",
        "peak_cuda_memory_allocated_mb",
        "max_current_cuda_memory_allocated_mb",
    )
    return {key: float(run.metrics[key]) for key in keys}


def _load_natural(
    root: Path,
    variant: str,
    count: int,
) -> tuple[list[dict[str, Any]], list[Any]]:
    paths = _evidence_files(root, variant)
    if len(paths) != count:
        raise CandidateAnalysisError(
            f"natural {variant} requires {count} fresh processes, got {len(paths)}"
        )
    runs = []
    evidence_rows = []
    uuids = set()
    for index, path in enumerate(paths):
        evidence = _json(path)
        if evidence.get("schema_version") != RUN_SCHEMA_VERSION:
            raise CandidateAnalysisError(f"natural evidence schema mismatch: {path}")
        if evidence.get("mode") != "natural" or bool(evidence.get("candidate")) != (variant == "candidate"):
            raise CandidateAnalysisError(f"natural evidence identity mismatch: {path}")
        _runtime_identity(evidence, variant=variant)
        run_uuid = evidence.get("run_uuid")
        if not isinstance(run_uuid, str) or run_uuid in uuids:
            raise CandidateAnalysisError("natural runs must have unique process UUIDs")
        uuids.add(run_uuid)
        profile = Path(str(evidence.get("profile_path")))
        runs.append(_parse_run(f"natural_{variant}_{index}", profile, None))
        evidence_rows.append(evidence)
    identities = {
        json.dumps(_runtime_identity(row, variant=variant), sort_keys=True)
        for row in evidence_rows
    }
    if len(identities) != 1:
        raise CandidateAnalysisError(f"natural {variant} runtime identity changed")
    return evidence_rows, runs


def _fixed_profile_metrics(profile_path: Path) -> tuple[float, float, float]:
    payload = _json(profile_path)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise CandidateAnalysisError(f"fixed profile lacks metadata: {profile_path}")
    if metadata.get("profile_pass_kind") != "untraced_control" or metadata.get("authoritative_performance") is not True:
        raise CandidateAnalysisError("fixed-work performance must be authoritative untraced")
    generation = payload.get("generation")
    stages = payload.get("stages")
    if not isinstance(generation, list) or not isinstance(stages, list):
        raise CandidateAnalysisError("fixed profile lacks generation/stage arrays")
    main_model = sum(
        float(record["model_elapsed_seconds"])
        for record in generation
        if isinstance(record, dict) and record.get("profile_label") == "main_generation"
    )
    validate = next((stage for stage in stages if stage.get("name") == "validate_inputs"), None)
    final = next((stage for stage in reversed(stages) if stage.get("name") in {"write_osu", "write_osz"}), None)
    if main_model <= 0.0 or not isinstance(validate, dict) or not isinstance(final, dict):
        raise CandidateAnalysisError("fixed profile cannot compute main/request wall")
    request_wall = float(final["finished_at_perf_counter_seconds"]) - float(
        validate["started_at_perf_counter_seconds"]
    )
    cold_outer_wall = float(final["finished_at_perf_counter_seconds"]) - float(
        stages[0]["started_at_perf_counter_seconds"]
    )
    if (
        not math.isfinite(request_wall)
        or request_wall <= 0.0
        or not math.isfinite(cold_outer_wall)
        or cold_outer_wall <= 0.0
    ):
        raise CandidateAnalysisError("fixed profile request/cold wall is invalid")
    return main_model, request_wall, cold_outer_wall


def _load_fixed(root: Path, variant: str, count: int) -> list[dict[str, float]]:
    paths = _evidence_files(root, variant)
    if len(paths) != count:
        raise CandidateAnalysisError(
            f"fixed-work {variant} requires {count} fresh processes, got {len(paths)}"
        )
    rows = []
    manifest_hashes = set()
    runtime_identities = set()
    uuids = set()
    for path in paths:
        evidence = _json(path)
        if evidence.get("schema_version") != RUN_SCHEMA_VERSION:
            raise CandidateAnalysisError(f"fixed evidence schema mismatch: {path}")
        if evidence.get("mode") != "replay-fixed-work" or bool(evidence.get("candidate")) != (variant == "candidate"):
            raise CandidateAnalysisError(f"fixed evidence identity mismatch: {path}")
        runtime_identities.add(
            json.dumps(_runtime_identity(evidence, variant=variant), sort_keys=True)
        )
        run_uuid = evidence.get("run_uuid")
        if not isinstance(run_uuid, str) or run_uuid in uuids:
            raise CandidateAnalysisError("fixed-work runs must have unique process UUIDs")
        uuids.add(run_uuid)
        main = evidence.get("labels", {}).get("main_generation", {})
        if main.get("logical_steps") != EXPECTED_MAIN_STEPS:
            raise CandidateAnalysisError(
                f"fixed-work run did not execute exactly {EXPECTED_MAIN_STEPS} main steps"
            )
        if any(
            record.get("target_steps") != record.get("logical_steps")
            for record in evidence.get("records", [])
        ):
            raise CandidateAnalysisError("fixed-work target/logical steps diverged")
        manifest_hashes.add(evidence.get("manifest_sha256"))
        profile_path = Path(evidence["profile_path"])
        model_seconds, request_wall, cold_outer_wall = _fixed_profile_metrics(profile_path)
        parsed = _parse_run(f"fixed_{variant}_{len(rows)}", profile_path, None)
        rows.append(
            {
                "main_steps": float(EXPECTED_MAIN_STEPS),
                "main_model_seconds": model_seconds,
                "fixed_work_tps": EXPECTED_MAIN_STEPS / model_seconds,
                "complete_request_wall_seconds": request_wall,
                "cold_process_outer_wall_seconds": cold_outer_wall,
                "peak_cuda_memory_allocated_mb": float(
                    evidence["cuda_memory"]["max_allocated_bytes"]
                )
                / 1024**2,
                "timing_model_seconds": parsed.metrics["timing_model_seconds"],
                "timing_tokens": parsed.metrics["timing_tokens"],
                "timing_tps": parsed.metrics["timing_tps"],
                "timing_outer_stage_wall_seconds": parsed.metrics[
                    "timing_outer_stage_wall_seconds"
                ],
                "main_outer_stage_wall_seconds": parsed.metrics[
                    "main_outer_stage_wall_seconds"
                ],
                "setup_stage_sum_seconds": parsed.metrics["setup_stage_sum_seconds"],
                "graph_capture_seconds": parsed.metrics["graph_capture_seconds"],
                "setup_plus_capture_seconds": parsed.metrics[
                    "setup_plus_capture_seconds"
                ],
            }
        )
    if len(manifest_hashes) != 1 or None in manifest_hashes:
        raise CandidateAnalysisError("fixed-work runs did not share one exact manifest")
    if len(runtime_identities) != 1:
        raise CandidateAnalysisError(f"fixed-work {variant} runtime identity changed")
    return rows


def _stable_natural(runs: list[Any], variant: str) -> None:
    for label in ("timing_context", "main_generation"):
        token_hashes = {run.label_signatures[label]["token_stream_sha256"] for run in runs}
        stopping_hashes = {run.label_signatures[label]["stopping_sha256"] for run in runs}
        if len(token_hashes) != 1 or len(stopping_hashes) != 1:
            raise CandidateAnalysisError(f"natural {variant} {label} is not deterministic")
    if len({run.osu_sha256 for run in runs}) != 1:
        raise CandidateAnalysisError(f"natural {variant} final map is not deterministic")
    if len({run.graph_signature["sha256"] for run in runs}) != 1:
        raise CandidateAnalysisError(
            f"natural {variant} dispatch/cache topology is not deterministic"
        )
    rng_rows = {json.dumps(_rng_material(run), sort_keys=True) for run in runs}
    if len(rng_rows) != 1 or not _rng_material(runs[0]):
        raise CandidateAnalysisError(f"natural {variant} RNG evidence is not deterministic")


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "count": float(len(values)),
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
        "mean": statistics.mean(values),
    }


def _load_suite(path: Path, candidate: bool) -> tuple[dict[str, Any], list[Any]]:
    evidence = _json(path)
    if evidence.get("schema_version") != SUITE_SCHEMA_VERSION or bool(evidence.get("candidate")) != candidate:
        raise CandidateAnalysisError(f"serial suite identity mismatch: {path}")
    _runtime_identity(
        evidence,
        variant="candidate" if candidate else "baseline",
    )
    raw_runs = evidence.get("runs")
    if not isinstance(raw_runs, list) or len(raw_runs) != 5:
        raise CandidateAnalysisError("serial suite requires exactly five songs")
    parsed = [
        _parse_run(f"suite_{'candidate' if candidate else 'baseline'}_{index}", Path(row["profile_path"]), None)
        for index, row in enumerate(raw_runs)
    ]
    return evidence, parsed


def _pair_divergence(reference: Any, candidate: Any) -> dict[str, Any]:
    dispatch_differences = _material_differences(
        reference.graph_signature["material"],
        candidate.graph_signature["material"],
    )
    baseline_rng = _rng_material(reference)
    candidate_rng = _rng_material(candidate)
    return {
        "timing_tokens": _token_divergence(reference, candidate, "timing_context"),
        "main_tokens": _token_divergence(reference, candidate, "main_generation"),
        "structure": _structure_divergence(reference, candidate),
        "osu": {
            "baseline_sha256": reference.osu_sha256,
            "candidate_sha256": candidate.osu_sha256,
            "byte_identical": reference.osu_sha256 == candidate.osu_sha256,
        },
        "rng": {
            "baseline": baseline_rng,
            "candidate": candidate_rng,
            "equal": baseline_rng == candidate_rng,
        },
        "dispatch_cache": {
            "baseline_sha256": reference.graph_signature["sha256"],
            "candidate_sha256": candidate.graph_signature["sha256"],
            "equal": reference.graph_signature["sha256"]
            == candidate.graph_signature["sha256"],
            "differences": dispatch_differences,
        },
    }


def _memory_stability(evidence: dict[str, Any], tolerance_mb: float) -> dict[str, Any]:
    tolerance = int(tolerance_mb * 1024**2)
    rows = evidence["runs"]
    before = [int(row["cuda_memory"]["allocated_bytes_before"]) for row in rows]
    after = [int(row["cuda_memory"]["allocated_bytes_after"]) for row in rows]
    deltas = [right - left for left, right in zip(before, after, strict=True)]
    drift = max(before + after) - min(before + after)
    return {
        "tolerance_mb": tolerance_mb,
        "allocated_before_after_range_mb": drift / 1024**2,
        "per_song_allocated_delta_mb": [value / 1024**2 for value in deltas],
        "last_after_minus_first_before_mb": (after[-1] - before[0]) / 1024**2,
        "pass": drift <= tolerance and after[-1] - before[0] <= tolerance,
    }


def analyze(
    *,
    natural_root: Path,
    fixed_root: Path,
    baseline_suite_path: Path,
    candidate_suite_path: Path,
    natural_repetitions: int,
    fixed_repetitions: int,
    memory_tolerance_mb: float,
) -> dict[str, Any]:
    baseline_natural_evidence, baseline_natural = _load_natural(
        natural_root,
        "baseline",
        natural_repetitions,
    )
    candidate_natural_evidence, candidate_natural = _load_natural(
        natural_root,
        "candidate",
        natural_repetitions,
    )
    _stable_natural(baseline_natural, "baseline")
    _stable_natural(candidate_natural, "candidate")
    natural_divergence = _pair_divergence(
        baseline_natural[0],
        candidate_natural[0],
    )

    baseline_fixed = _load_fixed(fixed_root, "baseline", fixed_repetitions)
    candidate_fixed = _load_fixed(fixed_root, "candidate", fixed_repetitions)
    fixed_tps = [row["fixed_work_tps"] for row in candidate_fixed]
    fixed_gate = statistics.median(fixed_tps) >= 500.0 and min(fixed_tps) >= 475.0

    baseline_suite_evidence, baseline_suite = _load_suite(baseline_suite_path, False)
    candidate_suite_evidence, candidate_suite = _load_suite(candidate_suite_path, True)
    baseline_names = [row["name"] for row in baseline_suite_evidence["runs"]]
    candidate_names = [row["name"] for row in candidate_suite_evidence["runs"]]
    if baseline_names != candidate_names:
        raise CandidateAnalysisError("baseline/candidate suite song order differs")
    suite_rows = []
    for index, (name, baseline, candidate) in enumerate(
        zip(baseline_names, baseline_suite, candidate_suite, strict=True)
    ):
        baseline_raw = baseline_suite_evidence["runs"][index]
        candidate_raw = candidate_suite_evidence["runs"][index]
        suite_rows.append(
            {
                "name": name,
                "baseline": _run_metrics(baseline),
                "candidate": _run_metrics(candidate),
                "divergence": _pair_divergence(baseline, candidate),
                "baseline_rng_state": baseline_raw.get("rng"),
                "candidate_rng_state": candidate_raw.get("rng"),
                "baseline_cuda_memory": baseline_raw["cuda_memory"],
                "candidate_cuda_memory": candidate_raw["cuda_memory"],
            }
        )
    aggregate_tokens = sum(row["candidate"]["main_tokens"] for row in suite_rows)
    aggregate_model = sum(row["candidate"]["main_model_seconds"] for row in suite_rows)
    aggregate_tps = aggregate_tokens / aggregate_model
    suite_gate = aggregate_tps >= 500.0 and all(
        row["candidate"]["main_tps"] >= 475.0 for row in suite_rows
    )
    candidate_memory = _memory_stability(candidate_suite_evidence, memory_tolerance_mb)
    if not candidate_memory["pass"]:
        raise CandidateAnalysisError(
            "candidate serial allocation growth exceeded the declared tolerance"
        )
    metric_keys = tuple(_run_metrics(baseline_natural[0]))

    def natural_variant(runs: list[Any]) -> dict[str, Any]:
        return {
            "runs": [_run_metrics(run) for run in runs],
            "summary": {
                key: _summary([float(run.metrics[key]) for run in runs])
                for key in metric_keys
            },
            "token_signatures": runs[0].label_signatures,
            "rng": _rng_material(runs[0]),
            "dispatch_cache_signature": runs[0].graph_signature,
            "result_sha256": runs[0].osu_sha256,
            "structure": runs[0].osu_structure,
        }

    fixed_target = {
        "median_at_least_500": statistics.median(fixed_tps) >= 500.0,
        "no_run_below_475": min(fixed_tps) >= 475.0,
    }
    suite_target = {
        "aggregate_at_least_500": aggregate_tps >= 500.0,
        "no_song_below_475": all(
            row["candidate"]["main_tps"] >= 475.0 for row in suite_rows
        ),
    }
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "validation": {
            "natural_fresh_process_repetitions": natural_repetitions,
            "fixed_work_fresh_process_repetitions": fixed_repetitions,
            "fixed_work_exact_main_steps": EXPECTED_MAIN_STEPS,
            "baseline_natural_repeatable": True,
            "candidate_natural_repeatable": True,
            "all_outputs_finite_and_well_formed": True,
            "serial_memory_stable": candidate_memory["pass"],
            "pass": True,
        },
        "target_500_tps": {
            "status": "met" if fixed_gate and suite_gate else "not_met",
            "fixed_work": fixed_target,
            "five_song": suite_target,
            "met": fixed_gate and suite_gate,
        },
        "runtimes": {
            "baseline": _runtime_identity(
                baseline_natural_evidence[0],
                variant="baseline",
            ),
            "candidate": _runtime_identity(
                candidate_natural_evidence[0],
                variant="candidate",
            ),
        },
        "fixed_work": {
            "baseline_runs": baseline_fixed,
            "candidate_runs": candidate_fixed,
            "baseline_tps": _summary(
                [row["fixed_work_tps"] for row in baseline_fixed]
            ),
            "candidate_tps": _summary(fixed_tps),
            "target": fixed_target,
            "candidate_request_wall_seconds": _summary(
                [row["complete_request_wall_seconds"] for row in candidate_fixed]
            ),
            "candidate_cold_process_outer_wall_seconds": _summary(
                [row["cold_process_outer_wall_seconds"] for row in candidate_fixed]
            ),
        },
        "natural_salvalai": {
            "baseline": natural_variant(baseline_natural),
            "candidate": natural_variant(candidate_natural),
            "divergence": natural_divergence,
        },
        "five_song_serial": {
            "songs": suite_rows,
            "candidate_aggregate_main_tokens": aggregate_tokens,
            "candidate_aggregate_main_model_seconds": aggregate_model,
            "candidate_aggregate_main_tps": aggregate_tps,
            "target": suite_target,
            "baseline_complete_request_wall_seconds": sum(
                row["baseline"]["complete_request_wall_seconds"] for row in suite_rows
            ),
            "candidate_complete_request_wall_seconds": sum(
                row["candidate"]["complete_request_wall_seconds"] for row in suite_rows
            ),
            "baseline_process_wall_seconds": (
                int(baseline_suite_evidence["process_finished_time_ns"])
                - int(baseline_suite_evidence["process_started_time_ns"])
            )
            / 1e9,
            "candidate_process_wall_seconds": (
                int(candidate_suite_evidence["process_finished_time_ns"])
                - int(candidate_suite_evidence["process_started_time_ns"])
            )
            / 1e9,
            "baseline_startup": baseline_suite_evidence["startup"],
            "candidate_startup": candidate_suite_evidence["startup"],
            "candidate_memory_stability": candidate_memory,
        },
    }


def text_report(report: dict[str, Any]) -> str:
    validation = report["validation"]
    target = report["target_500_tps"]
    fixed = report["fixed_work"]["candidate_tps"]
    suite = report["five_song_serial"]
    natural = report["natural_salvalai"]
    candidate_summary = natural["candidate"]["summary"]
    divergence = natural["divergence"]
    lines = [
        f"validation.pass={str(validation['pass']).lower()}",
        f"target_500_tps.status={target['status']}",
        f"fixed_work.main_steps={validation['fixed_work_exact_main_steps']}",
        f"fixed_work.candidate_tps_median={fixed['median']:.9f}",
        f"fixed_work.candidate_tps_minimum={fixed['minimum']:.9f}",
        f"natural.candidate_timing_model_seconds_median={candidate_summary['timing_model_seconds']['median']:.9f}",
        f"natural.candidate_timing_tps_median={candidate_summary['timing_tps']['median']:.9f}",
        f"natural.candidate_main_model_seconds_median={candidate_summary['main_model_seconds']['median']:.9f}",
        f"natural.candidate_main_tps_median={candidate_summary['main_tps']['median']:.9f}",
        f"natural.candidate_request_wall_median={candidate_summary['complete_request_wall_seconds']['median']:.9f}",
        f"natural.candidate_setup_capture_median={candidate_summary['setup_plus_capture_seconds']['median']:.9f}",
        f"natural.candidate_cold_process_outer_wall_median={candidate_summary['cold_process_outer_wall_seconds']['median']:.9f}",
        f"natural.timing_token_count_delta={divergence['timing_tokens']['token_count_delta']}",
        f"natural.main_token_count_delta={divergence['main_tokens']['token_count_delta']}",
        f"natural.main_stopping_equal={str(divergence['main_tokens']['stopping_equal']).lower()}",
        f"natural.osu_byte_identical={str(divergence['osu']['byte_identical']).lower()}",
        f"natural.dispatch_cache_equal={str(divergence['dispatch_cache']['equal']).lower()}",
        f"natural.rng_equal={str(divergence['rng']['equal']).lower()}",
        f"five_song.candidate_aggregate_main_tps={suite['candidate_aggregate_main_tps']:.9f}",
        f"five_song.baseline_complete_request_wall_seconds={suite['baseline_complete_request_wall_seconds']:.9f}",
        f"five_song.candidate_complete_request_wall_seconds={suite['candidate_complete_request_wall_seconds']:.9f}",
        f"five_song.candidate_process_wall_seconds={suite['candidate_process_wall_seconds']:.9f}",
        f"five_song.memory_stable={str(suite['candidate_memory_stability']['pass']).lower()}",
    ]
    for row in suite["songs"]:
        candidate = row["candidate"]
        row_divergence = row["divergence"]
        lines.append(
            f"song.{row['name']}.timing_model_seconds={candidate['timing_model_seconds']:.9f},"
            f"timing_tps={candidate['timing_tps']:.9f},"
            f"main_model_seconds={candidate['main_model_seconds']:.9f},"
            f"main_tps={candidate['main_tps']:.9f},"
            f"request_wall={candidate['complete_request_wall_seconds']:.9f},"
            f"main_token_count_delta={row_divergence['main_tokens']['token_count_delta']},"
            f"main_stopping_equal={str(row_divergence['main_tokens']['stopping_equal']).lower()},"
            f"osu_byte_identical={str(row_divergence['osu']['byte_identical']).lower()},"
            f"objects_delta={row_divergence['structure']['scalar_deltas']['hit_objects']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--natural-root", type=Path, required=True)
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--baseline-suite-evidence", type=Path, required=True)
    parser.add_argument("--candidate-suite-evidence", type=Path, required=True)
    parser.add_argument("--natural-repetitions", type=int, default=5)
    parser.add_argument("--fixed-repetitions", type=int, default=5)
    parser.add_argument("--memory-tolerance-mb", type=float, default=16.0)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        natural_root=args.natural_root,
        fixed_root=args.fixed_root,
        baseline_suite_path=args.baseline_suite_evidence,
        candidate_suite_path=args.candidate_suite_evidence,
        natural_repetitions=args.natural_repetitions,
        fixed_repetitions=args.fixed_repetitions,
        memory_tolerance_mb=args.memory_tolerance_mb,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.text_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.text_output.write_text(text_report(report), encoding="utf-8")


if __name__ == "__main__":
    main()
