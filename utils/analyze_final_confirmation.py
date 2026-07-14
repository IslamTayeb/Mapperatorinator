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
    _parse_run,
    _structure_divergence,
    _token_divergence,
)
from utils.run_final_confirmation import SCHEMA_VERSION as RUN_SCHEMA_VERSION
from utils.run_serial_final_confirmation import SCHEMA_VERSION as SUITE_SCHEMA_VERSION


ANALYSIS_SCHEMA_VERSION = "mapperatorinator.final-confirmation-analysis.v1"
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


def _load_natural(root: Path, variant: str, count: int) -> list[Any]:
    paths = _evidence_files(root, variant)
    if len(paths) != count:
        raise CandidateAnalysisError(
            f"natural {variant} requires {count} fresh processes, got {len(paths)}"
        )
    runs = []
    uuids = set()
    for index, path in enumerate(paths):
        evidence = _json(path)
        if evidence.get("schema_version") != RUN_SCHEMA_VERSION:
            raise CandidateAnalysisError(f"natural evidence schema mismatch: {path}")
        if evidence.get("mode") != "natural" or bool(evidence.get("candidate")) != (variant == "candidate"):
            raise CandidateAnalysisError(f"natural evidence identity mismatch: {path}")
        run_uuid = evidence.get("run_uuid")
        if not isinstance(run_uuid, str) or run_uuid in uuids:
            raise CandidateAnalysisError("natural runs must have unique process UUIDs")
        uuids.add(run_uuid)
        profile = Path(str(evidence.get("profile_path")))
        runs.append(_parse_run(f"natural_{variant}_{index}", profile, None))
    return runs


def _fixed_profile_metrics(profile_path: Path) -> tuple[float, float]:
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
    if not math.isfinite(request_wall) or request_wall <= 0.0:
        raise CandidateAnalysisError("fixed profile request wall is invalid")
    return main_model, request_wall


def _load_fixed(root: Path, variant: str, count: int) -> list[dict[str, float]]:
    paths = _evidence_files(root, variant)
    if len(paths) != count:
        raise CandidateAnalysisError(
            f"fixed-work {variant} requires {count} fresh processes, got {len(paths)}"
        )
    rows = []
    manifest_hashes = set()
    uuids = set()
    for path in paths:
        evidence = _json(path)
        if evidence.get("schema_version") != RUN_SCHEMA_VERSION:
            raise CandidateAnalysisError(f"fixed evidence schema mismatch: {path}")
        if evidence.get("mode") != "replay-fixed-work" or bool(evidence.get("candidate")) != (variant == "candidate"):
            raise CandidateAnalysisError(f"fixed evidence identity mismatch: {path}")
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
        model_seconds, request_wall = _fixed_profile_metrics(Path(evidence["profile_path"]))
        rows.append(
            {
                "main_steps": float(EXPECTED_MAIN_STEPS),
                "main_model_seconds": model_seconds,
                "fixed_work_tps": EXPECTED_MAIN_STEPS / model_seconds,
                "complete_request_wall_seconds": request_wall,
                "peak_cuda_memory_allocated_mb": float(
                    evidence["cuda_memory"]["max_allocated_bytes"]
                )
                / 1024**2,
            }
        )
    if len(manifest_hashes) != 1 or None in manifest_hashes:
        raise CandidateAnalysisError("fixed-work runs did not share one exact manifest")
    return rows


def _stable_natural(runs: list[Any], variant: str) -> None:
    for label in ("timing_context", "main_generation"):
        token_hashes = {run.label_signatures[label]["token_stream_sha256"] for run in runs}
        stopping_hashes = {run.label_signatures[label]["stopping_sha256"] for run in runs}
        if len(token_hashes) != 1 or len(stopping_hashes) != 1:
            raise CandidateAnalysisError(f"natural {variant} {label} is not deterministic")
    if len({run.osu_sha256 for run in runs}) != 1:
        raise CandidateAnalysisError(f"natural {variant} final map is not deterministic")


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
    raw_runs = evidence.get("runs")
    if not isinstance(raw_runs, list) or len(raw_runs) != 5:
        raise CandidateAnalysisError("serial suite requires exactly five songs")
    parsed = [
        _parse_run(f"suite_{'candidate' if candidate else 'baseline'}_{index}", Path(row["profile_path"]), None)
        for index, row in enumerate(raw_runs)
    ]
    return evidence, parsed


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
    baseline_natural = _load_natural(natural_root, "baseline", natural_repetitions)
    candidate_natural = _load_natural(natural_root, "candidate", natural_repetitions)
    _stable_natural(baseline_natural, "baseline")
    _stable_natural(candidate_natural, "candidate")
    natural_divergence = {
        "timing_context": _token_divergence(baseline_natural[0], candidate_natural[0], "timing_context"),
        "main_generation": _token_divergence(baseline_natural[0], candidate_natural[0], "main_generation"),
        "structure": _structure_divergence(baseline_natural[0], candidate_natural[0]),
    }

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
    for name, baseline, candidate in zip(baseline_names, baseline_suite, candidate_suite, strict=True):
        suite_rows.append(
            {
                "name": name,
                "baseline_main_tokens": int(baseline.metrics["main_tokens"]),
                "candidate_main_tokens": int(candidate.metrics["main_tokens"]),
                "baseline_main_model_seconds": baseline.metrics["main_model_seconds"],
                "candidate_main_model_seconds": candidate.metrics["main_model_seconds"],
                "candidate_main_tps": candidate.metrics["main_tps"],
                "baseline_complete_request_wall_seconds": baseline.metrics["complete_request_wall_seconds"],
                "candidate_complete_request_wall_seconds": candidate.metrics["complete_request_wall_seconds"],
                "token_divergence": _token_divergence(baseline, candidate, "main_generation"),
                "structure_divergence": _structure_divergence(baseline, candidate),
            }
        )
    aggregate_tokens = sum(row["candidate_main_tokens"] for row in suite_rows)
    aggregate_model = sum(row["candidate_main_model_seconds"] for row in suite_rows)
    aggregate_tps = aggregate_tokens / aggregate_model
    suite_gate = aggregate_tps >= 500.0 and all(row["candidate_main_tps"] >= 475.0 for row in suite_rows)
    candidate_memory = _memory_stability(candidate_suite_evidence, memory_tolerance_mb)
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "acceptance": {
            "fixed_work_exact_main_steps": EXPECTED_MAIN_STEPS,
            "fixed_work_median_tps_at_least_500": statistics.median(fixed_tps) >= 500.0,
            "fixed_work_no_run_below_475": min(fixed_tps) >= 475.0,
            "five_song_aggregate_tps_at_least_500": aggregate_tps >= 500.0,
            "five_song_no_song_below_475": all(row["candidate_main_tps"] >= 475.0 for row in suite_rows),
            "serial_memory_stable": candidate_memory["pass"],
            "pass": fixed_gate and suite_gate and candidate_memory["pass"],
        },
        "fixed_work": {
            "baseline_runs": baseline_fixed,
            "candidate_runs": candidate_fixed,
            "candidate_tps": _summary(fixed_tps),
            "candidate_request_wall_seconds": _summary(
                [row["complete_request_wall_seconds"] for row in candidate_fixed]
            ),
        },
        "natural_salvalai": {
            "baseline_main_tps": _summary([run.metrics["main_tps"] for run in baseline_natural]),
            "candidate_main_tps": _summary([run.metrics["main_tps"] for run in candidate_natural]),
            "baseline_request_wall_seconds": _summary(
                [run.metrics["complete_request_wall_seconds"] for run in baseline_natural]
            ),
            "candidate_request_wall_seconds": _summary(
                [run.metrics["complete_request_wall_seconds"] for run in candidate_natural]
            ),
            "divergence": natural_divergence,
        },
        "five_song_serial": {
            "songs": suite_rows,
            "candidate_aggregate_main_tokens": aggregate_tokens,
            "candidate_aggregate_main_model_seconds": aggregate_model,
            "candidate_aggregate_main_tps": aggregate_tps,
            "baseline_complete_request_wall_seconds": sum(
                row["baseline_complete_request_wall_seconds"] for row in suite_rows
            ),
            "candidate_complete_request_wall_seconds": sum(
                row["candidate_complete_request_wall_seconds"] for row in suite_rows
            ),
            "candidate_memory_stability": candidate_memory,
        },
    }


def text_report(report: dict[str, Any]) -> str:
    acceptance = report["acceptance"]
    fixed = report["fixed_work"]["candidate_tps"]
    suite = report["five_song_serial"]
    natural = report["natural_salvalai"]
    lines = [
        f"acceptance.pass={str(acceptance['pass']).lower()}",
        f"fixed_work.main_steps={acceptance['fixed_work_exact_main_steps']}",
        f"fixed_work.candidate_tps_median={fixed['median']:.9f}",
        f"fixed_work.candidate_tps_minimum={fixed['minimum']:.9f}",
        f"natural.candidate_main_tps_median={natural['candidate_main_tps']['median']:.9f}",
        f"natural.candidate_request_wall_median={natural['candidate_request_wall_seconds']['median']:.9f}",
        f"five_song.candidate_aggregate_main_tps={suite['candidate_aggregate_main_tps']:.9f}",
        f"five_song.baseline_complete_request_wall_seconds={suite['baseline_complete_request_wall_seconds']:.9f}",
        f"five_song.candidate_complete_request_wall_seconds={suite['candidate_complete_request_wall_seconds']:.9f}",
        f"five_song.memory_stable={str(suite['candidate_memory_stability']['pass']).lower()}",
    ]
    for row in suite["songs"]:
        lines.append(
            f"song.{row['name']}.candidate_main_tps={row['candidate_main_tps']:.9f},"
            f"candidate_request_wall={row['candidate_complete_request_wall_seconds']:.9f},"
            f"token_count_delta={row['token_divergence']['token_count_delta']},"
            f"objects_delta={row['structure_divergence']['scalar_deltas']['hit_objects']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--natural-root", type=Path, required=True)
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--baseline-suite-evidence", type=Path, required=True)
    parser.add_argument("--candidate-suite-evidence", type=Path, required=True)
    parser.add_argument("--natural-repetitions", type=int, default=2)
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
