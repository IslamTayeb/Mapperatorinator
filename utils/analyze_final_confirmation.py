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
    _sha256_file,
    _structure_divergence,
    _token_divergence,
)
from utils.run_final_confirmation import SCHEMA_VERSION as RUN_SCHEMA_VERSION
from utils.run_serial_final_confirmation import SCHEMA_VERSION as SUITE_SCHEMA_VERSION
from utils.validate_k4_profile_contract import validate as validate_k4_profile_contract


ANALYSIS_SCHEMA_VERSION = "mapperatorinator.final-confirmation-analysis.v1"
EXPECTED_MAIN_STEPS = 8294
K4_BLOCK_SIZE = 4


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


def _verified_artifact(evidence: dict[str, Any], kind: str) -> Path:
    raw_path = evidence.get(f"{kind}_path")
    expected_sha = evidence.get(f"{kind}_sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise CandidateAnalysisError(f"evidence is missing {kind}_path")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise CandidateAnalysisError(f"evidence is missing {kind}_sha256")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file() or _sha256_file(path) != expected_sha:
        raise CandidateAnalysisError(f"{kind} artifact hash mismatch: {path}")
    return path


def _validate_initialization(evidence: dict[str, Any], *, candidate: bool) -> None:
    initialization = evidence.get("initialization")
    if candidate:
        if not isinstance(initialization, dict):
            raise CandidateAnalysisError(
                "candidate lacks weight-only initialization evidence"
            )
        required_identity = {
            "version": "approximate-fp16-weights-fp32-state-v2",
            "result_class": "documented-drift",
            "exactness_claim": False,
            "fp32_activations_caches_reductions_logits": True,
        }
        for key, expected in required_identity.items():
            if initialization.get(key) != expected:
                raise CandidateAnalysisError(
                    f"candidate initialization has wrong {key}"
                )
        fp16_regions = initialization.get("fp16_weight_regions")
        if not isinstance(fp16_regions, list) or any(
            not isinstance(region, str) for region in fp16_regions
        ):
            raise CandidateAnalysisError(
                "candidate initialization has invalid FP16 regions"
            )
        if set(fp16_regions) != {
            "self_qkv",
            "self_output",
            "mlp_fc1",
            "mlp_fc2",
            "final_logits",
        }:
            raise CandidateAnalysisError(
                "candidate initialization has wrong FP16 regions"
            )
        fp32_regions = initialization.get("fp32_selected_decode_matrix_regions")
        if not isinstance(fp32_regions, list) or any(
            not isinstance(region, str) for region in fp32_regions
        ):
            raise CandidateAnalysisError(
                "candidate initialization has invalid FP32 regions"
            )
        if set(fp32_regions) != {
            "cross_query",
            "cross_output",
        }:
            raise CandidateAnalysisError(
                "candidate initialization has wrong FP32 regions"
            )
        numeric_fields = (
            "initialization_wall_seconds",
            "extension_init_seconds",
            "weight_pack_seconds",
        )
        numeric: dict[str, float] = {}
        for key in numeric_fields:
            value = initialization.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise CandidateAnalysisError(
                    f"candidate initialization {key} is invalid"
                )
            numeric[key] = float(value)
        if (
            numeric["extension_init_seconds"] + numeric["weight_pack_seconds"]
            > numeric["initialization_wall_seconds"] + 1e-3
        ):
            raise CandidateAnalysisError(
                "candidate initialization components exceed its wall"
            )
        for key in ("retained_fp32_source_weight_bytes", "packed_weight_bytes"):
            value = initialization.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise CandidateAnalysisError(
                    f"candidate initialization {key} is invalid"
                )
        if not isinstance(initialization.get("initialization_cuda_memory"), dict):
            raise CandidateAnalysisError(
                "candidate initialization lacks CUDA memory evidence"
            )
    elif initialization is not None:
        raise CandidateAnalysisError(
            "baseline unexpectedly contains candidate initialization"
        )


def _validate_runtime_count(evidence: dict[str, Any], run: Any) -> None:
    expected = (
        2 if any(stage.name == "load_timing_model" for stage in run.stages) else 1
    )
    if evidence.get("loaded_runtime_count") != expected:
        raise CandidateAnalysisError(
            "fresh-process loaded runtime count disagrees with timing-model topology"
        )


def _k4_contract(
    profile_path: Path,
    *,
    candidate: bool,
    expected_fixed_main_steps: int | None = None,
) -> dict[str, Any]:
    payload = _json(profile_path)
    try:
        contract = validate_k4_profile_contract(
            payload,
            role="candidate" if candidate else "baseline",
            block_size=K4_BLOCK_SIZE,
        )
    except (TypeError, ValueError) as exc:
        raise CandidateAnalysisError(f"K4 profile contract failed: {exc}") from exc
    if not candidate or expected_fixed_main_steps is None:
        return contract

    main = {
        "physical_steps": 0,
        "logical_steps": 0,
        "wasted_steps": 0,
    }
    for record in payload["generation"]:
        if record.get("profile_label") != "main_generation":
            continue
        stats = record["optimized_cuda_graphs"]["k8_candidate"]
        for key in main:
            main[key] += int(stats[key])
    if main != {
        "physical_steps": expected_fixed_main_steps,
        "logical_steps": expected_fixed_main_steps,
        "wasted_steps": 0,
    }:
        raise CandidateAnalysisError(
            "fixed-work K4 main work must have identical physical/logical steps "
            f"and zero waste; expected {expected_fixed_main_steps}, got {main}"
        )
    return {**contract, "fixed_main_work": main}


def _wall_comparison(baseline: list[float], candidate: list[float]) -> dict[str, Any]:
    baseline_summary = _summary(baseline)
    candidate_summary = _summary(candidate)
    return {
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "candidate_minus_baseline_median_seconds": (
            candidate_summary["median"] - baseline_summary["median"]
        ),
        "pass": candidate_summary["median"] <= baseline_summary["median"],
    }


def _process_wall(evidence: dict[str, Any], *, name: str) -> float:
    started = evidence.get("process_started_time_ns")
    finished = evidence.get("process_finished_time_ns")
    if (
        isinstance(started, bool)
        or not isinstance(started, int)
        or isinstance(finished, bool)
        or not isinstance(finished, int)
        or started <= 0
        or finished <= started
    ):
        raise CandidateAnalysisError(f"{name} process timestamps are invalid")
    return (finished - started) / 1e9


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
        if evidence.get("mode") != "natural" or bool(evidence.get("candidate")) != (
            variant == "candidate"
        ):
            raise CandidateAnalysisError(f"natural evidence identity mismatch: {path}")
        run_uuid = evidence.get("run_uuid")
        if not isinstance(run_uuid, str) or run_uuid in uuids:
            raise CandidateAnalysisError("natural runs must have unique process UUIDs")
        uuids.add(run_uuid)
        candidate = variant == "candidate"
        _validate_initialization(evidence, candidate=candidate)
        profile = _verified_artifact(evidence, "profile")
        result = _verified_artifact(evidence, "result")
        run = _parse_run(f"natural_{variant}_{index}", profile, None)
        if run.osu_path.resolve() != result:
            raise CandidateAnalysisError("natural evidence points at the wrong result")
        _validate_runtime_count(evidence, run)
        _k4_contract(profile, candidate=candidate)
        runs.append(run)
    return runs


def _load_fixed(root: Path, variant: str, count: int) -> list[dict[str, Any]]:
    paths = _evidence_files(root, variant)
    if len(paths) != count:
        raise CandidateAnalysisError(
            f"fixed-work {variant} requires {count} fresh processes, got {len(paths)}"
        )
    rows: list[dict[str, Any]] = []
    parsed_runs = []
    manifest_hashes = set()
    uuids = set()
    for path in paths:
        evidence = _json(path)
        if evidence.get("schema_version") != RUN_SCHEMA_VERSION:
            raise CandidateAnalysisError(f"fixed evidence schema mismatch: {path}")
        if evidence.get("mode") != "replay-fixed-work" or bool(
            evidence.get("candidate")
        ) != (variant == "candidate"):
            raise CandidateAnalysisError(f"fixed evidence identity mismatch: {path}")
        run_uuid = evidence.get("run_uuid")
        if not isinstance(run_uuid, str) or run_uuid in uuids:
            raise CandidateAnalysisError(
                "fixed-work runs must have unique process UUIDs"
            )
        uuids.add(run_uuid)
        if evidence.get("expected_main_steps") != EXPECTED_MAIN_STEPS:
            raise CandidateAnalysisError("fixed-work expected_main_steps changed")
        labels = evidence.get("labels")
        if not isinstance(labels, dict):
            raise CandidateAnalysisError("fixed-work evidence lacks labels")
        if set(labels) != {"timing_context", "main_generation"}:
            raise CandidateAnalysisError("fixed-work evidence has wrong label set")
        main = labels.get("main_generation", {})
        if main.get("logical_steps") != EXPECTED_MAIN_STEPS:
            raise CandidateAnalysisError(
                f"fixed-work run did not execute exactly {EXPECTED_MAIN_STEPS} main steps"
            )
        records = evidence.get("records")
        if not isinstance(records, list) or not records:
            raise CandidateAnalysisError("fixed-work evidence lacks execution records")
        if any(
            not isinstance(record, dict)
            or record.get("target_steps") != record.get("logical_steps")
            for record in records
        ):
            raise CandidateAnalysisError("fixed-work target/logical steps diverged")
        for label, summary in labels.items():
            label_records = [
                record for record in records if record.get("profile_label") == label
            ]
            if not isinstance(summary, dict) or summary != {
                "windows": len(label_records),
                "logical_steps": sum(
                    int(record["logical_steps"]) for record in label_records
                ),
            }:
                raise CandidateAnalysisError(
                    f"fixed-work evidence summary diverged for {label}"
                )
        candidate = variant == "candidate"
        _validate_initialization(evidence, candidate=candidate)
        profile = _verified_artifact(evidence, "profile")
        result = _verified_artifact(evidence, "result")
        manifest = _verified_artifact(evidence, "manifest")
        manifest_hashes.add(_sha256_file(manifest))
        parsed = _parse_run(f"fixed_{variant}_{len(rows)}", profile, None)
        if parsed.osu_path.resolve() != result:
            raise CandidateAnalysisError(
                "fixed-work evidence points at the wrong result"
            )
        _validate_runtime_count(evidence, parsed)
        contract = _k4_contract(
            profile,
            candidate=candidate,
            expected_fixed_main_steps=EXPECTED_MAIN_STEPS,
        )
        parsed_runs.append(parsed)
        model_seconds = parsed.metrics["main_model_seconds"]
        rows.append(
            {
                "main_steps": EXPECTED_MAIN_STEPS,
                "main_model_seconds": model_seconds,
                "fixed_work_tps": EXPECTED_MAIN_STEPS / model_seconds,
                "timing_tokens": int(parsed.metrics["timing_tokens"]),
                "timing_model_seconds": parsed.metrics["timing_model_seconds"],
                "timing_tps": parsed.metrics["timing_tps"],
                "timing_outer_stage_wall_seconds": parsed.metrics[
                    "timing_outer_stage_wall_seconds"
                ],
                "main_outer_stage_wall_seconds": parsed.metrics[
                    "main_outer_stage_wall_seconds"
                ],
                "complete_request_wall_seconds": parsed.metrics[
                    "complete_request_wall_seconds"
                ],
                "cold_process_outer_wall_seconds": parsed.metrics[
                    "cold_process_outer_wall_seconds"
                ],
                "setup_plus_capture_seconds": parsed.metrics[
                    "setup_plus_capture_seconds"
                ],
                "peak_cuda_memory_allocated_mb": float(
                    evidence["cuda_memory"]["max_allocated_bytes"]
                )
                / 1024**2,
                "k4_contract": contract,
            }
        )
    if len(manifest_hashes) != 1:
        raise CandidateAnalysisError("fixed-work runs did not share one exact manifest")
    _stable_natural(parsed_runs, f"fixed-work {variant}")
    return rows


def _stable_natural(runs: list[Any], variant: str) -> None:
    for label in ("timing_context", "main_generation"):
        token_hashes = {
            run.label_signatures[label]["token_stream_sha256"] for run in runs
        }
        stopping_hashes = {
            run.label_signatures[label]["stopping_sha256"] for run in runs
        }
        if len(token_hashes) != 1 or len(stopping_hashes) != 1:
            raise CandidateAnalysisError(
                f"natural {variant} {label} is not deterministic"
            )
    if len({run.osu_sha256 for run in runs}) != 1:
        raise CandidateAnalysisError(
            f"natural {variant} final map is not deterministic"
        )


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
    if (
        evidence.get("schema_version") != SUITE_SCHEMA_VERSION
        or bool(evidence.get("candidate")) != candidate
    ):
        raise CandidateAnalysisError(f"serial suite identity mismatch: {path}")
    run_uuid = evidence.get("run_uuid")
    if not isinstance(run_uuid, str) or not run_uuid:
        raise CandidateAnalysisError("serial suite lacks a process UUID")
    _validate_initialization(evidence, candidate=candidate)
    _verified_artifact(evidence, "song_manifest")
    startup = evidence.get("startup")
    if not isinstance(startup, dict) or not isinstance(
        startup.get("separate_timing_model"), bool
    ):
        raise CandidateAnalysisError("serial suite startup topology is invalid")
    expected_runtimes = 2 if startup["separate_timing_model"] else 1
    if startup.get("loaded_runtime_count") != expected_runtimes:
        raise CandidateAnalysisError("serial suite loaded runtime count is invalid")
    raw_runs = evidence.get("runs")
    if not isinstance(raw_runs, list) or len(raw_runs) != 5:
        raise CandidateAnalysisError("serial suite requires exactly five songs")
    parsed = []
    for index, row in enumerate(raw_runs):
        if not isinstance(row, dict):
            raise CandidateAnalysisError("serial suite run must be an object")
        profile = _verified_artifact(row, "profile")
        result = _verified_artifact(row, "result")
        run = _parse_run(
            f"suite_{'candidate' if candidate else 'baseline'}_{index}",
            profile,
            None,
        )
        if run.osu_path.resolve() != result:
            raise CandidateAnalysisError("serial suite points at the wrong result")
        _k4_contract(profile, candidate=candidate)
        if not isinstance(row.get("name"), str) or not row["name"]:
            raise CandidateAnalysisError("serial suite run lacks a song name")
        if not isinstance(row.get("audio_path"), str) or not isinstance(
            row.get("audio_sha256"), str
        ):
            raise CandidateAnalysisError("serial suite run lacks audio identity")
        parsed.append(run)
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
        "timing_context": _token_divergence(
            baseline_natural[0], candidate_natural[0], "timing_context"
        ),
        "main_generation": _token_divergence(
            baseline_natural[0], candidate_natural[0], "main_generation"
        ),
        "structure": _structure_divergence(baseline_natural[0], candidate_natural[0]),
    }

    baseline_fixed = _load_fixed(fixed_root, "baseline", fixed_repetitions)
    candidate_fixed = _load_fixed(fixed_root, "candidate", fixed_repetitions)
    fixed_tps = [row["fixed_work_tps"] for row in candidate_fixed]
    fixed_gate = statistics.median(fixed_tps) >= 500.0 and min(fixed_tps) >= 475.0

    fixed_wall_acceptance = {
        metric: _wall_comparison(
            [row[metric] for row in baseline_fixed],
            [row[metric] for row in candidate_fixed],
        )
        for metric in (
            "complete_request_wall_seconds",
            "cold_process_outer_wall_seconds",
            "setup_plus_capture_seconds",
        )
    }
    natural_wall_acceptance = {
        metric: _wall_comparison(
            [run.metrics[metric] for run in baseline_natural],
            [run.metrics[metric] for run in candidate_natural],
        )
        for metric in (
            "complete_request_wall_seconds",
            "cold_process_outer_wall_seconds",
            "setup_plus_capture_seconds",
        )
    }

    baseline_suite_evidence, baseline_suite = _load_suite(baseline_suite_path, False)
    candidate_suite_evidence, candidate_suite = _load_suite(candidate_suite_path, True)
    if not isinstance(
        baseline_suite_evidence.get("song_manifest_sha256"), str
    ) or baseline_suite_evidence[
        "song_manifest_sha256"
    ] != candidate_suite_evidence.get(
        "song_manifest_sha256"
    ):
        raise CandidateAnalysisError("baseline/candidate suite manifest differs")
    baseline_names = [row["name"] for row in baseline_suite_evidence["runs"]]
    candidate_names = [row["name"] for row in candidate_suite_evidence["runs"]]
    if baseline_names != candidate_names:
        raise CandidateAnalysisError("baseline/candidate suite song order differs")
    for baseline_row, candidate_row in zip(
        baseline_suite_evidence["runs"],
        candidate_suite_evidence["runs"],
        strict=True,
    ):
        if (
            baseline_row["audio_path"] != candidate_row["audio_path"]
            or baseline_row["audio_sha256"] != candidate_row["audio_sha256"]
        ):
            raise CandidateAnalysisError(
                "baseline/candidate suite audio identity differs"
            )
    suite_rows = []
    for name, baseline, candidate in zip(
        baseline_names, baseline_suite, candidate_suite, strict=True
    ):
        request_wall_pass = (
            candidate.metrics["complete_request_wall_seconds"]
            <= baseline.metrics["complete_request_wall_seconds"]
        )
        cold_wall_pass = (
            candidate.metrics["cold_process_outer_wall_seconds"]
            <= baseline.metrics["cold_process_outer_wall_seconds"]
        )
        setup_wall_pass = (
            candidate.metrics["setup_plus_capture_seconds"]
            <= baseline.metrics["setup_plus_capture_seconds"]
        )
        suite_rows.append(
            {
                "name": name,
                "baseline_timing_tokens": int(baseline.metrics["timing_tokens"]),
                "candidate_timing_tokens": int(candidate.metrics["timing_tokens"]),
                "baseline_timing_model_seconds": baseline.metrics[
                    "timing_model_seconds"
                ],
                "candidate_timing_model_seconds": candidate.metrics[
                    "timing_model_seconds"
                ],
                "baseline_timing_tps": baseline.metrics["timing_tps"],
                "candidate_timing_tps": candidate.metrics["timing_tps"],
                "baseline_timing_outer_stage_wall_seconds": baseline.metrics[
                    "timing_outer_stage_wall_seconds"
                ],
                "candidate_timing_outer_stage_wall_seconds": candidate.metrics[
                    "timing_outer_stage_wall_seconds"
                ],
                "baseline_main_tokens": int(baseline.metrics["main_tokens"]),
                "candidate_main_tokens": int(candidate.metrics["main_tokens"]),
                "baseline_main_model_seconds": baseline.metrics["main_model_seconds"],
                "candidate_main_model_seconds": candidate.metrics["main_model_seconds"],
                "baseline_main_tps": baseline.metrics["main_tps"],
                "candidate_main_tps": candidate.metrics["main_tps"],
                "baseline_main_outer_stage_wall_seconds": baseline.metrics[
                    "main_outer_stage_wall_seconds"
                ],
                "candidate_main_outer_stage_wall_seconds": candidate.metrics[
                    "main_outer_stage_wall_seconds"
                ],
                "baseline_complete_request_wall_seconds": baseline.metrics[
                    "complete_request_wall_seconds"
                ],
                "candidate_complete_request_wall_seconds": candidate.metrics[
                    "complete_request_wall_seconds"
                ],
                "baseline_cold_process_outer_wall_seconds": baseline.metrics[
                    "cold_process_outer_wall_seconds"
                ],
                "candidate_cold_process_outer_wall_seconds": candidate.metrics[
                    "cold_process_outer_wall_seconds"
                ],
                "baseline_setup_plus_capture_seconds": baseline.metrics[
                    "setup_plus_capture_seconds"
                ],
                "candidate_setup_plus_capture_seconds": candidate.metrics[
                    "setup_plus_capture_seconds"
                ],
                "complete_request_wall_nonregression": request_wall_pass,
                "cold_process_wall_nonregression": cold_wall_pass,
                "setup_plus_capture_nonregression": setup_wall_pass,
                "timing_token_divergence": _token_divergence(
                    baseline,
                    candidate,
                    "timing_context",
                ),
                "main_token_divergence": _token_divergence(
                    baseline,
                    candidate,
                    "main_generation",
                ),
                "structure_divergence": _structure_divergence(baseline, candidate),
            }
        )
    aggregate_tokens = sum(row["candidate_main_tokens"] for row in suite_rows)
    aggregate_model = sum(row["candidate_main_model_seconds"] for row in suite_rows)
    aggregate_tps = aggregate_tokens / aggregate_model
    aggregate_timing_tokens = sum(row["candidate_timing_tokens"] for row in suite_rows)
    aggregate_timing_model = sum(
        row["candidate_timing_model_seconds"] for row in suite_rows
    )
    aggregate_timing_tps = aggregate_timing_tokens / aggregate_timing_model
    suite_gate = aggregate_tps >= 500.0 and all(
        row["candidate_main_tps"] >= 475.0 for row in suite_rows
    )
    candidate_memory = _memory_stability(candidate_suite_evidence, memory_tolerance_mb)
    baseline_suite_request_wall = sum(
        row["baseline_complete_request_wall_seconds"] for row in suite_rows
    )
    candidate_suite_request_wall = sum(
        row["candidate_complete_request_wall_seconds"] for row in suite_rows
    )
    baseline_suite_process_wall = _process_wall(
        baseline_suite_evidence, name="baseline suite"
    )
    candidate_suite_process_wall = _process_wall(
        candidate_suite_evidence, name="candidate suite"
    )
    suite_wall_acceptance = {
        "every_song_complete_request_wall": all(
            row["complete_request_wall_nonregression"] for row in suite_rows
        ),
        "every_song_cold_process_wall": all(
            row["cold_process_wall_nonregression"] for row in suite_rows
        ),
        "every_song_setup_plus_capture": all(
            row["setup_plus_capture_nonregression"] for row in suite_rows
        ),
        "aggregate_complete_request_wall": (
            candidate_suite_request_wall <= baseline_suite_request_wall
        ),
        "serial_process_wall": candidate_suite_process_wall
        <= baseline_suite_process_wall,
    }
    wall_pass = all(
        comparison["pass"]
        for family in (fixed_wall_acceptance, natural_wall_acceptance)
        for comparison in family.values()
    ) and all(suite_wall_acceptance.values())
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "acceptance": {
            "fixed_work_exact_main_steps": EXPECTED_MAIN_STEPS,
            "fixed_work_median_tps_at_least_500": statistics.median(fixed_tps) >= 500.0,
            "fixed_work_no_run_below_475": min(fixed_tps) >= 475.0,
            "five_song_aggregate_tps_at_least_500": aggregate_tps >= 500.0,
            "five_song_no_song_below_475": all(
                row["candidate_main_tps"] >= 475.0 for row in suite_rows
            ),
            "serial_memory_stable": candidate_memory["pass"],
            "complete_request_cold_setup_nonregression": wall_pass,
            "pass": fixed_gate
            and suite_gate
            and candidate_memory["pass"]
            and wall_pass,
        },
        "fixed_work": {
            "baseline_runs": baseline_fixed,
            "candidate_runs": candidate_fixed,
            "baseline_tps": _summary([row["fixed_work_tps"] for row in baseline_fixed]),
            "candidate_tps": _summary(fixed_tps),
            "baseline_timing_tps": _summary(
                [row["timing_tps"] for row in baseline_fixed]
            ),
            "candidate_timing_tps": _summary(
                [row["timing_tps"] for row in candidate_fixed]
            ),
            "wall_acceptance": fixed_wall_acceptance,
        },
        "natural_salvalai": {
            "baseline_timing_tps": _summary(
                [run.metrics["timing_tps"] for run in baseline_natural]
            ),
            "candidate_timing_tps": _summary(
                [run.metrics["timing_tps"] for run in candidate_natural]
            ),
            "baseline_timing_model_seconds": _summary(
                [run.metrics["timing_model_seconds"] for run in baseline_natural]
            ),
            "candidate_timing_model_seconds": _summary(
                [run.metrics["timing_model_seconds"] for run in candidate_natural]
            ),
            "baseline_main_tps": _summary(
                [run.metrics["main_tps"] for run in baseline_natural]
            ),
            "candidate_main_tps": _summary(
                [run.metrics["main_tps"] for run in candidate_natural]
            ),
            "baseline_request_wall_seconds": _summary(
                [
                    run.metrics["complete_request_wall_seconds"]
                    for run in baseline_natural
                ]
            ),
            "candidate_request_wall_seconds": _summary(
                [
                    run.metrics["complete_request_wall_seconds"]
                    for run in candidate_natural
                ]
            ),
            "baseline_cold_process_outer_wall_seconds": _summary(
                [
                    run.metrics["cold_process_outer_wall_seconds"]
                    for run in baseline_natural
                ]
            ),
            "candidate_cold_process_outer_wall_seconds": _summary(
                [
                    run.metrics["cold_process_outer_wall_seconds"]
                    for run in candidate_natural
                ]
            ),
            "wall_acceptance": natural_wall_acceptance,
            "divergence": natural_divergence,
        },
        "five_song_serial": {
            "songs": suite_rows,
            "candidate_aggregate_main_tokens": aggregate_tokens,
            "candidate_aggregate_main_model_seconds": aggregate_model,
            "candidate_aggregate_main_tps": aggregate_tps,
            "candidate_aggregate_timing_tokens": aggregate_timing_tokens,
            "candidate_aggregate_timing_model_seconds": aggregate_timing_model,
            "candidate_aggregate_timing_tps": aggregate_timing_tps,
            "baseline_complete_request_wall_seconds": baseline_suite_request_wall,
            "candidate_complete_request_wall_seconds": candidate_suite_request_wall,
            "baseline_process_wall_seconds": baseline_suite_process_wall,
            "candidate_process_wall_seconds": candidate_suite_process_wall,
            "baseline_startup": baseline_suite_evidence["startup"],
            "candidate_startup": candidate_suite_evidence["startup"],
            "candidate_memory_stability": candidate_memory,
            "wall_acceptance": suite_wall_acceptance,
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
        f"fixed_work.wall_nonregression={str(all(item['pass'] for item in report['fixed_work']['wall_acceptance'].values())).lower()}",
        f"natural.candidate_timing_tps_median={natural['candidate_timing_tps']['median']:.9f}",
        f"natural.candidate_main_tps_median={natural['candidate_main_tps']['median']:.9f}",
        f"natural.candidate_request_wall_median={natural['candidate_request_wall_seconds']['median']:.9f}",
        f"natural.candidate_cold_process_outer_wall_median={natural['candidate_cold_process_outer_wall_seconds']['median']:.9f}",
        f"natural.wall_nonregression={str(all(item['pass'] for item in natural['wall_acceptance'].values())).lower()}",
        f"five_song.candidate_aggregate_main_tps={suite['candidate_aggregate_main_tps']:.9f}",
        f"five_song.candidate_aggregate_timing_tps={suite['candidate_aggregate_timing_tps']:.9f}",
        f"five_song.baseline_complete_request_wall_seconds={suite['baseline_complete_request_wall_seconds']:.9f}",
        f"five_song.candidate_complete_request_wall_seconds={suite['candidate_complete_request_wall_seconds']:.9f}",
        f"five_song.candidate_process_wall_seconds={suite['candidate_process_wall_seconds']:.9f}",
        f"five_song.wall_nonregression={str(all(suite['wall_acceptance'].values())).lower()}",
        f"five_song.memory_stable={str(suite['candidate_memory_stability']['pass']).lower()}",
    ]
    for row in suite["songs"]:
        lines.append(
            f"song.{row['name']}.candidate_main_tps={row['candidate_main_tps']:.9f},"
            f"candidate_request_wall={row['candidate_complete_request_wall_seconds']:.9f},"
            f"token_count_delta={row['main_token_divergence']['token_count_delta']},"
            f"stopping_equal={str(row['main_token_divergence']['stopping_equal']).lower()},"
            f"objects_delta={row['structure_divergence']['scalar_deltas']['hit_objects']}"
        )
    return "\n".join(lines) + "\n"


def analysis_exit_code(report: dict[str, Any]) -> int:
    acceptance = report.get("acceptance")
    if not isinstance(acceptance, dict) or not isinstance(acceptance.get("pass"), bool):
        raise CandidateAnalysisError("analysis report lacks a boolean acceptance.pass")
    return 0 if acceptance["pass"] else 3


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
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.text_output.write_text(text_report(report), encoding="utf-8")
    raise SystemExit(analysis_exit_code(report))


if __name__ == "__main__":
    main()
