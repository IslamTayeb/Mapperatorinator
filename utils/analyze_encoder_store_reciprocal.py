"""Analyze the strict-FP32 all-window encoder-precompute ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from osuT5.osuT5.inference.optimized.scout.encoder_store import (
    STORE_VERSION,
    SUPPORTED_LABELS,
)
from utils import analyze_fp32_fresh_baseline as baseline
from utils import nsight_agent_profile as nsight


SCHEMA_VERSION = "mapperatorinator.strict-fp32-encoder-precompute.v1"
DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16)
MINIMUM_REQUEST_SAVING_FRACTION = 0.05
RTX_2080_TI_TOTAL_BYTES = 11_264 * 1024 * 1024


class EncoderGateError(RuntimeError):
    pass


def _load(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EncoderGateError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EncoderGateError(f"{name} must contain a JSON object")
    return value


def _finite(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EncoderGateError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        raise EncoderGateError(f"{name} must be finite and non-negative")
    return result


def _integer(value: Any, *, name: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EncoderGateError(f"{name} must be an integer")
    if value < 0 or (positive and value <= 0):
        raise EncoderGateError(f"{name} must be non-negative")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_and_result(run_dir: Path) -> tuple[dict[str, Any], Path]:
    profile_path = baseline._path_from_pointer(run_dir, "profile-path.txt")
    result_path = baseline._path_from_pointer(run_dir, "result-path.txt")
    return nsight._load_inference_profile(profile_path), result_path


def _strict_runtime(
    profile: Mapping[str, Any],
    *,
    expected_commit: str,
    expected_branch: str,
    audit: bool,
) -> None:
    try:
        baseline._runtime_contract(
            profile,
            expected_commit=expected_commit,
            expected_branch=expected_branch,
            exactness_audit=audit,
        )
    except baseline.BaselineError as exc:
        raise EncoderGateError(str(exc)) from exc


def _store_contract(
    profile: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    batch_size: int,
) -> dict[str, Any]:
    metadata = profile.get("metadata")
    if not isinstance(metadata, dict):
        raise EncoderGateError("candidate profile metadata is missing")
    profile_store = metadata.get("optimized_batched_encoder_store")
    external_store = evidence.get("encoder_store")
    if not isinstance(profile_store, dict) or not isinstance(external_store, dict):
        raise EncoderGateError("candidate encoder-store evidence is missing")
    if profile_store != external_store:
        raise EncoderGateError("profile and external encoder-store evidence differ")
    required = {
        "version": STORE_VERSION,
        "production_default": False,
        "selector_changes": False,
        "decoder_changes": False,
        "model_forward_changes": False,
        "server_changes": False,
        "precision": "fp32",
        "strict_fp32": True,
        "strict_exactness_required_for_promotion": True,
        "batch_size": batch_size,
        "labels_completed": list(SUPPORTED_LABELS),
        "active_processor_count": 0,
    }
    failures = {
        key: {"expected": expected, "actual": external_store.get(key)}
        for key, expected in required.items()
        if external_store.get(key) != expected
    }
    if failures:
        raise EncoderGateError(f"candidate store contract changed: {failures}")
    entries = external_store.get("entries")
    if not isinstance(entries, list) or len(entries) not in {1, 2}:
        raise EncoderGateError("candidate must expose one shared or two stage stores")
    total_complete = 0.0
    total_bytes = 0
    labels: list[str] = []
    charged_entries: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise EncoderGateError(f"store entry {index} must be an object")
        times = {
            key: _finite(entry.get(key), name=f"entry[{index}].{key}")
            for key in (
                "batch_setup_seconds",
                "input_copy_seconds",
                "encoder_synchronized_seconds",
                "storage_allocation_seconds",
                "output_store_copy_seconds",
                "complete_precompute_seconds",
            )
        }
        if times["encoder_synchronized_seconds"] <= 0:
            raise EncoderGateError(f"store entry {index} encoder time is not positive")
        accounted = sum(
            times[key]
            for key in (
                "batch_setup_seconds",
                "input_copy_seconds",
                "encoder_synchronized_seconds",
                "storage_allocation_seconds",
                "output_store_copy_seconds",
            )
        )
        if accounted > times["complete_precompute_seconds"] + 0.02:
            raise EncoderGateError(
                f"store entry {index} component timings exceed complete wall"
            )
        memory = {
            key: _integer(entry.get(key), name=f"entry[{index}].{key}")
            for key in (
                "output_store_bytes",
                "baseline_allocated_vram_bytes",
                "baseline_reserved_vram_bytes",
                "peak_allocated_vram_bytes",
                "peak_reserved_vram_bytes",
                "incremental_peak_allocated_vram_bytes",
                "incremental_peak_reserved_vram_bytes",
            )
        }
        if memory["output_store_bytes"] <= 0:
            raise EncoderGateError(f"store entry {index} output storage is empty")
        if (
            memory["peak_allocated_vram_bytes"]
            - memory["baseline_allocated_vram_bytes"]
            != memory["incremental_peak_allocated_vram_bytes"]
        ):
            raise EncoderGateError(f"store entry {index} allocated delta is invalid")
        if memory["incremental_peak_allocated_vram_bytes"] < memory["output_store_bytes"]:
            raise EncoderGateError(f"store entry {index} does not charge output storage")
        uses = entry.get("uses")
        entry_labels = entry.get("labels")
        if not isinstance(uses, dict) or not isinstance(entry_labels, list):
            raise EncoderGateError(f"store entry {index} reuse evidence is missing")
        for label in entry_labels:
            row = uses.get(label)
            if not isinstance(row, dict):
                raise EncoderGateError(f"store entry {index} lacks use {label}")
            if _integer(row.get("rows_reused"), name=f"entry[{index}].{label}.rows", positive=True) <= 0:
                raise EncoderGateError(f"store entry {index} reused no rows")
            if row.get("per_window_device_copy") is not False:
                raise EncoderGateError("encoder store performs a per-window device copy")
        labels.extend(str(label) for label in entry_labels)
        total_complete += times["complete_precompute_seconds"]
        total_bytes += memory["output_store_bytes"]
        charged_entries.append({**times, **memory, "labels": entry_labels})
    if sorted(labels) != sorted(SUPPORTED_LABELS):
        raise EncoderGateError(f"store stage labels changed: {labels}")
    if not math.isclose(
        _finite(
            external_store.get("total_complete_precompute_seconds"),
            name="total_complete_precompute_seconds",
        ),
        total_complete,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise EncoderGateError("store total precompute wall does not reconcile")
    if _integer(
        external_store.get("total_output_store_bytes"),
        name="total_output_store_bytes",
        positive=True,
    ) != total_bytes:
        raise EncoderGateError("store total output bytes do not reconcile")
    return {
        "entry_count": len(entries),
        "shared_across_timing_main": external_store.get(
            "shared_across_timing_main"
        ),
        "complete_precompute_seconds": total_complete,
        "output_store_bytes": total_bytes,
        "peak_allocated_vram_bytes": max(
            entry["peak_allocated_vram_bytes"] for entry in charged_entries
        ),
        "incremental_peak_allocated_vram_bytes": sum(
            entry["incremental_peak_allocated_vram_bytes"]
            for entry in charged_entries
        ),
        "entries": charged_entries,
    }


def _label_signature(profile: Mapping[str, Any], label: str) -> dict[str, Any]:
    value = nsight._profile_label_signature(profile, label)
    if value.get("status") != "available" or not value.get("self_consistent"):
        raise EncoderGateError(f"{label} token/stopping signature is unavailable")
    return value


def _candidate_metrics(profile: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        label: baseline._label_metrics(profile, label) for label in SUPPORTED_LABELS
    }
    result["request_to_final_osu_wall_seconds"] = baseline._request_wall_seconds(
        profile
    )
    result["peak_cuda_memory_mb"] = baseline._peak_memory_mb(profile)
    return result


def _drift_row(audit: Mapping[str, Any], batch_size: int, *, name: str) -> dict[str, Any]:
    metadata = audit.get("metadata")
    variants = audit.get("variants")
    if not isinstance(metadata, dict) or not isinstance(variants, dict):
        raise EncoderGateError(f"{name} encoder audit is malformed")
    if metadata.get("batch_sizes") != list(DEFAULT_BATCH_SIZES):
        raise EncoderGateError(f"{name} encoder audit batch ladder changed")
    row = variants.get(str(batch_size))
    if not isinstance(row, dict):
        raise EncoderGateError(f"{name} encoder audit lacks B{batch_size}")
    drift = row.get("encoder_drift_vs_b1")
    if not isinstance(drift, dict):
        raise EncoderGateError(f"{name} B{batch_size} lacks drift")
    windows = _integer(drift.get("window_count"), name=f"{name}.windows", positive=True)
    exact_windows = _integer(
        drift.get("exact_window_count"), name=f"{name}.exact_windows"
    )
    max_abs = _finite(drift.get("max_abs"), name=f"{name}.max_abs")
    return {
        "live_window_count": windows,
        "exact_window_count": exact_windows,
        "max_abs": max_abs,
        "mean_window_max_abs": _finite(
            drift.get("mean_window_max_abs"), name=f"{name}.mean_window_max_abs"
        ),
        "encoder_exact": exact_windows == windows and max_abs == 0.0,
        "complete_precompute_seconds": _finite(
            row.get("complete_precompute_seconds"),
            name=f"{name}.complete_precompute_seconds",
            positive=True,
        ),
        "complete_seconds_saved_vs_b1": float(
            row.get("complete_seconds_saved_vs_b1")
        ),
        "input_copy_seconds": _finite(
            row.get("input_copy_seconds"), name=f"{name}.input_copy_seconds"
        ),
        "storage_allocation_seconds": _finite(
            row.get("storage_allocation_seconds"),
            name=f"{name}.storage_allocation_seconds",
        ),
        "output_store_copy_seconds": _finite(
            row.get("output_store_copy_seconds"),
            name=f"{name}.output_store_copy_seconds",
        ),
        "output_store_bytes": _integer(
            row.get("output_store_bytes"),
            name=f"{name}.output_store_bytes",
            positive=True,
        ),
        "incremental_peak_vram_bytes": _integer(
            row.get("incremental_peak_vram_bytes"),
            name=f"{name}.incremental_peak_vram_bytes",
            positive=True,
        ),
    }


def analyze(
    baseline_run_root: Path,
    candidate_run_root: Path,
    *,
    main_audit_path: Path,
    timing_audit_path: Path,
    expected_commit: str,
    expected_branch: str,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
) -> dict[str, Any]:
    batch_sizes = tuple(batch_sizes)
    if batch_sizes != DEFAULT_BATCH_SIZES:
        raise EncoderGateError(
            f"strict encoder ladder must be {DEFAULT_BATCH_SIZES}, got {batch_sizes}"
        )
    if len(expected_commit) != 40:
        raise EncoderGateError("expected commit must be a full SHA")
    baseline_analysis = _load(
        baseline_run_root / "analysis.json", name="baseline analysis"
    )
    if baseline_analysis.get("status") != "PASS":
        raise EncoderGateError("fresh strict-FP32 baseline did not pass")
    baseline_control, baseline_result = _profile_and_result(
        baseline_run_root / "run-01"
    )
    baseline_audit, baseline_audit_result = _profile_and_result(
        baseline_run_root / "exactness-audit"
    )
    baseline_exactness = baseline._strict_exactness_contract(baseline_audit)
    baseline_workload = baseline._workload_contract(baseline_control)
    baseline_preset = baseline._preset_contract(baseline_control)
    baseline_records = baseline._record_contract(baseline_control)
    baseline_output_sha = _sha256_file(baseline_result)
    baseline_structure = nsight.summarize_osu_structure(baseline_result)
    if _sha256_file(baseline_audit_result) != baseline_output_sha:
        raise EncoderGateError("baseline control and exactness output bytes differ")
    baseline_signatures = {
        label: _label_signature(baseline_control, label)
        for label in SUPPORTED_LABELS
    }
    baseline_request_median = _finite(
        baseline_analysis["aggregates"]["request_to_final_osu_wall_seconds"][
            "median"
        ],
        name="baseline request median",
        positive=True,
    )
    required_request_saving = (
        baseline_request_median * MINIMUM_REQUEST_SAVING_FRACTION
    )
    main_audit = _load(main_audit_path, name="main encoder audit")
    timing_audit = _load(timing_audit_path, name="timing encoder audit")

    variants: dict[str, Any] = {}
    for batch_size in batch_sizes:
        variant_dir = candidate_run_root / f"b{batch_size}"
        control_dir = variant_dir / "control"
        audit_dir = variant_dir / "exactness-audit"
        control, control_result = _profile_and_result(control_dir)
        exactness_profile, exactness_result = _profile_and_result(audit_dir)
        _strict_runtime(
            control,
            expected_commit=expected_commit,
            expected_branch=expected_branch,
            audit=False,
        )
        _strict_runtime(
            exactness_profile,
            expected_commit=expected_commit,
            expected_branch=expected_branch,
            audit=True,
        )
        control_evidence = _load(
            control_dir / "evidence.json", name=f"B{batch_size} control evidence"
        )
        audit_evidence = _load(
            audit_dir / "evidence.json", name=f"B{batch_size} audit evidence"
        )
        if control_evidence.get("profile_pass_kind") != "untraced_control":
            raise EncoderGateError(f"B{batch_size} control pass kind changed")
        if audit_evidence.get("profile_pass_kind") != "exactness_audit":
            raise EncoderGateError(f"B{batch_size} audit pass kind changed")
        control_store = _store_contract(
            control, control_evidence, batch_size=batch_size
        )
        audit_store = _store_contract(
            exactness_profile, audit_evidence, batch_size=batch_size
        )
        if (
            control_store["output_store_bytes"]
            != audit_store["output_store_bytes"]
        ):
            raise EncoderGateError(f"B{batch_size} store size is not repeatable")

        control_signatures = {
            label: _label_signature(control, label) for label in SUPPORTED_LABELS
        }
        audit_signatures = {
            label: _label_signature(exactness_profile, label)
            for label in SUPPORTED_LABELS
        }
        control_output_sha = _sha256_file(control_result)
        audit_output_sha = _sha256_file(exactness_result)
        control_structure = nsight.summarize_osu_structure(control_result)
        audit_structure = nsight.summarize_osu_structure(exactness_result)
        candidate_exactness = baseline._strict_exactness_contract(
            exactness_profile
        )
        main_drift = _drift_row(main_audit, batch_size, name="main")
        timing_drift = _drift_row(timing_audit, batch_size, name="timing")
        checks = {
            "encoder_main_exact": main_drift["encoder_exact"],
            "encoder_timing_exact": timing_drift["encoder_exact"],
            "workload_control": baseline._workload_contract(control)
            == baseline_workload,
            "workload_audit": baseline._workload_contract(exactness_profile)
            == baseline_workload,
            "preset_control": baseline._preset_contract(control) == baseline_preset,
            "preset_audit": baseline._preset_contract(exactness_profile)
            == baseline_preset,
            "dispatch_and_graph_control": baseline._record_contract(control)
            == baseline_records,
            "dispatch_and_graph_audit": baseline._record_contract(
                exactness_profile
            )
            == baseline_records,
            "strict_rng_and_cache": candidate_exactness == baseline_exactness,
            "output_bytes_control": control_output_sha == baseline_output_sha,
            "output_bytes_audit": audit_output_sha == baseline_output_sha,
            "output_structure_control": control_structure == baseline_structure,
            "output_structure_audit": audit_structure == baseline_structure,
        }
        for label in SUPPORTED_LABELS:
            checks[f"{label}_tokens_control"] = (
                control_signatures[label]["token_stream_sha256"]
                == baseline_signatures[label]["token_stream_sha256"]
            )
            checks[f"{label}_stopping_control"] = (
                control_signatures[label]["stopping_sha256"]
                == baseline_signatures[label]["stopping_sha256"]
            )
            checks[f"{label}_tokens_audit"] = (
                audit_signatures[label]["token_stream_sha256"]
                == baseline_signatures[label]["token_stream_sha256"]
            )
            checks[f"{label}_stopping_audit"] = (
                audit_signatures[label]["stopping_sha256"]
                == baseline_signatures[label]["stopping_sha256"]
            )
        exact_pass = all(checks.values())
        metrics = _candidate_metrics(control)
        request_saving = (
            baseline_request_median
            - metrics["request_to_final_osu_wall_seconds"]
        )
        request_saving_pass = request_saving >= required_request_saving
        vram_pass = (
            metrics["peak_cuda_memory_mb"] * 1024 * 1024
            <= RTX_2080_TI_TOTAL_BYTES
            and control_store["peak_allocated_vram_bytes"]
            <= RTX_2080_TI_TOTAL_BYTES
        )
        promotion_pass = exact_pass and request_saving_pass and vram_pass
        variants[str(batch_size)] = {
            "classification": (
                "strict-exact-and-fast"
                if promotion_pass
                else "strict-exact-insufficient-complete-wall-gain"
                if exact_pass
                else "documented-drift-not-promotion-eligible"
            ),
            "strict_exactness_pass": exact_pass,
            "request_saving_pass": request_saving_pass,
            "vram_pass": vram_pass,
            "promotion_pass": promotion_pass,
            "checks": checks,
            "failed_checks": sorted(
                name for name, passed in checks.items() if not passed
            ),
            "encoder_drift": {"main": main_drift, "timing": timing_drift},
            "performance": {
                "baseline_request_median_seconds": baseline_request_median,
                "candidate_request_seconds": metrics[
                    "request_to_final_osu_wall_seconds"
                ],
                "request_seconds_saved": request_saving,
                "request_speedup_pct": request_saving
                / baseline_request_median
                * 100.0,
                "required_request_seconds_saved": required_request_saving,
                "runner_wall_seconds": _finite(
                    control_evidence.get("run_wall_seconds"),
                    name=f"B{batch_size}.runner_wall_seconds",
                    positive=True,
                ),
                "peak_cuda_memory_mb": metrics["peak_cuda_memory_mb"],
                "timing": metrics["timing_context"],
                "main": metrics["main_generation"],
            },
            "store": control_store,
            "output": {
                "baseline_sha256": baseline_output_sha,
                "control_sha256": control_output_sha,
                "audit_sha256": audit_output_sha,
                "structure": control_structure,
            },
        }

    winners = [
        int(batch_size)
        for batch_size, row in variants.items()
        if row["promotion_pass"]
    ]
    exact_variants = [
        int(batch_size)
        for batch_size, row in variants.items()
        if row["strict_exactness_pass"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": (
            "PASS_STRICT_FP32_ENCODER_PRECOMPUTE"
            if winners
            else "STOP_ENCODER_PRECOMPUTE"
        ),
        "production_changes": False,
        "decoder_changes": False,
        "server_changes": False,
        "batch_sizes": list(batch_sizes),
        "baseline_run_root": str(baseline_run_root),
        "candidate_run_root": str(candidate_run_root),
        "baseline_request_median_seconds": baseline_request_median,
        "minimum_request_saving_fraction": MINIMUM_REQUEST_SAVING_FRACTION,
        "minimum_request_saving_seconds": required_request_saving,
        "strict_exact_batch_sizes": exact_variants,
        "promotion_batch_sizes": winners,
        "variants": variants,
    }


def _text(report: Mapping[str, Any]) -> str:
    lines = [
        f"decision={report['decision']}",
        f"baseline_request_median_seconds={report['baseline_request_median_seconds']:.6f}",
        f"minimum_request_saving_seconds={report['minimum_request_saving_seconds']:.6f}",
        "strict_exact_batch_sizes="
        + ",".join(str(value) for value in report["strict_exact_batch_sizes"]),
        "promotion_batch_sizes="
        + ",".join(str(value) for value in report["promotion_batch_sizes"]),
    ]
    for batch_size, row in report["variants"].items():
        lines.extend(
            (
                f"b{batch_size}_classification={row['classification']}",
                f"b{batch_size}_strict_exact={str(row['strict_exactness_pass']).lower()}",
                f"b{batch_size}_request_seconds_saved={row['performance']['request_seconds_saved']:.6f}",
                f"b{batch_size}_main_encoder_max_abs={row['encoder_drift']['main']['max_abs']}",
                f"b{batch_size}_timing_encoder_max_abs={row['encoder_drift']['timing']['max_abs']}",
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run-root", type=Path, required=True)
    parser.add_argument("--candidate-run-root", type=Path, required=True)
    parser.add_argument("--main-audit", type=Path, required=True)
    parser.add_argument("--timing-audit", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(
        args.baseline_run_root,
        args.candidate_run_root,
        main_audit_path=args.main_audit,
        timing_audit_path=args.timing_audit,
        expected_commit=args.expected_commit,
        expected_branch=args.expected_branch,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.text_output.write_text(_text(report), encoding="utf-8")
    print(_text(report), end="")
    return 0 if report["decision"] == "PASS_STRICT_FP32_ENCODER_PRECOMPUTE" else 3


if __name__ == "__main__":
    raise SystemExit(main())
