"""Analyze a same-precision all-window encoder-precompute ladder."""

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
from utils import analyze_fp16_fresh_baseline as fp16_baseline
from utils import nsight_agent_profile as nsight


SCHEMA_VERSION = "mapperatorinator.shared-precision-encoder-precompute.v2"
DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16)
SUPPORTED_PRECISIONS = ("fp32", "fp16")
EXPECTED_PRESET_VERSIONS = {
    "fp32": "accepted-fp32-native-cross-mlp-289-v3",
    "fp16": fp16_baseline.EXPECTED_PRESET_VERSION,
}
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


def _signed_finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EncoderGateError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EncoderGateError(f"{name} must be finite")
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


def _output_contract(
    profile: Mapping[str, Any],
    result_path: Path,
    *,
    name: str,
) -> tuple[str, dict[str, Any]]:
    metadata = profile.get("metadata")
    if not isinstance(metadata, dict):
        raise EncoderGateError(f"{name} profile metadata is missing")
    output_sha = _sha256_file(result_path)
    if metadata.get("result_file_sha256") != output_sha:
        raise EncoderGateError(f"{name} profile/output hash mismatch")
    structure = nsight.summarize_osu_structure(result_path)
    if structure.get("malformed_by_section") != {
        "TimingPoints": 0,
        "HitObjects": 0,
    } or structure.get("nonfinite_values") != 0:
        raise EncoderGateError(f"{name} produced an invalid map")
    return output_sha, structure


def _same_precision_runtime(
    profile: Mapping[str, Any],
    *,
    expected_commit: str,
    expected_branch: str,
    audit: bool,
    precision: str,
) -> None:
    try:
        if precision == "fp32":
            baseline._runtime_contract(
                profile,
                expected_commit=expected_commit,
                expected_branch=expected_branch,
                exactness_audit=audit,
            )
        elif precision == "fp16":
            fp16_baseline._runtime_contract(
                profile,
                expected_commit=expected_commit,
                expected_branch=expected_branch,
                audit=audit,
            )
        else:
            raise EncoderGateError(f"unsupported precision {precision!r}")
    except (baseline.BaselineError, fp16_baseline.FP16BaselineError) as exc:
        raise EncoderGateError(str(exc)) from exc


def _store_contract(
    profile: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    batch_size: int,
    precision: str = "fp32",
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
        "precision": precision,
        "strict_fp32": precision == "fp32",
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
        live_window_count = _integer(
            entry.get("live_window_count"),
            name=f"entry[{index}].live_window_count",
            positive=True,
        )
        if _integer(
            entry.get("batch_size"),
            name=f"entry[{index}].batch_size",
            positive=True,
        ) != batch_size:
            raise EncoderGateError(f"store entry {index} batch size changed")
        _integer(
            entry.get("batch_count"),
            name=f"entry[{index}].batch_count",
            positive=True,
        )
        frame_contract = entry.get("frame_contract")
        if not isinstance(frame_contract, dict) or (
            frame_contract.get("dtype") != "torch.float32"
            or frame_contract.get("device") != "cpu"
        ):
            raise EncoderGateError(
                f"store entry {index} source-frame contract changed"
            )
        expected_output_dtype = {
            "fp32": "torch.float32",
            "fp16": "torch.float16",
        }[precision]
        output_shape = entry.get("output_shape")
        if (
            not isinstance(output_shape, list)
            or not output_shape
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in output_shape
            )
            or output_shape[0] != live_window_count
            or entry.get("output_dtype") != expected_output_dtype
        ):
            raise EncoderGateError(
                f"store entry {index} output tensor contract changed"
            )
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
        expected_store_bytes = math.prod(output_shape) * (
            4 if precision == "fp32" else 2
        )
        if memory["output_store_bytes"] != expected_store_bytes:
            raise EncoderGateError(
                f"store entry {index} output storage bytes do not match shape/dtype"
            )
        if (
            memory["peak_allocated_vram_bytes"]
            - memory["baseline_allocated_vram_bytes"]
            != memory["incremental_peak_allocated_vram_bytes"]
        ):
            raise EncoderGateError(f"store entry {index} allocated delta is invalid")
        if memory["incremental_peak_allocated_vram_bytes"] < memory["output_store_bytes"]:
            raise EncoderGateError(f"store entry {index} does not charge output storage")
        if (
            memory["peak_reserved_vram_bytes"]
            - memory["baseline_reserved_vram_bytes"]
            != memory["incremental_peak_reserved_vram_bytes"]
        ):
            raise EncoderGateError(f"store entry {index} reserved delta is invalid")
        if (
            memory["peak_allocated_vram_bytes"] > RTX_2080_TI_TOTAL_BYTES
            or memory["peak_reserved_vram_bytes"] > RTX_2080_TI_TOTAL_BYTES
        ):
            raise EncoderGateError(f"store entry {index} exceeds RTX 2080 Ti VRAM")
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


def _drift_row(
    audit: Mapping[str, Any],
    batch_size: int,
    *,
    name: str,
    precision: str = "fp32",
) -> dict[str, Any]:
    metadata = audit.get("metadata")
    variants = audit.get("variants")
    if not isinstance(metadata, dict) or not isinstance(variants, dict):
        raise EncoderGateError(f"{name} encoder audit is malformed")
    if metadata.get("batch_sizes") != list(DEFAULT_BATCH_SIZES):
        raise EncoderGateError(f"{name} encoder audit batch ladder changed")
    if metadata.get("source_precision") != precision:
        raise EncoderGateError(
            f"{name} encoder audit precision changed: "
            f"{metadata.get('source_precision')!r}"
        )
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
        "complete_seconds_saved_vs_b1": _signed_finite(
            row.get("complete_seconds_saved_vs_b1"),
            name=f"{name}.complete_seconds_saved_vs_b1",
        ),
        "batch_setup_seconds": _finite(
            row.get("batch_setup_seconds"), name=f"{name}.batch_setup_seconds"
        ),
        "input_copy_seconds": _finite(
            row.get("input_copy_seconds"), name=f"{name}.input_copy_seconds"
        ),
        "encoder_synchronized_seconds": _finite(
            row.get("encoder_synchronized_seconds"),
            name=f"{name}.encoder_synchronized_seconds",
            positive=True,
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
        "baseline_allocated_vram_bytes": _integer(
            row.get("baseline_allocated_vram_bytes"),
            name=f"{name}.baseline_allocated_vram_bytes",
        ),
        "peak_allocated_vram_bytes": _integer(
            row.get("peak_allocated_vram_bytes"),
            name=f"{name}.peak_allocated_vram_bytes",
            positive=True,
        ),
        "verification_device_to_cpu_seconds_excluded": _finite(
            row.get("verification_device_to_cpu_seconds_excluded"),
            name=f"{name}.verification_device_to_cpu_seconds_excluded",
        ),
    }


def _component_audit_contract(
    audit: Mapping[str, Any],
    *,
    name: str,
    precision: str,
) -> None:
    metadata = audit.get("metadata")
    variants = audit.get("variants")
    manifest = audit.get("input_manifest")
    if not all(isinstance(value, dict) for value in (metadata, variants, manifest)):
        raise EncoderGateError(f"{name} encoder audit is malformed")
    required = {
        "result_class": "same-precision-drift-component-ceiling",
        "production_wiring": False,
        "server_changes": False,
        "source_engine": "optimized",
        "source_precision": precision,
        "source_attn_implementation": "sdpa",
        "batch_sizes": list(DEFAULT_BATCH_SIZES),
    }
    failures = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in required.items()
        if metadata.get(key) != expected
    }
    if failures:
        raise EncoderGateError(f"{name} encoder audit contract changed: {failures}")
    if "2080 Ti" not in str(metadata.get("cuda_device", "")):
        raise EncoderGateError(f"{name} encoder audit did not use an RTX 2080 Ti")
    if _integer(
        metadata.get("max_batch_size"),
        name=f"{name}.max_batch_size",
        positive=True,
    ) < max(DEFAULT_BATCH_SIZES):
        raise EncoderGateError(f"{name} encoder audit cannot cover B16")
    live_windows = _integer(
        metadata.get("live_window_count"),
        name=f"{name}.live_window_count",
        positive=True,
    )
    if manifest.get("live_window_count") != live_windows:
        raise EncoderGateError(f"{name} encoder input manifest count changed")
    combined_sha = manifest.get("combined_sha256")
    if not isinstance(combined_sha, str) or len(combined_sha) != 64:
        raise EncoderGateError(f"{name} encoder input manifest hash is invalid")
    if set(variants) != {str(value) for value in DEFAULT_BATCH_SIZES}:
        raise EncoderGateError(f"{name} encoder audit variants changed")
    runtime = metadata.get("runtime")
    effective = runtime.get("optimized_effective_config") if isinstance(runtime, dict) else None
    if not isinstance(effective, dict) or (
        effective.get("version") != EXPECTED_PRESET_VERSIONS[precision]
        or effective.get("precision") != precision
        or effective.get("attn_implementation") != "sdpa"
        or effective.get("batch_size") != 1
    ):
        raise EncoderGateError(f"{name} encoder audit source preset changed")


def analyze(
    baseline_run_root: Path,
    candidate_run_root: Path,
    *,
    main_audit_path: Path,
    timing_audit_path: Path,
    expected_commit: str,
    expected_branch: str,
    precision: str = "fp32",
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
) -> dict[str, Any]:
    batch_sizes = tuple(batch_sizes)
    if batch_sizes != DEFAULT_BATCH_SIZES:
        raise EncoderGateError(
            f"encoder ladder must be {DEFAULT_BATCH_SIZES}, got {batch_sizes}"
        )
    if precision not in SUPPORTED_PRECISIONS:
        raise EncoderGateError(
            f"precision must be one of {SUPPORTED_PRECISIONS}, got {precision!r}"
        )
    if len(expected_commit) != 40:
        raise EncoderGateError("expected commit must be a full SHA")
    baseline_analysis = _load(
        baseline_run_root / "analysis.json", name="baseline analysis"
    )
    if baseline_analysis.get("status") != "PASS":
        raise EncoderGateError(f"fresh accepted-{precision} baseline did not pass")
    if precision == "fp16" and baseline_analysis.get("precision") != "fp16":
        raise EncoderGateError("FP16 baseline analysis precision changed")
    baseline_control, baseline_result = _profile_and_result(
        baseline_run_root / "run-01"
    )
    baseline_metadata = baseline_control.get("metadata")
    if not isinstance(baseline_metadata, dict) or baseline_metadata.get(
        "precision"
    ) != precision:
        raise EncoderGateError("baseline profile precision changed")
    baseline_audit_name = (
        "exactness-audit-01" if precision == "fp16" else "exactness-audit"
    )
    baseline_audit, baseline_audit_result = _profile_and_result(
        baseline_run_root / baseline_audit_name
    )
    baseline_exactness = baseline._strict_exactness_contract(baseline_audit)
    baseline_workload = baseline._workload_contract(baseline_control)
    baseline_preset = baseline._preset_contract(baseline_control)
    baseline_records = baseline._record_contract(baseline_control)
    baseline_output_sha, baseline_structure = _output_contract(
        baseline_control,
        baseline_result,
        name="baseline control",
    )
    baseline_audit_sha, baseline_audit_structure = _output_contract(
        baseline_audit,
        baseline_audit_result,
        name="baseline exactness audit",
    )
    if (
        baseline_audit_sha != baseline_output_sha
        or baseline_audit_structure != baseline_structure
    ):
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
    _component_audit_contract(main_audit, name="main", precision=precision)
    _component_audit_contract(timing_audit, name="timing", precision=precision)

    variants: dict[str, Any] = {}
    for batch_size in batch_sizes:
        variant_dir = candidate_run_root / f"b{batch_size}"
        control_dir = variant_dir / "control"
        control, control_result = _profile_and_result(control_dir)
        audit_dirs = [
            variant_dir / "exactness-audit-01",
            variant_dir / "exactness-audit-02",
        ]
        audit_runs = [_profile_and_result(path) for path in audit_dirs]
        audit_profiles = [value[0] for value in audit_runs]
        audit_results = [value[1] for value in audit_runs]
        _same_precision_runtime(
            control,
            expected_commit=expected_commit,
            expected_branch=expected_branch,
            audit=False,
            precision=precision,
        )
        for exactness_profile in audit_profiles:
            _same_precision_runtime(
                exactness_profile,
                expected_commit=expected_commit,
                expected_branch=expected_branch,
                audit=True,
                precision=precision,
            )
        control_evidence = _load(
            control_dir / "evidence.json", name=f"B{batch_size} control evidence"
        )
        audit_evidence = [
            _load(path / "evidence.json", name=f"B{batch_size} audit evidence")
            for path in audit_dirs
        ]
        if control_evidence.get("profile_pass_kind") != "untraced_control":
            raise EncoderGateError(f"B{batch_size} control pass kind changed")
        if any(
            value.get("profile_pass_kind") != "exactness_audit"
            for value in audit_evidence
        ):
            raise EncoderGateError(f"B{batch_size} audit pass kind changed")
        control_store = _store_contract(
            control,
            control_evidence,
            batch_size=batch_size,
            precision=precision,
        )
        audit_stores = [
            _store_contract(
                profile,
                evidence,
                batch_size=batch_size,
                precision=precision,
            )
            for profile, evidence in zip(
                audit_profiles,
                audit_evidence,
                strict=True,
            )
        ]
        if len(
            {
                control_store["output_store_bytes"],
                *(value["output_store_bytes"] for value in audit_stores),
            }
        ) != 1:
            raise EncoderGateError(f"B{batch_size} store size is not repeatable")

        control_signatures = {
            label: _label_signature(control, label) for label in SUPPORTED_LABELS
        }
        audit_signatures = [
            {
                label: _label_signature(profile, label)
                for label in SUPPORTED_LABELS
            }
            for profile in audit_profiles
        ]
        candidate_profiles = [control, *audit_profiles]
        candidate_signatures = [control_signatures, *audit_signatures]
        output_contracts = [
            _output_contract(profile, path, name=f"B{batch_size} candidate run")
            for profile, path in zip(
                candidate_profiles,
                [control_result, *audit_results],
                strict=True,
            )
        ]
        candidate_output_hashes = [value[0] for value in output_contracts]
        candidate_structures = [value[1] for value in output_contracts]
        candidate_exactness = [
            baseline._strict_exactness_contract(profile)
            for profile in audit_profiles
        ]
        main_drift = _drift_row(
            main_audit,
            batch_size,
            name="main",
            precision=precision,
        )
        timing_drift = _drift_row(
            timing_audit,
            batch_size,
            name="timing",
            precision=precision,
        )
        determinism_checks = {
            "output_bytes": len(set(candidate_output_hashes)) == 1,
            "output_structure": all(
                value == candidate_structures[0]
                for value in candidate_structures[1:]
            ),
            "strict_rng_and_cache_repeat": (
                candidate_exactness[0] == candidate_exactness[1]
            ),
            "workload_repeat": all(
                baseline._workload_contract(profile)
                == baseline._workload_contract(control)
                for profile in audit_profiles
            ),
            "preset_repeat": all(
                baseline._preset_contract(profile)
                == baseline._preset_contract(control)
                for profile in audit_profiles
            ),
            "dispatch_and_graph_repeat": all(
                baseline._record_contract(profile)
                == baseline._record_contract(control)
                for profile in audit_profiles
            ),
        }
        baseline_checks = {
            "encoder_main_exact": main_drift["encoder_exact"],
            "encoder_timing_exact": timing_drift["encoder_exact"],
            "workload": all(
                baseline._workload_contract(profile) == baseline_workload
                for profile in candidate_profiles
            ),
            "preset": all(
                baseline._preset_contract(profile) == baseline_preset
                for profile in candidate_profiles
            ),
            "dispatch_and_graph": all(
                baseline._record_contract(profile) == baseline_records
                for profile in candidate_profiles
            ),
            "strict_rng_and_cache": all(
                value == baseline_exactness for value in candidate_exactness
            ),
            "output_bytes": all(
                value == baseline_output_sha for value in candidate_output_hashes
            ),
            "output_structure": all(
                value == baseline_structure for value in candidate_structures
            ),
        }
        for label in SUPPORTED_LABELS:
            determinism_checks[f"{label}_tokens"] = len(
                {
                    value[label]["token_stream_sha256"]
                    for value in candidate_signatures
                }
            ) == 1
            determinism_checks[f"{label}_stopping"] = len(
                {
                    value[label]["stopping_sha256"]
                    for value in candidate_signatures
                }
            ) == 1
            baseline_checks[f"{label}_tokens"] = all(
                value[label]["token_stream_sha256"]
                == baseline_signatures[label]["token_stream_sha256"]
                for value in candidate_signatures
            )
            baseline_checks[f"{label}_stopping"] = all(
                value[label]["stopping_sha256"]
                == baseline_signatures[label]["stopping_sha256"]
                for value in candidate_signatures
            )
        determinism_pass = all(determinism_checks.values())
        same_precision_exactness_pass = all(baseline_checks.values())
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
        promotion_pass = (
            determinism_pass
            and same_precision_exactness_pass
            and request_saving_pass
            and vram_pass
        )
        variants[str(batch_size)] = {
            "classification": (
                "same-precision-exact-and-fast"
                if promotion_pass
                else "same-precision-exact-insufficient-complete-wall-gain"
                if determinism_pass and same_precision_exactness_pass
                else "deterministic-documented-drift-not-promotion-eligible"
                if determinism_pass
                else "nondeterministic-candidate-rejected"
            ),
            "determinism_pass": determinism_pass,
            "same_precision_exactness_pass": same_precision_exactness_pass,
            "strict_exactness_pass": same_precision_exactness_pass,
            "request_saving_pass": request_saving_pass,
            "vram_pass": vram_pass,
            "promotion_pass": promotion_pass,
            "determinism_checks": determinism_checks,
            "baseline_equivalence_checks": baseline_checks,
            "failed_determinism_checks": sorted(
                name for name, passed in determinism_checks.items() if not passed
            ),
            "failed_baseline_equivalence_checks": sorted(
                name for name, passed in baseline_checks.items() if not passed
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
                "control_sha256": candidate_output_hashes[0],
                "audit_sha256": candidate_output_hashes[1:],
                "structure": candidate_structures[0],
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
    deterministic_variants = [
        int(batch_size)
        for batch_size, row in variants.items()
        if row["determinism_pass"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": (
            f"PASS_{precision.upper()}_ENCODER_PRECOMPUTE"
            if winners
            else "STOP_ENCODER_PRECOMPUTE"
        ),
        "precision": precision,
        "production_changes": False,
        "decoder_changes": False,
        "server_changes": False,
        "batch_sizes": list(batch_sizes),
        "baseline_run_root": str(baseline_run_root),
        "candidate_run_root": str(candidate_run_root),
        "baseline_request_median_seconds": baseline_request_median,
        "minimum_request_saving_fraction": MINIMUM_REQUEST_SAVING_FRACTION,
        "minimum_request_saving_seconds": required_request_saving,
        "deterministic_batch_sizes": deterministic_variants,
        "strict_exact_batch_sizes": exact_variants,
        "promotion_batch_sizes": winners,
        "variants": variants,
    }


def _text(report: Mapping[str, Any]) -> str:
    lines = [
        f"decision={report['decision']}",
        f"baseline_request_median_seconds={report['baseline_request_median_seconds']:.6f}",
        f"minimum_request_saving_seconds={report['minimum_request_saving_seconds']:.6f}",
        f"precision={report.get('precision', 'fp32')}",
        "deterministic_batch_sizes="
        + ",".join(str(value) for value in report["deterministic_batch_sizes"]),
        "strict_exact_batch_sizes="
        + ",".join(str(value) for value in report["strict_exact_batch_sizes"]),
        "promotion_batch_sizes="
        + ",".join(str(value) for value in report["promotion_batch_sizes"]),
    ]
    for batch_size, row in report["variants"].items():
        lines.extend(
            (
                f"b{batch_size}_classification={row['classification']}",
                f"b{batch_size}_deterministic={str(row['determinism_pass']).lower()}",
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
    parser.add_argument("--precision", choices=SUPPORTED_PRECISIONS, required=True)
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
        precision=args.precision,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.text_output.write_text(_text(report), encoding="utf-8")
    print(_text(report), end="")
    return 0 if report["promotion_batch_sizes"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
