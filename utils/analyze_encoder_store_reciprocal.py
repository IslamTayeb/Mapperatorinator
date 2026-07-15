"""Analyze reciprocal full-song encoder-store profiles and the B1/B16 drift audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from osuT5.osuT5.inference.optimized.scout.encoder_store import (
    STORE_VERSION,
    SUPPORTED_LABELS,
)
from utils.fixed_seed_inference import SEED_POLICY_VERSION, stage_seed
from utils.run_batched_encoder_store_full_song import SELECTED_STACK_VERSION
from utils.summarize_inference_profile import compare_reciprocal_profiles


MIN_COMPLETE_REQUEST_SAVING_SECONDS = 0.3
MAX_ABS_ENCODER_DRIFT = 1e-3
RTX_2080_TI_TOTAL_BYTES = 11_264 * 1024 * 1024
STORE_ACCOUNTING_TOLERANCE_SECONDS = 0.02
STORE_TIME_FIELDS = (
    "batch_setup_seconds",
    "input_copy_seconds",
    "encoder_synchronized_seconds",
    "storage_allocation_seconds",
    "output_store_copy_seconds",
)
STORE_MEMORY_FIELDS = (
    "baseline_allocated_vram_bytes",
    "baseline_reserved_vram_bytes",
    "peak_allocated_vram_bytes",
    "peak_reserved_vram_bytes",
    "incremental_peak_allocated_vram_bytes",
    "incremental_peak_reserved_vram_bytes",
    "output_store_bytes",
)


def _load(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _number(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _signed_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: Any, *, name: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0 or (positive and value <= 0):
        raise ValueError(f"{name} must be non-negative")
    return value


def _records(profile: dict[str, Any], label: str) -> list[dict[str, Any]]:
    result = [
        row
        for row in profile.get("generation", [])
        if isinstance(row, dict) and row.get("profile_label") == label
    ]
    if not result:
        raise ValueError(f"profile is missing generation label {label!r}")
    return result


def _tokens(profile: dict[str, Any], label: str) -> list[int]:
    result: list[int] = []
    for index, row in enumerate(_records(profile, label)):
        value = row.get("generated_token_ids")
        if not isinstance(value, list):
            raise ValueError(f"{label}[{index}] lacks generated_token_ids")
        result.extend(int(token) for token in value)
    return result


def _token_comparison(reference: list[int], candidate: list[int]) -> dict[str, Any]:
    aligned = min(len(reference), len(candidate))
    mismatches = sum(
        left != right
        for left, right in zip(reference[:aligned], candidate[:aligned], strict=True)
    )
    first = next(
        (
            index
            for index, (left, right) in enumerate(
                zip(reference[:aligned], candidate[:aligned], strict=True)
            )
            if left != right
        ),
        None,
    )
    if first is None and len(reference) != len(candidate):
        first = aligned
    return {
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "count_delta": len(candidate) - len(reference),
        "aligned_count": aligned,
        "aligned_mismatch_count": mismatches,
        "aligned_mismatch_fraction": mismatches / aligned if aligned else 0.0,
        "first_mismatch": first,
        "exact": reference == candidate,
    }


def _stage(profile: dict[str, Any], name: str) -> float:
    matches = [
        row
        for row in profile.get("stages", [])
        if isinstance(row, dict) and row.get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError(f"profile requires exactly one {name!r} stage")
    return _number(matches[0].get("wall_seconds"), name=f"stage.{name}")


def _complete_wall(profile: dict[str, Any]) -> float:
    stages = [row for row in profile.get("stages", []) if isinstance(row, dict)]
    if not stages:
        raise ValueError("profile stages are empty")
    started = _number(
        stages[0].get("started_at_perf_counter_seconds"),
        name="first stage start",
    )
    finished = _number(
        stages[-1].get("finished_at_perf_counter_seconds"),
        name="last stage finish",
    )
    if finished <= started:
        raise ValueError("profile complete wall is not positive")
    return finished - started


def _runner_complete_wall(evidence: dict[str, Any], *, role: str) -> float:
    """Return the outer cold-request wall recorded by the isolated runner."""

    return _number(
        evidence.get("run_wall_seconds"),
        name=f"{role}.run_wall_seconds",
        positive=True,
    )


def _label_metrics(profile: dict[str, Any], label: str) -> dict[str, Any]:
    rows = _records(profile, label)
    tokens = sum(int(row.get("generated_tokens", 0) or 0) for row in rows)
    model = sum(
        _number(
            row.get("model_elapsed_seconds"),
            name=f"{label}.model_elapsed_seconds",
            positive=True,
        )
        for row in rows
    )
    outer = sum(
        _number(row.get("wall_seconds"), name=f"{label}.wall_seconds")
        for row in rows
    )
    return {
        "records": len(rows),
        "tokens": tokens,
        "model_seconds": model,
        "outer_seconds": outer,
        "tokens_per_second": tokens / model,
        "per_window_generated_tokens": [
            int(row.get("generated_tokens", 0) or 0) for row in rows
        ],
    }


def _profile_metrics(profile: dict[str, Any]) -> dict[str, Any]:
    memory_rows = [
        row
        for key in ("stages", "generation")
        for row in profile.get(key, [])
        if isinstance(row, dict) and "cuda_max_memory_allocated_mb" in row
    ]
    if not memory_rows:
        raise ValueError("profile lacks CUDA peak-memory evidence")
    peak_allocated_mb = max(
        _number(
            row["cuda_max_memory_allocated_mb"],
            name="cuda_max_memory_allocated_mb",
        )
        for row in memory_rows
    )
    return {
        "timing": _label_metrics(profile, "timing_context"),
        "main": _label_metrics(profile, "main_generation"),
        "timing_stage_seconds": _stage(profile, "timing_context_generation"),
        "main_stage_seconds": _stage(profile, "main_generation"),
        "complete_request_seconds": _complete_wall(profile),
        "peak_cuda_memory_allocated_bytes": int(
            peak_allocated_mb * 1024 * 1024
        ),
    }


def _validate_seed_metadata(profile: dict[str, Any], *, base_seed: int) -> None:
    metadata = profile.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("profile metadata is missing")
    if metadata.get("reciprocal_seed_policy") != SEED_POLICY_VERSION:
        raise ValueError("profile reciprocal seed policy is missing")
    for label in SUPPORTED_LABELS:
        if metadata.get(f"reciprocal_seed_{label}") != stage_seed(base_seed, label):
            raise ValueError(f"profile reciprocal seed changed for {label}")


def _validate_store_charges(store: dict[str, Any]) -> dict[str, Any]:
    entries = store.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("candidate encoder-store entries are missing")
    charged_entries = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"candidate encoder-store entry {index} is invalid")
        complete = _number(
            entry.get("complete_precompute_seconds"),
            name=f"entry[{index}].complete_precompute_seconds",
            positive=True,
        )
        times = {
            field: _number(
                entry.get(field),
                name=f"entry[{index}].{field}",
            )
            for field in STORE_TIME_FIELDS
        }
        accounted = sum(times.values())
        if accounted > complete + STORE_ACCOUNTING_TOLERANCE_SECONDS:
            raise ValueError(
                f"candidate encoder-store entry {index} component time exceeds "
                "complete precompute wall"
            )
        memory = {
            field: _integer(
                entry.get(field),
                name=f"entry[{index}].{field}",
                positive=field == "output_store_bytes",
            )
            for field in STORE_MEMORY_FIELDS
        }
        if (
            memory["peak_allocated_vram_bytes"]
            - memory["baseline_allocated_vram_bytes"]
            != memory["incremental_peak_allocated_vram_bytes"]
        ):
            raise ValueError(
                f"candidate encoder-store entry {index} allocated VRAM delta changed"
            )
        if (
            memory["peak_reserved_vram_bytes"]
            - memory["baseline_reserved_vram_bytes"]
            != memory["incremental_peak_reserved_vram_bytes"]
        ):
            raise ValueError(
                f"candidate encoder-store entry {index} reserved VRAM delta changed"
            )
        if (
            memory["incremental_peak_allocated_vram_bytes"]
            < memory["output_store_bytes"]
        ):
            raise ValueError(
                f"candidate encoder-store entry {index} peak allocation does not "
                "charge output storage"
            )
        charged_entries.append(
            {
                "entry_index": index,
                "complete_precompute_seconds": complete,
                "accounted_precompute_seconds": accounted,
                "unattributed_precompute_seconds": max(0.0, complete - accounted),
                **times,
                **memory,
            }
        )
    total_complete = sum(
        entry["complete_precompute_seconds"] for entry in charged_entries
    )
    total_store = sum(entry["output_store_bytes"] for entry in charged_entries)
    if not math.isclose(
        _number(
            store.get("total_complete_precompute_seconds"),
            name="total_complete_precompute_seconds",
        ),
        total_complete,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("candidate total precompute wall does not match entries")
    if _integer(
        store.get("total_output_store_bytes"),
        name="total_output_store_bytes",
        positive=True,
    ) != total_store:
        raise ValueError("candidate total output storage does not match entries")
    return {
        "entries": charged_entries,
        "total_complete_precompute_seconds": total_complete,
        "total_accounted_precompute_seconds": sum(
            entry["accounted_precompute_seconds"] for entry in charged_entries
        ),
        "total_output_store_bytes": total_store,
        "total_incremental_peak_allocated_vram_bytes": sum(
            entry["incremental_peak_allocated_vram_bytes"]
            for entry in charged_entries
        ),
        "total_incremental_peak_reserved_vram_bytes": sum(
            entry["incremental_peak_reserved_vram_bytes"]
            for entry in charged_entries
        ),
    }


def _validate_candidate_store(
    profile: dict[str, Any],
    evidence: dict[str, Any],
    *,
    batch_size: int,
) -> dict[str, Any]:
    metadata = profile.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("candidate profile metadata is missing")
    store = metadata.get("optimized_batched_encoder_store")
    if not isinstance(store, dict):
        raise ValueError("candidate profile lacks encoder-store metadata")
    external = evidence.get("encoder_store")
    if not isinstance(external, dict):
        raise ValueError("candidate evidence lacks encoder_store")
    required = {
        "version": STORE_VERSION,
        "batch_size": batch_size,
        "labels_completed": list(SUPPORTED_LABELS),
        "production_default": False,
    }
    failures = {
        key: {"expected": expected, "profile": store.get(key), "evidence": external.get(key)}
        for key, expected in required.items()
        if store.get(key) != expected or external.get(key) != expected
    }
    for key in (
        "store_count",
        "shared_across_timing_main",
        "total_complete_precompute_seconds",
        "total_output_store_bytes",
    ):
        if store.get(key) != external.get(key):
            failures[key] = {
                "profile": store.get(key),
                "evidence": external.get(key),
            }
    if failures:
        raise ValueError(f"candidate encoder-store evidence mismatch: {failures}")
    profile_entries = store.get("entries")
    external_entries = external.get("entries")
    if not isinstance(profile_entries, list) or not isinstance(external_entries, list):
        raise ValueError("candidate encoder-store entries are missing")
    if len(profile_entries) != len(external_entries):
        raise ValueError("candidate encoder-store entry count changed")
    labels_seen: list[str] = []
    for index, (profile_entry, external_entry) in enumerate(
        zip(profile_entries, external_entries, strict=True)
    ):
        for key in (
            "entry_index",
            "labels",
            "live_window_count",
            "output_store_bytes",
            "conditioning_sha256",
            "complete_precompute_seconds",
            *STORE_TIME_FIELDS,
            *STORE_MEMORY_FIELDS,
        ):
            if profile_entry.get(key) != external_entry.get(key):
                raise ValueError(
                    f"candidate encoder-store entry {index} field {key} changed"
                )
        labels_seen.extend(external_entry.get("labels", []))
        uses = external_entry.get("uses")
        if not isinstance(uses, dict):
            raise ValueError(f"candidate encoder-store entry {index} uses are missing")
        for label in external_entry.get("labels", []):
            row = uses.get(label)
            if not isinstance(row, dict) or row.get("rows_reused", 0) <= 0:
                raise ValueError(f"candidate did not reuse encoder rows for {label}")
            if row.get("per_window_device_copy") is not False:
                raise ValueError(
                    f"candidate unexpectedly copied encoder rows for {label}"
                )
    if sorted(labels_seen) != sorted(SUPPORTED_LABELS):
        raise ValueError(f"candidate store labels changed: {labels_seen}")
    _validate_store_charges(external)
    return external


def _baseline_has_no_store(profile: dict[str, Any]) -> None:
    metadata = profile.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("baseline profile metadata is missing")
    if "optimized_batched_encoder_store" in metadata:
        raise ValueError("baseline unexpectedly contains encoder-store metadata")


def _osu_structure(path: Path) -> dict[str, int]:
    section = None
    counts = {"hit_objects": 0, "timing_points": 0}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line
            continue
        if not line or line.startswith("//"):
            continue
        if section == "[HitObjects]":
            counts["hit_objects"] += 1
        elif section == "[TimingPoints]":
            counts["timing_points"] += 1
    return counts


def analyze(
    profile_paths: dict[str, Path],
    evidence_paths: dict[str, Path],
    audit_paths: dict[str, Path],
    *,
    batch_size: int,
    base_seed: int,
    minimum_complete_request_saving_seconds: float = (
        MIN_COMPLETE_REQUEST_SAVING_SECONDS
    ),
) -> dict[str, Any]:
    minimum_complete_request_saving_seconds = _number(
        minimum_complete_request_saving_seconds,
        name="minimum_complete_request_saving_seconds",
        positive=True,
    )
    profiles = {
        role: _load(path, name=f"{role} profile")
        for role, path in profile_paths.items()
    }
    evidence = {
        role: _load(path, name=f"{role} evidence")
        for role, path in evidence_paths.items()
    }
    selected_topologies = {}
    for role, payload in evidence.items():
        if payload.get("schema_version") != 2:
            raise ValueError(f"{role} evidence schema changed")
        topology = payload.get("selected_topology")
        if not isinstance(topology, dict):
            raise ValueError(f"{role} evidence lacks selected topology")
        if topology.get("version") != SELECTED_STACK_VERSION:
            raise ValueError(f"{role} selected topology version changed")
        if not isinstance(payload.get("selected_initialization"), dict):
            raise ValueError(f"{role} selected initialization is missing")
        selected_topologies[role] = topology
    for profile in profiles.values():
        _validate_seed_metadata(profile, base_seed=base_seed)
    for role in ("baseline_first", "baseline_second"):
        _baseline_has_no_store(profiles[role])
        if evidence[role].get("mode") != "baseline":
            raise ValueError(f"{role} evidence mode changed")
    stores = {
        role: _validate_candidate_store(
            profiles[role],
            evidence[role],
            batch_size=batch_size,
        )
        for role in ("candidate_second", "candidate_first")
    }
    store_charges = {
        role: _validate_store_charges(store) for role, store in stores.items()
    }

    audits = {
        role: _load(path, name=f"{role} encoder drift audit")
        for role, path in audit_paths.items()
    }
    audit_rows: dict[str, dict[str, Any]] = {}
    audit_window_counts = set()
    for role, audit in audits.items():
        audit_variant = audit.get("variants", {}).get(str(batch_size))
        if not isinstance(audit_variant, dict):
            raise ValueError(f"{role} drift audit lacks B{batch_size}")
        drift = audit_variant.get("encoder_drift_vs_b1")
        if not isinstance(drift, dict):
            raise ValueError(f"{role} drift audit lacks B1 comparison")
        audit_window_counts.add(audit.get("metadata", {}).get("live_window_count"))
        audit_rows[role] = {
            "b1_complete_precompute_seconds": _number(
                audit["variants"]["1"]["complete_precompute_seconds"],
                name=f"{role}.b1_complete_precompute_seconds",
                positive=True,
            ),
            "candidate_complete_precompute_seconds": _number(
                audit_variant["complete_precompute_seconds"],
                name=f"{role}.candidate_complete_precompute_seconds",
                positive=True,
            ),
            "complete_seconds_saved_vs_b1": _signed_number(
                audit_variant["complete_seconds_saved_vs_b1"],
                name=f"{role}.complete_seconds_saved_vs_b1",
            ),
            "max_abs_encoder_drift": _number(
                drift.get("max_abs"),
                name=f"{role}.max_abs_encoder_drift",
            ),
            "mean_window_max_abs_encoder_drift": _number(
                drift.get("mean_window_max_abs"),
                name=f"{role}.mean_window_max_abs_encoder_drift",
            ),
            "exact_window_count": _integer(
                drift.get("exact_window_count"),
                name=f"{role}.exact_window_count",
            ),
            "window_count": _integer(
                drift.get("window_count"),
                name=f"{role}.window_count",
                positive=True,
            ),
            "incremental_peak_vram_bytes": _integer(
                audit_variant.get("incremental_peak_vram_bytes"),
                name=f"{role}.incremental_peak_vram_bytes",
                positive=True,
            ),
            "output_store_bytes": _integer(
                audit_variant.get("output_store_bytes"),
                name=f"{role}.output_store_bytes",
                positive=True,
            ),
        }
    if len(audit_window_counts) != 1:
        raise ValueError("main and timing drift audits have different window counts")
    audit_windows = next(iter(audit_window_counts))
    for store in stores.values():
        if any(
            entry.get("live_window_count") != audit_windows
            for entry in store["entries"]
        ):
            raise ValueError("drift audit and candidate store window counts differ")

    reciprocal = compare_reciprocal_profiles(
        profile_paths["baseline_first"],
        profile_paths["candidate_second"],
        profile_paths["candidate_first"],
        profile_paths["baseline_second"],
        labels=["main_generation", "timing_context"],
        regression_tolerance_pct=100.0,
    )
    metrics = {role: _profile_metrics(profile) for role, profile in profiles.items()}
    for role, role_metrics in metrics.items():
        role_metrics["runner_complete_wall_seconds"] = _runner_complete_wall(
            evidence[role],
            role=role,
        )
    order_pairs = (
        ("baseline_first", "candidate_second"),
        ("baseline_second", "candidate_first"),
    )
    orders: list[dict[str, Any]] = []
    for baseline_role, candidate_role in order_pairs:
        baseline = metrics[baseline_role]
        candidate = metrics[candidate_role]
        request_saved = (
            baseline["runner_complete_wall_seconds"]
            - candidate["runner_complete_wall_seconds"]
        )
        orders.append(
            {
                "baseline": baseline_role,
                "candidate": candidate_role,
                "baseline_complete_request_seconds": baseline[
                    "runner_complete_wall_seconds"
                ],
                "candidate_complete_request_seconds": candidate[
                    "runner_complete_wall_seconds"
                ],
                "complete_request_seconds_saved": request_saved,
                "complete_request_speedup_pct": (
                    request_saved
                    / baseline["runner_complete_wall_seconds"]
                    * 100.0
                ),
                "timing_stage_seconds_saved": (
                    baseline["timing_stage_seconds"]
                    - candidate["timing_stage_seconds"]
                ),
                "main_stage_seconds_saved": (
                    baseline["main_stage_seconds"]
                    - candidate["main_stage_seconds"]
                ),
                "timing_tokens": _token_comparison(
                    _tokens(profiles[baseline_role], "timing_context"),
                    _tokens(profiles[candidate_role], "timing_context"),
                ),
                "main_tokens": _token_comparison(
                    _tokens(profiles[baseline_role], "main_generation"),
                    _tokens(profiles[candidate_role], "main_generation"),
                ),
            }
        )

    repeatability = {}
    for label in SUPPORTED_LABELS:
        repeatability[f"baseline_{label}"] = _token_comparison(
            _tokens(profiles["baseline_first"], label),
            _tokens(profiles["baseline_second"], label),
        )
        repeatability[f"candidate_{label}"] = _token_comparison(
            _tokens(profiles["candidate_first"], label),
            _tokens(profiles["candidate_second"], label),
        )
    candidate_artifacts = [
        Path(profiles[role]["metadata"]["result_file_path"])
        for role in ("candidate_first", "candidate_second")
    ]
    baseline_artifacts = [
        Path(profiles[role]["metadata"]["result_file_path"])
        for role in ("baseline_first", "baseline_second")
    ]
    repeatability_pass = all(row["exact"] for row in repeatability.values())
    complete_wall_saving_pass = all(
        order["complete_request_seconds_saved"]
        >= minimum_complete_request_saving_seconds
        for order in orders
    )
    drift_pass = all(
        row["max_abs_encoder_drift"] <= MAX_ABS_ENCODER_DRIFT
        for row in audit_rows.values()
    )
    candidate_vram_pass = all(
        metrics[role]["peak_cuda_memory_allocated_bytes"]
        <= RTX_2080_TI_TOTAL_BYTES
        for role in ("candidate_first", "candidate_second")
    ) and all(
        entry["peak_allocated_vram_bytes"] <= RTX_2080_TI_TOTAL_BYTES
        for charges in store_charges.values()
        for entry in charges["entries"]
    )
    storage_pass = all(
        charges["total_output_store_bytes"] > 0
        and charges["total_incremental_peak_allocated_vram_bytes"]
        >= charges["total_output_store_bytes"]
        for charges in store_charges.values()
    )
    baseline_output_repeat = (
        profiles["baseline_first"]["metadata"].get("result_file_sha256")
        == profiles["baseline_second"]["metadata"].get("result_file_sha256")
    )
    candidate_output_repeat = (
        profiles["candidate_first"]["metadata"].get("result_file_sha256")
        == profiles["candidate_second"]["metadata"].get("result_file_sha256")
    )
    output_repeatability_pass = baseline_output_repeat and candidate_output_repeat
    pass_gate = (
        repeatability_pass
        and output_repeatability_pass
        and complete_wall_saving_pass
        and drift_pass
        and candidate_vram_pass
        and storage_pass
    )
    report = {
        "schema_version": 2,
        "decision": (
            "PASS_FULL_SONG_GATE" if pass_gate else "STOP_ENCODER_STORE"
        ),
        "batch_size": batch_size,
        "minimum_complete_request_saving_seconds": (
            minimum_complete_request_saving_seconds
        ),
        "profiles": {key: str(value) for key, value in profile_paths.items()},
        "evidence": {key: str(value) for key, value in evidence_paths.items()},
        "selected_topologies": selected_topologies,
        "audit_paths": {key: str(value) for key, value in audit_paths.items()},
        "audits": audit_rows,
        "store_charges": store_charges,
        "metrics": metrics,
        "orders": orders,
        "mean_complete_request_seconds_saved": sum(
            order["complete_request_seconds_saved"] for order in orders
        )
        / len(orders),
        "repeatability": repeatability,
        "repeatability_pass": repeatability_pass,
        "complete_wall_saving_pass": complete_wall_saving_pass,
        "max_abs_encoder_drift": MAX_ABS_ENCODER_DRIFT,
        "drift_pass": drift_pass,
        "rtx_2080_ti_total_bytes": RTX_2080_TI_TOTAL_BYTES,
        "candidate_vram_pass": candidate_vram_pass,
        "storage_pass": storage_pass,
        "output_repeatability_pass": output_repeatability_pass,
        "baseline_output_repeat": {
            "sha256_equal": baseline_output_repeat,
            "structure": [_osu_structure(path) for path in baseline_artifacts],
        },
        "candidate_output_repeat": {
            "sha256_equal": candidate_output_repeat,
            "structure": [_osu_structure(path) for path in candidate_artifacts],
        },
        "reciprocal_exactness": reciprocal,
    }
    return report


def _text(report: dict[str, Any]) -> str:
    lines = [
        f"decision={report['decision']}",
        f"batch_size={report['batch_size']}",
        f"minimum_complete_request_saving_seconds={report['minimum_complete_request_saving_seconds']:.6f}",
        f"mean_complete_request_seconds_saved={report['mean_complete_request_seconds_saved']:.6f}",
        f"main_encoder_max_abs_drift={report['audits']['main']['max_abs_encoder_drift']}",
        f"timing_encoder_max_abs_drift={report['audits']['timing']['max_abs_encoder_drift']}",
        f"main_encoder_precompute_seconds_saved={report['audits']['main']['complete_seconds_saved_vs_b1']}",
        f"timing_encoder_precompute_seconds_saved={report['audits']['timing']['complete_seconds_saved_vs_b1']}",
        f"repeatability_pass={str(report['repeatability_pass']).lower()}",
        f"output_repeatability_pass={str(report['output_repeatability_pass']).lower()}",
        f"complete_wall_saving_pass={str(report['complete_wall_saving_pass']).lower()}",
        f"drift_pass={str(report['drift_pass']).lower()}",
        f"storage_pass={str(report['storage_pass']).lower()}",
        f"candidate_vram_pass={str(report['candidate_vram_pass']).lower()}",
    ]
    for index, order in enumerate(report["orders"]):
        lines.append(
            f"order_{index}_complete_request_seconds_saved="
            f"{order['complete_request_seconds_saved']:.6f}"
        )
        lines.append(
            f"order_{index}_main_token_count_delta={order['main_tokens']['count_delta']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    for role in (
        "baseline-first",
        "candidate-second",
        "candidate-first",
        "baseline-second",
    ):
        parser.add_argument(f"--{role}-profile", type=Path, required=True)
        parser.add_argument(f"--{role}-evidence", type=Path, required=True)
    parser.add_argument("--main-audit", type=Path, required=True)
    parser.add_argument("--timing-audit", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument(
        "--minimum-complete-request-saving-seconds",
        type=float,
        default=MIN_COMPLETE_REQUEST_SAVING_SECONDS,
    )
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    args = parser.parse_args()
    keys = (
        "baseline_first",
        "candidate_second",
        "candidate_first",
        "baseline_second",
    )
    report = analyze(
        {
            key: getattr(args, f"{key}_profile")
            for key in keys
        },
        {
            key: getattr(args, f"{key}_evidence")
            for key in keys
        },
        {"main": args.main_audit, "timing": args.timing_audit},
        batch_size=args.batch_size,
        base_seed=args.base_seed,
        minimum_complete_request_saving_seconds=(
            args.minimum_complete_request_saving_seconds
        ),
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.text_output.write_text(_text(report), encoding="utf-8")
    print(_text(report), end="")
    if report["decision"] != "PASS_FULL_SONG_GATE":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
