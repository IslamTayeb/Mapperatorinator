"""Fail-loud dispatch validation for the mixed-weight full-song scout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROFILE_SCHEMA_VERSION = 1
WEIGHT_ONLY_METADATA_VERSION = "approximate-fp16-weights-fp32-state-v2"
WEIGHT_ONLY_DISPATCHES = (
    "weight_only_self_attention_block",
    "weight_only_mlp_tail",
    "weight_only_final_projection",
)
FP16_WEIGHT_REGIONS = (
    "self_qkv",
    "self_output",
    "mlp_fc1",
    "mlp_fc2",
    "final_logits",
)
FP32_SELECTED_DECODE_MATRIX_REGIONS = ("cross_query", "cross_output")
PROFILE_ROLES = ("baseline", "candidate")
PROFILE_LABELS = ("timing_context", "main_generation")


class WeightOnlyProfileError(ValueError):
    pass


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WeightOnlyProfileError(f"{name} must be an object")
    return value


def _metadata(value: Any, *, name: str) -> dict[str, Any]:
    metadata = _object(value, name=name)
    required = {
        "version": WEIGHT_ONLY_METADATA_VERSION,
        "result_class": "documented-drift",
        "exactness_claim": False,
        "fp32_activations_caches_reductions_logits": True,
        "fp16_weight_regions": list(FP16_WEIGHT_REGIONS),
        "fp32_selected_decode_matrix_regions": list(
            FP32_SELECTED_DECODE_MATRIX_REGIONS
        ),
    }
    failures = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in required.items()
        if metadata.get(key) != expected
    }
    if "fp32_weight_regions" in metadata:
        failures["fp32_weight_regions"] = {
            "expected": "absent",
            "actual": metadata["fp32_weight_regions"],
        }
    if failures:
        raise WeightOnlyProfileError(
            f"{name} has invalid mixed-weight metadata: {failures}"
        )
    return metadata


def _dispatch_counts(record: dict[str, Any], *, name: str) -> dict[str, int]:
    raw = _object(record.get("optimized_dispatch_capture_hits"), name=f"{name}.hits")
    counts: dict[str, int] = {}
    for key in WEIGHT_ONLY_DISPATCHES:
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise WeightOnlyProfileError(
                f"{name}.hits.{key} must be a non-negative integer"
            )
        counts[key] = value
    return counts


def validate_profile(payload: Any, *, role: str) -> dict[str, Any]:
    if role not in PROFILE_ROLES:
        raise WeightOnlyProfileError(f"role must be one of {PROFILE_ROLES}")
    root = _object(payload, name="profile")
    if root.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise WeightOnlyProfileError(
            f"profile.schema_version must be {PROFILE_SCHEMA_VERSION}"
        )
    top_metadata = _object(root.get("metadata"), name="profile.metadata")
    candidate = role == "candidate"
    if candidate:
        _metadata(
            top_metadata.get("optimized_approximate_weight_only"),
            name="profile.metadata.optimized_approximate_weight_only",
        )
    elif "optimized_approximate_weight_only" in top_metadata:
        raise WeightOnlyProfileError(
            "baseline profile must not contain mixed-weight initialization metadata"
        )

    generation = root.get("generation")
    if not isinstance(generation, list) or not generation:
        raise WeightOnlyProfileError("profile.generation must be a non-empty list")
    by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in PROFILE_LABELS}
    for index, raw_record in enumerate(generation):
        record = _object(raw_record, name=f"profile.generation[{index}]")
        label = record.get("profile_label")
        if label in by_label:
            by_label[label].append(record)
    missing = [label for label, records in by_label.items() if not records]
    if missing:
        raise WeightOnlyProfileError(f"profile is missing generation labels: {missing}")

    totals = {key: 0 for key in WEIGHT_ONLY_DISPATCHES}
    for label, records in by_label.items():
        for index, record in enumerate(records):
            name = f"profile.{label}[{index}]"
            policy = _object(
                record.get("optimized_dispatch_policy"),
                name=f"{name}.optimized_dispatch_policy",
            )
            candidate_policy = policy.get("approximate_weight_only")
            candidate_main = candidate and label == "main_generation"
            if not candidate_main:
                if candidate_policy is not None:
                    raise WeightOnlyProfileError(
                        f"{name} must not request mixed-weight dispatch"
                    )
                if "approximate_weight_only" in record:
                    raise WeightOnlyProfileError(
                        f"{name} must not contain record-level mixed-weight metadata"
                    )
                raw_hits = _object(
                    record.get("optimized_dispatch_capture_hits"),
                    name=f"{name}.hits",
                )
                unexpected = {
                    key: raw_hits[key]
                    for key in WEIGHT_ONLY_DISPATCHES
                    if key in raw_hits
                }
                if unexpected:
                    raise WeightOnlyProfileError(
                        f"{name} contains mixed-weight dispatch counters: "
                        f"{unexpected}"
                    )
                if candidate and record.get("optimized_dispatch_mode") != "accepted_batch1":
                    raise WeightOnlyProfileError(
                        f"{name}.optimized_dispatch_mode must be accepted_batch1"
                    )
                continue

            expected_policy = {
                "requested": True,
                "enabled": True,
                "disabled_reason": None,
                "result_class": "documented-drift",
                "exactness_claim": False,
            }
            if candidate_policy != expected_policy:
                raise WeightOnlyProfileError(
                    f"{name} has invalid mixed-weight dispatch policy: "
                    f"expected {expected_policy}, got {candidate_policy}"
                )
            expected_mode = "approximate_weight_only_batch1"
            if record.get("optimized_dispatch_mode") != expected_mode:
                raise WeightOnlyProfileError(
                    f"{name}.optimized_dispatch_mode must be {expected_mode}"
                )
            counts = _dispatch_counts(record, name=name)
            for key, value in counts.items():
                totals[key] += value

    if candidate:
        missing_dispatch = {key: value for key, value in totals.items() if value <= 0}
        if missing_dispatch:
            raise WeightOnlyProfileError(
                "candidate main generation did not execute every mixed-weight region: "
                f"{missing_dispatch}"
            )
    return {
        "role": role,
        "candidate_enabled_for_main": candidate,
        "candidate_disabled_for_timing": candidate,
        "main_weight_only_dispatch_counts": totals,
    }


def validate_profile_path(path: Path, *, role: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WeightOnlyProfileError(f"cannot read profile {path}: {exc}") from exc
    return validate_profile(payload, role=role)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--role", choices=PROFILE_ROLES, required=True)
    args = parser.parse_args()
    result = validate_profile_path(args.profile, role=args.role)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
