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
CROSS_MODES = (
    "accepted",
    "fp16_packed_projections",
)
CROSS_DISPATCHES = (
    "fp16_packed_cross_projection_candidate",
)
SPLIT_KV_DISPATCH = "native_q1_rope_cache_self_attention_split_kv_8"
SPLIT_KV_PREFIX = f"{SPLIT_KV_DISPATCH}_prefix_"
SPLIT_KV_PREFIXES = frozenset(range(192, 833, 64))
PROFILE_ROLES = ("baseline", "candidate")
PROFILE_LABELS = ("timing_context", "main_generation")


class WeightOnlyProfileError(ValueError):
    pass


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WeightOnlyProfileError(f"{name} must be an object")
    return value


def _metadata(
    value: Any,
    *,
    name: str,
    cross_mode: str = "accepted",
) -> dict[str, Any]:
    metadata = _object(value, name=name)
    if cross_mode not in CROSS_MODES:
        raise WeightOnlyProfileError(f"cross_mode must be one of {CROSS_MODES}")
    fp16_regions = list(FP16_WEIGHT_REGIONS)
    fp32_regions = list(FP32_SELECTED_DECODE_MATRIX_REGIONS)
    if cross_mode == "fp16_packed_projections":
        fp16_regions.extend(FP32_SELECTED_DECODE_MATRIX_REGIONS)
        fp32_regions = []
    required = {
        "version": WEIGHT_ONLY_METADATA_VERSION,
        "result_class": "documented-drift",
        "exactness_claim": False,
        "fp32_activations_caches_reductions_logits": True,
        "fp16_weight_regions": fp16_regions,
        "fp32_selected_decode_matrix_regions": fp32_regions,
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
    cross = _object(metadata.get("cross_candidate"), name=f"{name}.cross_candidate")
    common_cross = {
        "mode": cross_mode,
        "scope": "main-model-only",
        "attention_accumulation": "fp32",
        "production_selector_unchanged": True,
    }
    cross_failures = {
        key: {"expected": expected, "actual": cross.get(key)}
        for key, expected in common_cross.items()
        if cross.get(key) != expected
    }
    if cross_mode == "accepted":
        expected_specific = {
            "result_class": "control",
            "exactness_claim": False,
            "accepted_q1_bmm": True,
        }
    elif cross_mode == "fp16_packed_projections":
        expected_specific = {
            "result_class": "documented-drift",
            "exactness_claim": False,
            "packed_projection_weights": True,
            "projection_delta_only": True,
            "accepted_q1_bmm": True,
            "incremental_exactness_required": True,
        }
    cross_failures.update(
        {
            key: {"expected": expected, "actual": cross.get(key)}
            for key, expected in expected_specific.items()
            if cross.get(key) != expected
        }
    )
    if cross_failures:
        raise WeightOnlyProfileError(
            f"{name} has invalid cross candidate metadata: {cross_failures}"
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


def _effective_self_attention_counts(
    record: dict[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    raw = _object(record.get("optimized_dispatch_capture_hits"), name=f"{name}.hits")
    generic = raw.get("native_q1_rope_cache_self_attention")
    split = raw.get(SPLIT_KV_DISPATCH, 0)
    for key, value in (
        ("native_q1_rope_cache_self_attention", generic),
        (SPLIT_KV_DISPATCH, split),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise WeightOnlyProfileError(
                f"{name}.hits.{key} must be a non-negative integer"
            )
    prefixes: dict[int, int] = {}
    for key, value in raw.items():
        if not key.startswith(SPLIT_KV_PREFIX):
            continue
        suffix = key.removeprefix(SPLIT_KV_PREFIX)
        try:
            prefix = int(suffix)
        except ValueError as exc:
            raise WeightOnlyProfileError(
                f"{name}.hits contains malformed split-KV prefix key {key!r}"
            ) from exc
        if str(prefix) != suffix or prefix not in SPLIT_KV_PREFIXES:
            raise WeightOnlyProfileError(
                f"{name}.hits contains unsupported split-KV prefix {suffix!r}"
            )
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise WeightOnlyProfileError(
                f"{name}.hits.{key} must be a non-negative integer"
            )
        prefixes[prefix] = value
    if split != sum(prefixes.values()):
        raise WeightOnlyProfileError(
            f"{name}.hits split-KV aggregate does not equal per-prefix counts"
        )
    if split > generic:
        raise WeightOnlyProfileError(
            f"{name}.hits split-KV count exceeds effective native q1 count"
        )
    return {
        "native_q1_rope_cache_self_attention": generic,
        "native_q1_rope_cache_self_attention_split_kv_8": split,
        "split_kv_prefix_counts": dict(sorted(prefixes.items())),
    }


def validate_profile(
    payload: Any,
    *,
    role: str,
    cross_mode: str = "accepted",
) -> dict[str, Any]:
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
            cross_mode=cross_mode,
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
    cross_totals = {key: 0 for key in CROSS_DISPATCHES}
    candidate_q1_bmm_total = 0
    baseline_main_dispatch_totals = {
        "q1_bmm_cross_attention": 0,
        "native_cross_mlp_tail": 0,
    }
    effective_self_attention_totals = {
        label: {
            "native_q1_rope_cache_self_attention": 0,
            "native_q1_rope_cache_self_attention_split_kv_8": 0,
            "split_kv_prefix_counts": {},
        }
        for label in PROFILE_LABELS
    }
    for label, records in by_label.items():
        for index, record in enumerate(records):
            name = f"profile.{label}[{index}]"
            policy = _object(
                record.get("optimized_dispatch_policy"),
                name=f"{name}.optimized_dispatch_policy",
            )
            candidate_policy = policy.get("approximate_weight_only")
            candidate_main = candidate and label == "main_generation"
            if candidate:
                expected_candidate_policy = (
                    {
                        "requested": True,
                        "enabled": True,
                        "disabled_reason": None,
                        "result_class": "documented-drift",
                        "exactness_claim": False,
                    }
                    if candidate_main
                    else None
                )
                if candidate_policy != expected_candidate_policy:
                    raise WeightOnlyProfileError(
                        f"{name} has invalid mixed-weight dispatch policy: "
                        f"expected {expected_candidate_policy}, got {candidate_policy}"
                    )
                if "approximate_weight_only" in record:
                    raise WeightOnlyProfileError(
                        f"{name} must keep mixed-weight initialization metadata top-level"
                    )
            else:
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
            if not candidate:
                unexpected_keys = [
                    key for key in WEIGHT_ONLY_DISPATCHES if key in raw_hits
                ]
                if unexpected_keys:
                    raise WeightOnlyProfileError(
                        f"{name} contains mixed-weight dispatch counters: "
                        f"{unexpected_keys}"
                    )
            weight_counts = {
                key: raw_hits.get(key, 0) for key in WEIGHT_ONLY_DISPATCHES
            }
            invalid_weight_counts = {
                key: value
                for key, value in weight_counts.items()
                if isinstance(value, bool) or not isinstance(value, int) or value < 0
            }
            if invalid_weight_counts:
                raise WeightOnlyProfileError(
                    f"{name} contains invalid mixed-weight dispatch counters: "
                    f"{invalid_weight_counts}"
                )
            if not candidate_main:
                unexpected = {
                    key: value for key, value in weight_counts.items() if value != 0
                }
                if unexpected:
                    raise WeightOnlyProfileError(
                        f"{name} contains mixed-weight dispatch counters: {unexpected}"
                    )
            expected_effective_self_attention = (
                {
                    "requested": False,
                    "enabled": False,
                    "owner": None,
                    "kernel": None,
                    "standard_attention_hook_enabled": False,
                    "split_kv_selector_enabled": False,
                    "disabled_reason": "timing_context",
                }
                if label == "timing_context"
                else {
                    "requested": True,
                    "enabled": True,
                    "owner": (
                        "approximate_weight_only"
                        if candidate_main
                        else "accepted_attention_hook"
                    ),
                    "kernel": "native_q1_rope_cache_attention",
                    "standard_attention_hook_enabled": not candidate_main,
                    "split_kv_selector_enabled": True,
                    "disabled_reason": None,
                }
            )
            if policy.get("effective_native_q1_rope_cache_self_attention") != (
                expected_effective_self_attention
            ):
                raise WeightOnlyProfileError(
                    f"{name} has invalid effective native q1 policy"
                )
            expected_mode = (
                "approximate_weight_only_batch1"
                if candidate_main
                else "accepted_batch1"
            )
            if record.get("optimized_dispatch_mode") != expected_mode:
                raise WeightOnlyProfileError(
                    f"{name}.optimized_dispatch_mode must be {expected_mode}"
                )
            effective = _effective_self_attention_counts(record, name=name)
            label_totals = effective_self_attention_totals[label]
            label_totals["native_q1_rope_cache_self_attention"] += effective[
                "native_q1_rope_cache_self_attention"
            ]
            label_totals["native_q1_rope_cache_self_attention_split_kv_8"] += effective[
                "native_q1_rope_cache_self_attention_split_kv_8"
            ]
            prefix_totals = label_totals["split_kv_prefix_counts"]
            for prefix, value in effective["split_kv_prefix_counts"].items():
                prefix_totals[prefix] = prefix_totals.get(prefix, 0) + value

            if label == "timing_context":
                if any(
                    effective[key] != 0
                    for key in (
                        "native_q1_rope_cache_self_attention",
                        "native_q1_rope_cache_self_attention_split_kv_8",
                    )
                ):
                    raise WeightOnlyProfileError(
                        f"{name} timing context must not execute native self-attention"
                    )
                continue

            if candidate_main:
                counts = _dispatch_counts(record, name=name)
                for key, value in counts.items():
                    totals[key] += value
                if (
                    effective["native_q1_rope_cache_self_attention"]
                    != counts["weight_only_self_attention_block"]
                ):
                    raise WeightOnlyProfileError(
                        f"{name} native q1 count does not match weight-owned self-attention"
                    )
                expected_tail_count = counts["weight_only_mlp_tail"]
                cross_counts = {
                    key: raw_hits.get(key, 0) for key in CROSS_DISPATCHES
                }
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in cross_counts.values()
                ):
                    raise WeightOnlyProfileError(
                        f"{name} contains invalid cross candidate counters"
                    )
                for key, value in cross_counts.items():
                    cross_totals[key] += value
                q1_bmm = raw_hits.get("q1_bmm_cross_attention", 0)
                if isinstance(q1_bmm, bool) or not isinstance(q1_bmm, int):
                    raise WeightOnlyProfileError(
                        f"{name}.hits.q1_bmm_cross_attention must be an integer"
                    )
                expected = {
                    "accepted": (expected_tail_count, 0),
                    "fp16_packed_projections": (
                        expected_tail_count,
                        expected_tail_count,
                    ),
                }[cross_mode]
                actual = (
                    q1_bmm,
                    cross_counts["fp16_packed_cross_projection_candidate"],
                )
                if actual != expected:
                    if cross_mode == "accepted":
                        raise WeightOnlyProfileError(
                            f"{name} q1 BMM count does not match weight-owned MLP tail"
                        )
                    raise WeightOnlyProfileError(
                        f"{name} cross dispatch tuple must be {expected}, got {actual}"
                    )
                candidate_q1_bmm_total += q1_bmm
            else:
                for key in (
                    "native_q1_rope_cache_self_attention",
                    "q1_bmm_cross_attention",
                    "native_cross_mlp_tail",
                ):
                    value = raw_hits.get(key)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                    ):
                        raise WeightOnlyProfileError(
                            f"{name}.hits.{key} must be a non-negative integer"
                        )
                    if key in baseline_main_dispatch_totals:
                        baseline_main_dispatch_totals[key] += value

    main_effective = effective_self_attention_totals["main_generation"]
    if candidate:
        missing_dispatch = {key: value for key, value in totals.items() if value <= 0}
        if missing_dispatch:
            raise WeightOnlyProfileError(
                "candidate main generation did not execute every mixed-weight region: "
                f"{missing_dispatch}"
            )
        if main_effective["native_q1_rope_cache_self_attention_split_kv_8"] <= 0:
            raise WeightOnlyProfileError(
                "candidate main generation did not execute split-KV native q1"
            )
        accepted_fallback = (
            main_effective["native_q1_rope_cache_self_attention"]
            - main_effective["native_q1_rope_cache_self_attention_split_kv_8"]
        )
        if accepted_fallback <= 0:
            raise WeightOnlyProfileError(
                "candidate main generation did not execute accepted native q1 fallback"
            )
        main_effective["accepted_fallback"] = accepted_fallback
    else:
        missing_dispatch = {
            key: value
            for key, value in baseline_main_dispatch_totals.items()
            if value <= 0
        }
        if missing_dispatch:
            raise WeightOnlyProfileError(
                "baseline main generation did not execute every accepted region: "
                f"{missing_dispatch}"
            )
        if main_effective["native_q1_rope_cache_self_attention"] <= 0:
            raise WeightOnlyProfileError(
                "baseline main generation did not execute accepted native q1"
            )
        if main_effective["native_q1_rope_cache_self_attention_split_kv_8"] <= 0:
            raise WeightOnlyProfileError(
                "baseline main generation did not execute exact split-KV native q1"
            )
        accepted_fallback = (
            main_effective["native_q1_rope_cache_self_attention"]
            - main_effective["native_q1_rope_cache_self_attention_split_kv_8"]
        )
        if accepted_fallback <= 0:
            raise WeightOnlyProfileError(
                "baseline main generation did not execute accepted native q1 fallback"
            )
        main_effective["accepted_fallback"] = accepted_fallback
    return {
        "role": role,
        "cross_mode": cross_mode,
        "candidate_enabled_for_main": candidate,
        "candidate_disabled_for_timing": candidate,
        "main_weight_only_dispatch_counts": totals,
        "main_cross_candidate_dispatch_counts": cross_totals,
        "main_q1_bmm_cross_attention_count": candidate_q1_bmm_total,
        "baseline_main_dispatch_counts": baseline_main_dispatch_totals,
        "timing_effective_self_attention_counts": effective_self_attention_totals[
            "timing_context"
        ],
        "main_effective_self_attention_counts": main_effective,
    }


def validate_profile_path(
    path: Path,
    *,
    role: str,
    cross_mode: str = "accepted",
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WeightOnlyProfileError(f"cannot read profile {path}: {exc}") from exc
    return validate_profile(payload, role=role, cross_mode=cross_mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--role", choices=PROFILE_ROLES, required=True)
    parser.add_argument("--cross-mode", choices=CROSS_MODES, default="accepted")
    args = parser.parse_args()
    result = validate_profile_path(
        args.profile,
        role=args.role,
        cross_mode=args.cross_mode,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
