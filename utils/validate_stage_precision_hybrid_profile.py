"""Validate the isolated FP16-timing / mixed-FP32-main profile contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LABELS = ("timing_context", "main_generation")
HYBRID_VERSION = "timing-fp16-main-mixed-fp32-k4-v1"
WEIGHT_VERSION = "approximate-fp16-weights-fp32-state-v2"
FP32_PRESET = "accepted-fp32-native-cross-mlp-289-v3"
FP16_PRESET = "accepted-fp16-all-fused-v2"
SPLIT_DISPATCH = "native_q1_rope_cache_self_attention_split_kv_8"


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _count(hits: dict[str, Any], key: str, *, name: str) -> int:
    value = hits.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name}.{key} must be a non-negative integer")
    return value


def _hybrid_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    hybrid = _object(
        metadata.get("optimized_stage_precision_hybrid"),
        name="metadata.optimized_stage_precision_hybrid",
    )
    expected = {
        "version": HYBRID_VERSION,
        "result_class": "documented-drift",
        "exactness_claim": False,
        "load_order": ["main_gamemode_fp32", "base_timing_fp16"],
        "models_distinct": True,
        "decode_block_sizes": {
            "timing_context": 1,
            "main_generation": 4,
        },
    }
    failures = {
        key: {"expected": expected_value, "actual": hybrid.get(key)}
        for key, expected_value in expected.items()
        if hybrid.get(key) != expected_value
    }
    timing = _object(hybrid.get("timing"), name="hybrid.timing")
    main = _object(hybrid.get("main"), name="hybrid.main")
    timing_expected = {
        "model_role": "separate_base_timing_model",
        "precision": "fp16",
        "torch_dtype": "torch.float16",
        "preset_version": FP16_PRESET,
        "mixed_weight_state": False,
        "q1_bmm_cross_attention": True,
        "native_q1_rope_cache_self_attention": True,
        "native_cross_mlp_tail": True,
        "specialized_timing_dispatch_override": True,
    }
    main_expected = {
        "model_role": "gamemode_model",
        "precision": "fp32",
        "torch_dtype": "torch.float32",
        "preset_version": FP32_PRESET,
        "fp32_activations_caches_reductions_logits": True,
        "mixed_weight_state": True,
        "mixed_weight_version": WEIGHT_VERSION,
        "split_kv_q1": True,
    }
    for prefix, actual, required in (
        ("timing", timing, timing_expected),
        ("main", main, main_expected),
    ):
        failures.update(
            {
                f"{prefix}.{key}": {
                    "expected": expected_value,
                    "actual": actual.get(key),
                }
                for key, expected_value in required.items()
                if actual.get(key) != expected_value
            }
        )
    if failures:
        raise ValueError(f"invalid hybrid metadata: {failures}")
    return hybrid


def validate(payload: Any, *, role: str) -> dict[str, Any]:
    if role not in {"control", "candidate"}:
        raise ValueError("role must be control or candidate")
    root = _object(payload, name="profile")
    if root.get("schema_version") != 1:
        raise ValueError("profile.schema_version must be 1")
    metadata = _object(root.get("metadata"), name="profile.metadata")
    if metadata.get("precision") != "fp32":
        raise ValueError("public request precision must remain fp32")
    effective = _object(
        metadata.get("optimized_effective_config"),
        name="metadata.optimized_effective_config",
    )
    if effective.get("precision") != "fp32":
        raise ValueError("top-level optimized runtime must remain FP32 main")
    weights = _object(
        metadata.get("optimized_approximate_weight_only"),
        name="metadata.optimized_approximate_weight_only",
    )
    if (
        weights.get("version") != WEIGHT_VERSION
        or weights.get("exactness_claim") is not False
    ):
        raise ValueError("main mixed-weight initialization metadata is invalid")
    if role == "candidate":
        _hybrid_metadata(metadata)
    elif "optimized_stage_precision_hybrid" in metadata:
        raise ValueError("control profile must not contain hybrid metadata")

    generation = root.get("generation")
    if not isinstance(generation, list) or not generation:
        raise ValueError("profile.generation must be a non-empty list")
    by_label = {label: [] for label in LABELS}
    for index, raw in enumerate(generation):
        record = _object(raw, name=f"generation[{index}]")
        label = record.get("profile_label")
        if label in by_label:
            by_label[label].append(record)
    if any(not records for records in by_label.values()):
        raise ValueError("profile must contain timing_context and main_generation")

    totals = {
        label: {
            "records": len(records),
            "native_self": 0,
            "split_kv": 0,
            "q1_bmm_cross": 0,
            "native_cross_mlp": 0,
            "weight_self": 0,
            "weight_mlp": 0,
            "weight_logits": 0,
        }
        for label, records in by_label.items()
    }
    expected_timing_precision = "fp16" if role == "candidate" else "fp32"
    expected_timing_version = FP16_PRESET if role == "candidate" else FP32_PRESET
    for label, records in by_label.items():
        for index, record in enumerate(records):
            name = f"{label}[{index}]"
            expected_precision = (
                expected_timing_precision if label == "timing_context" else "fp32"
            )
            if record.get("precision") != expected_precision:
                raise ValueError(f"{name} has wrong stage precision")
            expected_version = (
                expected_timing_version if label == "timing_context" else FP32_PRESET
            )
            if record.get("optimized_effective_config_version") != expected_version:
                raise ValueError(f"{name} has wrong optimized preset version")
            expected_role = (
                "timing_fp16"
                if role == "candidate" and label == "timing_context"
                else "main_mixed_fp32"
                if role == "candidate"
                else None
            )
            if record.get("optimized_stage_precision_hybrid_role") != expected_role:
                raise ValueError(f"{name} has wrong hybrid stage role")

            policy = _object(
                record.get("optimized_dispatch_policy"),
                name=f"{name}.optimized_dispatch_policy",
            )
            hits = _object(
                record.get("optimized_dispatch_capture_hits"),
                name=f"{name}.optimized_dispatch_capture_hits",
            )
            timing = label == "timing_context"
            approximate = policy.get("approximate_weight_only")
            if timing and approximate is not None:
                raise ValueError(f"{name} timing model must not own mixed weights")
            if not timing and (
                not isinstance(approximate, dict)
                or approximate.get("enabled") is not True
            ):
                raise ValueError(f"{name} main model must enable mixed weights")

            specialized = _object(
                record.get("optimized_specialized_timing_dispatch"),
                name=f"{name}.optimized_specialized_timing_dispatch",
            )
            special_expected = role == "candidate" and timing
            if specialized.get("requested") is not special_expected:
                raise ValueError(f"{name} has wrong specialized timing request")
            if specialized.get("enabled") is not special_expected:
                raise ValueError(f"{name} has wrong specialized timing state")

            values = totals[label]
            values["native_self"] += _count(
                hits,
                "native_q1_rope_cache_self_attention",
                name=name,
            )
            values["split_kv"] += _count(hits, SPLIT_DISPATCH, name=name)
            values["q1_bmm_cross"] += _count(
                hits,
                "q1_bmm_cross_attention",
                name=name,
            )
            values["native_cross_mlp"] += _count(
                hits,
                "native_cross_mlp_tail",
                name=name,
            )
            values["weight_self"] += _count(
                hits,
                "weight_only_self_attention_block",
                name=name,
            )
            values["weight_mlp"] += _count(
                hits,
                "weight_only_mlp_tail",
                name=name,
            )
            values["weight_logits"] += _count(
                hits,
                "weight_only_final_projection",
                name=name,
            )

    timing = totals["timing_context"]
    main = totals["main_generation"]
    for key in ("q1_bmm_cross",):
        if timing[key] <= 0:
            raise ValueError(f"timing generation did not execute {key}")
    if timing["split_kv"] != 0:
        raise ValueError("timing generation must not claim FP32 split-KV")
    timing_native_expected = role == "candidate"
    for key in ("native_self", "native_cross_mlp"):
        if (timing[key] > 0) is not timing_native_expected:
            raise ValueError(f"timing generation has wrong {key} dispatch")
    if any(timing[key] != 0 for key in ("weight_self", "weight_mlp", "weight_logits")):
        raise ValueError("timing generation executed main mixed-weight kernels")
    for key in ("native_self", "split_kv", "q1_bmm_cross", "weight_self", "weight_mlp", "weight_logits"):
        if main[key] <= 0:
            raise ValueError(f"main generation did not execute {key}")
    if main["native_cross_mlp"] != 0:
        raise ValueError("mixed-weight main unexpectedly used accepted cross+MLP tail")

    return {
        "schema_version": 1,
        "role": role,
        "public_precision": "fp32",
        "stage_precisions": {
            "timing_context": expected_timing_precision,
            "main_generation": "fp32",
        },
        "dispatch_totals": totals,
        "pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--role", choices=("control", "candidate"), required=True)
    args = parser.parse_args()
    result = validate(
        json.loads(args.profile.read_text(encoding="utf-8")),
        role=args.role,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
