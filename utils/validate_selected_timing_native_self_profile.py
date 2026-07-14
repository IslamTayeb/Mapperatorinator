from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.run_selected_timing_native_self import (  # noqa: E402
    COMPOSITION_VERSION,
    SELECTED_MAIN_COMPOSITION,
)
from utils.validate_weight_only_full_song_profile import (  # noqa: E402
    WeightOnlyProfileError,
    validate_profile as validate_selected_profile,
)


VERSION = "fp32-timing-native-self-v1"
LABELS = ("timing_context", "main_generation")
WEIGHT_DISPATCHES = (
    "weight_only_self_attention_block",
    "weight_only_mlp_tail",
    "weight_only_final_projection",
    "int8_weight_mlp_tail",
    "fp16_packed_cross_projection_candidate",
)


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WeightOnlyProfileError(f"{name} must be an object")
    return value


def _count(hits: dict[str, Any], key: str, *, name: str) -> int:
    value = hits.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WeightOnlyProfileError(f"{name}.{key} must be non-negative integer")
    return value


def _metadata(value: Any, *, name: str) -> dict[str, Any]:
    metadata = _object(value, name=name)
    required = {
        "version": VERSION,
        "scope": "timing-context-only",
        "precision": "fp32",
        "torch_dtype": "torch.float32",
        "result_class": "exact",
        "exactness_claim": True,
        "native_q1_self_attention": True,
        "native_q1_rope_cache_self_attention": True,
        "split_kv_selector": True,
        "q1_bmm_cross_attention_retained": True,
        "native_cross_mlp_tail_retained_disabled": True,
        "approximate_weight_only_retained_disabled": True,
        "original_decoder_forward_retained": True,
        "production_selector_unchanged": True,
    }
    failures = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in required.items()
        if metadata.get(key) != expected
    }
    if failures:
        raise WeightOnlyProfileError(f"{name} has invalid metadata: {failures}")
    return metadata


def _validate_selected_initialization(root: dict[str, Any], *, role: str) -> None:
    if root.get("combined_runtime") != SELECTED_MAIN_COMPOSITION:
        raise WeightOnlyProfileError(
            f"{role} initialization lost selected main composition"
        )
    cross = _object(root.get("cross_candidate"), name=f"{role}.cross_candidate")
    if cross.get("mode") != "fp16_packed_projections":
        raise WeightOnlyProfileError(f"{role} initialization lost FP16-packed cross")
    if cross.get("accepted_q1_bmm") is not True:
        raise WeightOnlyProfileError(f"{role} initialization lost accepted q1 BMM")
    runtime = _object(root.get("cross_runtime"), name=f"{role}.cross_runtime")
    for key in (
        "incremental_exactness_required",
        "packed_projection_delta_only",
        "accepted_q1_bmm_required",
        "original_decoder_forward_required",
    ):
        if runtime.get(key) is not True:
            raise WeightOnlyProfileError(f"{role} cross runtime did not require {key}")
    overlay = _object(root.get("int8_mlp_overlay"), name=f"{role}.int8_mlp_overlay")
    if overlay.get("dispatch_counter") != "int8_weight_mlp_tail":
        raise WeightOnlyProfileError(f"{role} initialization lost INT8 MLP overlay")
    shared = _object(root.get("shared_rope"), name=f"{role}.shared_rope")
    if shared.get("scope") != "main-model-only":
        raise WeightOnlyProfileError(f"{role} shared RoPE scope changed")


def validate_initialization(payload: Any, *, role: str) -> dict[str, Any]:
    root = _object(payload, name="initialization")
    _validate_selected_initialization(root, role=role)
    timing = root.get("timing_native_self")
    if role == "control":
        if timing is not None:
            raise WeightOnlyProfileError(
                "control initialization must not enable timing-native-self"
            )
        return {
            "role": role,
            "selected_main_composition": SELECTED_MAIN_COMPOSITION,
            "timing_native_self": False,
        }
    row = _metadata(timing, name="initialization.timing_native_self")
    extras = {
        "composition_version": COMPOSITION_VERSION,
        "incremental_control": SELECTED_MAIN_COMPOSITION,
        "timing_model_distinct_from_main": True,
        "fixed_timing_work_required": True,
        "complete_request_wall_required": True,
    }
    failures = {
        key: {"expected": expected, "actual": row.get(key)}
        for key, expected in extras.items()
        if row.get(key) != expected
    }
    if failures:
        raise WeightOnlyProfileError(
            f"initialization timing-native-self composition is invalid: {failures}"
        )
    return {
        "role": role,
        "selected_main_composition": SELECTED_MAIN_COMPOSITION,
        "timing_native_self": True,
        "metadata": row,
    }


def validate(payload: Any, initialization: Any, *, role: str) -> dict[str, Any]:
    if role not in {"control", "candidate"}:
        raise WeightOnlyProfileError("role must be control or candidate")
    candidate = role == "candidate"
    root = _object(payload, name="profile")
    base = validate_selected_profile(
        root,
        role="candidate",
        cross_mode="fp16_packed_projections",
        timing_native_self=candidate,
    )
    init = validate_initialization(initialization, role=role)
    generation = root.get("generation")
    if not isinstance(generation, list):
        raise WeightOnlyProfileError("profile.generation must be a list")
    totals = {
        label: {
            "records": 0,
            "native_self": 0,
            "native_rope_cache_self": 0,
            "split_kv": 0,
            "q1_bmm_cross": 0,
            "native_cross_mlp": 0,
        }
        for label in LABELS
    }
    for index, raw in enumerate(generation):
        record = _object(raw, name=f"generation[{index}]")
        label = record.get("profile_label")
        if label not in totals:
            continue
        name = f"generation[{index}].{label}"
        if record.get("precision") != "fp32":
            raise WeightOnlyProfileError(f"{name} must retain FP32 precision")
        policy = _object(
            record.get("optimized_dispatch_policy"),
            name=f"{name}.optimized_dispatch_policy",
        )
        hits = _object(
            record.get("optimized_dispatch_capture_hits"),
            name=f"{name}.optimized_dispatch_capture_hits",
        )
        values = totals[label]
        values["records"] += 1
        values["native_self"] += _count(hits, "native_q1_self_attention", name=name)
        values["native_rope_cache_self"] += _count(
            hits, "native_q1_rope_cache_self_attention", name=name
        )
        values["split_kv"] += _count(
            hits, "native_q1_rope_cache_self_attention_split_kv_8", name=name
        )
        values["q1_bmm_cross"] += _count(hits, "q1_bmm_cross_attention", name=name)
        values["native_cross_mlp"] += _count(hits, "native_cross_mlp_tail", name=name)
        if label == "timing_context":
            if policy.get("q1_bmm_cross_attention", {}).get("enabled") is not True:
                raise WeightOnlyProfileError(f"{name} lost accepted q1 BMM cross")
            for key in (
                "native_q1_self_attention",
                "native_q1_rope_cache_self_attention",
            ):
                if policy.get(key, {}).get("enabled") is not candidate:
                    raise WeightOnlyProfileError(f"{name} has wrong {key} policy")
            if policy.get("native_cross_mlp_tail", {}).get("enabled") is not False:
                raise WeightOnlyProfileError(f"{name} enabled native cross/MLP")
            timing_policy = policy.get("timing_native_self")
            if candidate:
                _metadata(
                    record.get("optimized_timing_native_self"),
                    name=f"{name}.optimized_timing_native_self",
                )
                if timing_policy != {
                    "requested": True,
                    "enabled": True,
                    "disabled_reason": None,
                    "result_class": "exact",
                    "exactness_claim": True,
                }:
                    raise WeightOnlyProfileError(
                        f"{name} has invalid timing-native-self policy"
                    )
            elif timing_policy is not None or "optimized_timing_native_self" in record:
                raise WeightOnlyProfileError(
                    f"{name} control unexpectedly contains timing-native-self metadata"
                )
            for dispatch in WEIGHT_DISPATCHES:
                expected = 0 if dispatch != "q1_bmm_cross_attention" else None
                if expected == 0 and _count(hits, dispatch, name=name) != 0:
                    raise WeightOnlyProfileError(
                        f"{name} executed main-only dispatch {dispatch}"
                    )
        elif (
            "timing_native_self" in policy
            or "optimized_timing_native_self" in record
        ):
            raise WeightOnlyProfileError(
                f"{name} leaked timing-native-self into main generation"
            )

    timing = totals["timing_context"]
    main = totals["main_generation"]
    if timing["records"] <= 0 or main["records"] <= 0:
        raise WeightOnlyProfileError("profile must contain timing and main records")
    if timing["q1_bmm_cross"] <= 0:
        raise WeightOnlyProfileError("timing generation did not execute q1 BMM cross")
    if timing["native_cross_mlp"] != 0:
        raise WeightOnlyProfileError("timing generation executed native cross/MLP")
    if candidate and timing["native_rope_cache_self"] <= 0:
        raise WeightOnlyProfileError("candidate timing did not execute native self")
    if not candidate and any(
        timing[key] != 0
        for key in ("native_self", "native_rope_cache_self", "split_kv")
    ):
        raise WeightOnlyProfileError("control timing executed native self")
    return {
        "schema_version": 1,
        "role": role,
        "precision": "fp32",
        "base_composition": base,
        "initialization": init,
        "dispatch_totals": totals,
        "fixed_main_tokens": 8_294,
        "fixed_timing_tokens": 821,
        "complete_request_wall_required": True,
        "pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--role", choices=("control", "candidate"), required=True)
    args = parser.parse_args()
    result = validate(
        json.loads(args.profile.read_text(encoding="utf-8")),
        json.loads(args.initialization.read_text(encoding="utf-8")),
        role=args.role,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
