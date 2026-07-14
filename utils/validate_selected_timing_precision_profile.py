from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.inference.optimized.single.timing_precision_matrix import (  # noqa: E402
    FP16_WEIGHTS_FP32_STATE,
    FULL_FP16,
    MATRIX_VERSION,
    TIMING_PRECISION_MODES,
)
from utils.run_selected_timing_native_self import (  # noqa: E402
    SELECTED_MAIN_COMPOSITION,
)
from utils.validate_weight_only_full_song_profile import (  # noqa: E402
    WeightOnlyProfileError,
)


MAIN_REQUIRED_HITS = (
    "weight_only_self_attention_block",
    "int8_weight_mlp_tail",
    "weight_only_final_projection",
    "fp16_packed_cross_projection_candidate",
    "q1_bmm_cross_attention",
    "native_q1_rope_cache_self_attention",
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


def _matrix_metadata(value: Any, *, name: str, mode: str) -> dict[str, Any]:
    row = _object(value, name=name)
    full_fp16 = mode == FULL_FP16
    required = {
        "version": MATRIX_VERSION,
        "mode": mode,
        "scope": "separate-timing-model-only",
        "result_class": "documented-drift",
        "exactness_claim": False,
        "precision": "fp16" if full_fp16 else "fp32",
        "torch_dtype": "torch.float16" if full_fp16 else "torch.float32",
        "timing_block_size": 1,
        "q1_bmm_cross_attention": True,
        "original_decoder_forward_retained": True,
        "production_selector_unchanged": True,
        "main_runtime_unchanged": True,
        "full_fp16_model_storage": full_fp16,
        "fp16_packed_weights": not full_fp16,
        "fp32_activations_caches_norm_attention_reductions_logits": not full_fp16,
        "native_q1_rope_cache_self_attention": True,
        "native_cross_mlp_tail": full_fp16,
    }
    failures = {
        key: {"expected": expected, "actual": row.get(key)}
        for key, expected in required.items()
        if row.get(key) != expected
    }
    if failures:
        raise WeightOnlyProfileError(f"{name} has invalid metadata: {failures}")
    return row


def validate_initialization(payload: Any, *, mode: str) -> dict[str, Any]:
    root = _object(payload, name="initialization")
    if root.get("combined_runtime") != SELECTED_MAIN_COMPOSITION:
        raise WeightOnlyProfileError("initialization lost selected main composition")
    cross = _object(root.get("cross_candidate"), name="initialization.cross_candidate")
    if cross.get("mode") != "fp16_packed_projections":
        raise WeightOnlyProfileError("initialization lost selected FP16 main cross")
    runtime = _object(root.get("cross_runtime"), name="initialization.cross_runtime")
    for key in (
        "incremental_exactness_required",
        "packed_projection_delta_only",
        "accepted_q1_bmm_required",
        "original_decoder_forward_required",
    ):
        if runtime.get(key) is not True:
            raise WeightOnlyProfileError(f"selected cross runtime did not require {key}")
    timing = _matrix_metadata(
        root.get("timing_precision_matrix"),
        name="initialization.timing_precision_matrix",
        mode=mode,
    )
    for key, expected in {
        "composition_version": "selected-main-timing-precision-matrix-v1",
        "incremental_control": SELECTED_MAIN_COMPOSITION,
        "timing_model_distinct_from_main": True,
        "fixed_main_tokens": 8_294,
        "fixed_timing_tokens": 821,
        "complete_request_wall_required": True,
    }.items():
        if timing.get(key) != expected:
            raise WeightOnlyProfileError(
                f"initialization timing precision metadata has wrong {key}"
            )
    if mode == FP16_WEIGHTS_FP32_STATE:
        weights = _object(
            timing.get("weight_initialization"),
            name="initialization.timing_precision_matrix.weight_initialization",
        )
        if weights.get("cross_candidate", {}).get("mode") != "fp16_packed_projections":
            raise WeightOnlyProfileError("timing packed weights lost FP16 cross mode")
    elif "weight_initialization" in timing:
        raise WeightOnlyProfileError("full-FP16 timing must not pack separate weights")
    return timing


def validate(payload: Any, initialization: Any, *, mode: str) -> dict[str, Any]:
    if mode not in TIMING_PRECISION_MODES:
        raise WeightOnlyProfileError(
            f"mode must be one of {sorted(TIMING_PRECISION_MODES)}"
        )
    root = _object(payload, name="profile")
    timing_metadata = validate_initialization(initialization, mode=mode)
    generation = root.get("generation")
    if not isinstance(generation, list) or not generation:
        raise WeightOnlyProfileError("profile.generation must be a non-empty list")
    totals = {
        "main_records": 0,
        "timing_records": 0,
        "main_required_hits": {key: 0 for key in MAIN_REQUIRED_HITS},
        "timing": {
            "native_self": 0,
            "native_cross_mlp": 0,
            "q1_bmm_cross": 0,
            "weight_self": 0,
            "weight_mlp": 0,
            "weight_logits": 0,
            "fp16_cross": 0,
        },
    }
    for index, raw in enumerate(generation):
        record = _object(raw, name=f"generation[{index}]")
        label = record.get("profile_label")
        if label not in {"timing_context", "main_generation"}:
            continue
        name = f"generation[{index}].{label}"
        policy = _object(
            record.get("optimized_dispatch_policy"),
            name=f"{name}.optimized_dispatch_policy",
        )
        hits = _object(record.get("optimized_dispatch_capture_hits"), name=f"{name}.hits")
        if label == "main_generation":
            totals["main_records"] += 1
            if record.get("precision") != "fp32":
                raise WeightOnlyProfileError(f"{name} changed selected main precision")
            if "optimized_timing_precision_matrix" in record:
                raise WeightOnlyProfileError(f"{name} leaked timing precision metadata")
            approximate = policy.get("approximate_weight_only")
            if not isinstance(approximate, dict) or approximate.get("enabled") is not True:
                raise WeightOnlyProfileError(f"{name} lost selected mixed-weight main")
            effective = policy.get("effective_native_q1_rope_cache_self_attention")
            if not isinstance(effective, dict) or effective.get("owner") != "approximate_weight_only":
                raise WeightOnlyProfileError(f"{name} lost selected main self owner")
            for key in MAIN_REQUIRED_HITS:
                totals["main_required_hits"][key] += _count(hits, key, name=name)
            continue

        totals["timing_records"] += 1
        _matrix_metadata(
            record.get("optimized_timing_precision_matrix"),
            name=f"{name}.optimized_timing_precision_matrix",
            mode=mode,
        )
        expected_precision = "fp16" if mode == FULL_FP16 else "fp32"
        if record.get("precision") != expected_precision:
            raise WeightOnlyProfileError(
                f"{name} precision must be {expected_precision}"
            )
        timing = totals["timing"]
        timing["native_self"] += _count(
            hits, "native_q1_rope_cache_self_attention", name=name
        )
        timing["native_cross_mlp"] += _count(hits, "native_cross_mlp_tail", name=name)
        timing["q1_bmm_cross"] += _count(hits, "q1_bmm_cross_attention", name=name)
        timing["weight_self"] += _count(hits, "weight_only_self_attention_block", name=name)
        timing["weight_mlp"] += _count(hits, "weight_only_mlp_tail", name=name)
        timing["weight_logits"] += _count(hits, "weight_only_final_projection", name=name)
        timing["fp16_cross"] += _count(
            hits, "fp16_packed_cross_projection_candidate", name=name
        )
        specialized = _object(
            record.get("optimized_specialized_timing_dispatch"),
            name=f"{name}.optimized_specialized_timing_dispatch",
        )
        weight_dispatch = _object(
            record.get("optimized_timing_weight_only_dispatch"),
            name=f"{name}.optimized_timing_weight_only_dispatch",
        )
        if specialized.get("enabled") is not (mode == FULL_FP16):
            raise WeightOnlyProfileError(f"{name} specialized timing flag is wrong")
        if weight_dispatch.get("enabled") is not (mode == FP16_WEIGHTS_FP32_STATE):
            raise WeightOnlyProfileError(f"{name} timing weight-only flag is wrong")

    if totals["main_records"] <= 0 or totals["timing_records"] <= 0:
        raise WeightOnlyProfileError("profile must contain timing and main records")
    missing_main = {
        key: value
        for key, value in totals["main_required_hits"].items()
        if value <= 0
    }
    if missing_main:
        raise WeightOnlyProfileError(f"selected main dispatches are missing: {missing_main}")
    timing = totals["timing"]
    if timing["q1_bmm_cross"] <= 0 or timing["native_self"] <= 0:
        raise WeightOnlyProfileError("timing candidate did not run q1 BMM and native self")
    if mode == FULL_FP16:
        if timing["native_cross_mlp"] <= 0:
            raise WeightOnlyProfileError("full-FP16 timing did not run native cross/MLP")
        unexpected = {key: timing[key] for key in ("weight_self", "weight_mlp", "weight_logits", "fp16_cross") if timing[key]}
        if unexpected:
            raise WeightOnlyProfileError(f"full-FP16 timing ran weight-only hooks: {unexpected}")
    else:
        required = {
            key: timing[key]
            for key in ("weight_self", "weight_mlp", "weight_logits", "fp16_cross")
            if timing[key] <= 0
        }
        if required:
            raise WeightOnlyProfileError(f"timing packed-weight hooks are missing: {required}")
        if timing["native_cross_mlp"] != 0:
            raise WeightOnlyProfileError("timing packed weights ran native cross/MLP")
    return {
        "schema_version": 1,
        "mode": mode,
        "timing_metadata": timing_metadata,
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
    parser.add_argument("--mode", choices=sorted(TIMING_PRECISION_MODES), required=True)
    args = parser.parse_args()
    result = validate(
        json.loads(args.profile.read_text(encoding="utf-8")),
        json.loads(args.initialization.read_text(encoding="utf-8")),
        mode=args.mode,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
