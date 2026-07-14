import pytest

from utils.validate_stage_precision_hybrid_profile import validate


def _record(label: str, *, candidate: bool) -> dict:
    timing = label == "timing_context"
    precision = "fp16" if candidate and timing else "fp32"
    hits = {
        "native_q1_rope_cache_self_attention": 12 if candidate and timing else 0,
        "native_q1_self_attention": 0,
        "q1_bmm_cross_attention": 12,
        "native_cross_mlp_tail": 12 if candidate and timing else 0,
        "weight_only_self_attention_block": 0 if timing else 12,
        "weight_only_mlp_tail": 0 if timing else 12,
        "weight_only_final_projection": 0 if timing else 1,
    }
    if not timing:
        hits["native_q1_rope_cache_self_attention"] = 12
        hits["native_q1_rope_cache_self_attention_split_kv_8"] = 8
    policy = {
        "approximate_weight_only": (
            None
            if timing
            else {"requested": True, "enabled": True, "disabled_reason": None}
        )
    }
    return {
        "profile_label": label,
        "precision": precision,
        "optimized_effective_config_version": (
            "accepted-fp16-all-fused-v2"
            if candidate and timing
            else "accepted-fp32-native-cross-mlp-289-v3"
        ),
        "optimized_stage_precision_hybrid_role": (
            "timing_fp16"
            if candidate and timing
            else "main_mixed_fp32"
            if candidate
            else None
        ),
        "optimized_dispatch_policy": policy,
        "optimized_dispatch_capture_hits": hits,
        "optimized_specialized_timing_dispatch": {
            "requested": candidate and timing,
            "enabled": candidate and timing,
        },
    }


def _profile(*, candidate: bool) -> dict:
    metadata = {
        "precision": "fp32",
        "optimized_effective_config": {"precision": "fp32"},
        "optimized_approximate_weight_only": {
            "version": "approximate-fp16-weights-fp32-state-v2",
            "exactness_claim": False,
        },
    }
    if candidate:
        metadata["optimized_stage_precision_hybrid"] = {
            "version": "timing-fp16-main-mixed-fp32-k4-v1",
            "result_class": "documented-drift",
            "exactness_claim": False,
            "load_order": ["main_gamemode_fp32", "base_timing_fp16"],
            "models_distinct": True,
            "decode_block_sizes": {
                "timing_context": 1,
                "main_generation": 4,
            },
            "timing": {
                "model_role": "separate_base_timing_model",
                "precision": "fp16",
                "torch_dtype": "torch.float16",
                "preset_version": "accepted-fp16-all-fused-v2",
                "mixed_weight_state": False,
                "q1_bmm_cross_attention": True,
                "native_q1_rope_cache_self_attention": True,
                "native_cross_mlp_tail": True,
                "specialized_timing_dispatch_override": True,
            },
            "main": {
                "model_role": "gamemode_model",
                "precision": "fp32",
                "torch_dtype": "torch.float32",
                "preset_version": "accepted-fp32-native-cross-mlp-289-v3",
                "fp32_activations_caches_reductions_logits": True,
                "mixed_weight_state": True,
                "mixed_weight_version": "approximate-fp16-weights-fp32-state-v2",
                "split_kv_q1": True,
            },
        }
    return {
        "schema_version": 1,
        "metadata": metadata,
        "generation": [
            _record("timing_context", candidate=candidate),
            _record("main_generation", candidate=candidate),
        ],
    }


def test_validator_accepts_control_and_stage_hybrid_candidate():
    control = validate(_profile(candidate=False), role="control")
    candidate = validate(_profile(candidate=True), role="candidate")

    assert control["stage_precisions"] == {
        "timing_context": "fp32",
        "main_generation": "fp32",
    }
    assert candidate["stage_precisions"] == {
        "timing_context": "fp16",
        "main_generation": "fp32",
    }
    assert candidate["dispatch_totals"]["timing_context"]["native_self"] == 12


def test_validator_rejects_timing_model_that_owns_main_weights():
    payload = _profile(candidate=True)
    payload["generation"][0]["optimized_dispatch_policy"][
        "approximate_weight_only"
    ] = {"enabled": True}

    with pytest.raises(ValueError, match="timing model must not own mixed weights"):
        validate(payload, role="candidate")
