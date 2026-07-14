from copy import deepcopy

import pytest

from utils.validate_int8_mlp_full_song_profile import (
    Int8MlpProfileError,
    validate_profile,
)


def _overlay():
    return {
        "version": "per-row-symmetric-int8-mlp-v1",
        "result_class": "documented-drift",
        "exactness_claim": False,
        "scope": "main-model-decoder-mlp-only",
        "composition_control": "k4-split-kv-mixed-weight-shared-rope-v1",
        "replaces_effective_regions": ["mlp_fc1", "mlp_fc2"],
        "retained_unused_fp16_regions": ["mlp_fc1", "mlp_fc2"],
        "fp32_activations_norm_bias_reductions_residual_outputs": True,
        "quantization": "symmetric-per-output-row",
        "dispatch_counter": "int8_weight_mlp_tail",
        "extension_init_seconds": 1.0,
        "weight_pack_seconds": 0.2,
        "packed_weight_bytes": 1234,
    }


def _profile(*, candidate: bool):
    metadata = {"optimized_approximate_weight_only": {}}
    if candidate:
        metadata["optimized_approximate_weight_only"]["int8_mlp_overlay"] = _overlay()
    return {
        "schema_version": 1,
        "metadata": metadata,
        "generation": [
            {
                "profile_label": "timing_context",
                "optimized_dispatch_capture_hits": {
                    "weight_only_mlp_tail": 0,
                    "int8_weight_mlp_tail": 0,
                },
            },
            {
                "profile_label": "main_generation",
                "optimized_dispatch_capture_hits": {
                    "weight_only_mlp_tail": 120 if candidate else 110,
                    "int8_weight_mlp_tail": 120 if candidate else 0,
                },
            },
        ],
    }


def test_candidate_requires_overlay_and_one_int8_dispatch_per_mixed_mlp_hook() -> None:
    report = validate_profile(_profile(candidate=True), role="candidate")

    assert report["pass"] is True
    assert report["overlay_present"] is True
    assert report["dispatch_totals"]["main_generation"] == 120


def test_incremental_control_requires_no_int8_overlay_or_dispatch() -> None:
    report = validate_profile(_profile(candidate=False), role="baseline")

    assert report["pass"] is True
    assert report["overlay_present"] is False
    assert report["dispatch_totals"]["main_generation"] == 0


def test_candidate_rejects_partial_int8_dispatch() -> None:
    payload = _profile(candidate=True)
    payload["generation"][1]["optimized_dispatch_capture_hits"][
        "int8_weight_mlp_tail"
    ] = 119

    with pytest.raises(Int8MlpProfileError, match="must equal"):
        validate_profile(payload, role="candidate")


def test_timing_context_rejects_int8_dispatch() -> None:
    payload = deepcopy(_profile(candidate=True))
    payload["generation"][0]["optimized_dispatch_capture_hits"][
        "int8_weight_mlp_tail"
    ] = 1

    with pytest.raises(Int8MlpProfileError, match="timing context"):
        validate_profile(payload, role="candidate")
