import pytest

from utils.run_main_shared_rope_delegate import SHARED_STAGE_COMPOSITION_VERSION
from utils.validate_shared_rope_stage_initialization import validate


def _payload(role: str) -> dict:
    payload = {
        "combined_runtime": SHARED_STAGE_COMPOSITION_VERSION,
        "composition_arm": role,
        "result_class": "documented-drift",
        "exactness_claim": False,
        "initialization_wall_seconds": 1.25,
        "initialization_cuda_memory": {
            "allocated_bytes_before": 1,
            "allocated_bytes_after": 2,
            "allocated_bytes_delta": 1,
            "reserved_bytes_before": 3,
            "reserved_bytes_after": 4,
            "reserved_bytes_delta": 1,
            "max_allocated_bytes_before": 5,
            "max_allocated_bytes_after": 6,
            "max_allocated_bytes_delta": 1,
        },
        "shared_rope": {
            "version": "shared-decoder-rope-v1",
            "scope": "main-model-only",
            "incremental_exactness_claim": True,
            "original_decoder_forward_required": True,
            "stats": {
                "module_count": 12,
                "group_count": 1,
                "forwards": 9,
                "computes": 9,
                "reuses": 99,
                "expected_computes": 9,
                "expected_reuses": 99,
                "eliminated_per_forward": 11,
            },
        },
    }
    blocks = {"timing_context": 1, "main_generation": 4}
    if role == "control":
        payload["decode_block_sizes"] = blocks
    else:
        payload["stage_precision_hybrid"] = {
            "version": "timing-fp16-main-mixed-fp32-k4-v1",
            "result_class": "documented-drift",
            "exactness_claim": False,
            "decode_block_sizes": blocks,
        }
    return payload


@pytest.mark.parametrize("role", ["control", "candidate"])
def test_validator_accepts_both_composed_arms(role: str) -> None:
    report = validate(_payload(role), role=role)

    assert report["pass"] is True
    assert report["shared_rope"]["reuses"] == 99
    assert report["setup_seconds"]["initialization_wall_seconds"] == 1.25
    assert report["initialization_cuda_memory"]["allocated_bytes_delta"] == 1


def test_validator_rejects_shared_rope_accounting_drift() -> None:
    payload = _payload("candidate")
    payload["shared_rope"]["stats"]["reuses"] = 98

    with pytest.raises(ValueError, match="reuse accounting"):
        validate(payload, role="candidate")


def test_validator_rejects_hybrid_on_control() -> None:
    payload = _payload("candidate")
    payload["composition_arm"] = "control"

    with pytest.raises(ValueError, match="control must not initialize"):
        validate(payload, role="control")
