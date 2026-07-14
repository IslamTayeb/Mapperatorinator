from copy import deepcopy

import pytest

from utils.validate_k4_shared_rope_mask_reuse_initialization import ROLES, validate


def _initialization() -> dict:
    members = [f"layer-{index}" for index in range(12)]
    return {
        "result_class": "documented-drift",
        "exactness_claim": False,
        "combined_runtime": "k4-split-kv-mixed-weight-shared-rope-v1",
        "shared_rope": {
            "version": "shared-decoder-rope-v1",
            "scope": "main-model-only",
            "incremental_exactness_claim": True,
            "original_decoder_forward_required": True,
            "stats": {
                "module_count": 12,
                "group_count": 1,
                "forwards": 7,
                "computes": 7,
                "reuses": 77,
                "eliminated_per_forward": 11,
                "expected_computes": 7,
                "expected_reuses": 77,
                "group_computes": {"rope-0": 7},
                "member_names": members,
                "group_members": {"rope-0": members},
            },
        },
    }


def _reciprocal() -> dict:
    return {role: deepcopy(_initialization()) for role in ROLES}


def test_identical_shared_rope_stack_passes_all_reciprocal_arms() -> None:
    report = validate(_reciprocal())
    assert report["pass"] is True
    assert report["material"]["shared_rope"]["stats"]["reuses"] == 77


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("shared_rope", "original_decoder_forward_required"), False, "original"),
        (("shared_rope", "stats", "computes"), 6, "compute accounting"),
        (("shared_rope", "stats", "reuses"), 76, "reuse accounting"),
    ),
)
def test_invalid_shared_rope_invariants_fail_loudly(path, value, message) -> None:
    payloads = _reciprocal()
    target = payloads["candidate_first"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        validate(payloads)


def test_reciprocal_execution_count_difference_fails_loudly() -> None:
    payloads = _reciprocal()
    stats = payloads["candidate_second"]["shared_rope"]["stats"]
    stats["forwards"] = 8
    stats["computes"] = stats["expected_computes"] = 8
    stats["reuses"] = stats["expected_reuses"] = 88
    stats["group_computes"]["rope-0"] = 8

    with pytest.raises(ValueError, match="differs across reciprocal"):
        validate(payloads)
