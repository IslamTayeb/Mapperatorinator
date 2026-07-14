import pytest

from utils.validate_k4_mask_reuse_profile import validate


def _record(label: str, *, enabled: bool, block_replays: int = 3) -> dict:
    reuse = {"enabled": enabled}
    if enabled:
        reuse.update(
            {
                "allocation_count": 1,
                "buffer_capacity_elements": 1024,
                "buffer_bytes": 8192,
                "active_length": 512,
                "reset_calls": 1,
                "sync_calls": block_replays,
                "owned_reuse_calls": block_replays - 1,
                "external_copy_calls": 1,
                "external_copy_bytes": 128,
                "tail_fill_calls": 0,
                "activated_elements": block_replays * 4,
            }
        )
    return {
        "profile_label": label,
        "optimized_cuda_graphs": {
            "k8_candidate": {
                "block_size": 4,
                "block_replays": block_replays,
                "decoder_attention_mask_reuse_requested": enabled,
                "decoder_attention_mask_reuse": reuse,
            }
        },
    }


def _profile(*, enabled: bool) -> dict:
    return {
        "generation": [
            _record("timing_context", enabled=enabled),
            _record("main_generation", enabled=enabled, block_replays=7),
        ]
    }


def test_control_requires_explicit_disabled_reuse_evidence():
    report = validate(_profile(enabled=False), role="control")
    assert report["pass"] is True
    assert report["labels"]["main_generation"]["block_replays"] == 7


def test_candidate_requires_one_allocation_and_owned_reuse():
    report = validate(_profile(enabled=True), role="candidate")
    main = report["labels"]["main_generation"]
    assert main["allocation_count_max"] == 1
    assert main["sync_calls"] == main["block_replays"] == 7
    assert main["owned_reuse_calls"] == 6


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("allocation_count", 2, "more than one mask buffer"),
        ("sync_calls", 6, "sync/block accounting"),
        ("owned_reuse_calls", 0, "ownership accounting"),
        ("reset_calls", 0, "not request-reset"),
    ),
)
def test_candidate_fails_loudly_on_allocation_ownership_and_reset(
    field, value, message
):
    payload = _profile(enabled=True)
    payload["generation"][1]["optimized_cuda_graphs"]["k8_candidate"][
        "decoder_attention_mask_reuse"
    ][field] = value
    with pytest.raises(ValueError, match=message):
        validate(payload, role="candidate")


def test_role_mismatch_fails_loudly():
    with pytest.raises(ValueError, match="role mismatch"):
        validate(_profile(enabled=False), role="candidate")
