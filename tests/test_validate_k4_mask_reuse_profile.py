import pytest

from utils.validate_k4_mask_reuse_profile import validate


def _record(
    label: str,
    *,
    enabled: bool,
    block_replays: int = 3,
    external_copy_calls: int = 1,
) -> dict:
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
                "owned_reuse_calls": block_replays - external_copy_calls,
                "external_copy_calls": external_copy_calls,
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
    assert main["reuse_opportunities"] == 6
    assert main["records_with_reuse_opportunity"] == 1


def test_short_window_with_remainder_has_no_owned_reuse_opportunity() -> None:
    payload = _profile(enabled=True)
    # Distilled from job 49837362 timing sequence 43: two K4 handoffs were
    # separated by ordinary remainder work, so both arrived with external mask
    # storage and neither could reuse the owned buffer from the prior handoff.
    payload["generation"][0] = _record(
        "timing_context",
        enabled=True,
        block_replays=2,
        external_copy_calls=2,
    )

    report = validate(payload, role="candidate")

    timing = report["labels"]["timing_context"]
    assert timing["owned_reuse_calls"] == 0
    assert timing["reuse_opportunities"] == 0
    assert timing["records_without_reuse_opportunity"] == 1


def test_missing_owned_reuse_fails_when_an_opportunity_exists() -> None:
    payload = _profile(enabled=True)
    reuse = payload["generation"][1]["optimized_cuda_graphs"]["k8_candidate"][
        "decoder_attention_mask_reuse"
    ]
    reuse["owned_reuse_calls"] = 0
    reuse["external_copy_calls"] = 1

    with pytest.raises(ValueError, match="ownership accounting"):
        validate(payload, role="candidate")


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
