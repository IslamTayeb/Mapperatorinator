from __future__ import annotations

import pytest

from utils.validate_coalesced_split_kv_profile import (
    COALESCED,
    GENERIC,
    SPLIT,
    CoalescedProfileError,
    validate_profile,
)


def _profile(*, coalesced: int, split: int = 10, generic: int = 12):
    hits = {GENERIC: generic, SPLIT: split}
    if coalesced:
        hits[COALESCED] = coalesced
        hits[f"{COALESCED}_prefix_640"] = coalesced
    return {
        "schema_version": 1,
        "generation": [
            {
                "profile_label": "timing_context",
                "optimized_dispatch_capture_hits": {},
            },
            {
                "profile_label": "main_generation",
                "optimized_dispatch_capture_hits": hits,
            },
        ],
    }


def test_candidate_requires_all_split_dispatches_and_short_fallback() -> None:
    report = validate_profile(_profile(coalesced=10), role="candidate")
    assert report["pass"]
    assert report["main_counts"][COALESCED] == 10

    with pytest.raises(CoalescedProfileError, match="every eligible"):
        validate_profile(_profile(coalesced=9), role="candidate")
    with pytest.raises(CoalescedProfileError, match="short fallback"):
        validate_profile(
            _profile(coalesced=10, split=10, generic=10), role="candidate"
        )


def test_baseline_rejects_coalesced_and_candidate_requires_it() -> None:
    assert validate_profile(_profile(coalesced=0), role="baseline")["pass"]
    with pytest.raises(CoalescedProfileError, match="unexpectedly"):
        validate_profile(_profile(coalesced=10), role="baseline")
    with pytest.raises(CoalescedProfileError, match="did not execute"):
        validate_profile(_profile(coalesced=0), role="candidate")


def test_timing_context_must_remain_unmodified() -> None:
    payload = _profile(coalesced=10)
    payload["generation"][0]["optimized_dispatch_capture_hits"] = {COALESCED: 1}
    with pytest.raises(CoalescedProfileError, match="timing generation"):
        validate_profile(payload, role="candidate")
