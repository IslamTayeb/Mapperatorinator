from __future__ import annotations

import pytest

from utils.validate_coalesced_split_kv_profile import (
    COALESCED,
    GENERIC,
    SPLIT,
    CoalescedProfileError,
    validate_profile,
)


def _profile(
    *,
    coalesced: int,
    split: int = 10,
    generic: int = 12,
    timing_split: int = 0,
):
    hits = {GENERIC: generic, SPLIT: split}
    prefixes = tuple(range(192, 833, 64))
    quotient, remainder = divmod(split, len(prefixes))
    for index, prefix in enumerate(prefixes):
        hits[f"{SPLIT}_prefix_{prefix}"] = quotient + (index < remainder)
    if coalesced:
        hits[COALESCED] = coalesced
        quotient, remainder = divmod(coalesced, len(prefixes))
        for index, prefix in enumerate(prefixes):
            hits[f"{COALESCED}_prefix_{prefix}"] = quotient + (index < remainder)
    return {
        "schema_version": 1,
        "generation": [
            {
                "profile_label": "timing_context",
                "optimized_dispatch_capture_hits": {
                    SPLIT: timing_split,
                    f"{SPLIT}_prefix_192": timing_split,
                },
            },
            {
                "profile_label": "main_generation",
                "optimized_dispatch_capture_hits": hits,
            },
        ],
    }


def test_candidate_requires_all_split_dispatches_and_short_fallback() -> None:
    report = validate_profile(_profile(coalesced=11, split=11, generic=13), role="candidate")
    assert report["pass"]
    assert report["main_counts"][COALESCED] == 11

    with pytest.raises(CoalescedProfileError, match="every live eligible"):
        validate_profile(_profile(coalesced=11, split=12, generic=14), role="candidate")
    with pytest.raises(CoalescedProfileError, match="short fallback"):
        validate_profile(
            _profile(coalesced=11, split=11, generic=11), role="candidate"
        )


def test_baseline_rejects_coalesced_and_candidate_requires_it() -> None:
    assert validate_profile(_profile(coalesced=0), role="baseline")["pass"]
    with pytest.raises(CoalescedProfileError, match="unexpectedly"):
        validate_profile(_profile(coalesced=11, split=11, generic=13), role="baseline")
    with pytest.raises(CoalescedProfileError, match="did not execute"):
        validate_profile(_profile(coalesced=0), role="candidate")


def test_timing_context_must_remain_unmodified() -> None:
    payload = _profile(coalesced=11, split=11, generic=13)
    payload["generation"][0]["optimized_dispatch_capture_hits"] = {COALESCED: 1}
    with pytest.raises(CoalescedProfileError, match="timing generation"):
        validate_profile(payload, role="candidate")


def test_accepted_timing_split_counts_do_not_pollute_main_accounting() -> None:
    report = validate_profile(
        _profile(coalesced=11, split=11, generic=13, timing_split=2),
        role="candidate",
    )

    assert report["pass"]
    assert report["accepted_split_prefix_counts"] == {
        prefix: 1 for prefix in range(192, 833, 64)
    }
