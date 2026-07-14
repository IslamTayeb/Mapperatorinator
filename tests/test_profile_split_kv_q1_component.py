from __future__ import annotations

import pytest
import torch

from utils.profile_split_kv_q1_component import (
    MAX_ABS_DRIFT,
    split_counts_for_prefix,
    summarize_split_kv,
    validate_split_mask,
)


def _candidate(ms: float, *, drift: float = 0.0, passed: bool = True):
    return {
        "ms_per_call": ms,
        "memory_stable": True,
        "verifier": {"pass": passed},
        "drift": {
            "output_max_abs": drift,
            "cache_key_slot_max_abs": 0.0,
            "cache_value_slot_max_abs": 0.0,
        },
    }


def _bucket(accepted: float, count: int, candidates: dict):
    return {
        "accepted_ms_per_call": accepted,
        "accepted_memory_stable": True,
        "accepted_verifier": {"pass": True},
        "decode_replays": count,
        "candidates": candidates,
    }


def test_split_policy_keeps_128_unsplit_and_sweeps_larger_buckets() -> None:
    assert split_counts_for_prefix(128) == (1,)
    assert split_counts_for_prefix(576) == (2, 4, 8)
    assert split_counts_for_prefix(640) == (2, 4, 8)


def test_summary_selects_fastest_correct_split_and_weights_live_counts() -> None:
    report = summarize_split_kv(
        {
            "128": _bucket(0.10, 1000, {"split_1": _candidate(0.04)}),
            "640": _bucket(
                0.20,
                500,
                {
                    "split_2": _candidate(0.11),
                    "split_4": _candidate(0.08),
                    "split_8": _candidate(0.09),
                },
            ),
        },
        total_replays=3000,
    )

    assert report["correctness_pass"]
    assert report["selected"]["640"]["candidate"] == "split_4"
    assert report["measured_replays"] == 1500
    assert report["coverage_fraction"] == 0.5
    assert report["weighted_local_speedup"] == 2.5
    assert report["projected_main_saving_seconds"] == 1.44
    assert not report["sizing_pass"]


def test_summary_rejects_drift_and_falls_back_to_accepted_for_that_bucket() -> None:
    report = summarize_split_kv(
        {
            "128": _bucket(
                1.0,
                200,
                {"split_1": _candidate(0.1, drift=MAX_ABS_DRIFT + 1e-7)},
            )
        },
        total_replays=200,
    )

    assert not report["correctness_pass"]
    assert report["projected_main_saving_seconds"] == 0.0
    assert report["candidate_failures"]["128"] == {
        "split_1": [f"output_max_abs_above_{MAX_ABS_DRIFT}"]
    }


def test_summary_passes_both_declared_gates() -> None:
    report = summarize_split_kv(
        {"640": _bucket(1.0, 300, {"split_8": _candidate(0.25)})},
        total_replays=600,
    )

    assert report["weighted_local_speedup"] == 4.0
    assert report["projected_main_saving_seconds"] == 2.7
    assert report["promotion_pass"]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("accepted_memory_stable", False, "memory stable"),
        ("accepted_verifier", {"pass": False}, "cache verifier"),
    ],
)
def test_summary_rejects_an_invalid_accepted_control(field, value, error) -> None:
    bucket = _bucket(1.0, 300, {"split_8": _candidate(0.25)})
    bucket[field] = value

    with pytest.raises(ValueError, match=error):
        summarize_split_kv({"640": bucket}, total_replays=600)


def test_split_mask_rejects_an_all_masked_partition() -> None:
    mask = torch.zeros(8)
    mask[:2] = -torch.inf

    with pytest.raises(RuntimeError, match="all-masked partition"):
        validate_split_mask(mask, prefix=8, split_counts=(4,))

    validate_split_mask(torch.zeros(8), prefix=8, split_counts=(2, 4))
    validate_split_mask(None, prefix=8, split_counts=(2, 4))
