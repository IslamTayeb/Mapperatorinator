from __future__ import annotations

import copy

import pytest

from utils.profile_int8_projection_component import (
    AUDITED_REALISTIC_CEILING_SECONDS,
    CURRENT_BASELINE,
    FIXED_MAIN_SAVING_GATE_SECONDS,
    OUTPUTS_PER_BLOCK,
    REGIONS,
    summarize_component,
)


def _buckets(*, baseline_ms: float = 0.3, candidate_ms: float = 0.1):
    candidates = {
        str(opb): {
            "ms_per_call": candidate_ms + opb / 100_000,
            "finite": True,
            "repeat_deterministic": True,
            "memory_stable": True,
        }
        for opb in OUTPUTS_PER_BLOCK
    }
    return {
        "128": {
            "decode_replays": 100,
            "regions": {
                region: {
                    "baseline_ms_per_call": baseline_ms,
                    "baseline_finite": True,
                    "baseline_repeat_deterministic": True,
                    "baseline_memory_stable": True,
                    "candidates": copy.deepcopy(candidates),
                }
                for region in REGIONS
            },
        }
    }


def test_summary_uses_incremental_current_baselines_and_excludes_mlp() -> None:
    report = summarize_component(_buckets(), total_replays=200)

    assert set(report["regions"]) == set(REGIONS)
    assert "mlp" not in report["regions"]
    assert {
        region: entry["current_baseline"]
        for region, entry in report["regions"].items()
    } == CURRENT_BASELINE
    assert report["coverage_fraction"] == pytest.approx(0.5)
    assert report["selective_fixed_work_main_saving_seconds"] > 1.9
    assert report["saving_gate_seconds"] == FIXED_MAIN_SAVING_GATE_SECONDS
    assert report["sizing_pass"]
    assert report["promotion_pass"]


def test_preimplementation_realistic_ceiling_clears_half_second_gate() -> None:
    assert AUDITED_REALISTIC_CEILING_SECONDS == pytest.approx(0.532, abs=0.002)
    assert AUDITED_REALISTIC_CEILING_SECONDS >= FIXED_MAIN_SAVING_GATE_SECONDS


def test_summary_selectively_retains_only_positive_incremental_regions() -> None:
    buckets = _buckets(baseline_ms=0.1, candidate_ms=0.08)
    for candidate in buckets["128"]["regions"]["final_norm_logits"]["candidates"].values():
        candidate["ms_per_call"] = 0.12

    report = summarize_component(buckets, total_replays=100)

    assert not report["regions"]["final_norm_logits"]["retained_by_selective_policy"]
    assert report["regions"]["final_norm_logits"]["fixed_work_saving_seconds"] < 0
    assert report["selective_fixed_work_main_saving_seconds"] > 0


def test_summary_fails_loudly_when_every_configuration_is_invalid() -> None:
    buckets = _buckets()
    for candidate in buckets["128"]["regions"]["cross_norm_q"]["candidates"].values():
        candidate["finite"] = False

    with pytest.raises(RuntimeError, match="no valid INT8 configuration for cross_norm_q"):
        summarize_component(buckets, total_replays=100)


def test_summary_rejects_invalid_replay_counts() -> None:
    with pytest.raises(ValueError, match="bucket evidence"):
        summarize_component({}, total_replays=1)
    with pytest.raises(ValueError, match="replay counts"):
        summarize_component(_buckets(), total_replays=99)
