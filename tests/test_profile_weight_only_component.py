from __future__ import annotations

import math
from pathlib import Path

from utils.profile_weight_only_component import (
    BASELINE_MAIN_SECONDS,
    BASELINE_MAIN_TOKENS,
    DECODER_LAYERS,
    REGIONS,
    summarize_weight_only_component,
)


def _region(*, accepted: float, candidate: float, drift: float = 0.1):
    return {
        "accepted_ms_per_call": accepted,
        "candidate_ms_per_call": candidate,
        "saving_ms_per_call": accepted - candidate,
        "candidate_setup_seconds": 0.1,
        "candidate_peak_vram_bytes": 100,
        "candidate_memory_stable": True,
        "candidate_finite": True,
        "output_max_abs_drift": drift,
        "rounds": [],
    }


def _bucket(*, count: int, accepted: float = 0.5, candidate: float = 0.2):
    return {
        "decode_replays": count,
        "regions": {
            name: _region(accepted=0.1, candidate=0.05)
            for name in REGIONS
        },
        "prefix_total": {
            "accepted_ms_per_call": accepted,
            "candidate_ms_per_call": candidate,
            "saving_ms_per_call": accepted - candidate,
            "candidate_setup_seconds": 0.2,
            "candidate_peak_vram_bytes": 200,
            "candidate_memory_stable": True,
            "candidate_finite": True,
            "candidate_cache_behavior_pass": True,
            "output_max_abs_drift": 0.25,
            "cache_key_slot_max_abs_drift": 0.01,
            "cache_value_slot_max_abs_drift": 0.02,
            "candidate_checks": {"pass": True},
            "rounds": [],
        },
    }


def test_summary_uses_combined_prefix_not_sum_of_isolated_region_savings() -> None:
    buckets = {
        "128": _bucket(count=1_000),
        "640": _bucket(count=500),
    }

    report = summarize_weight_only_component(buckets, total_replays=3_000)

    expected = 1_500 * DECODER_LAYERS * (0.5 - 0.2) / 1_000
    expected += 1_500 * (0.1 - 0.05) / 1_000
    assert report["weighted_fixed_work_saving_seconds"] == expected
    assert report["coverage_fraction"] == 0.5
    assert report["sizing_pass"]
    assert report["correctness_pass"]
    assert report["maximum_observed_abs_drift"] == 0.25
    assert report["drift_is_documented_not_exactness"] is True
    assert report["projected_main_seconds"] == BASELINE_MAIN_SECONDS - expected
    assert report["projected_main_tps_at_8294_tokens"] == (
        BASELINE_MAIN_TOKENS / (BASELINE_MAIN_SECONDS - expected)
    )


def test_summary_does_not_treat_documented_numeric_drift_as_exactness() -> None:
    bucket = _bucket(count=100)
    bucket["regions"]["mlp"]["output_max_abs_drift"] = 4.25

    report = summarize_weight_only_component({"128": bucket}, total_replays=100)

    assert report["maximum_observed_abs_drift"] == 4.25
    assert report["correctness_pass"]


def test_summary_rejects_nonfinite_memory_and_cache_behavior_failures() -> None:
    bucket = _bucket(count=100)
    bucket["regions"]["mlp"]["candidate_finite"] = False
    bucket["regions"]["cross_norm_q"]["candidate_memory_stable"] = False
    bucket["prefix_total"]["candidate_cache_behavior_pass"] = False

    report = summarize_weight_only_component({"128": bucket}, total_replays=100)

    assert not report["correctness_pass"]
    assert report["correctness_failures"] == {
        "128": [
            "cross_norm_q:memory_unstable",
            "mlp:nonfinite",
            "prefix_total:cache_behavior_failed",
        ]
    }


def test_summary_fails_loudly_on_empty_or_invalid_counts() -> None:
    for buckets, total in (({}, 1), ({"128": _bucket(count=1)}, 0)):
        try:
            summarize_weight_only_component(buckets, total_replays=total)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid summary input did not fail")

    invalid = _bucket(count=0)
    try:
        summarize_weight_only_component({"128": invalid}, total_replays=1)
    except ValueError as error:
        assert "non-positive replay count" in str(error)
    else:
        raise AssertionError("zero replay count did not fail")


def test_projection_is_finite_for_a_passing_synthetic_case() -> None:
    report = summarize_weight_only_component(
        {"128": _bucket(count=1_000)}, total_replays=1_000
    )
    assert math.isfinite(report["projected_main_tps_at_8294_tokens"])


def test_dcc_wrapper_requires_clean_exact_branch_and_keeps_artifacts_outside_git() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/dcc/profile_weight_only_component.sbatch").read_text()

    assert "MAPPERATORINATOR_REPO:?" in source
    assert "MAPPERATORINATOR_COMMIT:?" in source
    assert "MAPPERATORINATOR_BRANCH:?" in source
    assert 'git -C "$REPO" status --porcelain' in source
    assert 'git -C "$REPO" rev-parse HEAD' in source
    assert 'git -C "$REPO" branch --show-current' in source
    assert "NVIDIA GeForce RTX 2080 Ti" in source
    assert 'RUN_ROOT="$WORK/runs/' in source
    assert "--bucket-mode sentinel" in source
    assert "--warmup 100" in source
    assert "--iters 1000" in source
    assert "sbatch" not in source
