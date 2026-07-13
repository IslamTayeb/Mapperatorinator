from __future__ import annotations

import copy

import pytest

from utils.summarize_fp16_native_prefix_component import (
    ALL_BUCKETS,
    CHECKS,
    DRIFT_FIELDS,
    SENTINEL_BUCKETS,
    summarize,
)


def _entry(
    *,
    replay_ms: float,
    capture_s: float = 0.0,
    prefix_ms: float = 0.1,
) -> dict:
    return {
        "full_model_replay_ms_per_call": replay_ms,
        "capture_setup_seconds": capture_s,
        "prefix_replay_ms_per_layer": prefix_ms,
        "checks": {check: True for check in CHECKS},
        "drift": {field: 0.0 for field in DRIFT_FIELDS},
    }


def _payload(
    buckets,
    *,
    baseline_ms: float = 1.0,
    framework_ms: float = 0.6,
    native_ms: float = 0.55,
    baseline_capture_s: float = 0.0,
    fp32_shared_capture_s: float = 0.0,
    framework_capture_s: float = 0.01,
    native_capture_s: float = 0.01,
    fp32_shared_prefix_ms: float = 0.12,
    framework_prefix_ms: float = 0.09,
    native_prefix_ms: float = 0.08,
) -> dict:
    return {
        "schema_version": 1,
        "variants": {
            "fp32_accepted": {
                "buckets": {
                    str(bucket): _entry(
                        replay_ms=baseline_ms,
                        capture_s=baseline_capture_s,
                        prefix_ms=0.12,
                    )
                    for bucket in buckets
                }
            },
            "fp32_shared_specialized": {
                "buckets": {
                    str(bucket): _entry(
                        replay_ms=baseline_ms,
                        capture_s=fp32_shared_capture_s,
                        prefix_ms=fp32_shared_prefix_ms,
                    )
                    for bucket in buckets
                }
            },
            "fp16_framework_self_bmm_cross": {
                "buckets": {
                    str(bucket): _entry(
                        replay_ms=framework_ms,
                        capture_s=framework_capture_s,
                        prefix_ms=framework_prefix_ms,
                    )
                    for bucket in buckets
                }
            },
            "fp16_native_self_bmm_cross": {
                "buckets": {
                    str(bucket): _entry(
                        replay_ms=native_ms,
                        capture_s=native_capture_s,
                        prefix_ms=native_prefix_ms,
                    )
                    for bucket in buckets
                }
            },
        },
    }


def test_all_bucket_projection_promotes_both_passing_candidates():
    report = summarize(_payload(ALL_BUCKETS))

    assert report["evidence"]["mode"] == "all_production_buckets"
    assert report["evidence"]["measured_replays"] == 7_597
    assert report["evidence"]["coverage_fraction"] == 1.0
    assert report["fp32_parity"]["pass"]
    assert report["native_prefix_comparison"]["native_advantage_pct"] == pytest.approx(
        (0.09 - 0.08) / 0.09 * 100
    )
    assert report["projections"]["fp16_framework_self_bmm_cross"]["projected_main_seconds"] == pytest.approx(
        28.243 + 7_597 * (0.6 - 1.0) / 1_000 + 0.12
    )
    assert report["candidate_decisions"]["fp16_framework_self_bmm_cross"]["promotion_pass"]
    assert report["candidate_decisions"]["fp16_native_self_bmm_cross"]["promotion_pass"]
    assert report["promotion_pass"]
    assert report["disposition"] == "PROMOTE_TO_FIXED_WORK_LOOP"


def test_sentinel_evidence_sizes_but_never_promotes():
    report = summarize(
        _payload(
            SENTINEL_BUCKETS,
            framework_ms=0.2,
            native_ms=0.15,
            framework_prefix_ms=0.04,
            native_prefix_ms=0.03,
        )
    )

    assert report["evidence"]["mode"] == "sentinel_only"
    assert report["evidence"]["measured_replays"] == 4_147
    assert report["evidence"]["coverage_fraction"] == pytest.approx(4_147 / 7_597)
    assert not report["evidence"]["promotion_coverage_pass"]
    assert report["sizing_pass"]
    assert not report["promotion_pass"]
    assert not report["candidate_decisions"]["fp16_framework_self_bmm_cross"]["promotion_pass"]
    assert report["disposition"] == "COLLECT_ALL_PRODUCTION_BUCKETS"


def test_arbitrary_partial_bucket_coverage_fails_loudly():
    payload = _payload(SENTINEL_BUCKETS | {704})

    with pytest.raises(ValueError, match="exactly sentinel buckets"):
        summarize(payload)


def test_variant_bucket_mismatch_fails_loudly():
    payload = _payload(SENTINEL_BUCKETS)
    del payload["variants"]["fp16_native_self_bmm_cross"]["buckets"]["640"]

    with pytest.raises(ValueError, match="identical buckets"):
        summarize(payload)


def test_capture_setup_regression_is_charged_separately_and_can_stop():
    report = summarize(
        _payload(
            ALL_BUCKETS,
            framework_ms=0.6,
            native_ms=0.55,
            framework_capture_s=0.15,
            native_capture_s=0.3,
        )
    )

    assert report["projections"]["fp16_framework_self_bmm_cross"]["capture_setup_delta_seconds"] == pytest.approx(
        1.8
    )
    assert report["projections"]["fp16_framework_self_bmm_cross"]["target_pass"] is False
    assert report["projections"]["fp16_native_self_bmm_cross"]["capture_setup_delta_seconds"] == pytest.approx(
        3.6
    )
    assert report["sizing_pass"] is False
    assert report["disposition"] == "STOP_COMPONENT_SCOUT"


def test_native_below_five_percent_is_rejected_without_blocking_framework():
    report = summarize(
        _payload(
            ALL_BUCKETS,
            native_prefix_ms=0.0864,
        )
    )

    assert report["native_prefix_comparison"]["native_advantage_pct"] == pytest.approx(
        4.0
    )
    assert report["native_prefix_comparison"]["retained"] is False
    assert report["candidate_decisions"]["fp16_native_self_bmm_cross"]["promotion_pass"] is False
    assert report["candidate_decisions"]["fp16_framework_self_bmm_cross"]["promotion_pass"] is True
    assert report["promotion_pass"]


def test_framework_correctness_failure_blocks_it_and_native_comparison():
    payload = _payload(ALL_BUCKETS)
    payload["variants"]["fp16_framework_self_bmm_cross"]["buckets"]["576"]["checks"][
        "future_slots_untouched"
    ] = False
    report = summarize(payload)

    assert report["correctness"]["variants"]["fp16_framework_self_bmm_cross"]["failures"] == {
        "576": ["future_slots_untouched"]
    }
    assert report["candidate_decisions"]["fp16_framework_self_bmm_cross"]["promotion_pass"] is False
    assert report["native_prefix_comparison"]["correctness_pair_pass"] is False
    assert report["candidate_decisions"]["fp16_native_self_bmm_cross"]["promotion_pass"] is False


def test_fp32_shared_regression_above_one_percent_stops_before_fp16():
    report = summarize(_payload(ALL_BUCKETS, baseline_ms=1.0))
    assert report["fp32_parity"]["pass"]

    payload = _payload(ALL_BUCKETS, baseline_ms=1.0)
    for entry in payload["variants"]["fp32_shared_specialized"]["buckets"].values():
        entry["full_model_replay_ms_per_call"] = 1.011
    report = summarize(payload)

    assert report["fp32_parity"]["replay_regression_pct"] == pytest.approx(1.1)
    assert not report["fp32_parity"]["pass"]
    assert report["disposition"] == "STOP_FP32_PARITY"


def test_baseline_correctness_failure_invalidates_every_candidate():
    payload = _payload(ALL_BUCKETS)
    payload["variants"]["fp32_accepted"]["buckets"]["128"]["checks"][
        "memory_stable"
    ] = False
    report = summarize(payload)

    assert report["correctness"]["baseline_pass"] is False
    assert report["sizing_pass"] is False
    assert report["promotion_pass"] is False


def test_fp32_shared_drift_blocks_exact_parity():
    payload = _payload(ALL_BUCKETS)
    payload["variants"]["fp32_shared_specialized"]["buckets"]["640"]["drift"][
        "cache_key_slot_max_abs"
    ] = 0.00101

    report = summarize(payload)

    assert not report["fp32_parity"]["exact_drift_pass"]
    assert not report["fp32_parity"]["pass"]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("full_model_replay_ms_per_call", 0.0, "must be positive"),
        ("capture_setup_seconds", -0.1, "must be non-negative"),
        ("prefix_replay_ms_per_layer", float("nan"), "must be finite"),
    ),
)
def test_invalid_timing_values_fail_loudly(field, value, match):
    payload = _payload(SENTINEL_BUCKETS)
    payload["variants"]["fp16_framework_self_bmm_cross"]["buckets"]["128"][field] = value

    with pytest.raises(ValueError, match=match):
        summarize(payload)


def test_missing_or_non_boolean_hard_checks_fail_loudly():
    missing = _payload(SENTINEL_BUCKETS)
    del missing["variants"]["fp16_framework_self_bmm_cross"]["buckets"]["128"]["checks"][
        "finite_outputs"
    ]
    with pytest.raises(ValueError, match="missing required checks"):
        summarize(missing)

    non_boolean = copy.deepcopy(missing)
    non_boolean["variants"]["fp16_framework_self_bmm_cross"]["buckets"]["128"]["checks"][
        "finite_outputs"
    ] = 1
    with pytest.raises(ValueError, match="JSON booleans"):
        summarize(non_boolean)
