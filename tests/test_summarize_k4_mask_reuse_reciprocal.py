import pytest

from utils.summarize_k4_mask_reuse_reciprocal import decide, text_report


def _metric(baseline: float, candidate: float, savings=(0.1, 0.1)):
    return {
        "baseline_median": baseline,
        "candidate_median": candidate,
        "candidate_minus_baseline_fraction": (candidate - baseline) / baseline,
        "reciprocal_improvement_median": sum(savings) / 2,
        "reciprocal_pairs": [
            {"baseline": baseline, "improvement": savings[0]},
            {"baseline": baseline, "improvement": savings[1]},
        ],
    }


def _analysis(
    *,
    complete=(0.1, 0.1),
    main=(0.1, 0.1),
    model_delta=0.0,
    timing_stage_delta=0.0,
):
    return {
        "mode": "relaxed",
        "parity": {
            "claim": "relaxed-nonexact",
            "cross_candidate_exact": True,
            "baseline_repeat_stable": True,
            "candidate_repeat_stable": True,
            "required_exact_labels_pass": True,
            "dispatch_cache_topology": {"pass": True},
        },
        "metrics": {
            "main_outer_stage_wall_seconds": _metric(24.0, 23.9, main),
            "complete_request_wall_seconds": _metric(32.0, 31.9, complete),
            "main_model_seconds": _metric(22.0, 22.0 * (1 + model_delta)),
            "timing_outer_stage_wall_seconds": _metric(
                8.0,
                8.0 * (1 + timing_stage_delta),
            ),
        },
    }


def test_incrementally_exact_pairwise_wall_win_promotes():
    report = decide(_analysis())
    assert report["promotion_pass"] is True
    assert report["result_class"] == "promotion"
    assert (
        report["incremental_exactness_scope"]
        == "mask-reuse-vs-relaxed-k4-split-weight-shared-rope-control"
    )
    assert "promotion_pass=true" in text_report(report)


def test_one_reciprocal_order_regression_blocks_promotion():
    report = decide(_analysis(complete=(-0.01, 0.2)))
    assert report["promotion_pass"] is False
    assert "regressed in a reciprocal order" in " ".join(report["failures"])


def test_small_wall_signal_and_model_regression_block_promotion():
    report = decide(
        _analysis(complete=(0.01, 0.01), main=(0.02, 0.02), model_delta=0.011)
    )
    assert report["result_class"] == "valid_negative"
    assert len(report["failures"]) == 3


def test_timing_stage_regression_blocks_promotion():
    report = decide(_analysis(timing_stage_delta=0.011))
    assert report["promotion_pass"] is False
    assert report["timing_stage"]["nonregression_pass"] is False
    assert "median timing outer stage regressed" in " ".join(report["failures"])


def test_one_reciprocal_timing_stage_regression_blocks_promotion():
    analysis = _analysis()
    analysis["metrics"]["timing_outer_stage_wall_seconds"] = _metric(
        8.0,
        8.0,
        savings=(-0.081, 0.1),
    )
    report = decide(analysis)
    assert report["promotion_pass"] is False
    assert report["timing_stage"]["pairwise_nonregression_pass"] is False
    assert "reciprocal order" in " ".join(report["failures"])


def test_global_exact_mode_is_rejected_for_relaxed_k4_mixed_control():
    analysis = _analysis()
    analysis["mode"] = "exact-fp32"
    with pytest.raises(ValueError, match="incremental exactness"):
        decide(analysis)


@pytest.mark.parametrize("value", (-1.0, float("nan"), True))
def test_invalid_threshold_fails_loudly(value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        decide(_analysis(), minimum_wall_saving_seconds=value)
