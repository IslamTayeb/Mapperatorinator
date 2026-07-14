from __future__ import annotations

import copy

import pytest

from utils.summarize_selected_timing_precision_matrix import ARMS, METRICS, summarize


def _analysis(improvement: float) -> dict:
    metrics = {}
    for index, metric in enumerate(METRICS, start=1):
        baseline = 10.0 + index
        candidate = baseline - improvement
        if metric.endswith("_tps"):
            candidate = baseline + improvement
        metrics[metric] = {
            "baseline_median": baseline,
            "candidate_median": candidate,
            "improvement": improvement,
            "reciprocal_improvement_median": improvement,
            "reciprocal_improvement_range": 0.1,
        }
    divergence = {
        label: {
            "token_stream_equal": improvement == 0.0,
            "stopping_equal": True,
            "aligned_mismatch_fraction": improvement / 10.0,
        }
        for label in ("timing_context", "main_generation")
    }
    return {
        "metrics": metrics,
        "parity": {
            "claim": "relaxed-nonexact",
            "candidate_repeat_stable": True,
            "token_and_stopping_divergence": divergence,
            "output_divergence": {"final_map_equal": improvement == 0.0},
        },
    }


def test_summarizer_ranks_product_wall_and_preserves_fixed_work() -> None:
    payloads = {
        "exact_fp32_native_self": _analysis(0.1),
        "full_fp16": _analysis(2.0),
        "fp16_weights_fp32_state": _analysis(1.0),
    }

    report = summarize(payloads)

    assert report["measurement"]["fixed_main_tokens"] == 8_294
    assert report["measurement"]["fixed_timing_tokens"] == 821
    assert report["rankings"]["complete_request_wall"] == [
        "full_fp16",
        "fp16_weights_fp32_state",
        "exact_fp32_native_self",
    ]


def test_summarizer_rejects_different_control_runs() -> None:
    payloads = {arm: _analysis(1.0) for arm in ARMS}
    payloads["full_fp16"] = copy.deepcopy(payloads["full_fp16"])
    payloads["full_fp16"]["metrics"]["main_model_seconds"]["baseline_median"] += 1

    with pytest.raises(ValueError, match="identical control"):
        summarize(payloads)
