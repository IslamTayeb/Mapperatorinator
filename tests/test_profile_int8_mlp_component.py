from __future__ import annotations

import pytest

from utils.profile_int8_mlp_component import DECODER_LAYERS, summarize_component


def _entry(*, fp16: float, int8: float, fp32: float = 0.3, count: int = 100):
    return {
        "decode_replays": count,
        "ms_per_call": {
            "fp32": fp32,
            "fp16_weight": fp16,
            "int8_weight": int8,
        },
        "checks": {
            "fp32_finite": True,
            "fp16_finite": True,
            "int8_finite": True,
            "fp32_repeat_deterministic": True,
            "fp16_repeat_deterministic": True,
            "int8_repeat_deterministic": True,
            "fp32_memory_stable": True,
            "fp16_memory_stable": True,
            "int8_memory_stable": True,
            "output_shape_preserved": True,
        },
    }


def test_summary_weights_live_counts_and_requires_1p4x_local_speedup() -> None:
    report = summarize_component(
        {
            "128": _entry(fp16=0.28, int8=0.18, count=1000),
            "640": _entry(fp16=0.32, int8=0.20, count=250),
        },
        total_replays=2500,
    )

    expected_fp16 = DECODER_LAYERS * (1000 * 0.28 + 250 * 0.32) / 1000
    expected_int8 = DECODER_LAYERS * (1000 * 0.18 + 250 * 0.20) / 1000
    assert report["measured_weighted_mlp_seconds"]["fp16_weight"] == pytest.approx(
        expected_fp16
    )
    assert report["measured_weighted_mlp_seconds"]["int8_weight"] == pytest.approx(
        expected_int8
    )
    assert report["int8_vs_fp16_local_speedup"] == pytest.approx(
        expected_fp16 / expected_int8
    )
    assert report["conservative_measured_main_saving_seconds"] == pytest.approx(
        expected_fp16 - expected_int8
    )
    assert report["fixed_work_main_saving_seconds"] == pytest.approx(
        2 * (expected_fp16 - expected_int8)
    )
    assert report["coverage_fraction"] == 0.5
    assert report["sizing_pass"]
    assert report["promotion_pass"]


def test_summary_stops_below_speed_gate_or_on_invariant_failure() -> None:
    slow = summarize_component(
        {"128": _entry(fp16=0.28, int8=0.21)}, total_replays=100
    )
    assert not slow["sizing_pass"]
    assert not slow["promotion_pass"]

    bad_entry = _entry(fp16=0.28, int8=0.18)
    bad_entry["checks"]["int8_repeat_deterministic"] = False
    bad = summarize_component({"128": bad_entry}, total_replays=100)
    assert bad["sizing_pass"]
    assert not bad["invariants_pass"]
    assert bad["invariant_failures"] == {"128": ["int8_repeat_deterministic"]}
    assert not bad["promotion_pass"]


def test_summary_fails_loudly_on_invalid_counts_or_timings() -> None:
    entry = _entry(fp16=0.28, int8=0.18)
    entry["decode_replays"] = 0
    with pytest.raises(ValueError, match="must be positive"):
        summarize_component({"128": entry}, total_replays=100)

    entry = _entry(fp16=0.28, int8=0.18)
    entry["ms_per_call"]["int8_weight"] = 0.0
    with pytest.raises(ValueError, match="timing must be positive"):
        summarize_component({"128": entry}, total_replays=100)
