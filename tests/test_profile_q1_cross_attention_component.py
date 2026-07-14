from __future__ import annotations

import pytest
from pathlib import Path

from utils.profile_q1_cross_attention_component import (
    FIXED_WORK_500_TPS_GAP_SECONDS,
    _launch_metadata,
    summarize_component,
    summarize_projection_component,
)


def _variant(ms: float, *, drift: float = 0.0, fp16_kv: bool = False):
    return {
        "ms_per_decode_step": ms,
        "max_abs_drift": drift,
        "checks_pass": True,
        "kv_storage_dtype": "torch.float16" if fp16_kv else "torch.float32",
    }


def test_summary_weights_timing_and_main_and_exposes_500_gap() -> None:
    report = summarize_component(
        {
            "accepted_q1_bmm": _variant(0.30),
            "one_pass_fp32": _variant(0.05),
            "split8_fp32": _variant(0.10),
            "split8_fp16_kv": _variant(0.08, drift=0.02, fp16_kv=True),
        },
        timing_decode_replays=1000,
        main_decode_replays=8000,
    )

    candidate = report["candidates"]["split8_fp32"]
    assert report["accepted_timing_seconds"] == pytest.approx(0.3)
    assert report["accepted_main_seconds"] == pytest.approx(2.4)
    assert candidate["main_saving_seconds"] == pytest.approx(1.6)
    assert candidate["timing_saving_seconds"] == pytest.approx(0.2)
    assert candidate["local_speedup"] == pytest.approx(3.0)
    assert candidate["fraction_of_500_tps_gap_closed"] == pytest.approx(
        1.6 / FIXED_WORK_500_TPS_GAP_SECONDS
    )
    assert candidate["promotion_pass"]
    assert not report["candidates"]["one_pass_fp32"]["promotion_eligible"]
    assert not report["candidates"]["one_pass_fp32"]["promotion_pass"]
    assert not report["candidates"]["split8_fp16_kv"]["promotion_eligible"]


def test_summary_rejects_fp32_drift_and_invalid_counts() -> None:
    report = summarize_component(
        {
            "accepted_q1_bmm": _variant(0.3),
            "split8_fp32": _variant(0.1, drift=0.00101),
        },
        timing_decode_replays=1000,
        main_decode_replays=8000,
    )

    assert not report["candidates"]["split8_fp32"]["correctness_pass"]
    assert not report["candidates"]["split8_fp32"]["promotion_pass"]
    with pytest.raises(ValueError, match="positive"):
        summarize_component(
            {"accepted_q1_bmm": _variant(0.3)},
            timing_decode_replays=0,
            main_decode_replays=1,
        )


def test_launch_metadata_distinguishes_underfilled_control_from_split() -> None:
    one_pass = _launch_metadata("one_pass_fp32")
    split8 = _launch_metadata("split8_fp32")

    assert one_pass["underfill_control"]
    assert one_pass["partial_grid"] == [12, 1, 1]
    assert split8["underfill_control"] is False
    assert split8["partial_grid"] == [96, 1, 1]
    assert split8["merge_grid"] == [12, 1, 1]
    assert split8["workspace_bytes_per_layer"] > 0


def test_projection_summary_is_independent_documented_drift_evidence() -> None:
    report = summarize_projection_component(
        baseline_ms=0.5,
        candidate_ms=0.3,
        max_abs_drift=0.04,
        checks_pass=True,
        timing_decode_replays=1000,
        main_decode_replays=8000,
    )

    assert report["result_class"] == "documented_drift_fp16_weights_fp32_state"
    assert report["main_saving_seconds"] == pytest.approx(1.6)
    assert report["timing_saving_seconds"] == pytest.approx(0.2)
    assert report["component_retention_pass"]
    assert report["sizing_pass"]
    assert report["production_promotion_pass"] is False


def test_dcc_wrapper_is_exact_commit_sm75_and_json_text_only() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts/dcc/profile_q1_cross_attention_component.sbatch"
    ).read_text(encoding="utf-8")

    assert "MAPPERATORINATOR_COMMIT" in source
    assert "MAPPERATORINATOR_BRANCH" in source
    assert '"$REMOTE/$BRANCH"' in source
    assert "NVIDIA GeForce RTX 2080 Ti" in source
    assert '"7.5"' in source
    assert "--warmup 100" in source
    assert "--iters 1000" in source
    assert "component.json" in source
    assert "component.txt" in source
    assert ".md" not in source
    assert ".html" not in source
    assert ".png" not in source
