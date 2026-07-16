from __future__ import annotations

from pathlib import Path

from utils.profile_cross_residual_epilogue_component import (
    BASE_TIP_CHOICE,
    FIXED_MAIN_DECODE_REPLAYS,
    SAVING_TARGET_SECONDS,
    summarize_component,
)


def _entry(*, baseline: float, candidate: float, unfused: float, count: int, drift: float = 0.0):
    return {
        "decode_replays": count,
        "baseline_ms_per_call": baseline,
        "candidate_ms_per_call": candidate,
        "unfused_residual_ms_per_call": unfused,
        "baseline_memory_stable": True,
        "candidate_memory_stable": True,
        "unfused_memory_stable": True,
        "drift": {
            "layout_fused_vs_baseline_max_abs": drift,
            "unfused_vs_baseline_max_abs": drift,
        },
    }


def test_component_wrapper_fits_one_backfill_window() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts/dcc/profile_cross_residual_epilogue_component.sbatch"
    ).read_text(encoding="utf-8")
    assert "--time=00:30:00" in source
    assert "exclude=dcc-core-gpu-ferc-s-h36-9" in source
    assert "cross-residual-epilogue" in source
    assert "STOP_CROSS_RESIDUAL_EPILOGUE_COMPONENT" not in source


def test_summarize_component_promotes_only_above_gate() -> None:
    # Uniform 0.01ms/call save * 8207 * 12 / 1000 = 0.98484s → pass
    buckets = {
        "128": _entry(baseline=0.05, candidate=0.04, unfused=0.06, count=FIXED_MAIN_DECODE_REPLAYS)
    }
    summary = summarize_component(buckets, total_replays=FIXED_MAIN_DECODE_REPLAYS)
    assert summary["sizing_pass"] is True
    assert summary["promotion_pass"] is True
    assert summary["projected_main_saving_seconds"] > SAVING_TARGET_SECONDS

    # Tiny save below 0.08s gate
    buckets = {
        "128": _entry(
            baseline=0.050, candidate=0.0495, unfused=0.051, count=FIXED_MAIN_DECODE_REPLAYS
        )
    }
    summary = summarize_component(buckets, total_replays=FIXED_MAIN_DECODE_REPLAYS)
    assert summary["sizing_pass"] is False
    assert summary["promotion_pass"] is False
    assert summary["projected_main_saving_seconds"] < SAVING_TARGET_SECONDS


def test_base_tip_keeps_compiled_cross_out_of_scope() -> None:
    assert "0dbab9e5" in BASE_TIP_CHOICE["selected"]
    assert "without compiled-cross" in BASE_TIP_CHOICE["reason"]
