from __future__ import annotations

from pathlib import Path

from utils.profile_cross_residual_epilogue_component import (
    BASE_TIP,
    CANDIDATES,
    REGION_RELATIVE_GATE,
    SAVING_TARGET_SECONDS,
    summarize_component,
    summarize_variant,
)


def test_component_wrapper_fits_short_gpu_window() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts/dcc/profile_cross_residual_epilogue_component.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --gres=gpu:2080:1" in source
    assert "#SBATCH --time=00:20:00" in source
    assert "profile_cross_residual_epilogue_component.py" in source
    assert "--exclude=" in source
    assert "TMPDIR" in source
    assert "cross-residual-epilogue" in source


def test_base_tip_and_gates() -> None:
    assert "0dbab9e5" in BASE_TIP
    assert SAVING_TARGET_SECONDS == 0.08
    assert REGION_RELATIVE_GATE == 0.05
    assert "heads_view_fused" in CANDIDATES
    assert "split_linear_then_add" in CANDIDATES


def test_summary_requires_positive_saving_and_correctness() -> None:
    # 12 * 8532 * ms_delta / 1000 = projected seconds.
    # 5% of 0.004 ms → need ≥0.0002 ms; 0.08s E2E → ≥~0.000782 ms.
    strong = summarize_variant(
        name="heads_view_fused",
        accepted_ms=0.004,
        candidate_ms=0.003,  # 0.001 ms → ~0.102s, 25% region
        drift=0.0,
        exactness_class="exact",
        memory_stable=True,
        finite=True,
    )
    assert strong["region_pass"]
    assert strong["e2e_pass"]
    assert strong["component_pass"]
    assert not strong["promotion_pass"]

    flat = summarize_variant(
        name="heads_view_fused",
        accepted_ms=0.004,
        candidate_ms=0.00399,
        drift=0.0,
        exactness_class="exact",
        memory_stable=True,
        finite=True,
    )
    assert not flat["region_pass"]
    assert not flat["e2e_pass"]
    assert not flat["component_pass"]

    slower = summarize_variant(
        name="split_linear_then_add",
        accepted_ms=0.004,
        candidate_ms=0.005,
        drift=0.0,
        exactness_class="exact",
        memory_stable=True,
        finite=True,
    )
    assert slower["projected_main_saving_seconds"] < 0
    assert not slower["sizing_pass"]
    assert not slower["component_pass"]

    summary = summarize_component(
        {
            "heads_view_fused": flat,
            "split_linear_then_add": slower,
        }
    )
    assert summary["stop_condition"] == "STOP_CROSS_RESIDUAL_EPILOGUE_COMPONENT"
    assert not summary["any_component_pass"]
