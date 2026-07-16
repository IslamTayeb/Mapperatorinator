from __future__ import annotations

from pathlib import Path

import pytest

from utils.profile_cross_kv_layout_component import (
    BASE_TIP,
    CANDIDATES,
    EXACT_MAX_ABS_DRIFT,
    REGION_RELATIVE_GATE,
    SAVING_TARGET_SECONDS,
    summarize_component,
    summarize_variant,
)


def test_component_wrapper_fits_short_gpu_window() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts/dcc/profile_cross_kv_layout_component.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --gres=gpu:2080:1" in source
    assert "#SBATCH --time=00:20:00" in source
    assert "profile_cross_kv_layout_component.py" in source
    assert "--exclude=" in source
    assert "TMPDIR" in source


def test_base_tip_is_compiled_cross_selected_stack() -> None:
    assert "compiled-cross" in BASE_TIP
    assert "0dbab9e5" in BASE_TIP
    assert SAVING_TARGET_SECONDS == 0.2
    assert REGION_RELATIVE_GATE == 0.05
    assert EXACT_MAX_ABS_DRIFT == 1e-6
    assert "pretransposed_k_bmm" in CANDIDATES
    assert "fused_online_fp16_kv" in CANDIDATES


def test_summary_gates_region_or_e2e_and_correctness() -> None:
    # 12 * 8532 * 0.02 ms = 2.04768 s accepted region.
    # 5% region → candidate ≤ 0.019 ms; 0.2s E2E → delta ≥ ~0.00195 ms.
    strong = summarize_variant(
        name="fused_online_fp16_kv",
        accepted_ms=0.02,
        candidate_ms=0.018,
        drift=5e-4,
        exactness_class="relaxed-documented-drift",
        memory_stable=True,
        finite=True,
    )
    assert strong["region_pass"]
    assert strong["component_pass"]
    assert not strong["promotion_pass"]

    exact_ulp = summarize_variant(
        name="pretransposed_k_bmm",
        accepted_ms=0.02,
        candidate_ms=0.018,
        drift=6e-8,
        exactness_class="exact",
        memory_stable=True,
        finite=True,
    )
    assert exact_ulp["correctness_pass"]

    e2e_only = summarize_variant(
        name="pretransposed_k_bmm",
        accepted_ms=0.02,
        candidate_ms=0.0192,  # 4% region miss, but ~0.082s < 0.2s also miss
        drift=0.0,
        exactness_class="exact",
        memory_stable=True,
        finite=True,
    )
    assert not e2e_only["region_pass"]
    assert not e2e_only["e2e_pass"]
    assert not e2e_only["component_pass"]

    e2e_win = summarize_variant(
        name="pretransposed_k_bmm",
        accepted_ms=0.05,
        candidate_ms=0.048,  # 4% relative miss
        drift=0.0,
        exactness_class="exact",
        memory_stable=True,
        finite=True,
    )
    # 12 * 8532 * 0.002 ms = 0.204768 s ≥ 0.2
    assert not e2e_win["region_pass"]
    assert e2e_win["e2e_pass"]
    assert e2e_win["component_pass"]

    drifted = summarize_variant(
        name="fused_online_fp32",
        accepted_ms=0.02,
        candidate_ms=0.01,
        drift=1.1e-3,
        exactness_class="relaxed-documented-drift",
        memory_stable=True,
        finite=True,
    )
    assert not drifted["correctness_pass"]
    assert not drifted["component_pass"]

    summary = summarize_component(
        {
            "weak": e2e_only,
            "strong": strong,
        }
    )
    assert summary["any_component_pass"]
    assert summary["best_candidate"] == "fused_online_fp16_kv"
    assert summary["stop_condition"] is None

    fail_summary = summarize_component({"weak": e2e_only})
    assert not fail_summary["any_component_pass"]
    assert fail_summary["stop_condition"] == "STOP_CROSS_KV_LAYOUT_COMPONENT"


def test_scout_module_stays_import_cold_without_cuda_extension() -> None:
    pytest.importorskip("torch")
    import osuT5.osuT5.inference.optimized.scout.cross_kv_layout as scout

    assert scout._CROSS_KV_LAYOUT_EXTENSION is None
    meta = scout.scout_metadata()
    assert "not IFB" in meta["borrow"]
    assert meta["production_wiring"] is False
