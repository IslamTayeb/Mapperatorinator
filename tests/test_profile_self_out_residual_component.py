from __future__ import annotations

from pathlib import Path

from utils.profile_self_out_residual_component import (
    CANDIDATES,
    EXPECTED_HEAD_DIM,
    EXPECTED_HEADS,
    EXPECTED_HIDDEN,
    MAX_ABS_DRIFT,
    SIZING_FRACTION_OF_TIP,
    TIP_COMMIT,
    TIP_MAIN,
    saving_target_seconds,
    summarize_component,
    summarize_variant,
)


def test_sbatch_isolates_job_tmpdir_and_excludes_h36_9() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts/dcc/profile_self_out_residual_component.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --gres=gpu:2080:1" in source
    assert "profile_self_out_residual_component.py" in source
    assert "dcc-core-gpu-ferc-s-h36-9" in source
    assert "PRECISION" in source
    assert "TORCH_EXTENSIONS_DIR" in source
    assert "TMPDIR" in source
    assert "ALLOW_PARALLEL" in source


def test_tip_gates_use_five_percent_of_graduated_main() -> None:
    assert TIP_COMMIT == "55949274"
    assert TIP_MAIN["fp16"]["main_model_s"] == 21.330
    assert TIP_MAIN["fp16"]["tokens"] == 7809
    assert TIP_MAIN["fp32"]["main_model_s"] == 26.494
    assert TIP_MAIN["fp32"]["tokens"] == 8294
    assert SIZING_FRACTION_OF_TIP == 0.05
    assert abs(saving_target_seconds("fp16") - 1.0665) < 1e-9
    assert abs(saving_target_seconds("fp32") - 1.3247) < 1e-9
    assert MAX_ABS_DRIFT == 1e-3
    assert EXPECTED_HEADS == 12
    assert EXPECTED_HEAD_DIM == 64
    assert EXPECTED_HIDDEN == 768
    assert "native_fused" in CANDIDATES


def test_summary_requires_tip_fraction_and_correctness() -> None:
    # 12 layers * 7809 tokens * ms_delta / 1000 = projected seconds.
    # Need ≥1.0665 s → ms_delta ≥ 1.0665 * 1000 / (12 * 7809) ≈ 0.01138 ms.
    strong = summarize_variant(
        name="native_fused",
        accepted_ms=0.050,
        candidate_ms=0.030,  # 0.020 ms → ~1.87 s
        drift=0.0,
        memory_stable=True,
        finite=True,
        precision="fp16",
        decoder_layers=12,
    )
    assert strong["sizing_pass"]
    assert strong["component_pass"]
    assert strong["evidence_mode"] == "component_projection"
    assert not strong["promotion_pass"]

    weak = summarize_variant(
        name="native_fused",
        accepted_ms=0.050,
        candidate_ms=0.049,
        drift=0.0,
        memory_stable=True,
        finite=True,
        precision="fp16",
        decoder_layers=12,
    )
    assert not weak["sizing_pass"]
    assert not weak["component_pass"]

    drifted = summarize_variant(
        name="native_fused",
        accepted_ms=0.050,
        candidate_ms=0.030,
        drift=1.1e-3,
        memory_stable=True,
        finite=True,
        precision="fp32",
        decoder_layers=12,
    )
    assert not drifted["correctness_pass"]
    assert not drifted["component_pass"]

    summary = summarize_component(
        {"native_fused": weak},
        precision="fp16",
    )
    assert summary["stop_condition"] == "STOP_SELF_OUT_RESIDUAL_FUSION_COMPONENT"
    assert not summary["any_component_pass"]
    assert summary["evidence_mode"] == "component_projection"
