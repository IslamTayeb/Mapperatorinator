import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/dcc/verify_stage_precision_hybrid_full_song_reciprocal.sbatch"
COMPOSED_WRAPPER = (
    ROOT
    / "scripts/dcc/verify_k4_split_weight_shared_rope_stage_precision_full_song_reciprocal.sbatch"
)


def test_stage_precision_wrapper_has_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    subprocess.run(["bash", "-n", str(COMPOSED_WRAPPER)], check=True)


def test_stage_precision_wrapper_is_reciprocal_and_stage_explicit():
    source = WRAPPER.read_text(encoding="utf-8")

    assert source.count("run_profile baseline_first control") == 1
    assert source.count("run_profile candidate_first candidate") == 1
    assert source.count("run_profile candidate_second candidate") == 1
    assert source.count("run_profile baseline_second control") == 1
    assert source.index("run_profile baseline_first control") < source.index(
        "run_profile candidate_first candidate"
    )
    assert source.index("run_profile candidate_second candidate") < source.index(
        "run_profile baseline_second control"
    )
    assert "utils/run_stage_aware_k4_mixed_control.py" in source
    assert "utils/run_k4_stage_precision_hybrid.py" in source
    assert "--timing-block-size 1" in source
    assert "utils/validate_stage_precision_hybrid_profile.py" in source
    assert "--allow-mixed-stage-precision" in source
    assert "metric.timing_model_seconds" in source
    assert "metric.main_model_seconds" in source
    assert "metric.complete_request_wall_seconds" in source
    assert "divergence.timing_context" in source
    assert "divergence.main_generation" in source
    assert "divergence.final_map_equal" in source
    assert "fixed_main_generation_model_seconds" in source
    assert "metric.peak_cuda_memory_allocated_mb" in source
    assert "metric.setup_plus_capture_seconds" in source


def test_composed_wrapper_changes_only_stage_precision_over_shared_best() -> None:
    source = COMPOSED_WRAPPER.read_text(encoding="utf-8")

    assert "utils/run_k4_shared_rope_stage_aware_control.py" in source
    assert "utils/run_k4_shared_rope_stage_precision_hybrid.py" in source
    assert "REQUIRE_SHARED_ROPE=true" in source
    assert "FIXED_TIMING_TOKENS=821" in source
    assert "FIXED_MAIN_TOKENS=8294" in source
    assert "verify_stage_precision_hybrid_full_song_reciprocal.sbatch" in source

    general = WRAPPER.read_text(encoding="utf-8")
    assert "utils/validate_shared_rope_stage_initialization.py" in general
    assert "shared-rope-validation.json" in general
