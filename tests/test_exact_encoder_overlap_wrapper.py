from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "dcc" / "verify_exact_encoder_overlap_full_song_reciprocal.sbatch"


def test_wrapper_is_pinned_reciprocal_and_never_enables_batched_encoder():
    source = WRAPPER.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:2080:1" in source
    assert "precision=fp32" in source
    assert "inference_engine=optimized" in source
    assert "auto_select_gamemode_model=true" in source
    assert "parallel=false" in source
    assert source.count("run_profile ") == 4
    assert "minimum-request-saving-seconds 1.0" in source
    assert "profile_batched_encoder_precompute_ceiling" not in source
    assert "ENCODER_BATCH_SIZE" not in source


def test_overlap_module_remains_cold_and_unselected():
    package = (
        ROOT
        / "osuT5"
        / "osuT5"
        / "inference"
        / "optimized"
        / "scout"
        / "__init__.py"
    ).read_text(encoding="utf-8")
    selector = (ROOT / "inference.py").read_text(encoding="utf-8")
    assert "encoder_overlap" not in package
    assert "encoder_overlap" not in selector
