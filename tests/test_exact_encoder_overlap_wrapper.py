from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "dcc" / "verify_exact_encoder_overlap_full_song_reciprocal.sbatch"


def test_wrapper_is_pinned_reciprocal_and_never_enables_batched_encoder():
    source = WRAPPER.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:2080:1" in source
    assert "#SBATCH --time=00:30:00" in source
    assert "precision=fp32" in source
    assert "inference_engine=optimized" in source
    assert "auto_select_gamemode_model=true" in source
    assert "parallel=false" in source
    assert source.count("run_profile ") == 4
    assert "minimum-request-saving-seconds 1.0" in source
    assert "run_exact_encoder_overlap_full_song.py" in source
    assert 'export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"' in source
    assert source.count("--initialization-path") == 1
    assert source.count("validate_weight_only_full_song_profile.py") == 1
    assert source.count("validate_k4_profile_contract.py") == 1
    assert source.count("validate_k1_remainder_profile.py") == 1
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


def test_overlap_keeps_current_combined_runtime_and_original_decoder_forward():
    runner = (ROOT / "utils" / "run_exact_encoder_overlap_full_song.py").read_text(
        encoding="utf-8"
    )
    overlap = (
        ROOT
        / "osuT5"
        / "osuT5"
        / "inference"
        / "optimized"
        / "scout"
        / "encoder_overlap.py"
    ).read_text(encoding="utf-8")
    assert "run_k4_shared_rope_approximate_weight_only" in runner
    assert "run_current_topology" in runner
    assert "graph_remainders=True" in runner
    assert "VarWhisperDecoderLayer" not in overlap
    assert ".forward =" not in overlap
