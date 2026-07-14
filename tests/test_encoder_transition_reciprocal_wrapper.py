from pathlib import Path


WRAPPER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "dcc"
    / "verify_encoder_transition_full_song_reciprocal.sbatch"
)


def test_transition_wrapper_is_untraced_exact_and_current_main_pinned() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert "EXPECTED_BASELINE_COMMIT=367a9563a4d03dc15215c740fbd56cd9290d6d8b" in source
    assert "profile_pass_kind=untraced_control" in source
    assert "profile_detail_ranges=false" in source
    assert "profile_cuda_capture=false" in source
    assert "--mode exact-fp32" in source
    assert "--allow-dispatch-delta" not in source
    assert "dispatch.changed_paths=0" in source


def test_transition_wrapper_uses_stage_seed_and_candidate_skip_gate() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert "PYTHONHASHSEED=0" in source
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in source
    assert "utils/run_fixed_seed_inference.py" in source
    assert 'f"reciprocal_rng_{device}_sha256_{label}"' in source
    assert "stage_seed_policy" in source
    assert "profile_label_seed_fingerprints" in source
    assert "utils/validate_reciprocal_stage_seed_profiles.py" in source
    assert source.index("utils/validate_reciprocal_stage_seed_profiles.py") < source.index(
        "utils/analyze_reciprocal_full_song_candidate.py"
    )
    assert "utils/validate_encoder_stabilization_profile.py" in source
    assert "candidate_first.encoder-stabilization.json" in source
    assert "candidate_second.encoder-stabilization.json" in source
