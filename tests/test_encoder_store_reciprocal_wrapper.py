from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "dcc"
    / "verify_batched_encoder_store_full_song_reciprocal.sbatch"
)


def test_wrapper_pins_clean_pushed_2080ti_worktree_and_full_song():
    source = SCRIPT.read_text(encoding="utf-8")
    for fragment in (
        "#SBATCH --partition=gpu-common",
        "#SBATCH --gres=gpu:2080:1",
        'PROFILE_CONFIG=${PROFILE_CONFIG:-profile_salvalai}',
        'git -C "$REPO" status --porcelain',
        'git -C "$REPO" rev-parse "$REMOTE_REF"',
        'NVIDIA GeForce RTX 2080 Ti',
        "preload_weight_only_extension",
        "preload_int8_mlp_extension",
        'profile_pass_kind=untraced_control',
        'run_profile baseline_first baseline',
        'run_profile candidate_second candidate',
        'run_profile candidate_first candidate',
        'run_profile baseline_second baseline',
    ):
        assert fragment in source


def test_wrapper_charges_two_model_audits_and_writes_json_text_only():
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count("profile_batched_encoder_precompute_ceiling.py") == 2
    assert "auto_select_gamemode_model=false" in source
    assert '--main-audit "$RUN_ROOT/drift-audit/main/encoder-precompute.json"' in source
    assert '--timing-audit "$RUN_ROOT/drift-audit/timing/encoder-precompute.json"' in source
    assert "--minimum-complete-request-saving-seconds 0.3" in source
    assert '--json-output "$RUN_ROOT/analysis.json"' in source
    assert '--text-output "$RUN_ROOT/analysis.txt"' in source
    assert ".md" not in source
    assert ".html" not in source
