from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "dcc"
    / "verify_batched_encoder_store_full_song_reciprocal.sbatch"
)


def test_wrapper_pins_clean_pushed_strict_fp32_2080ti_worktree():
    source = SCRIPT.read_text(encoding="utf-8")
    for fragment in (
        "#SBATCH --partition=gpu-common",
        "#SBATCH --gres=gpu:2080:1",
        'BASELINE_RUN_ROOT=${MAPPERATORINATOR_BASELINE_RUN_ROOT:?',
        'git -C "$REPO" status --porcelain',
        'git -C "$REPO" rev-parse "$REMOTE_REF"',
        'export NVIDIA_TF32_OVERRIDE=0',
        'EXPECTED_GPU=${MAPPERATORINATOR_EXPECTED_GPU:-2080 Ti}',
        'precision=fp32',
        'profile_pass_kind="$pass_kind"',
    ):
        assert fragment in source


def test_wrapper_measures_all_batches_both_passes_and_two_encoder_stages():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "BATCH_SIZES=(1 2 4 8 16)" in source
    assert source.count("profile_batched_encoder_precompute_ceiling.py") == 2
    assert source.count("--batch-sizes 1,2,4,8,16") == 2
    assert "auto_select_gamemode_model=false" in source
    assert 'run_candidate "$batch_size" untraced_control control' in source
    assert 'run_candidate "$batch_size" exactness_audit exactness-audit' in source
    assert '--main-audit "$RUN_ROOT/component/main/encoder-precompute.json"' in source
    assert '--timing-audit "$RUN_ROOT/component/timing/encoder-precompute.json"' in source
    assert '--json-output "$RUN_ROOT/analysis.json"' in source
    assert '--text-output "$RUN_ROOT/analysis.txt"' in source
    assert ".md" not in source
    assert ".html" not in source
