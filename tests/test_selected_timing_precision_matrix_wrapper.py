from pathlib import Path


SCRIPT = Path("scripts/dcc/verify_selected_timing_precision_matrix.sbatch")


def test_wrapper_is_one_reciprocal_four_arm_full_song_gate() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "run_arm control_first control" in source
    assert "run_arm exact_first exact_fp32_native_self" in source
    assert "run_arm full_fp16_first full_fp16" in source
    assert "run_arm fp16_weights_first fp16_weights_fp32_state" in source
    assert "run_arm fp16_weights_second fp16_weights_fp32_state" in source
    assert "run_arm full_fp16_second full_fp16" in source
    assert "run_arm exact_second exact_fp32_native_self" in source
    assert "run_arm control_second control" in source
    assert "--fixed-timing-tokens 821" in source
    assert "--allow-mixed-stage-precision" in source
    assert "fixed_main_tokens=8294" in source
    assert "analysis.json" in source


def test_wrapper_requires_clean_pushed_exact_commit_and_2080ti() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "status --porcelain" in source
    assert "refs/remotes/$REMOTE/$BRANCH" in source
    assert '"$(git -C "$REPO" rev-parse HEAD)" != "$COMMIT"' in source
    assert "NVIDIA GeForce RTX 2080 Ti" in source
    assert "profile_pass_kind=untraced_control" in source
