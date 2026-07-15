from __future__ import annotations

from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = (
    REPO_ROOT
    / "scripts"
    / "dcc"
    / "verify_k1_shared_static_input_arena_reciprocal.sbatch"
)
RUNNER = REPO_ROOT / "utils" / "run_k1_shared_static_input_arena.py"


def test_wrapper_pins_clean_pushed_same_precision_2080ti_worktree():
    source = WRAPPER.read_text(encoding="utf-8")
    for fragment in (
        "#SBATCH --partition=gpu-common",
        "#SBATCH --gres=gpu:2080:1",
        'PRECISION=${MAPPERATORINATOR_PRECISION:?',
        'git -C "$REPO" status --porcelain',
        'git -C "$REPO" rev-parse "$REMOTE_REF"',
        "MAPPERATORINATOR_ALLOW_PARALLEL_RECIPROCAL:-0",
        "parallel_reciprocal_opt_in=$ALLOW_PARALLEL_RECIPROCAL",
        "expected exactly one visible GPU",
        "export NVIDIA_TF32_OVERRIDE=0",
        "profile_pass_kind=untraced_control",
    ):
        assert fragment in source


def test_wrapper_runs_fresh_reciprocal_orders_and_hard_parity_gates():
    source = WRAPPER.read_text(encoding="utf-8")
    assert source.count("run_profile baseline") == 2
    assert source.count("run_profile candidate") == 2
    assert source.index("run_profile baseline baseline_first") < source.index(
        "run_profile candidate candidate_second"
    )
    assert source.index("run_profile candidate candidate_second") < source.index(
        "run_profile candidate candidate_first"
    )
    assert source.index("run_profile candidate candidate_first") < source.index(
        "run_profile baseline baseline_second"
    )
    for fragment in (
        "utils.run_k1_shared_static_input_arena",
        "utils.summarize_inference_profile",
        "utils.validate_k1_shared_static_input_arena_reciprocal",
        '--json-output "$RUN_ROOT/decision.json"',
        '--text-output "$RUN_ROOT/decision.txt"',
    ):
        assert fragment in source
    assert ".md" not in source
    assert ".html" not in source


def test_wrapper_shell_syntax_is_valid():
    completed = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_runner_installs_candidate_only_around_normal_inference():
    source = RUNNER.read_text(encoding="utf-8")
    assert "install_shared_static_input_arena_candidate" in source
    assert "with install_shared_static_input_arena_candidate():" in source
    assert "inference_main()" in source
    assert "model.forward" not in source
    assert "decoder" not in source
