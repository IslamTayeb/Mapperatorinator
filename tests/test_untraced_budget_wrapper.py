from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/dcc/profile_untraced_budget.sbatch"


def test_wrapper_is_serial_strict_fp32_and_pins_pushed_worktree() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)

    assert "#SBATCH --gres=gpu:2080:1" in source
    assert "export NVIDIA_TF32_OVERRIDE=0" in source
    assert source.index("export NVIDIA_TF32_OVERRIDE=0") < source.index(
        '"$PYTHON" -c'
    )
    assert "precision=fp32" in source
    assert "precision=fp16" not in source
    assert "max_batch_size=1" not in source
    assert "run_profile untraced_control" in source
    assert "run_profile untraced_budget" in source
    assert source.index("run_profile untraced_control") < source.index(
        "run_profile untraced_budget"
    )
    assert 'git -C "$REPO" status --porcelain' in source
    assert 'git -C "$REPO" branch --show-current' in source
    assert 'git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF"' in source
    assert "another GPU job is running or pending" in source
    assert "MIN_AVAILABLE_KIB=$((20 * 1024 * 1024))" in source
    assert "compute_cap" in source and "7.5" in source
    assert "transparency" in source
    assert "summarize_untraced_budget.py" in source
    assert "reconcile within 2%" in source
    assert "sha256sums.txt" in source


def test_wrapper_charges_fresh_process_wall_for_both_passes() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert "started_ns=$(date +%s%N)" in source
    assert "finished_ns=$(date +%s%N)" in source
    assert '"elapsed_ns":%s' in source
    assert '[[ $exit_code -eq 0 ]]' in source
