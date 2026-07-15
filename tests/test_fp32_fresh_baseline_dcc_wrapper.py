from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/dcc/profile_fp32_fresh_baseline.sbatch"


def test_wrapper_is_valid_serial_fp32_only_slurm() -> None:
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
    assert "for run_index in 01 02 03 04 05" in source
    assert 'run_fresh_process "run-$run_index" untraced_control' in source
    assert "run_fresh_process exactness-audit exactness_audit" in source
    assert source.index("exactness-audit exactness_audit") > source.index(
        "for run_index in 01 02 03 04 05"
    )


def test_wrapper_pins_clean_pushed_worktree_and_captures_walls() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert 'git -C "$REPO" status --porcelain' in source
    assert 'git -C "$REPO" rev-parse HEAD' in source
    assert 'git -C "$REPO" branch --show-current' in source
    assert 'git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF"' in source
    assert 'git -C "$REPO" rev-parse "$REMOTE_REF"' in source
    assert "expected one visible GPU" in source
    assert "another GPU job is running or pending" in source
    assert "MIN_AVAILABLE_KIB=$((20 * 1024 * 1024))" in source
    assert 'AVAILABLE_KIB=$(df -Pk "$WORK"' in source
    assert "run-root filesystem has only" in source
    assert 'df -h "$WORK"' in source
    assert "started_ns=$(date +%s%N)" in source
    assert "finished_ns=$(date +%s%N)" in source
    assert '"elapsed_ns":%s' in source
    assert "analyze_fp32_fresh_baseline.py" in source
