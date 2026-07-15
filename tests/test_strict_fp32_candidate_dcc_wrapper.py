from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/dcc/profile_strict_fp32_candidate.sbatch"


def test_wrapper_is_serial_strict_fp32_and_has_two_modes() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    assert "#SBATCH --gres=gpu:2080:1" in source
    assert "export NVIDIA_TF32_OVERRIDE=0" in source
    assert source.index("export NVIDIA_TF32_OVERRIDE=0") < source.index(
        '"$PYTHON" -c'
    )
    assert "precision=fp32" in source
    assert "precision=fp16" not in source
    assert 'RUN_INDICES=(01)' in source
    assert 'RUN_INDICES=(01 02 03 04 05)' in source
    assert 'run_fresh_process "candidate-$run_index" untraced_control' in source
    assert "run_fresh_process exactness-audit exactness_audit" in source
    assert "Shell execution is deliberately serial" in source


def test_wrapper_pins_candidate_baseline_hardware_queue_and_storage() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert 'git -C "$REPO" status --porcelain' in source
    assert 'git -C "$REPO" rev-parse HEAD' in source
    assert 'git -C "$REPO" branch --show-current' in source
    assert 'git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF"' in source
    assert 'git -C "$REPO" rev-parse "$REMOTE_REF"' in source
    assert "another GPU job is running or pending" in source
    assert "expected one visible GPU" in source
    assert "MIN_AVAILABLE_KIB=$((20 * 1024 * 1024))" in source
    assert "MAPPERATORINATOR_BASELINE_RUN_ROOT" in source
    assert "MAPPERATORINATOR_BASELINE_ANALYSIS" in source
    assert "MAPPERATORINATOR_INITIAL_ANALYSIS" in source
    assert "MAPPERATORINATOR_DISPATCH_DELTA_SPEC" in source
    assert "dispatch-delta-spec.json" in source
    assert "--dispatch-delta-spec" in source
    assert "dispatch delta spec must be tracked inside the candidate worktree" in source
    assert 'git -C "$REPO" ls-files --error-unmatch' in source
    assert 'git -C "$REPO" show "$COMMIT:$SPEC_RELATIVE"' in source
    assert "started_ns=$(date +%s%N)" in source
    assert "finished_ns=$(date +%s%N)" in source
    assert "analyze_strict_fp32_candidate.py" in source
    assert 'require_nonempty "$RUN_ROOT/analysis.json"' in source
    assert 'require_nonempty "$RUN_ROOT/analysis.txt"' in source
