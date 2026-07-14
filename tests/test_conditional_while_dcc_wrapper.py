from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "scripts/dcc/probe_conditional_while.sbatch"
SUBMIT = ROOT / "scripts/dcc/submit_conditional_while_probe.sh"


def test_conditional_dcc_scripts_are_valid_bash():
    subprocess.run(["bash", "-n", str(SBATCH)], check=True)
    subprocess.run(["bash", "-n", str(SUBMIT)], check=True)


def test_probe_wrapper_is_exact_clean_and_staged():
    source = SBATCH.read_text(encoding="utf-8")
    assert "git -C \"$REPO\" status --porcelain" in source
    assert "git -C \"$REPO\" rev-parse HEAD" in source
    assert "refs/remotes/$REMOTE/$BRANCH" in source
    assert "TORCH_CUDA_ARCH_LIST=7.5" in source
    assert "probe_cuda_conditional_while.py" in source
    assert "profile_conditional_while_real_prefix.py" in source
    assert source.index("probe_cuda_conditional_while.py") < source.index(
        "profile_conditional_while_real_prefix.py"
    )
    assert "--forced-stops 1,3,7" in source
    assert "find \"$RUN_ROOT\" -type f -name '*.osu' -delete" in source


def test_submit_wrapper_refuses_competing_user_jobs():
    source = SUBMIT.read_text(encoding="utf-8")
    assert "squeue -h -u \"$USER\"" in source
    assert "refusing to submit" in source
    assert "sinfo -h -p gpu-common" in source
    assert "exec sbatch" in source
