from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/dcc/probe_exact_k4_multinomial.sbatch"


def test_wrapper_is_strict_fp32_serial_and_pinned() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)

    assert "#SBATCH --gres=gpu:2080:1" in source
    assert "#SBATCH --time=00:05:00" in source
    assert "export NVIDIA_TF32_OVERRIDE=0" in source
    assert source.index("export NVIDIA_TF32_OVERRIDE=0") < source.index(
        '"$PYTHON" utils/probe_exact_k4_multinomial.py'
    )
    assert 'git -C "$REPO" status --porcelain' in source
    assert 'git -C "$REPO" branch --show-current' in source
    assert 'git -C "$REPO" rev-parse "$REMOTE_REF"' in source
    assert "another GPU job is running or pending" in source
    assert "2080 Ti" in source and "7.5" in source
    assert "--seeds 7,12345,987654,42" in source
    assert "--eos-positions 1,2,3,4" in source
    assert '[[ -s "$RUN_ROOT/analysis.json" ]]' in source
