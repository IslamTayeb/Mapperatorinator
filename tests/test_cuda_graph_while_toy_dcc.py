from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "scripts/dcc/probe_cuda_graph_while_toy.sbatch"


def test_while_toy_sbatch_is_valid_bash():
    subprocess.run(["bash", "-n", str(SBATCH)], check=True)


def test_while_toy_sbatch_is_rung1_only_and_isolated():
    source = SBATCH.read_text(encoding="utf-8")
    assert "probe_cuda_conditional_while.py" in source
    assert "profile_conditional_while_real_prefix.py" not in source
    assert "TORCH_EXTENSIONS_DIR=" in source
    assert "TMPDIR=" in source
    assert "torch_extensions-while-toy-" in source
    assert "TORCH_CUDA_ARCH_LIST=7.5" in source
    assert 'required-gpu-substring "2080 Ti"' in source
    assert "ledger=section_32" in source
    assert "rung=1_toy_while" in source
    assert "inductor" not in source.lower()
