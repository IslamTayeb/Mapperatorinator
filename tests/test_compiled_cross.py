from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_compiled_cross_import_is_cold() -> None:
    code = (
        "import sys; "
        "import osuT5.osuT5.inference.optimized.single.engine; "
        "assert 'osuT5.osuT5.inference.optimized.kernels.compiled_cross' "
        "not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_compiled_cross_contract_keeps_outer_graph_ownership() -> None:
    source = (
        ROOT
        / "osuT5/osuT5/inference/optimized/kernels/compiled_cross.py"
    ).read_text(encoding="utf-8")
    assert 'mode="max-autotune-no-cudagraphs"' in source
    assert "fullgraph=True" in source
    assert "dynamic=False" in source
    assert '"outer_cuda_graph_owned": True' in source
