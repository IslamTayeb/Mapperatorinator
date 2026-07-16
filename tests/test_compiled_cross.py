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


def test_compiled_cross_activation_is_request_only() -> None:
    source = (
        ROOT
        / "osuT5/osuT5/inference/optimized/kernels/compiled_cross_activation.py"
    ).read_text(encoding="utf-8")
    assert "torch.compile" not in source
    assert "compiled_cross_candidate_context" in source


def test_exact_compiled_cross_runner_keeps_device_state_on_both_roles() -> None:
    source = (
        ROOT / "utils/run_exact_compiled_cross_candidate.py"
    ).read_text(encoding="utf-8")
    assert "device_sequence_state_candidate_context" in source
    assert "compiled_cross_candidate_context" in source
    assert 'device_sequence_state_enabled": True' in source
