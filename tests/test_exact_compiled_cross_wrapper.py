from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_exact_compiled_cross_wrapper_uses_node_local_tmpdir() -> None:
    source = (
        ROOT
        / "scripts/dcc/profile_exact_compiled_cross_reciprocal.sbatch"
    ).read_text(encoding="utf-8")
    assert "run_exact_compiled_cross_candidate.py" in source
    assert "SLURM_TMPDIR" in source
    assert "torch_extensions-exact-compiled-cross-" in source
    assert "--allow-dispatch-delta" in source
    assert "exact_compiled_cross_reciprocal=PASS" in source
