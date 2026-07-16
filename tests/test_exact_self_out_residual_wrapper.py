from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_exact_self_out_residual_wrapper_uses_unique_tmpdir() -> None:
    source = (
        ROOT / "scripts/dcc/profile_exact_self_out_residual_reciprocal.sbatch"
    ).read_text(encoding="utf-8")
    assert "run_exact_self_out_residual_candidate.py" in source
    assert "SLURM_TMPDIR" in source
    assert "torch_extensions-exact-self-out-residual-" in source
    assert "--allow-dispatch-delta" in source
    assert "*native_self_out_residual*" in source
    assert "*optimized_cuda_graphs*" in source
    assert "exact_self_out_residual_reciprocal=PASS" in source
    assert "MAPPERATORINATOR_ALLOW_PARALLEL_RECIPROCAL:-0" in source


def test_self_out_residual_activation_cold() -> None:
    from osuT5.osuT5.inference.optimized.kernels.self_out_residual_activation import (
        self_out_residual_candidate_context,
        self_out_residual_requested,
    )

    assert self_out_residual_requested() is False
    with self_out_residual_candidate_context():
        assert self_out_residual_requested() is True
    assert self_out_residual_requested() is False
