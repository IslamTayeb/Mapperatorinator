from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_exact_decode_cast_copy_wrapper_uses_unique_tmpdir() -> None:
    source = (
        ROOT / "scripts/dcc/profile_exact_decode_cast_copy_reciprocal.sbatch"
    ).read_text(encoding="utf-8")
    assert "run_exact_decode_cast_copy_candidate.py" in source
    assert "SLURM_TMPDIR" in source
    assert "torch_extensions-exact-decode-cast-copy-" in source
    assert "--allow-dispatch-delta" in source
    assert "exact_decode_cast_copy_reciprocal=PASS" in source
    assert "MAPPERATORINATOR_ALLOW_PARALLEL_RECIPROCAL:-0" in source


def test_decode_cast_copy_activation_cold() -> None:
    from osuT5.osuT5.inference.optimized.scout.decode_cast_copy_activation import (
        decode_cast_copy_candidate_context,
        decode_cast_copy_requested,
    )

    assert decode_cast_copy_requested() is False
    with decode_cast_copy_candidate_context():
        assert decode_cast_copy_requested() is True
    assert decode_cast_copy_requested() is False
