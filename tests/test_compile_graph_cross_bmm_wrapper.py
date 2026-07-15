from pathlib import Path


SCRIPT = Path("scripts/dcc/profile_compile_graph_cross_bmm.sbatch")


def test_wrapper_isolates_build_and_temp_and_keeps_cuda_graph_measurement() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'EXTENSION_ROOT="$WORK/torch_extensions/compile-graph-cross-${COMMIT}"' in source
    assert 'JOB_TMP="$WORK/tmp/compile-graph-cross-${SLURM_JOB_ID}"' in source
    assert 'TORCH_EXTENSIONS_DIR="$EXTENSION_ROOT"' in source
    assert 'TMPDIR="$JOB_TMP"' in source
    assert "profile_compile_graph_cross_bmm.py" in source
    assert "--iterations 1000" in source
    assert "MAPPERATORINATOR_ALLOW_PARALLEL" in source
