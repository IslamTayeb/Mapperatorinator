from pathlib import Path


SCRIPT = Path("scripts/dcc/profile_int8_projection_component.sbatch")


def test_wrapper_is_bounded_and_pins_clean_pushed_worktree() -> None:
    source = SCRIPT.read_text()

    assert "#SBATCH --time=00:30:00" in source
    assert "MAPPERATORINATOR_REPO:?" in source
    assert "MAPPERATORINATOR_COMMIT:?" in source
    assert "MAPPERATORINATOR_BRANCH:?" in source
    assert 'status --porcelain' in source
    assert 'rev-parse HEAD' in source
    assert 'branch --show-current' in source
    assert 'NVIDIA GeForce RTX 2080 Ti' in source


def test_wrapper_runs_only_the_opt_in_component_scout() -> None:
    source = SCRIPT.read_text()

    assert "utils/profile_int8_projection_component.py" in source
    assert "--bucket-mode sentinel" in source
    assert "--warmup 100" in source
    assert "--iters 1000" in source
    assert "component.json" in source
    assert "component.txt" in source
    assert "diagnostics.jsonl" in source
    assert "MAPPERATORINATOR_CUDA_LAUNCH_BLOCKING" in source
    assert 'export CUDA_LAUNCH_BLOCKING="$CUDA_LAUNCH_BLOCKING_VALUE"' in source
    assert "sbatch " not in source
