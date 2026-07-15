from pathlib import Path


WRAPPER = Path("scripts/dcc/probe_strict_fp32_conditional_while.sbatch")


def test_wrapper_is_serial_strict_fp32_sm75_and_exact_checkpoint() -> None:
    source = WRAPPER.read_text()

    assert "#SBATCH --partition=gpu-common" in source
    assert "#SBATCH --account=romerolab" in source
    assert "#SBATCH --gres=gpu:2080:1" in source
    assert "EXPECTED_GPU=${MAPPERATORINATOR_EXPECTED_GPU:-2080 Ti}" in source
    assert '[[ "${GPU_ROWS[0]}" == *"7.5"* ]]' in source
    assert "another GPU job is running or pending" in source
    assert 'git -C "$REPO" status --porcelain' in source
    assert "MAPPERATORINATOR_COMMIT" in source
    assert "MAPPERATORINATOR_BRANCH" in source
    assert "MAPPERATORINATOR_REMOTE_REF" in source
    assert 'rev-parse "$REMOTE_REF"' in source


def test_tf32_override_precedes_every_python_process() -> None:
    source = WRAPPER.read_text()

    override = source.index("export NVIDIA_TF32_OVERRIDE=0")
    assert override < source.index('"$PYTHON" -c')
    assert override < source.index('"${PROBE_COMMAND[@]}"')


def test_wrapper_distinguishes_negative_result_from_setup_and_control_failure() -> None:
    source = WRAPPER.read_text()

    assert "0|2" in source
    assert "VALID_NEGATIVE_CONDITIONAL_WHILE_RNG" in source
    assert "EXACT_CONDITIONAL_WHILE_FEASIBLE" in source
    assert "FAIL_INVALID_ORDINARY_GRAPH_CONTROL" in source
    assert "SETUP_FAILURE" in source
    assert "utils/analyze_conditional_while_multinomial_probe.py" in source


def test_run_artifacts_are_json_or_text_only() -> None:
    source = WRAPPER.read_text()

    assert "probe-result.json" in source
    assert "analysis.json" in source
    assert "analysis.txt" in source
    assert "process-wall.json" in source
    assert "! -name '*.json' ! -name '*.txt'" in source
    assert ".csv" not in source
