from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_wrapper_runs_smoke_before_full_and_keeps_strict_check_diagnostic():
    source = (ROOT / "scripts/dcc/verify_k8_reciprocal.sbatch").read_text()

    assert source.index("analyze_scope smoke") < source.index("analyze_scope full")
    assert "run_profile \"$scope\" inference.py baseline_first" in source
    assert "run_profile \"$scope\" utils/run_k8_candidate.py candidate_second" in source
    assert "run_profile \"$scope\" utils/run_k8_candidate.py candidate_first" in source
    assert "run_profile \"$scope\" inference.py baseline_second" in source
    assert "set +e" in source
    assert "strict-reciprocal.exit-code.txt" in source
    assert "profile_pass_kind=untraced_control" in source
    assert "profile_detail_ranges=false" in source
    assert "profile_cuda_capture=false" in source
    assert "--scope \"$scope\"" in source


def test_wrapper_and_submitter_fail_loudly_on_provenance_and_gpu_state():
    wrapper = (ROOT / "scripts/dcc/verify_k8_reciprocal.sbatch").read_text()
    submitter = (ROOT / "scripts/dcc/submit_k8_reciprocal.sh").read_text()

    for needle in (
        "git -C \"$REPO\" status --porcelain",
        "git -C \"$REPO\" rev-parse HEAD",
        "git -C \"$REPO\" branch --show-current",
        "refs/remotes/$REMOTE/$BRANCH",
    ):
        assert needle in wrapper
        assert needle in submitter
    assert "expected RTX 2080 Ti" in wrapper
    assert "squeue -h -u \"$USER\"" in submitter
    assert "refusing to submit" in submitter
    assert "sinfo -h -p gpu-common" in submitter


def test_wrapper_keeps_generated_artifacts_outside_git_and_json_text_only():
    source = (ROOT / "scripts/dcc/verify_k8_reciprocal.sbatch").read_text()

    assert 'RUN_ROOT="$WORK/runs/k8-reciprocal-' in source
    assert "gate.json" in source
    assert "gate.txt" in source
    assert ".md" not in source
    assert ".html" not in source
    assert ".png" not in source


def test_validator_is_directly_executable_from_repo_root():
    result = subprocess.run(
        [sys.executable, "utils/validate_k8_reciprocal.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
