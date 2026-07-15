from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/dcc/verify_exact_k4_reciprocal.sbatch"


def test_wrapper_pins_strict_fp32_clean_pushed_2080ti_contract():
    source = WRAPPER.read_text(encoding="utf-8")

    assert "export NVIDIA_TF32_OVERRIDE=0" in source
    assert "precision=fp32" in source
    assert "attn_implementation=sdpa" in source
    assert '[[ -z "$(git -C "$REPO" status --porcelain)" ]]' in source
    assert 'rev-parse "$REMOTE_REF"' in source
    assert 'rev-parse HEAD' in source
    assert "2080 Ti" in source
    assert "another GPU job is running or pending" in source
    assert "MIN_AVAILABLE_KIB" in source


def test_wrapper_runs_reciprocal_performance_and_separate_exactness_audits():
    source = WRAPPER.read_text(encoding="utf-8")

    ordered = (
        "run_fresh_process baseline_first inference.py untraced_control",
        "run_fresh_process candidate_second utils/run_exact_k4_candidate.py untraced_control",
        "run_fresh_process candidate_first utils/run_exact_k4_candidate.py untraced_control",
        "run_fresh_process baseline_second inference.py untraced_control",
    )
    positions = [source.index(line) for line in ordered]
    assert positions == sorted(positions)
    assert "run_fresh_process exactness_baseline inference.py exactness_audit" in source
    assert (
        "run_fresh_process exactness_candidate utils/run_exact_k4_candidate.py "
        "exactness_audit"
    ) in source
    assert "utils/validate_exact_k4_reciprocal.py" in source


def test_wrapper_has_independent_smoke_and_full_scopes_and_non_docs_artifacts():
    source = WRAPPER.read_text(encoding="utf-8")

    assert "EXACT_K4_SCOPE=${EXACT_K4_SCOPE:-smoke}" in source
    assert "PROFILE_CONFIG=profile_salvalai_smoke15" in source
    assert "PROFILE_CONFIG=profile_salvalai" in source
    assert "strict-exact-k4-${EXACT_K4_SCOPE}-${SLURM_JOB_ID}" in source
    assert "analysis.json" in source
    assert "analysis.txt" in source
    assert ".md" not in source
    assert ".html" not in source


def test_candidate_runner_installs_only_exact_k4_context():
    source = (ROOT / "utils/run_exact_k4_candidate.py").read_text(encoding="utf-8")

    assert "with install_exact_k4_candidate():" in source
    assert "counter" not in source.lower()
    assert "runpy.run_path" in source


def test_validator_cli_is_importable():
    result = subprocess.run(
        [sys.executable, "utils/validate_exact_k4_reciprocal.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--exactness-baseline" in result.stdout
    assert "--full-saving-seconds" in result.stdout
