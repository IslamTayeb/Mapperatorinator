from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from utils.analyze_fp16_fresh_baseline import (
    EXPECTED_PRESET_VERSION,
    SCHEMA_VERSION as FP16_RUN_SCHEMA_VERSION,
)
from utils.analyze_fp16_reciprocal_candidate import (
    CANDIDATE_PRESET_VERSION,
    FP16ReciprocalError,
    analyze,
)


HASH = "a" * 64


def _analysis(*, candidate: bool, run_count: int, model_seconds: float) -> dict:
    signature = {
        "generated_tokens": 3,
        "token_stream_sha256": HASH,
        "stopping_sha256": HASH,
    }
    run = {
        "process_wall_seconds": model_seconds + 5.0,
        "request_to_final_osu_wall_seconds": model_seconds + 2.0,
        "generation": {
            "timing_context": {
                "generated_tokens": 3,
                "synchronized_model_seconds": model_seconds / 4,
            },
            "main_generation": {
                "generated_tokens": 3,
                "synchronized_model_seconds": model_seconds * 3 / 4,
            },
        },
        "token_signatures": {
            "timing_context": dict(signature),
            "main_generation": dict(signature),
        },
        "output_sha256": HASH,
        "output_structure": {"hit_objects": 1, "timing_points": 1},
    }
    exactness = {
        "rng_before": HASH,
        "rng_after": HASH,
        "cache_writes": HASH,
    }
    return {
        "schema_version": FP16_RUN_SCHEMA_VERSION,
        "status": "PASS",
        "precision": "fp16",
        "fresh_processes": True,
        "authoritative_performance": True,
        "run_count": run_count,
        "runtime_identity": {"commit": "c" * 40, "branch": "codex/candidate"},
        "preset_version": (
            CANDIDATE_PRESET_VERSION if candidate else EXPECTED_PRESET_VERSION
        ),
        "device_conditional_temperature": candidate,
        "fixed_work": {"timing_tokens": 3, "main_tokens": 3},
        "checks": {"all": True},
        "runs": [dict(run) for _ in range(run_count)],
        "exactness_audits": [
            {"output_sha256": HASH, "strict_exactness": exactness},
            {"output_sha256": HASH, "strict_exactness": exactness},
        ],
    }


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_initial_gate_requires_exact_parity_and_reports_wall_and_model_delta(tmp_path):
    baseline = _write(
        tmp_path / "baseline.json",
        _analysis(candidate=False, run_count=5, model_seconds=10.0),
    )
    candidate = _write(
        tmp_path / "candidate.json",
        _analysis(candidate=True, run_count=1, model_seconds=9.0),
    )

    result = analyze(baseline, candidate, mode="initial")

    assert result["status"] == "PASS_INITIAL"
    assert result["exact_same_precision_parity"] is True
    assert result["deltas"]["combined_synchronized_model_seconds"][
        "saved_seconds"
    ] == pytest.approx(1.0)
    assert result["deltas"]["request_to_final_osu_wall_seconds"][
        "saved_seconds"
    ] == pytest.approx(1.0)


def test_gate_fails_on_same_precision_token_drift(tmp_path):
    baseline = _write(
        tmp_path / "baseline.json",
        _analysis(candidate=False, run_count=5, model_seconds=10.0),
    )
    candidate_payload = _analysis(candidate=True, run_count=1, model_seconds=9.0)
    candidate_payload["runs"][0]["token_signatures"]["main_generation"][
        "token_stream_sha256"
    ] = "b" * 64
    candidate = _write(tmp_path / "candidate.json", candidate_payload)

    with pytest.raises(FP16ReciprocalError, match="parity failed"):
        analyze(baseline, candidate, mode="initial")


def test_wrapper_runs_candidate_and_two_audits_without_docs():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/dcc/profile_fp16_reciprocal_candidate.sbatch"
    ).read_text(encoding="utf-8")
    assert 'initial) RUN_INDICES=(01)' in script
    assert 'confirm) RUN_INDICES=(01 02 03 04 05)' in script
    assert "run_fresh_process exactness-audit-01 exactness_audit" in script
    assert "run_fresh_process exactness-audit-02 exactness_audit" in script
    assert "--expected-device-conditional-temperature" in script
    assert ".md" not in script
    assert ".html" not in script


@pytest.mark.parametrize(
    "script_name",
    (
        "analyze_fp16_fresh_baseline.py",
        "analyze_fp16_reciprocal_candidate.py",
    ),
)
def test_analyzers_support_direct_script_launch_without_repo_pythonpath(script_name):
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, str(root / "utils" / script_name), "--help"],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
