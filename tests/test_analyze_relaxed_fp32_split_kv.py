from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from utils.analyze_relaxed_fp32_split_kv import (
    RelaxedSplitKVDiagnosticError,
    analyze,
)
from utils.analyze_strict_fp32_candidate import CandidateGateError


ROOT = Path(__file__).resolve().parents[1]


def _fixture_module():
    path = ROOT / "tests/test_analyze_strict_fp32_candidate.py"
    spec = importlib.util.spec_from_file_location("strict_candidate_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _relaxed_fixture(tmp_path: Path):
    fixture = _fixture_module()
    baseline_root, baseline_analysis = fixture._baseline(tmp_path)
    candidate = fixture._candidate(tmp_path, count=1)
    fixture._declare_split_kv(candidate)
    spec = fixture._split_kv_spec(tmp_path / "split-kv-delta.json")
    audit_path = candidate / "exactness-audit/output/result.osu.profile.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    main = next(
        record
        for record in audit["generation"]
        if record["profile_label"] == "main_generation"
    )
    main["strict_exactness"]["cache_writes"]["layers"][0]["self_keys"] = "d" * 64
    main["strict_exactness"]["cache_writes"]["aggregate_sha256"] = "e" * 64
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    (candidate / "preflight.txt").write_text(
        "job_id=123\n"
        "             JOBID NAME STATE TRES_PER_NODE\n"
        "                123 current RUNNING gres/gpu:2080:1\n"
        "                456 other RUNNING gres/gpu:2080:1\n",
        encoding="utf-8",
    )
    return fixture, baseline_root, baseline_analysis, candidate, spec


def test_relaxed_gate_allows_only_main_self_cache_hash_drift(tmp_path: Path) -> None:
    fixture, baseline_root, baseline_analysis, candidate, spec = _relaxed_fixture(
        tmp_path
    )

    result = analyze(
        baseline_root,
        baseline_analysis,
        candidate,
        candidate_commit=fixture.CANDIDATE_COMMIT,
        candidate_branch=fixture.CANDIDATE_BRANCH,
        mode="initial",
        dispatch_delta_spec_path=spec,
    )

    assert result["status"] == "PASS_DOCUMENTED_DRIFT_DIAGNOSTIC"
    assert result["exact_fixed_seed_tokens_rng_stopping_and_final_osu"] is True
    assert result["documented_cache_hash_drift"]["drift_count"] == 1
    assert result["documented_cache_hash_drift"]["main_cross_cache_exact"] is True
    assert result["documented_cache_hash_drift"]["timing_cache_exact"] is True
    assert result["documented_cache_hash_drift"]["rng_exact"] is True
    assert result["gpu_overlap"]["other_gpu_job_ids_observed_in_preflight"] == [
        "456"
    ]
    assert result["promotion_eligible"] is False
    assert result["authoritative_performance"] is False
    assert result["deltas"]["main_synchronized_model_seconds"]["saved_seconds"] == (
        pytest.approx(0.2)
    )

    with pytest.raises(CandidateGateError, match="rng_progression_and_cache_writes"):
        fixture.analyze(
            baseline_root,
            baseline_analysis,
            candidate,
            candidate_commit=fixture.CANDIDATE_COMMIT,
            candidate_branch=fixture.CANDIDATE_BRANCH,
            mode="initial",
            dispatch_delta_spec_path=spec,
        )


@pytest.mark.parametrize("field", ["rng_after", "cross_keys"])
def test_relaxed_gate_rejects_rng_or_cross_cache_drift(
    tmp_path: Path, field: str
) -> None:
    fixture, baseline_root, baseline_analysis, candidate, spec = _relaxed_fixture(
        tmp_path
    )
    audit_path = candidate / "exactness-audit/output/result.osu.profile.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    main = next(
        record
        for record in audit["generation"]
        if record["profile_label"] == "main_generation"
    )
    if field == "rng_after":
        main["strict_exactness"][field]["cuda_sha256"] = "f" * 64
    else:
        main["strict_exactness"]["cache_writes"]["layers"][0][field] = "f" * 64
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(RelaxedSplitKVDiagnosticError, match="changed"):
        analyze(
            baseline_root,
            baseline_analysis,
            candidate,
            candidate_commit=fixture.CANDIDATE_COMMIT,
            candidate_branch=fixture.CANDIDATE_BRANCH,
            mode="initial",
            dispatch_delta_spec_path=spec,
        )


@pytest.mark.parametrize(
    "script",
    (
        "utils/analyze_relaxed_fp32_split_kv.py",
        "utils/analyze_fp16_split_kv_reciprocal.py",
    ),
)
def test_analyzers_support_direct_help_without_repo_pythonpath(script: str) -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
