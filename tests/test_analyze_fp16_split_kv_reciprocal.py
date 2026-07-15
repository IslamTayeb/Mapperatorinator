from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from utils.analyze_fp16_fresh_baseline import (
    EXPECTED_PRESET_VERSION,
    SCHEMA_VERSION as FP16_RUN_SCHEMA_VERSION,
)
from utils.analyze_fp16_split_kv_reciprocal import (
    CANDIDATE_PRESET_VERSION,
    FP16SplitKVReciprocalError,
    analyze,
)


HASH = "a" * 64


def _run(*, token: int, model_seconds: float) -> dict:
    signature = {
        "generated_tokens": 3,
        "token_stream_sha256": HASH if token == 1 else "b" * 64,
        "stopping_sha256": HASH if token == 1 else "c" * 64,
        "token_groups_by_record": [[[token, token + 1, token + 2]]],
    }
    return {
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
        "output_sha256": HASH if token == 1 else "d" * 64,
        "output_structure": {"hit_objects": 1, "timing_points": 1},
    }


def _analysis(*, candidate: bool, run_count: int, model_seconds: float) -> dict:
    token = 9 if candidate else 1
    run = _run(token=token, model_seconds=model_seconds)
    exactness = {
        "rng_before": "e" * 64,
        "rng_after": "f" * 64,
        "cache_writes": "1" * 64,
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
        "split_kv": candidate,
        "fixed_work": {"timing_tokens": 3, "main_tokens": 3},
        "checks": {"all": True},
        "runs": [json.loads(json.dumps(run)) for _ in range(run_count)],
        "exactness_audits": [
            {
                "output_sha256": run["output_sha256"],
                "strict_exactness": json.loads(json.dumps(exactness)),
                "token_signatures": json.loads(
                    json.dumps(run["token_signatures"])
                ),
            },
            {
                "output_sha256": run["output_sha256"],
                "strict_exactness": json.loads(json.dumps(exactness)),
                "token_signatures": json.loads(
                    json.dumps(run["token_signatures"])
                ),
            },
        ],
    }


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _baseline_root(tmp_path: Path) -> Path:
    fixture_path = (
        Path(__file__).resolve().parent / "test_analyze_fp32_fresh_baseline.py"
    )
    spec = importlib.util.spec_from_file_location("fp32_reciprocal_fixture", fixture_path)
    assert spec is not None and spec.loader is not None
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    root = tmp_path / "baseline-root"
    fixture._run(root, "run-01")
    return root


def test_fp16_gate_allows_documented_drift_but_requires_repeatability(tmp_path):
    root = _baseline_root(tmp_path)
    baseline = _analysis(candidate=False, run_count=5, model_seconds=10.0)
    candidate = _analysis(candidate=True, run_count=1, model_seconds=9.0)
    baseline_path = _write(tmp_path / "baseline.json", baseline)
    candidate_path = _write(tmp_path / "candidate.json", candidate)

    result = analyze(
        root,
        baseline_path,
        candidate_path,
        mode="initial",
    )

    assert result["status"] == "PASS_INITIAL"
    assert result["candidate_deterministic_and_cache_safe"] is True
    assert not result["same_precision_drift"]["main_generation"]["tokens"][
        "exact"
    ]
    assert result["same_precision_drift"]["main_generation"]["tokens"][
        "aligned_mismatch_fraction"
    ] > 0
    assert result["deltas"]["main_synchronized_model_seconds"][
        "saved_seconds"
    ] == pytest.approx(0.75)


def test_fp16_gate_rejects_candidate_cache_or_rng_nondeterminism(tmp_path):
    root = _baseline_root(tmp_path)
    baseline_path = _write(
        tmp_path / "baseline.json",
        _analysis(candidate=False, run_count=5, model_seconds=10.0),
    )
    candidate = _analysis(candidate=True, run_count=1, model_seconds=9.0)
    candidate["exactness_audits"][1]["strict_exactness"]["cache_writes"] = (
        "2" * 64
    )
    candidate_path = _write(tmp_path / "candidate.json", candidate)

    with pytest.raises(FP16SplitKVReciprocalError, match="determinism/cache"):
        analyze(root, baseline_path, candidate_path, mode="initial")


def test_fp16_wrapper_is_serial_by_default_with_explicit_parallel_opt_in():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/dcc/profile_fp16_split_kv_reciprocal.sbatch"
    ).read_text(encoding="utf-8")
    assert 'initial) RUN_INDICES=(01)' in script
    assert 'confirm) RUN_INDICES=(01 02 03 04 05)' in script
    assert "MAPPERATORINATOR_ALLOW_PARALLEL_RECIPROCAL:-0" in script
    assert "non_authoritative_parallel_reciprocal" in script
    assert '"$ALLOW_PARALLEL_RECIPROCAL" != 1' in script
    assert "run_fresh_process exactness-audit-01 exactness_audit" in script
    assert "run_fresh_process exactness-audit-02 exactness_audit" in script
    assert "--expected-split-kv" in script
    assert "precision=fp16" in script
    assert '"$PYTHON" -m utils.analyze_fp16_fresh_baseline' in script
    assert '"$PYTHON" -m utils.analyze_fp16_split_kv_reciprocal' in script
    assert '"$PYTHON" utils/analyze_fp16_fresh_baseline.py' not in script
    assert '"$PYTHON" utils/analyze_fp16_split_kv_reciprocal.py' not in script
    assert ".md" not in script
    assert ".html" not in script
