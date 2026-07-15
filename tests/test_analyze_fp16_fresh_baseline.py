import importlib.util
import json
from pathlib import Path

import pytest

from utils.analyze_fp16_fresh_baseline import (
    EXPECTED_PRESET_VERSION,
    FP16BaselineError,
    analyze,
    _runtime_contract,
    _text,
)


COMMIT = "a" * 40
BRANCH = "codex/current-main-fp16-baseline"


def _profile(*, audit=False):
    return {
        "metadata": {
            "precision": "fp16",
            "inference_engine": "optimized",
            "attn_implementation": "sdpa",
            "profile_pass_kind": (
                "exactness_audit" if audit else "untraced_control"
            ),
            "authoritative_performance": not audit,
            "strict_exactness_evidence": audit,
            "nvidia_tf32_override": "0",
            "git_commit": COMMIT,
            "git_branch": BRANCH,
            "cuda_device_name": "NVIDIA GeForce RTX 2080 Ti",
            "cuda_device_capability": [7, 5],
            "optimized_effective_config": {
                "version": EXPECTED_PRESET_VERSION,
                "precision": "fp16",
                "attn_implementation": "sdpa",
                "decoder_loop_backend": "active_prefix_cuda_graph",
                "batch_size": 1,
                "q1_bmm_cross_attention": True,
                "native_q1_self_attention": True,
                "native_q1_rope_cache_self_attention": True,
                "native_cross_mlp_tail": True,
            },
        },
        "generation": [
            {"profile_label": "timing_context", "precision": "fp16"},
            {"profile_label": "main_generation", "precision": "fp16"},
        ],
    }


def test_runtime_requires_current_accepted_shared_specialized_fp16_preset():
    _runtime_contract(
        _profile(),
        expected_commit=COMMIT,
        expected_branch=BRANCH,
        audit=False,
    )
    _runtime_contract(
        _profile(audit=True),
        expected_commit=COMMIT,
        expected_branch=BRANCH,
        audit=True,
    )

    changed = _profile()
    changed["metadata"]["optimized_effective_config"][
        "native_q1_rope_cache_self_attention"
    ] = False
    with pytest.raises(FP16BaselineError, match="shared-specialized"):
        _runtime_contract(
            changed,
            expected_commit=COMMIT,
            expected_branch=BRANCH,
            audit=False,
        )


def test_runtime_rejects_fp32_so_fp16_cannot_mask_fp32_controls():
    changed = _profile()
    changed["metadata"]["precision"] = "fp32"
    with pytest.raises(FP16BaselineError, match="runtime contract"):
        _runtime_contract(
            changed,
            expected_commit=COMMIT,
            expected_branch=BRANCH,
            audit=False,
        )


def test_text_reports_fixed_work_tps_and_complete_song_wall():
    aggregate = {"median": 1.0, "minimum": 1.0, "maximum": 1.0, "range": 0.0}
    result = {
        "preset_version": EXPECTED_PRESET_VERSION,
        "fixed_work": {"timing_tokens": 821, "main_tokens": 7809},
        "aggregates": {
            "timing_synchronized_model_seconds": aggregate,
            "timing_tokens_per_second": aggregate,
            "main_synchronized_model_seconds": aggregate,
            "main_tokens_per_second": aggregate,
            "request_to_final_osu_wall_seconds": aggregate,
            "process_wall_seconds": aggregate,
        },
    }
    text = _text(result)
    assert "fixed_work_main_tokens=7809" in text
    assert "main_tps_median=" in text
    assert "request_wall_seconds_median=" in text


def test_wrapper_runs_five_controls_and_two_non_authoritative_audits():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/dcc/profile_fp16_fresh_baseline.sbatch"
    ).read_text(encoding="utf-8")
    assert "for run_index in 01 02 03 04 05" in script
    assert "run_fresh_process exactness-audit-01 exactness_audit" in script
    assert "run_fresh_process exactness-audit-02 exactness_audit" in script
    assert "precision=fp16" in script
    assert '"$PYTHON" -m utils.analyze_fp16_fresh_baseline' in script
    assert '"$PYTHON" utils/analyze_fp16_fresh_baseline.py' not in script
    assert "accepted-fp16" not in script
    assert ".md" not in script
    assert ".html" not in script


def test_full_analyzer_reuses_current_profiler_contract_without_fp32_mutation(
    tmp_path,
):
    fixture_path = (
        Path(__file__).resolve().parent / "test_analyze_fp32_fresh_baseline.py"
    )
    spec = importlib.util.spec_from_file_location("fp32_fixture", fixture_path)
    assert spec is not None and spec.loader is not None
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    root = tmp_path / "fp16-root"
    for name in ("run-01", "run-02", "run-03", "run-04", "run-05"):
        fixture._run(root, name)
    fixture._run(root, "exactness-audit-01", exactness=True)
    fixture._run(root, "exactness-audit-02", exactness=True)

    required_config = {
        "version": EXPECTED_PRESET_VERSION,
        "precision": "fp16",
        "attn_implementation": "sdpa",
        "decoder_loop_backend": "active_prefix_cuda_graph",
        "batch_size": 1,
        "q1_bmm_cross_attention": True,
        "native_q1_self_attention": True,
        "native_q1_rope_cache_self_attention": True,
        "native_cross_mlp_tail": True,
    }
    for path in root.glob("*/output/*.profile.json"):
        profile = json.loads(path.read_text(encoding="utf-8"))
        metadata = profile["metadata"]
        metadata["precision"] = "fp16"
        metadata["git_branch"] = BRANCH
        metadata["optimized_effective_config"] = required_config
        for record in profile["generation"]:
            record["precision"] = "fp16"
        path.write_text(json.dumps(profile), encoding="utf-8")

    result = analyze(root, expected_commit=COMMIT, expected_branch=BRANCH)

    assert result["status"] == "PASS"
    assert result["run_count"] == 5
    assert result["fixed_work"] == {"timing_tokens": 2, "main_tokens": 3}
    assert result["checks"]["audit_rng_cache_repeat"]
    assert len(result["exactness_audits"]) == 2


def test_candidate_runtime_requires_consistent_fp16_split_kv_hits():
    profile = _profile()
    version = "candidate-fp16-all-fused-split-kv-v3"
    profile["metadata"]["optimized_effective_config"].update(
        {
            "version": version,
            "native_q1_rope_cache_split_kv": True,
            "native_q1_rope_cache_split_kv_split_count": 8,
            "native_q1_rope_cache_split_kv_prefix_buckets": list(
                range(192, 833, 64)
            ),
        }
    )
    aggregate = "native_q1_rope_cache_self_attention_split_kv_8"
    profile["generation"][1]["optimized_dispatch_capture_hits"] = {
        aggregate: 12,
        f"{aggregate}_prefix_640": 12,
    }
    _runtime_contract(
        profile,
        expected_commit=COMMIT,
        expected_branch=BRANCH,
        expected_preset_version=version,
        expected_split_kv=True,
        audit=False,
    )

    profile["generation"][1]["optimized_dispatch_capture_hits"][aggregate] = 13
    with pytest.raises(FP16BaselineError, match="hit metadata"):
        _runtime_contract(
            profile,
            expected_commit=COMMIT,
            expected_branch=BRANCH,
            expected_preset_version=version,
            expected_split_kv=True,
            audit=False,
        )


def test_full_split_kv_candidate_accepts_one_run_and_two_audits(tmp_path):
    fixture_path = (
        Path(__file__).resolve().parent / "test_analyze_fp32_fresh_baseline.py"
    )
    spec = importlib.util.spec_from_file_location("fp32_split_fixture", fixture_path)
    assert spec is not None and spec.loader is not None
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    root = tmp_path / "fp16-split-candidate"
    fixture._run(root, "run-01")
    fixture._run(root, "exactness-audit-01", exactness=True)
    fixture._run(root, "exactness-audit-02", exactness=True)

    version = "candidate-fp16-all-fused-split-kv-v3"
    required_config = {
        "version": version,
        "precision": "fp16",
        "attn_implementation": "sdpa",
        "decoder_loop_backend": "active_prefix_cuda_graph",
        "batch_size": 1,
        "q1_bmm_cross_attention": True,
        "native_q1_self_attention": True,
        "native_q1_rope_cache_self_attention": True,
        "native_cross_mlp_tail": True,
        "native_q1_rope_cache_split_kv": True,
        "native_q1_rope_cache_split_kv_split_count": 8,
        "native_q1_rope_cache_split_kv_prefix_buckets": list(
            range(192, 833, 64)
        ),
    }
    aggregate = "native_q1_rope_cache_self_attention_split_kv_8"
    for path in root.glob("*/output/*.profile.json"):
        profile = json.loads(path.read_text(encoding="utf-8"))
        metadata = profile["metadata"]
        metadata["precision"] = "fp16"
        metadata["git_branch"] = BRANCH
        metadata["optimized_effective_config"] = required_config
        for record in profile["generation"]:
            record["precision"] = "fp16"
            if record["profile_label"] == "main_generation":
                record["optimized_dispatch_capture_hits"] = {
                    aggregate: 12,
                    f"{aggregate}_prefix_640": 12,
                }
        path.write_text(json.dumps(profile), encoding="utf-8")

    result = analyze(
        root,
        expected_commit=COMMIT,
        expected_branch=BRANCH,
        expected_preset_version=version,
        expected_split_kv=True,
        run_count=1,
    )

    assert result["status"] == "PASS"
    assert result["run_count"] == 1
    assert result["preset_version"] == version
    assert result["split_kv"] is True
    assert result["runs"][0]["token_signatures"]["main_generation"][
        "token_groups_by_record"
    ]
