from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from utils.analyze_fp32_fresh_baseline import RUN_NAMES, analyze as analyze_baseline
from utils.analyze_strict_fp32_candidate import CandidateGateError, analyze, main


BASE_COMMIT = "a" * 40
CANDIDATE_COMMIT = "c" * 40
BASE_BRANCH = "codex/strict-fp32-frontier"
CANDIDATE_BRANCH = "codex/exact-candidate"
HASH = "b" * 64


def _osu() -> bytes:
    return (
        b"osu file format v14\n\n"
        b"[TimingPoints]\n"
        b"0,500,4,2,1,100,1,0\n\n"
        b"[HitObjects]\n"
        b"64,192,1000,1,0,0:0:0:0:\n"
    )


def _exactness(output_tokens: int) -> dict:
    return {
        "schema_version": 1,
        "timing_class": "non_authoritative_outside_model_elapsed",
        "rng_before": {
            "cpu_sha256": HASH,
            "cuda_device": "cuda:0",
            "cuda_sha256": HASH,
        },
        "rng_after": {
            "cpu_sha256": HASH,
            "cuda_device": "cuda:0",
            "cuda_sha256": HASH,
        },
        "cache_writes": {
            "schema_version": 1,
            "self_sequence_length": output_tokens - 1,
            "cross_sequence_length": 32,
            "layer_count": 12,
            "aggregate_sha256": HASH,
            "layers": [
                {
                    "layer_idx": index,
                    "self_keys": HASH,
                    "self_values": HASH,
                    "cross_keys": HASH,
                    "cross_values": HASH,
                }
                for index in range(12)
            ],
        },
    }


def _record(
    label: str,
    tokens: list[int],
    *,
    model_seconds: float,
    exactness: bool,
) -> dict:
    output_tokens = len(tokens) + 1
    record = {
        "profile_label": label,
        "context_type": "TIMING" if label == "timing_context" else "MAP",
        "mode": "sequential",
        "sequence_index": 0,
        "precision": "fp32",
        "generated_tokens": len(tokens),
        "generated_tokens_per_sample": [len(tokens)],
        "generated_token_ids": tokens,
        "output_tokens_per_sample": [output_tokens],
        "wall_seconds": model_seconds + 0.2,
        "model_elapsed_seconds": model_seconds,
        "decoder_loop_backend": "active_prefix_cuda_graph",
        "torch_compile_enabled": False,
        "optimized_effective_config_version": "accepted-fp32",
        "optimized_dispatch_mode": "accepted_batch1",
        "optimized_dispatch_policy": {
            "native_q1_self_attention": {"requested": True, "enabled": True}
        },
        "optimized_dispatch_capture_hits": {
            "native_q1_rope_cache_self_attention": 12,
            "q1_bmm_cross_attention": 12,
        },
        "optimized_cuda_graphs": {
            "graph_count": 1,
            "decode_replays": len(tokens),
            "capture_seconds": 0.5,
            "buckets": [
                {
                    "active_prefix_length": 128,
                    "decode_replays": len(tokens),
                    "capture_seconds": 0.5,
                }
            ],
        },
        "cuda_max_memory_allocated_mb": 900.0,
    }
    if exactness:
        record["strict_exactness"] = _exactness(output_tokens)
    return record


def _profile(
    result: Path,
    *,
    commit: str,
    branch: str,
    exactness: bool,
    timing_model: float,
    main_model: float,
    request_finish: float,
) -> dict:
    payload = result.read_bytes()
    metadata = {
        "model_path": "model",
        "audio_path": "/data/salvalai.mp3",
        "beatmap_path": None,
        "seed": 12345,
        "precision": "fp32",
        "attn_implementation": "sdpa",
        "inference_engine": "optimized",
        "use_server": False,
        "parallel": False,
        "max_batch_size": 1,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 0,
        "do_sample": True,
        "num_beams": 1,
        "cfg_scale": 1.0,
        "optimized_effective_config_version": "accepted-fp32",
        "optimized_effective_config": {
            "precision": "fp32",
            "native_q1_self_attention": True,
        },
        "optimized_runtime_owner": "optimized.single.engine",
        "optimized_result_class": "exact",
        "profile_pass_kind": "exactness_audit" if exactness else "untraced_control",
        "authoritative_performance": not exactness,
        "strict_exactness_evidence": exactness,
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "nvidia_tf32_override": "0",
        "cuda_device_name": "NVIDIA GeForce RTX 2080 Ti",
        "cuda_device_capability": [7, 5],
        "git_commit": commit,
        "git_branch": branch,
        "result_file_sha256": hashlib.sha256(payload).hexdigest(),
        "result_file_size_bytes": len(payload),
    }
    generation = [
        _record(
            "timing_context",
            [8, 9],
            model_seconds=timing_model,
            exactness=exactness,
        ),
        _record(
            "main_generation",
            [1, 2, 3],
            model_seconds=main_model,
            exactness=exactness,
        ),
    ]
    return {
        "schema_version": 1,
        "metadata": metadata,
        "summary": {
            "stage_wall_seconds": {
                "validate_inputs": 0.1,
                "write_osu": 0.2,
            },
            "generation_by_label": {
                "timing_context": {
                    "records": 1,
                    "wall_seconds": timing_model + 0.2,
                    "model_elapsed_seconds": timing_model,
                    "generated_tokens": 2,
                },
                "main_generation": {
                    "records": 1,
                    "wall_seconds": main_model + 0.2,
                    "model_elapsed_seconds": main_model,
                    "generated_tokens": 3,
                },
            },
        },
        "stages": [
            {
                "name": "validate_inputs",
                "wall_seconds": 0.1,
                "started_at_perf_counter_seconds": 11.0,
                "finished_at_perf_counter_seconds": 11.1,
                "cuda_max_memory_allocated_mb": 100.0,
            },
            {
                "name": "write_osu",
                "wall_seconds": 0.2,
                "started_at_perf_counter_seconds": request_finish - 0.2,
                "finished_at_perf_counter_seconds": request_finish,
                "cuda_max_memory_allocated_mb": 900.0,
            },
        ],
        "generation": generation,
    }


def _run(
    root: Path,
    name: str,
    *,
    commit: str,
    branch: str,
    exactness: bool = False,
    timing_model: float = 1.0,
    main_model: float = 1.0,
    request_finish: float = 15.2,
    process_wall: float = 7.0,
) -> Path:
    run_dir = root / name
    output = run_dir / "output"
    output.mkdir(parents=True)
    result = output / "result.osu"
    result.write_bytes(_osu())
    profile = output / "result.osu.profile.json"
    profile.write_text(
        json.dumps(
            _profile(
                result,
                commit=commit,
                branch=branch,
                exactness=exactness,
                timing_model=timing_model,
                main_model=main_model,
                request_finish=request_finish,
            )
        ),
        encoding="utf-8",
    )
    (run_dir / "profile-path.txt").write_text(
        str(profile.resolve()) + "\n", encoding="utf-8"
    )
    (run_dir / "result-path.txt").write_text(
        str(result.resolve()) + "\n", encoding="utf-8"
    )
    (run_dir / "command.txt").write_text("python inference.py\n", encoding="utf-8")
    (run_dir / "stdout.txt").write_text("complete\n", encoding="utf-8")
    (run_dir / "stderr.txt").write_text("", encoding="utf-8")
    elapsed_ns = int(process_wall * 1_000_000_000)
    (run_dir / "process-wall.json").write_text(
        json.dumps(
            {
                "started_unix_ns": 1_000_000_000,
                "finished_unix_ns": 1_000_000_000 + elapsed_ns,
                "elapsed_ns": elapsed_ns,
                "exit_code": 0,
            }
        ),
        encoding="utf-8",
    )
    return profile


def _baseline(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "baseline"
    for name in RUN_NAMES:
        _run(root, name, commit=BASE_COMMIT, branch=BASE_BRANCH)
    _run(
        root,
        "exactness-audit",
        commit=BASE_COMMIT,
        branch=BASE_BRANCH,
        exactness=True,
    )
    result = analyze_baseline(
        root,
        expected_commit=BASE_COMMIT,
        expected_branch=BASE_BRANCH,
        expected_main_tokens=3,
        expected_timing_tokens=2,
    )
    analysis = root / "analysis.json"
    analysis.write_text(json.dumps(result), encoding="utf-8")
    return root, analysis


def _candidate(
    tmp_path: Path,
    *,
    count: int,
    faster: bool = True,
) -> Path:
    root = tmp_path / f"candidate-{count}"
    for index in range(1, count + 1):
        _run(
            root,
            f"candidate-{index:02d}",
            commit=CANDIDATE_COMMIT,
            branch=CANDIDATE_BRANCH,
            timing_model=0.9 if faster else 1.1,
            main_model=0.8 if faster else 1.2,
            request_finish=14.8 if faster else 15.5,
            process_wall=6.5 if faster else 7.5,
        )
    _run(
        root,
        "exactness-audit",
        commit=CANDIDATE_COMMIT,
        branch=CANDIDATE_BRANCH,
        exactness=True,
    )
    return root


def _declare_split_kv(root: Path) -> None:
    for profile_path in root.glob("*/output/*.profile.json"):
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        metadata = profile["metadata"]
        metadata["optimized_effective_config_version"] = "accepted-fp32-split-kv"
        metadata["optimized_effective_config"]["split_kv"] = True
        for record in profile["generation"]:
            record["optimized_effective_config_version"] = "accepted-fp32-split-kv"
            record["optimized_dispatch_policy"]["native_q1_self_attention"][
                "split_kv"
            ] = True
            record["optimized_dispatch_capture_hits"][
                "native_q1_rope_cache_self_attention_split_kv"
            ] = 6
        profile_path.write_text(json.dumps(profile), encoding="utf-8")


def _split_kv_spec(path: Path) -> Path:
    paths = [
        "preset.optimized_effective_config.split_kv",
        "preset.optimized_effective_config_version",
        "records.*.optimized_dispatch_capture_hits.native_q1_rope_cache_self_attention_split_kv",
        "records.*.optimized_dispatch_policy.native_q1_self_attention.split_kv",
        "records.*.optimized_effective_config_version",
    ]
    path.write_text(
        json.dumps(
            {
                "schema_version": "mapperatorinator.dispatch-delta-allowlist.v1",
                "allowed_paths": paths,
                "required_paths": paths,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_initial_exact_candidate_is_confirmation_eligible(tmp_path: Path) -> None:
    baseline_root, baseline_analysis = _baseline(tmp_path)
    result = analyze(
        baseline_root,
        baseline_analysis,
        _candidate(tmp_path, count=1),
        candidate_commit=CANDIDATE_COMMIT,
        candidate_branch=CANDIDATE_BRANCH,
        mode="initial",
    )

    assert result["status"] == "PASS_INITIAL"
    assert result["promotion_eligible"]
    assert result["exact_fixed_seed_parity"]
    assert result["deltas"]["combined_synchronized_model_seconds"][
        "saved_seconds"
    ] == pytest.approx(0.3)
    assert result["deltas"]["request_to_final_osu_wall_seconds"][
        "saved_seconds"
    ] == pytest.approx(0.4)
    assert result["deltas"]["process_wall_seconds"]["saved_seconds"] == 0.5


def test_exact_but_slower_candidate_stops_without_confirmation(tmp_path: Path) -> None:
    baseline_root, baseline_analysis = _baseline(tmp_path)
    result = analyze(
        baseline_root,
        baseline_analysis,
        _candidate(tmp_path, count=1, faster=False),
        candidate_commit=CANDIDATE_COMMIT,
        candidate_branch=CANDIDATE_BRANCH,
        mode="initial",
    )

    assert result["status"] == "STOP_NO_GAIN"
    assert not result["promotion_eligible"]


def test_rejects_tf32_or_cache_write_drift(tmp_path: Path) -> None:
    baseline_root, baseline_analysis = _baseline(tmp_path)
    candidate = _candidate(tmp_path, count=1)
    profile_path = candidate / "candidate-01/output/result.osu.profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["metadata"]["cuda_matmul_allow_tf32"] = True
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(CandidateGateError, match="runtime metadata mismatch"):
        analyze(
            baseline_root,
            baseline_analysis,
            candidate,
            candidate_commit=CANDIDATE_COMMIT,
            candidate_branch=CANDIDATE_BRANCH,
            mode="initial",
        )

    candidate = _candidate(tmp_path / "cache", count=1)
    audit_path = candidate / "exactness-audit/output/result.osu.profile.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["generation"][1]["strict_exactness"]["cache_writes"]["layers"][0][
        "self_keys"
    ] = "d" * 64
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(CandidateGateError, match="rng_progression_and_cache_writes"):
        analyze(
            baseline_root,
            baseline_analysis,
            candidate,
            candidate_commit=CANDIDATE_COMMIT,
            candidate_branch=CANDIDATE_BRANCH,
            mode="initial",
        )


def test_accepts_only_explicit_required_dispatch_metadata_deltas(tmp_path: Path) -> None:
    baseline_root, baseline_analysis = _baseline(tmp_path)
    candidate = _candidate(tmp_path, count=1)
    _declare_split_kv(candidate)
    spec = _split_kv_spec(tmp_path / "split-kv-delta.json")

    result = analyze(
        baseline_root,
        baseline_analysis,
        candidate,
        candidate_commit=CANDIDATE_COMMIT,
        candidate_branch=CANDIDATE_BRANCH,
        mode="initial",
        dispatch_delta_spec_path=spec,
    )

    assert result["status"] == "PASS_INITIAL"
    assert result["declared_dispatch_delta"]["observed_paths"] == sorted(
        json.loads(spec.read_text(encoding="utf-8"))["required_paths"]
    )

    profile_path = candidate / "candidate-01/output/result.osu.profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["generation"][0]["optimized_dispatch_capture_hits"][
        "q1_bmm_cross_attention"
    ] = 11
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(CandidateGateError, match="undeclared=.*q1_bmm_cross_attention"):
        analyze(
            baseline_root,
            baseline_analysis,
            candidate,
            candidate_commit=CANDIDATE_COMMIT,
            candidate_branch=CANDIDATE_BRANCH,
            mode="initial",
            dispatch_delta_spec_path=spec,
        )

def test_confirmation_requires_matching_passed_initial_and_five_runs(tmp_path: Path) -> None:
    baseline_root, baseline_analysis = _baseline(tmp_path)
    initial = analyze(
        baseline_root,
        baseline_analysis,
        _candidate(tmp_path / "initial", count=1),
        candidate_commit=CANDIDATE_COMMIT,
        candidate_branch=CANDIDATE_BRANCH,
        mode="initial",
    )
    initial_path = tmp_path / "initial-analysis.json"
    initial_path.write_text(json.dumps(initial), encoding="utf-8")

    result = analyze(
        baseline_root,
        baseline_analysis,
        _candidate(tmp_path / "confirm", count=5),
        candidate_commit=CANDIDATE_COMMIT,
        candidate_branch=CANDIDATE_BRANCH,
        mode="confirm",
        initial_analysis_path=initial_path,
    )

    assert result["status"] == "PASS_CONFIRMED"
    assert result["candidate"]["run_count"] == 5
    with pytest.raises(CandidateGateError, match="requires --initial-analysis"):
        analyze(
            baseline_root,
            baseline_analysis,
            _candidate(tmp_path / "missing", count=5),
            candidate_commit=CANDIDATE_COMMIT,
            candidate_branch=CANDIDATE_BRANCH,
            mode="confirm",
        )


def test_cli_keeps_json_and_text_artifacts_for_a_no_gain_stop(tmp_path: Path) -> None:
    baseline_root, baseline_analysis = _baseline(tmp_path)
    candidate = _candidate(tmp_path, count=1, faster=False)
    output = tmp_path / "analysis.json"
    text = tmp_path / "analysis.txt"

    exit_code = main(
        [
            "--baseline-run-root",
            str(baseline_root),
            "--baseline-analysis",
            str(baseline_analysis),
            "--candidate-run-root",
            str(candidate),
            "--candidate-commit",
            CANDIDATE_COMMIT,
            "--candidate-branch",
            CANDIDATE_BRANCH,
            "--mode",
            "initial",
            "--output",
            str(output),
            "--text-output",
            str(text),
        ]
    )

    assert exit_code == 3
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "STOP_NO_GAIN"
    assert "promotion_eligible=false" in text.read_text(encoding="utf-8")
