from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from utils.analyze_fp32_fresh_baseline import BaselineError, RUN_NAMES, analyze


COMMIT = "a" * 40
BRANCH = "codex/strict-fp32-frontier"
HASH = "b" * 64


def _osu() -> bytes:
    return (
        b"osu file format v14\n\n"
        b"[TimingPoints]\n"
        b"0,500,4,2,1,100,1,0\n\n"
        b"[HitObjects]\n"
        b"64,192,1000,1,0,0:0:0:0:\n"
    )


def _strict_exactness(output_tokens: int) -> dict:
    layers = [
        {
            "layer_idx": index,
            "self_keys": HASH,
            "self_values": HASH,
            "cross_keys": HASH,
            "cross_values": HASH,
        }
        for index in range(12)
    ]
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
            "layers": layers,
        },
    }


def _record(label: str, tokens: list[int], *, exactness: bool) -> dict:
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
        "wall_seconds": 2.0,
        "model_elapsed_seconds": 1.0,
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
        record["strict_exactness"] = _strict_exactness(output_tokens)
    return record


def _profile(result: Path, *, exactness: bool) -> dict:
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
        "git_commit": COMMIT,
        "git_branch": BRANCH,
        "result_file_sha256": hashlib.sha256(payload).hexdigest(),
        "result_file_size_bytes": len(payload),
    }
    generation = [
        _record("timing_context", [8, 9], exactness=exactness),
        _record("main_generation", [1, 2, 3], exactness=exactness),
    ]
    stages = [
        {
            "name": "compile_args",
            "wall_seconds": 0.5,
            "started_at_perf_counter_seconds": 10.0,
            "finished_at_perf_counter_seconds": 10.5,
            "cuda_max_memory_allocated_mb": 50.0,
        },
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
            "started_at_perf_counter_seconds": 15.0,
            "finished_at_perf_counter_seconds": 15.2,
            "cuda_max_memory_allocated_mb": 900.0,
        },
    ]
    return {
        "schema_version": 1,
        "metadata": metadata,
        "summary": {
            "stage_wall_seconds": {
                "compile_args": 0.5,
                "validate_inputs": 0.1,
                "write_osu": 0.2,
            },
            "generation_by_label": {
                "timing_context": {
                    "records": 1,
                    "wall_seconds": 2.0,
                    "model_elapsed_seconds": 1.0,
                    "generated_tokens": 2,
                },
                "main_generation": {
                    "records": 1,
                    "wall_seconds": 2.0,
                    "model_elapsed_seconds": 1.0,
                    "generated_tokens": 3,
                },
            },
        },
        "stages": stages,
        "generation": generation,
    }


def _run(root: Path, name: str, *, exactness: bool = False) -> Path:
    run = root / name
    output = run / "output"
    output.mkdir(parents=True)
    result = output / "result.osu"
    result.write_bytes(_osu())
    profile = output / "result.osu.profile.json"
    profile.write_text(json.dumps(_profile(result, exactness=exactness)), encoding="utf-8")
    (run / "profile-path.txt").write_text(str(profile.resolve()) + "\n", encoding="utf-8")
    (run / "result-path.txt").write_text(str(result.resolve()) + "\n", encoding="utf-8")
    (run / "command.txt").write_text("python inference.py\n", encoding="utf-8")
    (run / "stdout.txt").write_text("complete\n", encoding="utf-8")
    (run / "stderr.txt").write_text("", encoding="utf-8")
    (run / "process-wall.json").write_text(
        json.dumps(
            {
                "started_unix_ns": 1_000_000_000,
                "finished_unix_ns": 8_000_000_000,
                "elapsed_ns": 7_000_000_000,
                "exit_code": 0,
            }
        ),
        encoding="utf-8",
    )
    return profile


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "run-root"
    for name in RUN_NAMES:
        _run(root, name)
    _run(root, "exactness-audit", exactness=True)
    return root


def test_analyzes_five_authoritative_controls_and_separate_exactness(tmp_path: Path) -> None:
    result = analyze(
        _root(tmp_path),
        expected_commit=COMMIT,
        expected_branch=BRANCH,
        expected_main_tokens=3,
        expected_timing_tokens=2,
    )

    assert result["status"] == "PASS"
    assert result["run_count"] == 5
    assert result["aggregates"]["process_wall_seconds"]["median"] == 7.0
    assert result["aggregates"]["request_to_final_osu_wall_seconds"][
        "median"
    ] == pytest.approx(4.2)
    assert result["aggregates"]["main_tokens_per_second"]["median"] == 3.0
    assert not result["exactness_audit"]["authoritative_performance"]
    assert len(result["exactness_audit"]["strict_exactness"]["main_generation"]) == 1


def test_rejects_dispatch_or_graph_drift(tmp_path: Path) -> None:
    root = _root(tmp_path)
    path = root / "run-05/output/result.osu.profile.json"
    profile = json.loads(path.read_text(encoding="utf-8"))
    profile["generation"][1]["optimized_dispatch_capture_hits"][
        "q1_bmm_cross_attention"
    ] = 11
    path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(BaselineError, match="dispatch_and_graph_contract"):
        analyze(
            root,
            expected_commit=COMMIT,
            expected_branch=BRANCH,
            expected_main_tokens=3,
            expected_timing_tokens=2,
        )


def test_rejects_tf32_or_output_drift(tmp_path: Path) -> None:
    root = _root(tmp_path)
    path = root / "run-03/output/result.osu.profile.json"
    profile = json.loads(path.read_text(encoding="utf-8"))
    profile["metadata"]["cuda_matmul_allow_tf32"] = True
    path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(BaselineError, match="strict FP32 runtime metadata mismatch"):
        analyze(
            root,
            expected_commit=COMMIT,
            expected_branch=BRANCH,
            expected_main_tokens=3,
            expected_timing_tokens=2,
        )


def test_rejects_missing_rng_or_cache_evidence(tmp_path: Path) -> None:
    root = _root(tmp_path)
    path = root / "exactness-audit/output/result.osu.profile.json"
    profile = json.loads(path.read_text(encoding="utf-8"))
    del profile["generation"][1]["strict_exactness"]["cache_writes"]
    path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(BaselineError, match="cache_writes must be an object"):
        analyze(
            root,
            expected_commit=COMMIT,
            expected_branch=BRANCH,
            expected_main_tokens=3,
            expected_timing_tokens=2,
        )
