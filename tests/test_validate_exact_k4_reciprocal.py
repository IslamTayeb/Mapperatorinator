from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils import validate_exact_k4_reciprocal as gate


COMMIT = "a" * 40
BRANCH = "codex/strict-fp32-exact-k4-runtime"
HASH = "b" * 64


def _metadata(*, exactness: bool) -> dict:
    values = {
        "model_path": "model",
        "audio_path": "/tmp/song.mp3",
        "seed": 12345,
        "precision": "fp32",
        "attn_implementation": "sdpa",
        "inference_engine": "optimized",
        "use_server": False,
        "parallel": False,
        "temperature": 0.9,
        "timing_temperature": 0.9,
        "mania_column_temperature": 0.9,
        "taiko_hit_temperature": 0.9,
        "timeshift_bias": 0.0,
        "top_p": 0.95,
        "top_k": 0,
        "do_sample": True,
        "num_beams": 1,
        "cfg_scale": 1.0,
        "lookback": 0,
        "lookahead": 0,
        "start_time": 0,
        "end_time": 15000,
        "in_context": [],
        "output_type": "osu",
        "profile_pass_kind": "exactness_audit" if exactness else "untraced_control",
        "authoritative_performance": not exactness,
        "strict_exactness_evidence": exactness,
        "profile_detail_ranges": False,
        "profile_cuda_capture": False,
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "nvidia_tf32_override": "0",
        "git_commit": COMMIT,
        "git_branch": BRANCH,
        "cuda_device_name": "NVIDIA GeForce RTX 2080 Ti",
        "cuda_device_capability": [7, 5],
        "optimized_effective_config": {"precision": "fp32"},
        "result_file_sha256": HASH,
        "result_file_size_bytes": 100,
    }
    return values


def _strict_exactness(tokens: list[int]) -> dict:
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
            "self_sequence_length": len(tokens),
            "cross_sequence_length": 32,
            "layer_count": 12,
            "aggregate_sha256": HASH,
            "layers": layers,
        },
    }


def _stats(*, logical: int = 4) -> dict:
    return {
        "block_size": 4,
        "prefill_steps": 1,
        "eligible_steps": 4,
        "block_replays": 1,
        "remainder_steps": 0,
        "status_reads": 1,
        "copy_calls": 4,
        "copy_bytes": 64,
        "capture_seconds": 0.1,
        "peak_vram_bytes": 1024,
        "physical_steps": 5,
        "logical_steps": logical,
        "wasted_steps": 5 - logical,
        "rng_policy": gate.RNG_POLICY,
        "rng_exact": True,
        "rng_drift": None,
        "rng_request_seed": 12345,
        "rng_window_identity": 1,
        "rng_early_eos_isolation": True,
        "sampling_mode": "sample",
        "prompt_seed_d2h_copy_calls": 0,
        "prompt_seed_d2h_copy_bytes": 0,
        "prompt_seed_setup_seconds": 0.0,
        "processor_signature_d2h_copy_calls": 0,
        "processor_signature_d2h_copy_bytes": 0,
        "processor_signature_setup_seconds": 0.0,
        "parent_backend": gate.PARENT_BACKEND,
        "capture_state_restore_synchronized": True,
        "terminal_rollbacks": 1,
        "generator_offset_corrections": 1,
        "generator_offset_increment": 4,
    }


def _record(label: str, *, candidate: bool, exactness: bool, seconds: float) -> dict:
    tokens = [10, 11, 12, 13]
    hits = {
        "native_q1_rope_cache_self_attention": 4 if candidate else 1,
        "native_q1_self_attention": 4 if candidate else 1,
        "q1_bmm_cross_attention": 4 if candidate else 1,
        "native_cross_mlp_tail": 4 if candidate else 1,
    }
    record = {
        "profile_label": label,
        "context_type": "TIMING" if label == "timing_context" else "MAP",
        "mode": "sequential",
        "sequence_index": 0,
        "precision": "fp32",
        "generated_tokens": len(tokens),
        "generated_tokens_per_sample": [len(tokens)],
        "generated_token_ids": tokens,
        "output_tokens_per_sample": [len(tokens) + 1],
        "wall_seconds": seconds + 0.1,
        "model_elapsed_seconds": seconds,
        "decoder_loop_backend": "active_prefix_cuda_graph",
        "torch_compile_enabled": False,
        "generation_compile_enabled": False,
        "optimized_effective_config_version": "accepted-fp32",
        "native_cross_mlp_tail_requested": True,
        "native_cross_mlp_tail_enabled": True,
        "native_cross_mlp_tail_disabled_reason": None,
        "optimized_dispatch_capture_hits": hits,
        "optimized_cuda_graphs": {
            "graph_count": 1,
            "decode_replays": 4,
            "capture_seconds": 0.1,
            "buckets": {"128": {"graph_count": 1, "decode_replays": 4}},
        },
    }
    if candidate:
        record["optimized_cuda_graphs"]["exact_k4_candidate"] = _stats()
    if exactness:
        record["strict_exactness"] = _strict_exactness(tokens)
    return record


def _profile(*, candidate: bool, exactness: bool, seconds: float = 1.0) -> dict:
    return {
        "metadata": _metadata(exactness=exactness),
        "generation": [
            _record("timing_context", candidate=candidate, exactness=exactness, seconds=seconds),
            _record("main_generation", candidate=candidate, exactness=exactness, seconds=seconds),
        ],
        "stages": [{"wall_seconds": seconds + 1.0}],
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _paths(tmp_path: Path) -> tuple[dict[str, Path], Path, Path]:
    performance = {
        "baseline_first": _write(tmp_path / "baseline-first.json", _profile(candidate=False, exactness=False)),
        "candidate_second": _write(tmp_path / "candidate-second.json", _profile(candidate=True, exactness=False, seconds=0.8)),
        "candidate_first": _write(tmp_path / "candidate-first.json", _profile(candidate=True, exactness=False, seconds=0.8)),
        "baseline_second": _write(tmp_path / "baseline-second.json", _profile(candidate=False, exactness=False)),
    }
    exactness_baseline = _write(
        tmp_path / "exactness-baseline.json", _profile(candidate=False, exactness=True)
    )
    exactness_candidate = _write(
        tmp_path / "exactness-candidate.json", _profile(candidate=True, exactness=True)
    )
    return performance, exactness_baseline, exactness_candidate


def _summarize(tmp_path: Path, **kwargs):
    performance, baseline, candidate = _paths(tmp_path)
    return gate.summarize(
        performance,
        exactness_baseline=baseline,
        exactness_candidate=candidate,
        expected_commit=COMMIT,
        expected_branch=BRANCH,
        scope="smoke",
        minimum_speedup_pct=0.0,
        full_saving_seconds=0.0,
        **kwargs,
    )


def test_exact_k4_reciprocal_passes_strict_contract(tmp_path: Path):
    report = _summarize(tmp_path)

    assert report["pass"] is True
    assert report["strict_exactness"]["pass"] is True
    assert report["reciprocal_main_tps_speedup_pct"] == 25.0
    assert report["reciprocal_complete_request_wall_saving_seconds"] == pytest.approx(0.2)
    assert report["candidate_contracts"]["candidate_first"]["pass"] is True


def test_token_change_fails_even_when_candidate_is_faster(tmp_path: Path):
    performance, baseline, candidate = _paths(tmp_path)
    payload = json.loads(performance["candidate_first"].read_text())
    payload["generation"][1]["generated_token_ids"][-1] = 99
    performance["candidate_first"] = _write(performance["candidate_first"], payload)

    report = gate.summarize(
        performance,
        exactness_baseline=baseline,
        exactness_candidate=candidate,
        expected_commit=COMMIT,
        expected_branch=BRANCH,
        scope="smoke",
        minimum_speedup_pct=0.0,
        full_saving_seconds=0.0,
    )

    assert report["pass"] is False
    assert any("changed exact work" in failure for failure in report["failures"])


def test_rng_offset_or_drift_change_fails(tmp_path: Path):
    performance, baseline, candidate = _paths(tmp_path)
    payload = json.loads(performance["candidate_second"].read_text())
    stats = payload["generation"][1]["optimized_cuda_graphs"]["exact_k4_candidate"]
    stats["generator_offset_increment"] = 8
    stats["rng_exact"] = False
    performance["candidate_second"] = _write(performance["candidate_second"], payload)

    report = gate.summarize(
        performance,
        exactness_baseline=baseline,
        exactness_candidate=candidate,
        expected_commit=COMMIT,
        expected_branch=BRANCH,
        scope="smoke",
        minimum_speedup_pct=0.0,
        full_saving_seconds=0.0,
    )

    assert report["pass"] is False
    assert any("RNG offset" in failure for failure in report["failures"])
    assert any("strict RNG exactness" in failure for failure in report["failures"])


def test_cache_or_rng_signature_change_fails(tmp_path: Path):
    performance, baseline, candidate = _paths(tmp_path)
    payload = json.loads(candidate.read_text())
    payload["generation"][1]["strict_exactness"]["rng_after"]["cuda_sha256"] = "c" * 64
    candidate = _write(candidate, payload)

    report = gate.summarize(
        performance,
        exactness_baseline=baseline,
        exactness_candidate=candidate,
        expected_commit=COMMIT,
        expected_branch=BRANCH,
        scope="smoke",
        minimum_speedup_pct=0.0,
        full_saving_seconds=0.0,
    )

    assert report["pass"] is False
    assert report["strict_exactness"]["pass"] is False


def test_dispatch_topology_change_fails(tmp_path: Path):
    performance, baseline, candidate = _paths(tmp_path)
    payload = json.loads(performance["candidate_second"].read_text())
    payload["generation"][1]["optimized_dispatch_capture_hits"][
        "q1_bmm_cross_attention"
    ] = 0
    performance["candidate_second"] = _write(performance["candidate_second"], payload)

    report = gate.summarize(
        performance,
        exactness_baseline=baseline,
        exactness_candidate=candidate,
        expected_commit=COMMIT,
        expected_branch=BRANCH,
        scope="smoke",
        minimum_speedup_pct=0.0,
        full_saving_seconds=0.0,
    )

    assert report["pass"] is False
    assert any("dispatch policy" in failure for failure in report["failures"])


def test_full_gate_requires_real_saving(tmp_path: Path):
    performance, baseline, candidate = _paths(tmp_path)
    report = gate.summarize(
        performance,
        exactness_baseline=baseline,
        exactness_candidate=candidate,
        expected_commit=COMMIT,
        expected_branch=BRANCH,
        scope="full",
        minimum_speedup_pct=0.0,
        full_saving_seconds=1.0,
    )

    assert report["pass"] is False
    assert any("reciprocal main saving" in failure for failure in report["failures"])
