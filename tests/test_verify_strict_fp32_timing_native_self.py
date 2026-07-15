from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from utils.verify_strict_fp32_timing_native_self import (
    TimingNativeSelfError,
    verify,
)


HASH = "a" * 64


def _exactness(output_tokens: int) -> dict:
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
            "cross_sequence_length": 128,
            "layer_count": 12,
            "aggregate_sha256": HASH,
            "layers": layers,
        },
    }


def _policy(*, timing: bool, candidate: bool) -> dict:
    self_enabled = not timing or candidate
    policy = {
        "q1_bmm_cross_attention": {
            "requested": True,
            "enabled": True,
            "disabled_reason": None,
        },
        "native_q1_self_attention": {
            "requested": True,
            "enabled": self_enabled,
            "disabled_reason": None if self_enabled else "timing_context",
        },
        "native_q1_rope_cache_self_attention": {
            "requested": True,
            "enabled": self_enabled,
            "disabled_reason": None if self_enabled else "timing_context",
        },
        "native_cross_mlp_tail": {
            "requested": True,
            "enabled": not timing,
            "disabled_reason": "timing_context" if timing else None,
        },
    }
    if timing and candidate:
        policy["strict_fp32_timing_native_self"] = {
            "requested": True,
            "enabled": True,
            "disabled_reason": None,
            "result_class": "exact-incremental-candidate",
            "exactness_claim": True,
        }
    return policy


def _candidate_metadata() -> dict:
    return {
        "version": "strict-fp32-timing-native-self-v1",
        "scope": "standalone-timing-model-batch1-only",
        "precision": "fp32",
        "torch_dtype": "torch.float32",
        "tf32_disabled_required": True,
        "result_class": "exact-incremental-candidate",
        "exactness_claim": True,
        "native_q1_self_attention": True,
        "native_q1_rope_cache_self_attention": True,
        "q1_bmm_cross_attention_retained": True,
        "native_cross_mlp_tail_retained_disabled": True,
        "original_decoder_forward_retained": True,
        "accepted_kernel_implementation_reused": True,
        "owns_model_weights": False,
        "reduced_precision_weights": False,
        "reduced_precision_activations": False,
        "counter_rng": False,
        "production_selector_unchanged": True,
    }


def _record(label: str, tokens: list[int], *, candidate: bool) -> dict:
    timing = label == "timing_context"
    record = {
        "profile_label": label,
        "context_type": "TIMING" if timing else "MAP",
        "mode": "sequential",
        "sequence_index": 0,
        "precision": "fp32",
        "generated_tokens": len(tokens),
        "generated_tokens_per_sample": [len(tokens)],
        "generated_token_ids": tokens,
        "output_tokens_per_sample": [len(tokens) + 1],
        "model_elapsed_seconds": 0.8 if timing and candidate else 1.0,
        "optimized_effective_config_version": "accepted-fp32-v1",
        "optimized_dispatch_mode": (
            "strict_fp32_timing_native_self_batch1"
            if timing and candidate
            else "accepted_batch1"
        ),
        "optimized_dispatch_policy": _policy(timing=timing, candidate=candidate),
        "optimized_dispatch_capture_hits": {
            "native_q1_rope_cache_self_attention": (
                12 if (not timing or candidate) else 0
            ),
            "native_q1_self_attention": 0,
            "q1_bmm_cross_attention": 12,
            "native_cross_mlp_tail": 0 if timing else 12,
        },
        "strict_exactness": _exactness(len(tokens) + 1),
    }
    if timing and candidate:
        record["optimized_strict_fp32_timing_native_self"] = _candidate_metadata()
    return record


def _profile(*, candidate: bool) -> dict:
    return {
        "metadata": {
            "precision": "fp32",
            "attn_implementation": "sdpa",
            "inference_engine": "optimized",
            "use_server": False,
            "parallel": False,
            "num_beams": 1,
            "cfg_scale": 1.0,
            "profile_pass_kind": "exactness_audit",
            "authoritative_performance": False,
            "strict_exactness_evidence": True,
            "float32_matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "nvidia_tf32_override": "0",
            "cuda_device_name": "NVIDIA GeForce RTX 2080 Ti",
            "cuda_device_capability": [7, 5],
            "optimized_effective_config": {
                "version": "accepted-fp32-v1",
                "precision": "fp32",
                "native_q1_self_attention": True,
            },
        },
        "generation": [
            _record("timing_context", [3, 4], candidate=candidate),
            _record("main_generation", [5, 6, 7], candidate=False),
        ],
    }


def _initialization() -> dict:
    return {
        "schema_version": 1,
        "runner_version": "strict-fp32-timing-native-self-runner-v1",
        "timing_native_self": _candidate_metadata(),
        "model_loads": {
            "count": 2,
            "main_auto_select_gamemode_model": True,
            "timing_auto_select_gamemode_model": False,
            "owners_distinct": True,
        },
        "strict_fp32": {
            "NVIDIA_TF32_OVERRIDE": "0",
            "float32_matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
    }


def test_verifier_requires_exact_tokens_rng_cache_dispatch_and_osu(tmp_path) -> None:
    control_result = tmp_path / "control.osu"
    candidate_result = tmp_path / "candidate.osu"
    payload = b"osu file format v14\n"
    control_result.write_bytes(payload)
    candidate_result.write_bytes(payload)

    result = verify(
        _profile(candidate=False),
        _profile(candidate=True),
        _initialization(),
        control_result=control_result,
        candidate_result=candidate_result,
    )

    assert result["pass"] is True
    assert result["strict_fp32"] is True
    assert result["reduced_precision"] is False
    assert result["counter_rng"] is False
    assert result["dispatch"]["timing_context"]["control_native_self"] == 0
    assert result["dispatch"]["timing_context"]["candidate_native_self"] == 12
    assert result["timing_model_elapsed_seconds"]["speedup_pct"] == pytest.approx(20.0)
    assert result["timing_model_elapsed_seconds"]["authoritative"] is False
    assert result["final_osu_sha256"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("failure", ("tokens", "cache", "dispatch", "osu", "tf32"))
def test_verifier_fails_loudly_on_parity_or_policy_drift(tmp_path, failure) -> None:
    control = _profile(candidate=False)
    candidate = _profile(candidate=True)
    control_result = tmp_path / "control.osu"
    candidate_result = tmp_path / "candidate.osu"
    control_result.write_bytes(b"same")
    candidate_result.write_bytes(b"same")
    if failure == "tokens":
        candidate["generation"][0]["generated_token_ids"][0] = 99
    elif failure == "cache":
        candidate["generation"][0]["strict_exactness"]["cache_writes"][
            "aggregate_sha256"
        ] = "b" * 64
    elif failure == "dispatch":
        candidate["generation"][0]["optimized_dispatch_capture_hits"][
            "native_q1_rope_cache_self_attention"
        ] = 0
    elif failure == "osu":
        candidate_result.write_bytes(b"different")
    elif failure == "tf32":
        candidate["metadata"]["cuda_matmul_allow_tf32"] = True

    with pytest.raises(TimingNativeSelfError):
        verify(
            control,
            candidate,
            _initialization(),
            control_result=control_result,
            candidate_result=candidate_result,
        )
