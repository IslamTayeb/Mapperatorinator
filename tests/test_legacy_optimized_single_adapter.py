from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from config import InferenceConfig
from inference import validate_reserved_runtime_flags
from osuT5.osuT5.inference import server
from osuT5.osuT5.inference.legacy_single_adapter import (
    legacy_optimized_single_requested,
    load_legacy_optimized_single_runtime,
    validate_legacy_optimized_single,
)
from osuT5.osuT5.inference.optimized.single.engine import OptimizedSingleRuntime


def _canonical_args(**overrides):
    values = {
        "device": "cuda",
        "precision": "fp32",
        "attn_implementation": "sdpa",
        "inference_engine": "v32",
        "inference_generation_compile": True,
        "inference_active_prefix_decode_loop": True,
        "inference_active_prefix_decode_bucket_size": 64,
        "inference_active_prefix_decode_cuda_graph": True,
        "inference_active_prefix_decode_cuda_graph_warmup": 0,
        "inference_active_prefix_decode_cuda_graph_min_decode_steps": 1,
        "inference_stateful_monotonic_logits_processor": True,
        "inference_q1_bmm_cross_attention": True,
        "inference_decode_session_runtime": True,
        "inference_decode_session_cuda_graph": True,
        "inference_decode_session_chunk_size": 1,
        "inference_native_decode_kernels": True,
        "inference_native_q1_self_attention": True,
        "inference_native_q1_rope_cache_self_attention": True,
        "cfg_scale": 1.0,
        "num_beams": 1,
        "parallel": False,
        "use_server": False,
    }
    values.update(overrides)
    return InferenceConfig(**values)


def test_canonical_legacy_bundle_resolves_to_same_runtime():
    args = _canonical_args()

    assert legacy_optimized_single_requested(args) is True
    assert validate_legacy_optimized_single(args) is True
    validate_reserved_runtime_flags(args)
    runtime = load_legacy_optimized_single_runtime(args)

    assert isinstance(runtime, OptimizedSingleRuntime)
    assert runtime.config.version == "accepted-fp32-270.475-v1"


def test_partial_legacy_bundle_fails_with_migration_instruction():
    args = _canonical_args(inference_native_q1_rope_cache_self_attention=False)

    with pytest.raises(ValueError, match="complete accepted 270.475 tok/s bundle"):
        validate_reserved_runtime_flags(args)


def test_compile_only_v32_does_not_select_optimized_runtime():
    args = InferenceConfig(inference_generation_compile=True)

    assert legacy_optimized_single_requested(args) is False
    assert validate_legacy_optimized_single(args) is False
    assert load_legacy_optimized_single_runtime(args) is None


def test_fresh_compile_only_v32_adapter_is_optimized_cold():
    repo_root = Path(__file__).resolve().parents[1]
    code = r'''\
import sys
from config import InferenceConfig
from osuT5.osuT5.inference.legacy_single_adapter import load_legacy_optimized_single_runtime

args = InferenceConfig(inference_generation_compile=True)
assert load_legacy_optimized_single_runtime(args) is None
assert not any(name.startswith("osuT5.osuT5.inference.optimized") for name in sys.modules)
assert "torch.utils.cpp_extension" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_server_source_contains_no_optimized_runtime_implementation():
    source = inspect.getsource(server)

    assert ".optimized" not in source
    assert "active_prefix_decode_generate" not in source
    assert "cache_for_window" not in source
    assert "shared_graph_cache" not in source
    assert "stable_encoder_holder" not in source
    assert "native_q1_self_attention_requested =" not in source
    assert "q1_bmm_cross_attention=" not in source
