"""Default-cold compatibility adapter for the accepted legacy fast stack."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LEGACY_DEFAULTS = {
    "inference_active_prefix_decode_loop": False,
    "inference_active_prefix_decode_bucket_size": 128,
    "inference_active_prefix_decode_cuda_graph": False,
    "inference_active_prefix_decode_cuda_graph_warmup": 0,
    "inference_active_prefix_decode_cuda_graph_min_decode_steps": 1,
    "inference_stateful_monotonic_logits_processor": False,
    "inference_q1_bmm_cross_attention": False,
    "inference_decode_session_runtime": False,
    "inference_decode_session_cuda_graph": False,
    "inference_decode_session_chunk_size": 1,
    "inference_native_decode_kernels": False,
    "inference_native_q1_self_attention": False,
    "inference_native_q1_rope_cache_self_attention": False,
}

_ACCEPTED_LEGACY_STACK = {
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
}


def legacy_optimized_single_requested(args: Any) -> bool:
    return any(
        getattr(args, name) != default
        for name, default in _LEGACY_DEFAULTS.items()
    )


def validate_legacy_optimized_single(args: Any) -> bool:
    if getattr(args, "inference_engine", "v32") != "v32":
        return False
    if not legacy_optimized_single_requested(args):
        return False

    mismatches = [
        name
        for name, expected in _ACCEPTED_LEGACY_STACK.items()
        if getattr(args, name) != expected
    ]
    if mismatches:
        raise ValueError(
            "Legacy optimized single flags must reproduce the complete accepted "
            "270.475 tok/s bundle or migrate to inference_engine=optimized "
            "optimized_inference_mode=single. Mismatched flags: "
            + ", ".join(mismatches)
        )
    return True


def load_legacy_optimized_single_runtime(args: Any):
    if not validate_legacy_optimized_single(args):
        return None
    engine = import_module(
        "osuT5.osuT5.inference.optimized.single.engine"
    )
    return engine.OptimizedSingleRuntime()


def legacy_optimized_single_profile_metadata(args: Any) -> dict[str, Any]:
    runtime = load_legacy_optimized_single_runtime(args)
    if runtime is None:
        return {}
    metadata = runtime.profile_metadata()
    metadata["optimized_legacy_delegation"] = True
    return metadata
