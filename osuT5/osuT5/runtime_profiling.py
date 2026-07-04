from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch.profiler import record_function


_DETAIL_RANGES_ENABLED = False
_ACTIVE_PREFIX_SELF_ATTENTION_LENGTH: int | None = None
_Q1_BMM_CROSS_ATTENTION_ENABLED = False
_NATIVE_Q1_SELF_ATTENTION_ENABLED = False
_NATIVE_Q1_ROPE_CACHE_SELF_ATTENTION_ENABLED = False
_NATIVE_DECODER_LAYER_MLP_TAIL_ENABLED = False
_NATIVE_DECODER_LAYER_MLP_TAIL_OUTPUTS_PER_BLOCK = 4

_SDPA_BACKEND_ALIASES = {
    "flash": "FLASH_ATTENTION",
    "flash_attention": "FLASH_ATTENTION",
    "efficient": "EFFICIENT_ATTENTION",
    "efficient_attention": "EFFICIENT_ATTENTION",
    "mem_efficient": "EFFICIENT_ATTENTION",
    "memory_efficient": "EFFICIENT_ATTENTION",
    "math": "MATH",
    "cudnn": "CUDNN_ATTENTION",
    "cudnn_attention": "CUDNN_ATTENTION",
}


def detail_ranges_enabled() -> bool:
    return _DETAIL_RANGES_ENABLED


def active_prefix_self_attention_length() -> int | None:
    return _ACTIVE_PREFIX_SELF_ATTENTION_LENGTH


def q1_bmm_cross_attention_enabled() -> bool:
    return _Q1_BMM_CROSS_ATTENTION_ENABLED


def native_q1_self_attention_enabled() -> bool:
    return _NATIVE_Q1_SELF_ATTENTION_ENABLED


def native_q1_rope_cache_self_attention_enabled() -> bool:
    return _NATIVE_Q1_ROPE_CACHE_SELF_ATTENTION_ENABLED


def native_decoder_layer_mlp_tail_enabled() -> bool:
    return _NATIVE_DECODER_LAYER_MLP_TAIL_ENABLED


def native_decoder_layer_mlp_tail_outputs_per_block() -> int:
    return _NATIVE_DECODER_LAYER_MLP_TAIL_OUTPUTS_PER_BLOCK


@contextmanager
def generation_profile_context(
        *,
        detail_ranges: bool = False,
        sdpa_backend: str | None = None,
        active_prefix_self_attention_length: int | None = None,
        q1_bmm_cross_attention: bool = False,
        native_q1_self_attention: bool = False,
        native_q1_rope_cache_self_attention: bool = False,
        native_decoder_layer_mlp_tail: bool = False,
        native_decoder_layer_mlp_tail_outputs_per_block: int = 4,
) -> Iterator[None]:
    """Temporarily enable opt-in generation profiling controls."""
    global _DETAIL_RANGES_ENABLED, _ACTIVE_PREFIX_SELF_ATTENTION_LENGTH
    global _Q1_BMM_CROSS_ATTENTION_ENABLED, _NATIVE_Q1_SELF_ATTENTION_ENABLED
    global _NATIVE_Q1_ROPE_CACHE_SELF_ATTENTION_ENABLED
    global _NATIVE_DECODER_LAYER_MLP_TAIL_ENABLED, _NATIVE_DECODER_LAYER_MLP_TAIL_OUTPUTS_PER_BLOCK

    previous_detail_ranges = _DETAIL_RANGES_ENABLED
    previous_active_prefix_length = _ACTIVE_PREFIX_SELF_ATTENTION_LENGTH
    previous_q1_bmm_cross_attention = _Q1_BMM_CROSS_ATTENTION_ENABLED
    previous_native_q1_self_attention = _NATIVE_Q1_SELF_ATTENTION_ENABLED
    previous_native_q1_rope_cache_self_attention = _NATIVE_Q1_ROPE_CACHE_SELF_ATTENTION_ENABLED
    previous_native_decoder_layer_mlp_tail = _NATIVE_DECODER_LAYER_MLP_TAIL_ENABLED
    previous_native_decoder_layer_mlp_tail_outputs_per_block = _NATIVE_DECODER_LAYER_MLP_TAIL_OUTPUTS_PER_BLOCK
    _DETAIL_RANGES_ENABLED = bool(detail_ranges)
    _ACTIVE_PREFIX_SELF_ATTENTION_LENGTH = active_prefix_self_attention_length
    _Q1_BMM_CROSS_ATTENTION_ENABLED = bool(q1_bmm_cross_attention)
    _NATIVE_Q1_SELF_ATTENTION_ENABLED = bool(native_q1_self_attention)
    _NATIVE_Q1_ROPE_CACHE_SELF_ATTENTION_ENABLED = bool(native_q1_rope_cache_self_attention)
    _NATIVE_DECODER_LAYER_MLP_TAIL_ENABLED = bool(native_decoder_layer_mlp_tail)
    _NATIVE_DECODER_LAYER_MLP_TAIL_OUTPUTS_PER_BLOCK = int(native_decoder_layer_mlp_tail_outputs_per_block)
    try:
        if native_q1_self_attention or native_q1_rope_cache_self_attention:
            from osuT5.osuT5.inference.native_q1_attention import preload_native_q1_attention

            preload_native_q1_attention()
        if native_decoder_layer_mlp_tail:
            from osuT5.osuT5.inference.native_decoder_layer import preload_native_decoder_layer

            preload_native_decoder_layer()
        with sdpa_backend_context(sdpa_backend):
            yield
    finally:
        _DETAIL_RANGES_ENABLED = previous_detail_ranges
        _ACTIVE_PREFIX_SELF_ATTENTION_LENGTH = previous_active_prefix_length
        _Q1_BMM_CROSS_ATTENTION_ENABLED = previous_q1_bmm_cross_attention
        _NATIVE_Q1_SELF_ATTENTION_ENABLED = previous_native_q1_self_attention
        _NATIVE_Q1_ROPE_CACHE_SELF_ATTENTION_ENABLED = previous_native_q1_rope_cache_self_attention
        _NATIVE_DECODER_LAYER_MLP_TAIL_ENABLED = previous_native_decoder_layer_mlp_tail
        _NATIVE_DECODER_LAYER_MLP_TAIL_OUTPUTS_PER_BLOCK = previous_native_decoder_layer_mlp_tail_outputs_per_block


@contextmanager
def active_prefix_self_attention_context(prefix_length: int | None) -> Iterator[None]:
    """Temporarily trim SDPA self-attention K/V and mask to a fixed active prefix length."""
    global _ACTIVE_PREFIX_SELF_ATTENTION_LENGTH

    previous_active_prefix_length = _ACTIVE_PREFIX_SELF_ATTENTION_LENGTH
    _ACTIVE_PREFIX_SELF_ATTENTION_LENGTH = prefix_length
    try:
        yield
    finally:
        _ACTIVE_PREFIX_SELF_ATTENTION_LENGTH = previous_active_prefix_length


@contextmanager
def sdpa_backend_context(requested_backend: str | None) -> Iterator[None]:
    if not requested_backend:
        yield
        return

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError as exc:
        raise RuntimeError("This PyTorch build does not expose torch.nn.attention.sdpa_kernel") from exc

    backend_key = requested_backend.strip().lower()
    backend_name = _SDPA_BACKEND_ALIASES.get(backend_key)
    if backend_name is None:
        valid = ", ".join(sorted(_SDPA_BACKEND_ALIASES))
        raise ValueError(f"Unknown SDPA backend '{requested_backend}'. Expected one of: {valid}")

    try:
        backend = getattr(SDPBackend, backend_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"PyTorch SDPBackend does not expose {backend_name}; requested '{requested_backend}'"
        ) from exc

    with sdpa_kernel([backend]):
        yield


@contextmanager
def profile_range(name: str) -> Iterator[None]:
    if not _DETAIL_RANGES_ENABLED:
        yield
        return

    range_name = f"mapperatorinator.{name}"
    pushed = False
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(range_name)
        pushed = True
    with record_function(range_name):
        try:
            yield
        finally:
            if pushed:
                torch.cuda.nvtx.range_pop()
