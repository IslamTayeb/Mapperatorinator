from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Iterator

import torch
from torch.profiler import record_function


_DETAIL_RANGES_ENABLED = False

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


@contextmanager
def generation_profile_context(
        *,
        detail_ranges: bool = False,
        sdpa_backend: str | None = None,
        active_prefix_self_attention_length: int | None = None,
        q1_bmm_cross_attention: bool = False,
        native_q1_self_attention: bool = False,
        native_q1_rope_cache_self_attention: bool = False,
        native_cross_mlp_tail: bool = False,
) -> Iterator[None]:
    """Temporarily enable opt-in generation profiling controls."""
    global _DETAIL_RANGES_ENABLED

    previous_detail_ranges = _DETAIL_RANGES_ENABLED
    _DETAIL_RANGES_ENABLED = bool(detail_ranges)
    try:
        from osuT5.osuT5.inference.runtime_dispatch import (
            AttentionRuntimeHooks,
            DecoderLayerRuntimeHooks,
            attention_runtime_hooks_context,
            decoder_layer_runtime_hooks_context,
        )

        optimized_attention_context = attention_runtime_hooks_context(
            AttentionRuntimeHooks()
        )
        if (
            q1_bmm_cross_attention
            or native_q1_self_attention
            or native_q1_rope_cache_self_attention
        ):
            from osuT5.osuT5.inference.optimized.single.runtime_context import (
                attention_runtime_context,
            )

            optimized_attention_context = attention_runtime_context(
                q1_bmm_cross_attention=q1_bmm_cross_attention,
                native_q1_self_attention=native_q1_self_attention,
                native_q1_rope_cache_self_attention=(
                    native_q1_rope_cache_self_attention
                ),
            )
        optimized_decoder_layer_context = decoder_layer_runtime_hooks_context(
            DecoderLayerRuntimeHooks()
        )
        if native_cross_mlp_tail:
            from osuT5.osuT5.inference.optimized.single.runtime_context import (
                decoder_layer_runtime_context,
            )

            optimized_decoder_layer_context = decoder_layer_runtime_context(
                native_cross_mlp_tail=True,
            )
        optimized_active_prefix_context = nullcontext()
        if (
            active_prefix_self_attention_length is not None
            or q1_bmm_cross_attention
            or native_q1_self_attention
            or native_q1_rope_cache_self_attention
        ):
            from osuT5.osuT5.inference.optimized.single.runtime_context import (
                active_prefix_self_attention_context as optimized_prefix_context,
            )

            optimized_active_prefix_context = optimized_prefix_context(
                active_prefix_self_attention_length
            )
        with (
            optimized_attention_context,
            optimized_decoder_layer_context,
            optimized_active_prefix_context,
            sdpa_backend_context(sdpa_backend),
        ):
            yield
    finally:
        _DETAIL_RANGES_ENABLED = previous_detail_ranges


@contextmanager
def active_prefix_self_attention_context(prefix_length: int | None) -> Iterator[None]:
    """Lazy compatibility adapter for optimized active-prefix state."""
    from osuT5.osuT5.inference.optimized.single.runtime_context import (
        active_prefix_self_attention_context as optimized_prefix_context,
    )

    with optimized_prefix_context(prefix_length):
        yield


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
