"""Runtime hook installation for optimized single inference."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from typing import Iterator

import torch

from ...runtime_dispatch import (
    AttentionRuntimeHooks,
    DecoderLayerRuntimeHooks,
    attention_runtime_hooks,
    attention_runtime_hooks_context,
    decoder_layer_runtime_hooks_context,
)


_ACTIVE_PREFIX_SELF_ATTENTION_LENGTH: int | None = None


def active_prefix_self_attention_length() -> int | None:
    return _ACTIVE_PREFIX_SELF_ATTENTION_LENGTH


@contextmanager
def active_prefix_self_attention_context(
    prefix_length: int | None,
) -> Iterator[None]:
    """Own active-prefix state and install the neutral model input hook."""
    global _ACTIVE_PREFIX_SELF_ATTENTION_LENGTH

    from ..kernels.dispatch import active_prefix_attention_inputs

    previous_length = _ACTIVE_PREFIX_SELF_ATTENTION_LENGTH
    previous_hooks = attention_runtime_hooks()
    _ACTIVE_PREFIX_SELF_ATTENTION_LENGTH = prefix_length
    hooks = replace(
        previous_hooks,
        sdpa_attention_inputs=(
            active_prefix_attention_inputs
            if prefix_length is not None
            else None
        ),
    )
    try:
        with attention_runtime_hooks_context(hooks):
            yield
    finally:
        _ACTIVE_PREFIX_SELF_ATTENTION_LENGTH = previous_length


@contextmanager
def attention_runtime_context(
    *,
    q1_bmm_cross_attention: bool = False,
    native_q1_self_attention: bool = False,
    native_q1_rope_cache_self_attention: bool = False,
    skip_out_proj: bool = False,
    expected_dtype: torch.dtype = torch.float32,
    dispatch_counts: dict[str, int] | None = None,
) -> Iterator[None]:
    if expected_dtype not in {torch.float32, torch.float16}:
        raise TypeError("optimized attention supports only float32 or float16")
    from ..kernels.dispatch import (
        q1_rope_cache_self_attention_forward,
        sdpa_q1_attention_forward,
    )

    native_q1_attention = None
    native_q1_rope_cache_attention = None
    if native_q1_self_attention or native_q1_rope_cache_self_attention:
        from ..kernels import q1_attention

        q1_attention.preload_native_q1_attention()
        native_q1_attention = q1_attention.native_q1_attention
        native_q1_rope_cache_attention = (
            q1_attention.native_q1_rope_cache_attention
        )
    sdpa_attention_forward = None
    if q1_bmm_cross_attention or native_q1_self_attention:
        sdpa_attention_forward = partial(
            sdpa_q1_attention_forward,
            q1_bmm_cross_attention=q1_bmm_cross_attention,
            native_q1_self_attention=native_q1_self_attention,
            native_q1_attention=native_q1_attention,
            expected_dtype=expected_dtype,
            dispatch_counts=dispatch_counts,
        )
    q1_rope_cache_self_attention = None
    if native_q1_rope_cache_self_attention:
        q1_rope_cache_self_attention = partial(
            q1_rope_cache_self_attention_forward,
            native_q1_rope_cache_attention=(
                native_q1_rope_cache_attention
            ),
            expected_dtype=expected_dtype,
            dispatch_counts=dispatch_counts,
        )
    hooks = AttentionRuntimeHooks(
        sdpa_attention_inputs=attention_runtime_hooks().sdpa_attention_inputs,
        sdpa_attention_forward=sdpa_attention_forward,
        q1_rope_cache_self_attention_forward=(
            q1_rope_cache_self_attention
        ),
        skip_out_proj=skip_out_proj,
    )
    with attention_runtime_hooks_context(hooks):
        yield


@contextmanager
def decoder_layer_runtime_context(
    *,
    native_cross_mlp_tail: bool = False,
    native_self_wo_linear: bool = False,
    dispatch_counts: dict[str, int] | None = None,
) -> Iterator[None]:
    cross_mlp_tail_forward = None
    self_wo_linear_forward = None
    if native_cross_mlp_tail or native_self_wo_linear:
        from ..kernels import decoder_layer

        decoder_layer.preload_native_decoder_layer()
    if native_cross_mlp_tail:
        from ..kernels.cross_mlp import native_cross_mlp_tail_forward

        cross_mlp_tail_forward = partial(
            native_cross_mlp_tail_forward,
            dispatch_counts=dispatch_counts,
        )
    if native_self_wo_linear:
        from ..scout.self_wo_linear import native_self_wo_linear_forward

        self_wo_linear_forward = partial(
            native_self_wo_linear_forward,
            dispatch_counts=dispatch_counts,
        )
    hooks = DecoderLayerRuntimeHooks(
        cross_mlp_tail_forward=cross_mlp_tail_forward,
        self_wo_linear_forward=self_wo_linear_forward,
    )
    with decoder_layer_runtime_hooks_context(hooks):
        yield
