"""Runtime hook installation for optimized single inference."""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from typing import Iterator

from ...runtime_dispatch import (
    AttentionRuntimeHooks,
    attention_runtime_hooks_context,
)


@contextmanager
def attention_runtime_context(
    *,
    q1_bmm_cross_attention: bool,
    native_q1_self_attention: bool,
    native_q1_rope_cache_self_attention: bool,
) -> Iterator[None]:
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
        )
    q1_rope_cache_self_attention = None
    if native_q1_rope_cache_self_attention:
        q1_rope_cache_self_attention = partial(
            q1_rope_cache_self_attention_forward,
            native_q1_rope_cache_attention=(
                native_q1_rope_cache_attention
            ),
        )
    hooks = AttentionRuntimeHooks(
        sdpa_attention_forward=sdpa_attention_forward,
        q1_rope_cache_self_attention_forward=(
            q1_rope_cache_self_attention
        ),
    )
    with attention_runtime_hooks_context(hooks):
        yield
