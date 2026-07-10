"""Runtime hook installation for optimized single inference."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from ...runtime_dispatch import (
    AttentionRuntimeHooks,
    attention_runtime_hooks_context,
)


@contextmanager
def native_attention_runtime_context() -> Iterator[None]:
    from ..kernels.q1_attention import (
        native_q1_attention,
        native_q1_rope_cache_attention,
        preload_native_q1_attention,
    )

    preload_native_q1_attention()
    hooks = AttentionRuntimeHooks(
        native_q1_attention=native_q1_attention,
        native_q1_rope_cache_attention=native_q1_rope_cache_attention,
    )
    with attention_runtime_hooks_context(hooks):
        yield
