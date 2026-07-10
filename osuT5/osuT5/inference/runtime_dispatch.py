"""Neutral runtime hooks shared by model code and opt-in inference engines."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator


@dataclass(frozen=True)
class AttentionRuntimeHooks:
    sdpa_attention_forward: Callable[..., Any] | None = None
    q1_rope_cache_self_attention_forward: Callable[..., Any] | None = None


_EMPTY_ATTENTION_RUNTIME_HOOKS = AttentionRuntimeHooks()
_ATTENTION_RUNTIME_HOOKS = _EMPTY_ATTENTION_RUNTIME_HOOKS


def attention_runtime_hooks() -> AttentionRuntimeHooks:
    return _ATTENTION_RUNTIME_HOOKS


@contextmanager
def attention_runtime_hooks_context(
    hooks: AttentionRuntimeHooks,
) -> Iterator[None]:
    global _ATTENTION_RUNTIME_HOOKS

    previous = _ATTENTION_RUNTIME_HOOKS
    _ATTENTION_RUNTIME_HOOKS = hooks
    try:
        yield
    finally:
        _ATTENTION_RUNTIME_HOOKS = previous
