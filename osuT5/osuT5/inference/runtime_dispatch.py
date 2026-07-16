"""Neutral runtime hooks shared by model code and opt-in inference engines."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator


@dataclass(frozen=True)
class AttentionRuntimeHooks:
    sdpa_attention_inputs: Callable[..., Any] | None = None
    sdpa_attention_forward: Callable[..., Any] | None = None
    q1_rope_cache_self_attention_forward: Callable[..., Any] | None = None
    compiled_out_proj: Callable[..., Any] | None = None


@dataclass(frozen=True)
class DecoderLayerRuntimeHooks:
    cross_mlp_tail_forward: Callable[..., Any] | None = None


_EMPTY_ATTENTION_RUNTIME_HOOKS = AttentionRuntimeHooks()
_ATTENTION_RUNTIME_HOOKS = _EMPTY_ATTENTION_RUNTIME_HOOKS
_EMPTY_DECODER_LAYER_RUNTIME_HOOKS = DecoderLayerRuntimeHooks()
_DECODER_LAYER_RUNTIME_HOOKS = _EMPTY_DECODER_LAYER_RUNTIME_HOOKS


def attention_runtime_hooks() -> AttentionRuntimeHooks:
    return _ATTENTION_RUNTIME_HOOKS


def decoder_layer_runtime_hooks() -> DecoderLayerRuntimeHooks:
    return _DECODER_LAYER_RUNTIME_HOOKS


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


@contextmanager
def decoder_layer_runtime_hooks_context(
    hooks: DecoderLayerRuntimeHooks,
) -> Iterator[None]:
    global _DECODER_LAYER_RUNTIME_HOOKS

    previous = _DECODER_LAYER_RUNTIME_HOOKS
    _DECODER_LAYER_RUNTIME_HOOKS = hooks
    try:
        yield
    finally:
        _DECODER_LAYER_RUNTIME_HOOKS = previous
