"""Opt-in request for tip-exact decode logits workspace + compiled softmax."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_REQUESTED = False
_ACTIVE_FINALIZE: ContextVar[object | None] = ContextVar(
    "compiled_decode_logits_finalize", default=None
)


def compiled_decode_logits_finalize_requested() -> bool:
    return _REQUESTED


def active_decode_logits_finalize() -> object | None:
    return _ACTIVE_FINALIZE.get()


@contextmanager
def compiled_decode_logits_finalize_candidate_context() -> Iterator[None]:
    """Request owned logits workspace + compiled softmax on tip decode."""

    global _REQUESTED
    previous = _REQUESTED
    _REQUESTED = True
    try:
        yield
    finally:
        _REQUESTED = previous


@contextmanager
def decode_logits_finalize_activation_context(finalize: object | None) -> Iterator[None]:
    token = _ACTIVE_FINALIZE.set(finalize)
    try:
        yield
    finally:
        _ACTIVE_FINALIZE.reset(token)


__all__ = [
    "active_decode_logits_finalize",
    "compiled_decode_logits_finalize_candidate_context",
    "compiled_decode_logits_finalize_requested",
    "decode_logits_finalize_activation_context",
]
