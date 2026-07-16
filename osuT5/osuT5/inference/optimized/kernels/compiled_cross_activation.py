"""Opt-in request flag for exact compiled q1 cross BMM (cold-import safe)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


_COMPILED_CROSS_REQUESTED = False


def compiled_cross_requested() -> bool:
    return _COMPILED_CROSS_REQUESTED


@contextmanager
def compiled_cross_candidate_context() -> Iterator[None]:
    """Request owned compile-before-capture for the exact optimized path."""

    global _COMPILED_CROSS_REQUESTED
    previous = _COMPILED_CROSS_REQUESTED
    _COMPILED_CROSS_REQUESTED = True
    try:
        yield
    finally:
        _COMPILED_CROSS_REQUESTED = previous


__all__ = [
    "compiled_cross_candidate_context",
    "compiled_cross_requested",
]
