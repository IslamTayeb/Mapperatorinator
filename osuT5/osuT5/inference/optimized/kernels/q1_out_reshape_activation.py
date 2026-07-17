"""Opt-in request for tip-exact q1 attn-out reshape (skip transpose+contiguous)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

_REQUESTED = False


def q1_out_reshape_requested() -> bool:
    return _REQUESTED


@contextmanager
def q1_out_reshape_candidate_context() -> Iterator[None]:
    """Request reshape of native q1 [1,H,1,D] → [1,1,H*D] without D2D copy."""

    global _REQUESTED
    previous = _REQUESTED
    _REQUESTED = True
    try:
        yield
    finally:
        _REQUESTED = previous


__all__ = [
    "q1_out_reshape_candidate_context",
    "q1_out_reshape_requested",
]
