"""Opt-in request flag for exact compiled self Wqkv/Wo GEMVs (cold-import safe)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


_REQUESTED = False


def compiled_self_proj_requested() -> bool:
    return _REQUESTED


@contextmanager
def compiled_self_proj_candidate_context() -> Iterator[None]:
    """Request owned compile-before-capture for tip self-attn Wqkv/Wo."""

    global _REQUESTED
    previous = _REQUESTED
    _REQUESTED = True
    try:
        yield
    finally:
        _REQUESTED = previous


__all__ = [
    "compiled_self_proj_candidate_context",
    "compiled_self_proj_requested",
]
