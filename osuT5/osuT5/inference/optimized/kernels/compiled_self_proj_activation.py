"""Opt-in request for decode-only compiled self Wqkv/Wo GEMVs (cold-import safe)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


_REQUESTED = False


def compiled_self_proj_requested() -> bool:
    return _REQUESTED


def compiled_self_proj_decode_only_requested() -> bool:
    """Alias: §14 confines compile to fixed (1,1) decode shapes."""

    return _REQUESTED


@contextmanager
def compiled_self_proj_decode_only_candidate_context() -> Iterator[None]:
    """Request owned compile-before-capture for tip self-attn Wqkv/Wo decode GEMVs only."""

    global _REQUESTED
    previous = _REQUESTED
    _REQUESTED = True
    try:
        yield
    finally:
        _REQUESTED = previous


# Back-compat alias used by §13 scout scripts; same request flag.
compiled_self_proj_candidate_context = (
    compiled_self_proj_decode_only_candidate_context
)


__all__ = [
    "compiled_self_proj_candidate_context",
    "compiled_self_proj_decode_only_candidate_context",
    "compiled_self_proj_decode_only_requested",
    "compiled_self_proj_requested",
]
