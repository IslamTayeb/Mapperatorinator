"""Opt-in decode cast/copy elimination (cold-import safe)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


_DECODE_CAST_COPY_REQUESTED = False


def decode_cast_copy_requested() -> bool:
    return _DECODE_CAST_COPY_REQUESTED


@contextmanager
def decode_cast_copy_candidate_context() -> Iterator[None]:
    global _DECODE_CAST_COPY_REQUESTED
    previous = _DECODE_CAST_COPY_REQUESTED
    _DECODE_CAST_COPY_REQUESTED = True
    try:
        yield
    finally:
        _DECODE_CAST_COPY_REQUESTED = previous


__all__ = [
    "decode_cast_copy_candidate_context",
    "decode_cast_copy_requested",
]
