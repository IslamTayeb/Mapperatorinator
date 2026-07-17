"""Opt-in active-prefix exact-length CUDA-graph decode (bucket=1).

Scheduling / CUDA-graph family reformulation from §24 evidence: raising the
bucket 64→256 cut unique captures (12→4) but regressed main_tps because pad
KV attention dominated capture savings. Candidate forces bucket_size=1 so
each decode step graphs the true prefix length (no pad), accepting more
captures to remove pad FLOPs. Tip default ACTIVE_PREFIX_BUCKET_SIZE=64 stays
cold. Not reshape/allocator/proj_out/Wo-residual/mask-workspace; not a same-
direction larger-bucket retune.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

# Tip default remains 64 in engine.py; candidate scout uses exact-length=1.
CANDIDATE_ACTIVE_PREFIX_BUCKET_SIZE = 1

_REQUESTED = False
_BUCKET: ContextVar[int | None] = ContextVar(
    "active_prefix_exact_length_override", default=None
)


def active_prefix_exact_length_requested() -> bool:
    return _REQUESTED


def active_prefix_exact_length_override() -> int | None:
    return _BUCKET.get()


def effective_active_prefix_bucket_size(default: int) -> int:
    override = _BUCKET.get()
    if override is None:
        return int(default)
    if (
        isinstance(override, bool)
        or not isinstance(override, int)
        or override <= 0
    ):
        raise ValueError("active-prefix exact-length override must be a positive int")
    return int(override)


@contextmanager
def active_prefix_exact_length_candidate_context(
    bucket_size: int = CANDIDATE_ACTIVE_PREFIX_BUCKET_SIZE,
) -> Iterator[None]:
    global _REQUESTED
    if (
        isinstance(bucket_size, bool)
        or not isinstance(bucket_size, int)
        or bucket_size <= 0
    ):
        raise ValueError("active-prefix exact-length bucket_size must be a positive int")
    previous = _REQUESTED
    token = _BUCKET.set(int(bucket_size))
    _REQUESTED = True
    try:
        yield
    finally:
        _BUCKET.reset(token)
        _REQUESTED = previous


__all__ = [
    "CANDIDATE_ACTIVE_PREFIX_BUCKET_SIZE",
    "active_prefix_exact_length_candidate_context",
    "active_prefix_exact_length_override",
    "active_prefix_exact_length_requested",
    "effective_active_prefix_bucket_size",
]
