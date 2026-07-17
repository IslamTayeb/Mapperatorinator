"""Opt-in persistent sampling-tail CUDA graph (W-BASE / §46).

Graphs fixed-shape temperature + top-p + softmax + multinomial on
session-static workspaces. Capture is warmup only: Philox generator state is
registered and RNG is restored so the first real sample is always a replay
(§29b Philox pattern). Forward graphs stay separate; sample graphs persist in
the session ``shared_graph_cache`` across windows.

V32 cold default remains unchanged. Not a bare §29 whole-token retry.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

_REQUESTED = False
_HITS = 0


def sampling_tail_cuda_graph_requested() -> bool:
    return _REQUESTED


def sampling_tail_cuda_graph_hits() -> int:
    return _HITS


def record_sampling_tail_cuda_graph_hit() -> None:
    global _HITS
    _HITS += 1


def reset_sampling_tail_cuda_graph_hits() -> None:
    global _HITS
    _HITS = 0


@contextmanager
def sampling_tail_cuda_graph_candidate_context() -> Iterator[None]:
    global _REQUESTED
    previous = _REQUESTED
    reset_sampling_tail_cuda_graph_hits()
    _REQUESTED = True
    try:
        yield
    finally:
        _REQUESTED = previous


__all__ = [
    "record_sampling_tail_cuda_graph_hit",
    "reset_sampling_tail_cuda_graph_hits",
    "sampling_tail_cuda_graph_candidate_context",
    "sampling_tail_cuda_graph_hits",
    "sampling_tail_cuda_graph_requested",
]
