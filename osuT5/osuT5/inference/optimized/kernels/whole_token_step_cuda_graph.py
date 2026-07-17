"""Opt-in whole-token-step CUDA graph capture (structure lever).

Tip today graphs only the decoder forward, then runs logits / top-p /
softmax / multinomial / append / EOS on the host side of each token.
Candidate expands the replayed region to include those post-forward
steps against preallocated buffers (index_copy_ / static workspaces)
and avoids per-token ``torch.cuda.synchronize``.

V32 cold default remains unchanged. Exactness still required for
graduate. Speculative / WHILE / occupancy levers are separate sections.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

_REQUESTED = False
_HITS = 0
_ACTIVE_LENGTH_TENSOR = None


def whole_token_step_cuda_graph_requested() -> bool:
    return _REQUESTED


def whole_token_step_cuda_graph_hits() -> int:
    return _HITS


def record_whole_token_step_cuda_graph_hit() -> None:
    global _HITS
    _HITS += 1


def reset_whole_token_step_cuda_graph_hits() -> None:
    global _HITS
    _HITS = 0


def whole_token_active_length_tensor():
    """Optional ``[1]`` int64 length tensor for capture-safe processor scans."""

    return _ACTIVE_LENGTH_TENSOR


def set_whole_token_active_length_tensor(length_tensor) -> None:
    global _ACTIVE_LENGTH_TENSOR
    _ACTIVE_LENGTH_TENSOR = length_tensor


@contextmanager
def whole_token_step_cuda_graph_candidate_context() -> Iterator[None]:
    global _REQUESTED
    previous = _REQUESTED
    previous_length = _ACTIVE_LENGTH_TENSOR
    reset_whole_token_step_cuda_graph_hits()
    set_whole_token_active_length_tensor(None)
    _REQUESTED = True
    try:
        yield
    finally:
        _REQUESTED = previous
        set_whole_token_active_length_tensor(previous_length)


__all__ = [
    "record_whole_token_step_cuda_graph_hit",
    "reset_whole_token_step_cuda_graph_hits",
    "set_whole_token_active_length_tensor",
    "whole_token_active_length_tensor",
    "whole_token_step_cuda_graph_candidate_context",
    "whole_token_step_cuda_graph_hits",
    "whole_token_step_cuda_graph_requested",
]
