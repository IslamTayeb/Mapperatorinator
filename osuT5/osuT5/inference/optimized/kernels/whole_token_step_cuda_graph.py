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


@contextmanager
def whole_token_step_cuda_graph_candidate_context() -> Iterator[None]:
    global _REQUESTED
    previous = _REQUESTED
    reset_whole_token_step_cuda_graph_hits()
    _REQUESTED = True
    try:
        yield
    finally:
        _REQUESTED = previous


__all__ = [
    "record_whole_token_step_cuda_graph_hit",
    "reset_whole_token_step_cuda_graph_hits",
    "whole_token_step_cuda_graph_candidate_context",
    "whole_token_step_cuda_graph_hits",
    "whole_token_step_cuda_graph_requested",
]
