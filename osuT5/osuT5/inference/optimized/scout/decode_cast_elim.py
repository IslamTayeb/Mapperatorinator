"""Opt-in decode cast/copy elimination for exact FP16/FP32 tip scouts.

Importing this module does not change production dispatch. The candidate
context patches ``engine.active_prefix_decode_generate`` to reuse a static
FP32 logits workspace (avoids per-token ``.to(copy=True)`` allocation) and
sets a process-local flag so q1 attention paths prefer zero-copy ``reshape``
over ``transpose().contiguous().view`` when layouts allow.

No INT8 paths. Graph static ``copy_`` traffic is intentionally untouched.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Iterator

import torch


_REUSE_LOGITS_WORKSPACE = False
_ZERO_COPY_ATTN_LAYOUT = False


def reuse_logits_workspace_enabled() -> bool:
    return _REUSE_LOGITS_WORKSPACE


def zero_copy_attn_layout_enabled() -> bool:
    return _ZERO_COPY_ATTN_LAYOUT


def materialize_next_logits(
    logits: torch.Tensor,
    *,
    device: torch.device,
    workspace: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return float32 last-token logits, optionally via a reused buffer."""

    if not isinstance(logits, torch.Tensor) or logits.ndim < 2:
        raise TypeError("logits must be a rank-2+ tensor")
    row = logits[:, -1, :]
    if not reuse_logits_workspace_enabled():
        return row.to(copy=True, dtype=torch.float32, device=device)

    buffer = workspace.get("next_logits")
    if (
        buffer is None
        or buffer.shape != row.shape
        or buffer.device != device
        or buffer.dtype != torch.float32
    ):
        buffer = torch.empty(row.shape, dtype=torch.float32, device=device)
        workspace["next_logits"] = buffer
    buffer.copy_(row)
    return buffer


def flatten_bh1d(attn_output: torch.Tensor) -> torch.Tensor:
    """Collapse [1, heads, 1, head_dim] to [1, 1, hidden] with optional copy."""

    if attn_output.dim() != 4 or attn_output.shape[0] != 1 or attn_output.shape[2] != 1:
        raise ValueError("expected attention output shape [1, heads, 1, head_dim]")
    if zero_copy_attn_layout_enabled():
        return attn_output.reshape(1, 1, -1)
    return attn_output.transpose(1, 2).contiguous().view(1, 1, -1)


@dataclass(slots=True)
class DecodeCastElimActivation:
    calls: int = 0


@contextmanager
def decode_cast_elim_candidate_context() -> Iterator[DecodeCastElimActivation]:
    """Enable logits workspace reuse + zero-copy attn flatten on tip decode."""

    global _REUSE_LOGITS_WORKSPACE, _ZERO_COPY_ATTN_LAYOUT
    from ..single import engine

    original = engine.active_prefix_decode_generate
    activation = DecodeCastElimActivation()
    previous_logits = _REUSE_LOGITS_WORKSPACE
    previous_layout = _ZERO_COPY_ATTN_LAYOUT

    @wraps(original)
    def candidate(*args, **kwargs):
        if "reuse_logits_workspace" in kwargs:
            raise RuntimeError(
                "decode cast-elim candidate owns reuse_logits_workspace"
            )
        activation.calls += 1
        return original(*args, reuse_logits_workspace=True, **kwargs)

    engine.active_prefix_decode_generate = candidate
    _REUSE_LOGITS_WORKSPACE = True
    _ZERO_COPY_ATTN_LAYOUT = True
    try:
        yield activation
    finally:
        engine.active_prefix_decode_generate = original
        _REUSE_LOGITS_WORKSPACE = previous_logits
        _ZERO_COPY_ATTN_LAYOUT = previous_layout


def scout_metadata() -> dict[str, Any]:
    return {
        "version": "decode-cast-elim-v1",
        "scope": "logits-workspace-and-attn-layout",
        "evidence_mode": "full_song_reciprocal",
        "int8": False,
        "candidates": (
            "reuse_fp32_logits_workspace",
            "zero_copy_bh1d_reshape",
        ),
        "untouched": (
            "cuda_graph_static_copy_",
            "fp16_cross_bmm_v_float_policy",
        ),
    }


__all__ = [
    "DecodeCastElimActivation",
    "decode_cast_elim_candidate_context",
    "flatten_bh1d",
    "materialize_next_logits",
    "reuse_logits_workspace_enabled",
    "scout_metadata",
    "zero_copy_attn_layout_enabled",
]
