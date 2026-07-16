"""Opt-in request for exact self out_proj+residual fusion (cold-import safe)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


_SELF_OUT_RESIDUAL_REQUESTED = False
_BOUND_RESIDUAL: torch.Tensor | None = None
_FUSED_ON_LAST_CALL = False


def self_out_residual_requested() -> bool:
    return _SELF_OUT_RESIDUAL_REQUESTED


def bound_self_attn_residual() -> torch.Tensor | None:
    return _BOUND_RESIDUAL


def mark_self_out_residual_fused() -> None:
    global _FUSED_ON_LAST_CALL
    _FUSED_ON_LAST_CALL = True


def consume_self_out_residual_fused() -> bool:
    global _FUSED_ON_LAST_CALL
    fused = _FUSED_ON_LAST_CALL
    _FUSED_ON_LAST_CALL = False
    return fused


@contextmanager
def self_out_residual_candidate_context() -> Iterator[None]:
    """Request owned self Wo+residual fusion on the exact optimized path."""

    global _SELF_OUT_RESIDUAL_REQUESTED
    previous = _SELF_OUT_RESIDUAL_REQUESTED
    _SELF_OUT_RESIDUAL_REQUESTED = True
    try:
        yield
    finally:
        _SELF_OUT_RESIDUAL_REQUESTED = previous


@contextmanager
def self_attn_residual_bind(residual: torch.Tensor) -> Iterator[None]:
    """Bind the pre-norm residual for one self-attention call."""

    global _BOUND_RESIDUAL, _FUSED_ON_LAST_CALL
    if not isinstance(residual, torch.Tensor):
        raise TypeError("self-attn residual bind requires a tensor")
    previous = _BOUND_RESIDUAL
    previous_fused = _FUSED_ON_LAST_CALL
    _BOUND_RESIDUAL = residual
    _FUSED_ON_LAST_CALL = False
    try:
        yield
    finally:
        _BOUND_RESIDUAL = previous
        _FUSED_ON_LAST_CALL = previous_fused


__all__ = [
    "bound_self_attn_residual",
    "consume_self_out_residual_fused",
    "mark_self_out_residual_fused",
    "self_attn_residual_bind",
    "self_out_residual_candidate_context",
    "self_out_residual_requested",
]
