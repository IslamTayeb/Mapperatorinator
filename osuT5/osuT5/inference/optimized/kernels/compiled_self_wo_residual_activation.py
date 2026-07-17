"""Opt-in request for tip-exact decode-only compiled self Wo+residual."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator

import torch

CompiledWoResidual = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None],
    torch.Tensor,
]

_REQUESTED = False
_ACTIVE: ContextVar[CompiledWoResidual | None] = ContextVar(
    "compiled_self_wo_residual",
    default=None,
)


def compiled_self_wo_residual_requested() -> bool:
    return _REQUESTED


def active_compiled_self_wo_residual() -> CompiledWoResidual | None:
    return _ACTIVE.get()


@contextmanager
def compiled_self_wo_residual_candidate_context() -> Iterator[None]:
    """Request owned compile of decode-only self Wo linear + residual add."""

    global _REQUESTED
    previous = _REQUESTED
    _REQUESTED = True
    try:
        yield
    finally:
        _REQUESTED = previous


@contextmanager
def compiled_self_wo_residual_activation_context(
    compiled: CompiledWoResidual | None,
) -> Iterator[None]:
    token = _ACTIVE.set(compiled)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


__all__ = [
    "CompiledWoResidual",
    "active_compiled_self_wo_residual",
    "compiled_self_wo_residual_activation_context",
    "compiled_self_wo_residual_candidate_context",
    "compiled_self_wo_residual_requested",
]
