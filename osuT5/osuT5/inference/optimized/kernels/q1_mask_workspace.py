"""Opt-in request-local float32 q1 attention-mask workspace.

Elementwise/memory family lever: reuse one contiguous float32 buffer for
native q1 `_native_mask` materialization instead of per-call
`.to(float32).contiguous()` allocations. Not reshape (§19), not headgroup
scheduling (§12), not cast-elim/copy (§4/§5), not allocator env (§16/§17).
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator

import torch


@dataclass
class Q1MaskWorkspace:
    """Request-local float32 contiguous mask buffer for native q1."""

    capacity: int
    device: torch.device
    buffer: torch.Tensor = field(init=False)
    hits: int = 0
    reallocs: int = 0
    copy_bytes: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.capacity, bool)
            or not isinstance(self.capacity, int)
            or self.capacity <= 0
        ):
            raise ValueError("q1 mask workspace capacity must be a positive int")
        if not isinstance(self.device, torch.device) or self.device.type != "cuda":
            raise RuntimeError("q1 mask workspace requires one CUDA device")
        self.buffer = torch.empty(
            self.capacity,
            dtype=torch.float32,
            device=self.device,
        )

    def materialize(
        self,
        attention_mask: torch.Tensor,
        *,
        expected_numel: int,
    ) -> torch.Tensor:
        if expected_numel <= 0:
            raise ValueError("q1 mask workspace expected_numel must be positive")
        flat = attention_mask.reshape(-1)
        if flat.numel() != expected_numel:
            raise ValueError(
                f"attention_mask must contain {expected_numel} elements, "
                f"got {flat.numel()}"
            )
        if expected_numel > int(self.buffer.numel()):
            self.buffer = torch.empty(
                expected_numel,
                dtype=torch.float32,
                device=self.device,
            )
            self.capacity = expected_numel
            self.reallocs += 1
        out = self.buffer[:expected_numel]
        if flat.dtype == torch.float32 and flat.is_contiguous():
            out.copy_(flat)
        else:
            out.copy_(flat.to(dtype=torch.float32))
        self.hits += 1
        self.copy_bytes += int(expected_numel) * 4
        return out

    def evidence(self) -> dict[str, int | str]:
        return {
            "capacity": int(self.capacity),
            "hits": int(self.hits),
            "reallocs": int(self.reallocs),
            "copy_bytes": int(self.copy_bytes),
            "device": str(self.device),
            "dtype": "float32",
        }


_REQUESTED = False
_ACTIVE: ContextVar[Q1MaskWorkspace | None] = ContextVar(
    "q1_mask_workspace",
    default=None,
)


def q1_mask_workspace_requested() -> bool:
    return _REQUESTED


def active_q1_mask_workspace() -> Q1MaskWorkspace | None:
    return _ACTIVE.get()


@contextmanager
def q1_mask_workspace_candidate_context() -> Iterator[None]:
    """Request owned q1 float32 mask workspace for decode."""

    global _REQUESTED
    previous = _REQUESTED
    _REQUESTED = True
    try:
        yield
    finally:
        _REQUESTED = previous


@contextmanager
def q1_mask_workspace_activation_context(
    workspace: Q1MaskWorkspace | None,
) -> Iterator[None]:
    token = _ACTIVE.set(workspace)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def prepare_q1_mask_workspace(
    *,
    device: torch.device,
    capacity: int,
) -> Q1MaskWorkspace:
    return Q1MaskWorkspace(capacity=capacity, device=device)


__all__ = [
    "Q1MaskWorkspace",
    "active_q1_mask_workspace",
    "prepare_q1_mask_workspace",
    "q1_mask_workspace_activation_context",
    "q1_mask_workspace_candidate_context",
    "q1_mask_workspace_requested",
]
