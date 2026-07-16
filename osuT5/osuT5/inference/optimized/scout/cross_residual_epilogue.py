"""Opt-in cross out_proj+residual epilogue layout helpers.

Component-only: importing this module does not change production dispatch.
Compiled-cross BMM ownership is intentionally untouched.

Accepted selected path materializes attention output via
``transpose(1, 2).contiguous().view`` before ``weight_only_linear_residual``.
Candidates under test:

- ``heads_view_fused``: zero-copy ``reshape`` into the same fused residual kernel
- ``split_linear_then_add``: diagnostic unfused linear + residual add
"""

from __future__ import annotations

from typing import Any

import torch

from ..kernels.weight_only import (
    PackedLinear,
    weight_only_linear,
    weight_only_linear_residual,
)


def _require_bh1d(attn_output: torch.Tensor) -> torch.Tensor:
    if not isinstance(attn_output, torch.Tensor):
        raise TypeError("attention output must be a tensor")
    if attn_output.dim() != 4 or attn_output.shape[0] != 1 or attn_output.shape[2] != 1:
        raise ValueError("attention output must have shape [1, heads, 1, head_dim]")
    if attn_output.dtype != torch.float32:
        raise TypeError("attention output must retain FP32 storage")
    if attn_output.device.type != "cuda":
        raise RuntimeError("attention output must be a CUDA tensor")
    return attn_output


def accepted_cross_out_residual(
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    packed: PackedLinear,
    *,
    outputs_per_block: int = 8,
) -> torch.Tensor:
    """Mirror the selected-stack reshape + fused linear residual sequence."""

    attn_output = _require_bh1d(attn_output)
    flat = attn_output.transpose(1, 2).contiguous().view(1, 1, -1)
    return weight_only_linear_residual(
        flat,
        residual,
        packed,
        outputs_per_block=outputs_per_block,
    )


def heads_view_cross_out_residual(
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    packed: PackedLinear,
    *,
    outputs_per_block: int = 8,
) -> torch.Tensor:
    """Fuse layout conversion via zero-copy reshape into fused residual."""

    attn_output = _require_bh1d(attn_output)
    flat = attn_output.reshape(1, 1, -1)
    return weight_only_linear_residual(
        flat,
        residual,
        packed,
        outputs_per_block=outputs_per_block,
    )


def split_linear_then_add_cross_out(
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    packed: PackedLinear,
    *,
    outputs_per_block: int = 8,
) -> torch.Tensor:
    """Diagnostic: accepted reshape + unfused linear, then residual add."""

    attn_output = _require_bh1d(attn_output)
    flat = attn_output.transpose(1, 2).contiguous().view(1, 1, -1)
    return weight_only_linear(
        flat,
        packed,
        outputs_per_block=outputs_per_block,
    ) + residual


def scout_metadata() -> dict[str, Any]:
    return {
        "version": "cross-out-proj-residual-epilogue-v1",
        "scope": "cross-attention-out-proj-residual-only",
        "compiled_cross_bmm": False,
        "candidates": (
            "heads_view_fused",
            "split_linear_then_add",
        ),
    }


__all__ = [
    "accepted_cross_out_residual",
    "heads_view_cross_out_residual",
    "scout_metadata",
    "split_linear_then_add_cross_out",
]
