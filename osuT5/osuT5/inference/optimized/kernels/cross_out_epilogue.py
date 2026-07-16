"""Cross out_proj layout + residual epilogue helpers for the selected stack.

Residual fusion itself already lives in ``weight_only_linear_residual``.
These helpers own only the BMM→Wo bridge layout choice and the opt-in
layout-fused flatten that skips a pure contiguous copy.
"""

from __future__ import annotations

import os

import torch

from .weight_only import PackedLinear, weight_only_linear, weight_only_linear_residual


LAYOUT_FUSED_ENV = "MAPPERATORINATOR_CROSS_OUT_LAYOUT_FUSED"


def cross_out_layout_fused_enabled() -> bool:
    raw = os.environ.get(LAYOUT_FUSED_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def flatten_cross_attention_output(
    cross_output: torch.Tensor,
    *,
    layout_fused: bool,
) -> torch.Tensor:
    """Flatten q1 cross BMM output to ``[1, 1, hidden]`` for Wo."""

    if cross_output.dim() != 4 or cross_output.shape[0] != 1 or cross_output.shape[2] != 1:
        raise ValueError("cross BMM output must have shape [1, heads, 1, head_dim]")
    hidden = int(cross_output.shape[1] * cross_output.shape[3])
    if layout_fused:
        if not cross_output.is_contiguous():
            raise ValueError("layout-fused flatten requires contiguous BMM output")
        return cross_output.reshape(1, 1, hidden)
    return cross_output.transpose(1, 2).contiguous().view(1, 1, hidden)


def cross_out_proj_residual(
    cross_output: torch.Tensor,
    residual: torch.Tensor,
    packed: PackedLinear,
    *,
    layout_fused: bool,
    fuse_residual: bool = True,
    outputs_per_block: int = 8,
) -> torch.Tensor:
    """Run selected-stack cross Wo (+ optional residual) with optional layout fusion."""

    flat = flatten_cross_attention_output(cross_output, layout_fused=layout_fused)
    if fuse_residual:
        return weight_only_linear_residual(
            flat,
            residual,
            packed,
            outputs_per_block=outputs_per_block,
        )
    projected = weight_only_linear(
        flat,
        packed,
        outputs_per_block=outputs_per_block,
    )
    return residual + projected
