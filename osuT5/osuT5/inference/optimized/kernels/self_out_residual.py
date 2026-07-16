"""Fuse self-attention out_proj + residual with the owned one-token kernel."""

from __future__ import annotations

import torch
from torch import nn

from ....runtime_profiling import profile_range
from .decoder_layer import (
    native_one_token_linear_residual,
    preload_native_decoder_layer,
)
from .self_out_residual_activation import (
    bound_self_attn_residual,
    mark_self_out_residual_fused,
    self_out_residual_requested,
)


def apply_self_out_residual_if_requested(
    *,
    module,
    attn_output: torch.Tensor,
    range_prefix: str,
    profile_ranges: bool,
    dispatch_counts: dict[str, int] | None = None,
) -> torch.Tensor | None:
    """Replace Wo(+dropout) + residual when the candidate residual is bound."""

    if not self_out_residual_requested():
        return None
    residual = bound_self_attn_residual()
    if residual is None:
        return None
    if module.training:
        return None
    if not isinstance(module.out_drop, nn.Identity):
        return None
    if not isinstance(attn_output, torch.Tensor) or not isinstance(residual, torch.Tensor):
        return None
    if attn_output.shape[0] != 1 or residual.shape[0] != 1:
        return None
    if attn_output.dtype != residual.dtype or attn_output.device != residual.device:
        return None

    flat = attn_output
    if flat.dim() == 4:
        # [1, heads, 1, dim] from native attention → [1, 1, hidden]
        flat = flat.reshape(1, 1, -1)
    elif flat.dim() == 3:
        if flat.shape[1] != 1:
            return None
        flat = flat.reshape(1, 1, -1)
    else:
        return None
    if residual.shape != flat.shape:
        if residual.numel() != flat.numel():
            return None
        residual = residual.reshape_as(flat)

    preload_native_decoder_layer()
    if profile_ranges:
        with profile_range(f"{range_prefix}.self_out_residual_fused"):
            fused = native_one_token_linear_residual(
                flat,
                residual,
                module.Wo.weight,
                module.Wo.bias,
                outputs_per_block=8,
            )
    else:
        fused = native_one_token_linear_residual(
            flat,
            residual,
            module.Wo.weight,
            module.Wo.bias,
            outputs_per_block=8,
        )
    mark_self_out_residual_fused()
    if dispatch_counts is not None:
        dispatch_counts["native_self_out_residual"] = (
            dispatch_counts.get("native_self_out_residual", 0) + 1
        )
    return fused


__all__ = ["apply_self_out_residual_if_requested"]
