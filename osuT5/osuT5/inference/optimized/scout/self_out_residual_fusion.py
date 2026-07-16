"""Opt-in self-attn out_proj+residual fusion for exact FP16/FP32 tip scouts.

Importing this module does not change production dispatch. The candidate
context patches ``engine.generation_profile_context`` so the graduated tip
(shared-RoPE + device state + native cross+MLP) additionally:

1. skips ``Wo`` inside attention (``skip_out_proj``)
2. fuses ``Wo + residual`` via ``native_one_token_linear_residual``

No INT8 / weight-only paths.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from ..kernels.decoder_layer import native_one_token_linear_residual


_SUPPORTED_DTYPES = (torch.float32, torch.float16)


def _require_bh1d_or_flat(
    attn_output: torch.Tensor,
    *,
    expected_dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(attn_output, torch.Tensor):
        raise TypeError("attention output must be a tensor")
    if attn_output.dtype != expected_dtype:
        raise TypeError(
            f"attention output must have dtype {expected_dtype}, got {attn_output.dtype}"
        )
    if attn_output.device.type != "cuda":
        raise RuntimeError("attention output must be a CUDA tensor")
    if attn_output.dim() == 4 and attn_output.shape[0] == 1 and attn_output.shape[2] == 1:
        return attn_output.reshape(1, 1, -1)
    if attn_output.dim() == 3 and attn_output.shape[:2] == (1, 1):
        return attn_output
    raise ValueError(
        "attention output must have shape [1, heads, 1, head_dim] or [1, 1, hidden]"
    )


def accepted_self_out_residual(
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Mirror accepted tip: flatten → Wo → residual add."""

    flat = _require_bh1d_or_flat(attn_output, expected_dtype=residual.dtype)
    return residual + F.linear(flat, weight, bias)


def native_fused_self_out_residual(
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    outputs_per_block: int = 8,
) -> torch.Tensor:
    """Fuse out_proj+residual via existing native_one_token_linear_residual."""

    flat = _require_bh1d_or_flat(attn_output, expected_dtype=residual.dtype)
    return native_one_token_linear_residual(
        flat,
        residual,
        weight,
        bias,
        outputs_per_block=outputs_per_block,
    )


def native_self_out_residual_forward(
    *,
    module: torch.nn.Module,
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    layer_name: str,
    dispatch_counts: dict[str, int] | None = None,
) -> torch.Tensor | None:
    """Decoder-layer hook: fuse Wo+residual when decode inputs qualify."""

    del layer_name  # reserved for NVTX naming in later iterations
    self_attn = module.self_attn
    if (
        module.training
        or self_attn.training
        or getattr(module, "dropout", 0) != 0
        or getattr(self_attn, "dropout", 0) != 0
        or residual.device.type != "cuda"
        or residual.shape[:2] != (1, 1)
        or residual.dtype not in _SUPPORTED_DTYPES
        or attn_output.dtype != residual.dtype
        or attn_output.device != residual.device
    ):
        return None
    if getattr(self_attn, "out_drop", None) is not None:
        # Identity dropout modules expose p=0; refuse stochastic drop.
        p = getattr(self_attn.out_drop, "p", 0.0)
        if float(p) != 0.0:
            return None

    hidden = native_fused_self_out_residual(
        attn_output,
        residual,
        self_attn.Wo.weight,
        self_attn.Wo.bias,
        outputs_per_block=8,
    )
    if dispatch_counts is not None:
        dispatch_counts["native_self_out_residual"] = (
            dispatch_counts.get("native_self_out_residual", 0) + 1
        )
    return hidden


def scout_metadata() -> dict[str, Any]:
    return {
        "version": "self-out-proj-residual-fusion-v1",
        "scope": "self-attention-out-proj-residual-only",
        "evidence_mode": "full_song_reciprocal",
        "int8": False,
        "candidates": ("native_fused",),
        "accepted_path": "Wo(attn) → residual + hidden",
        "candidate_path": "skip Wo → native_one_token_linear_residual(Wo, residual)",
    }


@dataclass(slots=True)
class SelfOutResidualFusionActivation:
    calls: int = 0


@contextmanager
def self_out_residual_fusion_candidate_context() -> Iterator[SelfOutResidualFusionActivation]:
    """Enable native self out+residual fusion on top of the tip engine."""

    from ..single import engine

    original = engine.generation_profile_context
    activation = SelfOutResidualFusionActivation()

    @contextmanager
    @wraps(original)
    def candidate(*args, **kwargs):
        if "native_self_out_residual" in kwargs:
            raise RuntimeError(
                "self-out residual candidate owns native_self_out_residual"
            )
        activation.calls += 1
        with original(*args, native_self_out_residual=True, **kwargs) as ctx:
            yield ctx

    engine.generation_profile_context = candidate
    try:
        yield activation
    finally:
        engine.generation_profile_context = original


__all__ = [
    "SelfOutResidualFusionActivation",
    "accepted_self_out_residual",
    "native_fused_self_out_residual",
    "native_self_out_residual_forward",
    "scout_metadata",
    "self_out_residual_fusion_candidate_context",
]
