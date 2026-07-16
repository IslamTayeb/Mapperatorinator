"""Opt-in self-attn RMSNorm+Wqkv fusion for exact FP16/FP32 tip scouts.

Importing this module does not change production dispatch. The candidate
context patches ``engine.generation_profile_context`` so the graduated tip
(shared-RoPE + device state + native cross+MLP) additionally:

1. skips ``self_attn_layer_norm`` in the decoder layer
2. fuses ``RMSNorm + Wqkv`` via ``native_one_token_rmsnorm_linear`` (3×d)

No INT8 / weight-only paths.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Iterator

import torch

from ..kernels.decoder_layer import native_one_token_rmsnorm_linear


_SUPPORTED_DTYPES = (torch.float32, torch.float16)


def _norm_eps(module: torch.nn.Module, dtype: torch.dtype) -> float:
    eps = getattr(module, "eps", None)
    return float(torch.finfo(dtype).eps if eps is None else eps)


def accepted_self_norm_wqkv(
    hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor | None,
    *,
    eps: float,
) -> torch.Tensor:
    """Mirror accepted tip: RMSNorm → linear."""

    # Match nn.RMSNorm: rsqrt(mean(x^2) + eps) * weight, then linear.
    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    hidden = hidden_states * torch.rsqrt(variance + eps)
    hidden = hidden * norm_weight
    return torch.nn.functional.linear(hidden, linear_weight, linear_bias)


def native_fused_self_norm_wqkv(
    hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor | None,
    *,
    eps: float,
    outputs_per_block: int = 8,
) -> torch.Tensor:
    return native_one_token_rmsnorm_linear(
        hidden_states,
        norm_weight,
        linear_weight,
        linear_bias,
        eps=eps,
        outputs_per_block=outputs_per_block,
    )


def attach_decoder_norm_for_fuse(
    *,
    self_attn: torch.nn.Module,
    self_attn_layer_norm: torch.nn.Module,
    hidden_dtype: torch.dtype,
) -> None:
    """Stash decoder RMSNorm params on attention for the fused QKV path."""

    self_attn._scout_self_norm_weight = self_attn_layer_norm.weight
    self_attn._scout_self_norm_eps = _norm_eps(self_attn_layer_norm, hidden_dtype)


def clear_decoder_norm_for_fuse(self_attn: torch.nn.Module) -> None:
    if hasattr(self_attn, "_scout_self_norm_weight"):
        delattr(self_attn, "_scout_self_norm_weight")
    if hasattr(self_attn, "_scout_self_norm_eps"):
        delattr(self_attn, "_scout_self_norm_eps")


def scout_metadata() -> dict[str, Any]:
    return {
        "version": "self-rmsnorm-wqkv-fusion-v1",
        "scope": "self-attention-rmsnorm-wqkv-only",
        "evidence_mode": "full_song_reciprocal",
        "int8": False,
        "candidates": ("native_fused",),
        "accepted_path": "RMSNorm → Wqkv",
        "candidate_path": "native_one_token_rmsnorm_linear(norm, Wqkv)",
        "supported_dtypes": [str(dtype) for dtype in _SUPPORTED_DTYPES],
    }


@dataclass(slots=True)
class SelfNormWqkvFusionActivation:
    calls: int = 0


@contextmanager
def self_norm_wqkv_fusion_candidate_context() -> Iterator[SelfNormWqkvFusionActivation]:
    """Enable native self RMSNorm+Wqkv fusion on top of the tip engine."""

    from ..single import engine

    original = engine.generation_profile_context
    activation = SelfNormWqkvFusionActivation()

    @contextmanager
    @wraps(original)
    def candidate(*args, **kwargs):
        if "native_self_norm_wqkv" in kwargs:
            raise RuntimeError(
                "self-norm Wqkv candidate owns native_self_norm_wqkv"
            )
        activation.calls += 1
        with original(*args, native_self_norm_wqkv=True, **kwargs) as ctx:
            yield ctx

    engine.generation_profile_context = candidate
    try:
        yield activation
    finally:
        engine.generation_profile_context = original


__all__ = [
    "SelfNormWqkvFusionActivation",
    "accepted_self_norm_wqkv",
    "attach_decoder_norm_for_fuse",
    "clear_decoder_norm_for_fuse",
    "native_fused_self_norm_wqkv",
    "scout_metadata",
    "self_norm_wqkv_fusion_candidate_context",
]
