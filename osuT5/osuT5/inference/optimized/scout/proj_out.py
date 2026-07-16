"""Opt-in final decoder RMSNorm + proj_out (LM-head) fusion for exact tip scouts.

Importing this module does not change production dispatch. The candidate
context patches ``engine.generation_profile_context`` so the graduated tip
(shared-RoPE + device state + native cross+MLP) additionally:

1. skips ``decoder.layer_norm`` on one-token decode ``(1, 1, H)``
2. fuses ``RMSNorm + proj_out`` via ``native_one_token_rmsnorm_linear``

No INT8 / weight-only paths. Prefill / multi-token keeps separate norm+linear.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Iterator

import torch

from ..kernels.decoder_layer import native_one_token_rmsnorm_linear


_SUPPORTED_DTYPES = (torch.float32, torch.float16)
_PENDING_ATTR = "_scout_proj_out_fuse_pending"
_EPS_ATTR = "_scout_proj_out_fuse_eps"


def _norm_eps(module: torch.nn.Module, dtype: torch.dtype) -> float:
    eps = getattr(module, "eps", None)
    return float(torch.finfo(dtype).eps if eps is None else eps)


def accepted_final_norm_proj_out(
    hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor | None,
    *,
    eps: float,
) -> torch.Tensor:
    """Mirror accepted tip: RMSNorm → proj_out linear."""

    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    hidden = hidden_states * torch.rsqrt(variance + eps)
    hidden = hidden * norm_weight
    return torch.nn.functional.linear(hidden, linear_weight, linear_bias)


def native_fused_final_norm_proj_out(
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


def attach_final_norm_for_fuse(
    decoder: torch.nn.Module,
    *,
    hidden_dtype: torch.dtype,
) -> None:
    """Mark that decoder.layer_norm was skipped for a fused proj_out call."""

    layer_norm = decoder.layer_norm
    setattr(decoder, _PENDING_ATTR, True)
    setattr(decoder, _EPS_ATTR, _norm_eps(layer_norm, hidden_dtype))


def clear_final_norm_for_fuse(decoder: torch.nn.Module) -> None:
    if hasattr(decoder, _PENDING_ATTR):
        delattr(decoder, _PENDING_ATTR)
    if hasattr(decoder, _EPS_ATTR):
        delattr(decoder, _EPS_ATTR)


def try_native_fused_proj_out(
    *,
    lm_module: torch.nn.Module,
    hidden_states: torch.Tensor,
    dispatch_counts: dict[str, int] | None = None,
) -> torch.Tensor | None:
    """Apply fused final-norm+proj_out when decoder deferred the RMSNorm."""

    model = getattr(lm_module, "model", None)
    decoder = getattr(model, "decoder", None) if model is not None else None
    proj_out = getattr(lm_module, "proj_out", None)
    if decoder is None or proj_out is None:
        return None
    if not bool(getattr(decoder, _PENDING_ATTR, False)):
        return None
    if (
        hidden_states.device.type != "cuda"
        or hidden_states.shape[:2] != (1, 1)
        or hidden_states.dtype not in _SUPPORTED_DTYPES
        or proj_out.weight.dtype != hidden_states.dtype
        or proj_out.weight.device != hidden_states.device
    ):
        raise RuntimeError(
            "proj_out fuse pending but hidden/proj_out are not one-token CUDA "
            f"fp16/fp32 decode inputs (shape={tuple(hidden_states.shape)}, "
            f"dtype={hidden_states.dtype}, device={hidden_states.device})"
        )

    eps = float(getattr(decoder, _EPS_ATTR))
    try:
        logits = native_fused_final_norm_proj_out(
            hidden_states,
            decoder.layer_norm.weight,
            proj_out.weight,
            proj_out.bias,
            eps=eps,
            outputs_per_block=8,
        )
    finally:
        clear_final_norm_for_fuse(decoder)

    if dispatch_counts is not None:
        dispatch_counts["native_final_norm_proj_out"] = (
            dispatch_counts.get("native_final_norm_proj_out", 0) + 1
        )
    return logits


def scout_metadata() -> dict[str, Any]:
    return {
        "version": "final-norm-proj-out-fusion-v1",
        "scope": "decoder-final-rmsnorm-proj-out-only",
        "evidence_mode": "full_song_reciprocal",
        "int8": False,
        "candidates": ("native_fused",),
        "accepted_path": "decoder.layer_norm → proj_out",
        "candidate_path": (
            "skip decoder.layer_norm → native_one_token_rmsnorm_linear"
            "(norm, proj_out)"
        ),
        "supported_dtypes": [str(dtype) for dtype in _SUPPORTED_DTYPES],
    }


@dataclass(slots=True)
class ProjOutFusionActivation:
    calls: int = 0


@contextmanager
def proj_out_fusion_candidate_context() -> Iterator[ProjOutFusionActivation]:
    """Enable native final-norm+proj_out fusion on top of the tip engine."""

    from ..single import engine

    original = engine.generation_profile_context
    activation = ProjOutFusionActivation()

    @contextmanager
    @wraps(original)
    def candidate(*args, **kwargs):
        if "native_proj_out" in kwargs:
            raise RuntimeError("proj_out candidate owns native_proj_out")
        activation.calls += 1
        with original(*args, native_proj_out=True, **kwargs) as ctx:
            yield ctx

    engine.generation_profile_context = candidate
    try:
        yield activation
    finally:
        engine.generation_profile_context = original


__all__ = [
    "ProjOutFusionActivation",
    "accepted_final_norm_proj_out",
    "attach_final_norm_for_fuse",
    "clear_final_norm_for_fuse",
    "native_fused_final_norm_proj_out",
    "proj_out_fusion_candidate_context",
    "scout_metadata",
    "try_native_fused_proj_out",
]
