"""Opt-in self-attn Wo (attention-out) one-token linear — without residual.

Distinct from self-out residual fusion (§6 STOP_NO_PROMOTE): residual stays a
separate add. Importing this module does not change production dispatch.

Candidate context patches ``engine.generation_profile_context`` so tip
(shared-RoPE + device state + native cross+MLP) additionally:

1. skips ``Wo`` inside attention (``skip_out_proj``)
2. applies owned ``native_one_token_linear`` for Wo
3. adds residual separately in the decoder layer

No INT8 / weight-only / proj_out paths.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from ..kernels.decoder_layer import native_one_token_linear


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


def accepted_self_wo_linear(
    attn_output: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Mirror accepted tip: flatten → Wo (residual owned by decoder)."""

    flat = _require_bh1d_or_flat(attn_output, expected_dtype=weight.dtype)
    return F.linear(flat, weight, bias)


def native_self_wo_linear(
    attn_output: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    outputs_per_block: int = 8,
) -> torch.Tensor:
    flat = _require_bh1d_or_flat(attn_output, expected_dtype=weight.dtype)
    return native_one_token_linear(
        flat,
        weight,
        bias,
        outputs_per_block=outputs_per_block,
    )


def native_self_wo_linear_forward(
    *,
    module: torch.nn.Module,
    attn_output: torch.Tensor,
    layer_name: str,
    dispatch_counts: dict[str, int] | None = None,
) -> torch.Tensor | None:
    """Decoder-layer hook: replace Wo only when decode inputs qualify."""

    del layer_name
    self_attn = module.self_attn
    if (
        module.training
        or self_attn.training
        or getattr(module, "dropout", 0) != 0
        or getattr(self_attn, "dropout", 0) != 0
        or attn_output.device.type != "cuda"
        or attn_output.dtype not in _SUPPORTED_DTYPES
        or self_attn.Wo.weight.dtype != attn_output.dtype
        or self_attn.Wo.weight.device != attn_output.device
    ):
        return None
    if getattr(self_attn, "out_drop", None) is not None:
        p = getattr(self_attn.out_drop, "p", 0.0)
        if float(p) != 0.0:
            return None
    # Qualify one-token decode shapes before allocating native work.
    try:
        _require_bh1d_or_flat(attn_output, expected_dtype=attn_output.dtype)
    except (TypeError, ValueError, RuntimeError):
        return None

    hidden = native_self_wo_linear(
        attn_output,
        self_attn.Wo.weight,
        self_attn.Wo.bias,
        outputs_per_block=8,
    )
    if dispatch_counts is not None:
        dispatch_counts["native_self_wo_linear"] = (
            dispatch_counts.get("native_self_wo_linear", 0) + 1
        )
    return hidden


def scout_metadata() -> dict[str, Any]:
    return {
        "version": "self-wo-linear-v1",
        "scope": "self-attention-out-proj-only-no-residual",
        "evidence_mode": "full_song_reciprocal",
        "int8": False,
        "candidates": ("native_one_token_linear",),
        "accepted_path": "Wo(attn) → residual + hidden",
        "candidate_path": (
            "skip Wo → native_one_token_linear(Wo) → residual + hidden"
        ),
        "supported_dtypes": [str(dtype) for dtype in _SUPPORTED_DTYPES],
    }


@dataclass(slots=True)
class SelfWoLinearActivation:
    calls: int = 0


@contextmanager
def self_wo_linear_candidate_context() -> Iterator[SelfWoLinearActivation]:
    """Enable native self Wo (no residual) on top of the tip engine."""

    from ..single import engine

    original = engine.generation_profile_context
    activation = SelfWoLinearActivation()

    @contextmanager
    @wraps(original)
    def candidate(*args, **kwargs):
        if "native_self_wo_linear" in kwargs:
            raise RuntimeError("self-wo linear candidate owns native_self_wo_linear")
        activation.calls += 1
        with original(*args, native_self_wo_linear=True, **kwargs) as ctx:
            yield ctx

    engine.generation_profile_context = candidate
    try:
        yield activation
    finally:
        engine.generation_profile_context = original


__all__ = [
    "SelfWoLinearActivation",
    "accepted_self_wo_linear",
    "native_self_wo_linear",
    "native_self_wo_linear_forward",
    "scout_metadata",
    "self_wo_linear_candidate_context",
]
