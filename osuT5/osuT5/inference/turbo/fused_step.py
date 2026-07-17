"""§38 TIER2 fused decoder-step reference (fp32 accumulate).

Seven logical stages per layer (launch-collapse target):
  1. norm + Wqkv
  2. q1 self-attn
  3. Wo + residual
  4. cross block (norm + Wq/Wkv + attn + Wo + residual)
  5. fc1 (+ GELU)
  6. fc2 + residual
  7. glue / dtype restore

This module provides the **relaxed-numerics reference** used by rung-1
teacher-forced logit agreement. CUDA 7-kernel collapse comes after the
TIER2 quality gate. Opt-in only via ``inference_engine=turbo``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


def rmsnorm_fp32(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """RMSNorm with fp32 accumulate; restore storage dtype."""
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    y = x_f * torch.rsqrt(var + eps)
    return (y * weight.float()).to(dtype=x.dtype)


def linear_fp32(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Linear with fp32 accumulate; restore storage dtype."""
    out = F.linear(x.float(), weight.float(), None if bias is None else bias.float())
    return out.to(dtype=x.dtype)


def rmsnorm_linear_fp32(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """Stage 1 / cross-Q style: fused RMSNorm + Linear, fp32 accumulate."""
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    y = x_f * torch.rsqrt(var + eps) * norm_weight.float()
    out = F.linear(y, linear_weight.float(), None if linear_bias is None else linear_bias.float())
    return out.to(dtype=x.dtype)


def linear_residual_fp32(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Stage 3 / 6 style: Linear + residual add in fp32, restore dtype."""
    out = F.linear(x.float(), weight.float(), None if bias is None else bias.float())
    return (out + residual.float()).to(dtype=residual.dtype)


def gelu_exact_fp32(x: torch.Tensor) -> torch.Tensor:
    """Exact GELU in fp32 (matches tip native ``erff`` path)."""
    x_f = x.float()
    return (0.5 * x_f * (1.0 + torch.erf(x_f * 0.7071067811865475))).to(dtype=x.dtype)


def _norm_eps(norm: nn.Module, dtype: torch.dtype) -> float:
    eps = getattr(norm, "eps", None)
    if eps is None:
        eps = getattr(norm, "variance_epsilon", 1e-6)
    return float(eps)


class _FusedLinear(nn.Module):
    """Drop-in Linear that always accumulates in fp32."""

    def __init__(self, inner: nn.Linear):
        super().__init__()
        self.inner = inner

    @property
    def weight(self):
        return self.inner.weight

    @property
    def bias(self):
        return self.inner.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear_fp32(x, self.inner.weight, self.inner.bias)


class _FusedRMSNorm(nn.Module):
    """Drop-in RMSNorm that always accumulates in fp32."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    @property
    def weight(self):
        return self.inner.weight

    @property
    def eps(self):
        return _norm_eps(self.inner, torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm_fp32(x, self.inner.weight, self.eps)


def _wrap_module_linears_and_norms(module: nn.Module) -> list[tuple[nn.Module, str, nn.Module]]:
    """Replace decoder Linear/RMSNorm children with fp32-accumulate wrappers.

    Returns list of (parent, attr_name, original) for restore.
    """
    saved: list[tuple[nn.Module, str, nn.Module]] = []
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            saved.append((module, child_name, child))
            setattr(module, child_name, _FusedLinear(child))
        elif child.__class__.__name__ == "RMSNorm" or isinstance(child, nn.RMSNorm):
            saved.append((module, child_name, child))
            setattr(module, child_name, _FusedRMSNorm(child))
        else:
            saved.extend(_wrap_module_linears_and_norms(child))
    return saved


def _restore_saved(saved: list[tuple[nn.Module, str, nn.Module]]) -> None:
    for parent, name, orig in saved:
        setattr(parent, name, orig)


@contextmanager
def install_tier2_fused_numerics(raw_model) -> Iterator[dict[str, int]]:
    """Install fp32-accumulate Linear/RMSNorm wrappers on decoder layers only.

    Encoder / LM head stay tip-bitexact. Optimized engine code paths are not
    modified; this only mutates the bound module tree while the context is open.
    """
    decoder = None
    for path in (
        ("transformer", "model", "decoder"),
        ("model", "decoder"),
        ("decoder",),
    ):
        obj = raw_model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            decoder = obj
            break
    if decoder is None or not hasattr(decoder, "layers"):
        raise RuntimeError("tier2 fused install: decoder.layers not found on raw_model")
    layers = decoder.layers
    saved: list[tuple[nn.Module, str, nn.Module]] = []
    for layer in layers:
        saved.extend(_wrap_module_linears_and_norms(layer))
    meta = {
        "wrapped_modules": len(saved),
        "decoder_layers": len(layers),
        "accumulate": "fp32",
        "stages": 7,
    }
    try:
        yield meta
    finally:
        _restore_saved(saved)


def relative_logit_delta(
    teacher: torch.Tensor,
    candidate: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-position max relative |Δ| / max(|teacher|, eps) over vocab."""
    # teacher/candidate: [T, V] or [B, T, V]
    t = teacher.float()
    c = candidate.float()
    denom = t.abs().clamp_min(eps)
    rel = (c - t).abs() / denom
    if rel.dim() == 3:
        return rel.amax(dim=-1).reshape(-1)
    if rel.dim() == 2:
        return rel.amax(dim=-1)
    raise ValueError("expected logits rank 2 or 3")


def top1_agreement(teacher: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Boolean agreement of argmax over last dim; flattened positions."""
    t = teacher.float().argmax(dim=-1).reshape(-1)
    c = candidate.float().argmax(dim=-1).reshape(-1)
    return t == c
