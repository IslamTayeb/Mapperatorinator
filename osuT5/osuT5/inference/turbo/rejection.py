"""Leviathan-style rejection sampling for turbo speculation (TIER1 primitive).

Distribution-equivalent when verify uses the same sampling distribution as the
target engine. Not bit-exact. Not a TPS claim.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def expected_accepted(alpha: float, gamma: int) -> float:
    if alpha >= 1.0 - 1e-12:
        return float(gamma + 1)
    if alpha <= 0.0:
        return 1.0
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def acceptance_alpha(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Σ_v min(p(v), q(v)) over vocab. p,q shape [..., V]."""
    return torch.minimum(p, q).sum(dim=-1)


def residual_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Normalize max(p-q, 0). p,q shape [V] or [..., V]."""
    resid = (p - q).clamp_min(0.0)
    denom = resid.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return resid / denom


@torch.no_grad()
def reject_sample_prefix(
    *,
    p_probs: torch.Tensor,
    q_probs: torch.Tensor,
    draft_token_ids: Sequence[int] | torch.Tensor,
    rng: torch.Generator | None = None,
) -> tuple[int, int | None]:
    """Accept a prefix of drafted tokens under Leviathan rejection sampling.

    Args:
        p_probs: target probs for positions 0..γ-1, shape [γ, V]
        q_probs: draft probs for positions 0..γ-1, shape [γ, V]
        draft_token_ids: length γ drafted token ids
        rng: optional CPU/CUDA generator for bernoulli draws

    Returns:
        (n_accepted, residual_token_or_None)
        If n_accepted < γ, residual is sampled from norm(max(p-q,0)) at the
        reject index. If all γ accepted, residual is None (bonus target sample
        is the caller's responsibility).
    """
    if p_probs.ndim != 2 or q_probs.ndim != 2:
        raise ValueError("p_probs and q_probs must be [gamma, vocab]")
    gamma = int(p_probs.shape[0])
    if q_probs.shape[0] != gamma:
        raise ValueError("p/q gamma mismatch")
    ids = torch.as_tensor(draft_token_ids, device=p_probs.device, dtype=torch.long)
    if ids.numel() != gamma:
        raise ValueError(f"expected {gamma} draft tokens, got {ids.numel()}")

    for i in range(gamma):
        tok = int(ids[i].item())
        p_i = p_probs[i, tok]
        q_i = q_probs[i, tok].clamp_min(1e-12)
        accept_prob = torch.minimum(p_i / q_i, torch.ones((), device=p_i.device, dtype=p_i.dtype))
        u = torch.rand((), device=p_i.device, generator=rng, dtype=p_i.dtype)
        if float(u.item()) > float(accept_prob.item()):
            resid = residual_distribution(p_probs[i], q_probs[i])
            residual_tok = int(torch.multinomial(resid, 1, generator=rng).item())
            return i, residual_tok
    return gamma, None


def apply_temp_top_p(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    scores = logits.float() / max(float(temperature), 1e-5)
    sorted_scores, sorted_idx = torch.sort(scores, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_scores, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    remove = cumsum - sorted_probs > top_p
    remove[..., 0] = False
    sorted_scores = sorted_scores.masked_fill(remove, torch.finfo(sorted_scores.dtype).min)
    restored = torch.full_like(scores, torch.finfo(scores.dtype).min)
    restored.scatter_(-1, sorted_idx, sorted_scores)
    return F.softmax(restored, dim=-1)
