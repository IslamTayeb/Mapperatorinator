"""§51 EAGLE-style draft head scaffold (Track C).

Target: draft cost c_d ≈ 0.05–0.1× tip decode step, with offline/runtime
E[acc] ≥ 1.7 so keep-KV + verify ceilings restore ≥420 TPS.

Not wired into ``generate_window`` yet. Not a 500 / TIER1 claim.
Campaign tip remains ``55949274`` / FP16 366.11.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Tip step ≈ 1.85 ms @ 366.11 TPS (post-rung bound script).
TIP_STEP_MS = 1.85
C_DRAFT_TARGET_LO = 0.05 * TIP_STEP_MS  # ~0.0925 ms
C_DRAFT_TARGET_HI = 0.10 * TIP_STEP_MS  # ~0.185 ms
C_VERIFY_MS = 3.075  # §48 in-loop absolute
GATE_E = 1.7
GATE_CEILING_TPS = 420.0


def tip_step_ms() -> float:
    return TIP_STEP_MS


def draft_budget_ms(*, fraction: float) -> float:
    if fraction <= 0.0:
        raise ValueError("fraction must be > 0")
    return float(fraction) * TIP_STEP_MS


def step_ms_keep_kv(*, c_draft_ms: float, c_verify_ms: float = C_VERIFY_MS) -> float:
    """One speculative cycle with keep-KV (no crop rebuild)."""
    return float(c_draft_ms) + float(c_verify_ms)


def ceiling_tps(*, c_draft_ms: float, e_acc: float, c_verify_ms: float = C_VERIFY_MS) -> float:
    step = step_ms_keep_kv(c_draft_ms=c_draft_ms, c_verify_ms=c_verify_ms)
    return 1000.0 * float(e_acc) / step


def budget_table(e_values: tuple[float, ...] = (1.7, 2.0, 2.4)) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for frac in (0.05, 0.10):
        c_d = draft_budget_ms(fraction=frac)
        for e in e_values:
            rows.append(
                {
                    "c_d_frac": frac,
                    "c_draft_ms": c_d,
                    "E": float(e),
                    "step_ms": step_ms_keep_kv(c_draft_ms=c_d),
                    "ceiling_tps": ceiling_tps(c_draft_ms=c_d, e_acc=e),
                }
            )
    return rows


@dataclass(frozen=True)
class EagleProbeGate:
    """Cheap-probe promote / kill for §51 before heavy train."""

    teacher_force_E: float
    in_loop_E: float | None
    c_draft_ms_est: float
    ceiling_at_E17: float

    @property
    def pass_teacher_force(self) -> bool:
        return self.teacher_force_E >= GATE_E

    @property
    def pass_in_loop(self) -> bool:
        if self.in_loop_E is None:
            return False
        return self.in_loop_E >= GATE_E

    @property
    def pass_budget(self) -> bool:
        return C_DRAFT_TARGET_LO <= self.c_draft_ms_est <= C_DRAFT_TARGET_HI * 1.5

    @property
    def pass_ceiling(self) -> bool:
        return self.ceiling_at_E17 >= GATE_CEILING_TPS

    def decision(self) -> str:
        """GO_HEAVY_TRAIN only if TF E and budget/ceiling look recoverable.

        §52 lesson: teacher-force tip-dump E overstates in-loop E — never
        promote on TF alone. Smoke train may proceed on TF≥1.7 + budget, but
        runtime gate remains in_loop_E≥1.7.
        """
        if not self.pass_budget or not self.pass_ceiling:
            return "STOP_BUDGET"
        if not self.pass_teacher_force:
            return "STOP_TF_E"
        if self.in_loop_E is not None and not self.pass_in_loop:
            return "STOP_IN_LOOP_E"
        if self.pass_in_loop:
            return "GO_RUNTIME_WIRE"
        return "GO_SMOKE_TRAIN"


class EagleDraftHead(nn.Module):
    """Minimal EAGLE-style autoregressive draft head.

    Consumes previous decoder hidden state ``h_{t-1}`` (d_model) and predicts
    next-token logits. Optional one-layer self-attn stub reserved for later;
    v0 is a 2-layer MLP + tied/vocab projection — cheap enough for c_d budget
    probes before heavy train.
    """

    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        hidden_mult: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model < 1 or vocab_size < 2:
            raise ValueError("invalid d_model / vocab_size")
        hid = int(d_model * hidden_mult)
        self.d_model = int(d_model)
        self.vocab_size = int(vocab_size)
        self.fc1 = nn.Linear(d_model, hid, bias=True)
        self.fc2 = nn.Linear(hid, d_model, bias=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Args:
            hidden_states: [..., d_model] previous-step features.
        Returns:
            logits: [..., vocab]
        """
        x = self.norm(hidden_states)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = hidden_states + self.drop(x)
        return self.lm_head(x)

    def init_from_teacher_proj(self, teacher: nn.Module) -> dict[str, Any]:
        """Copy teacher ``proj_out`` / ``lm_head`` weights when shapes match."""
        meta: dict[str, Any] = {"copied": False, "source": None}
        for name in ("proj_out", "lm_head"):
            mod = getattr(teacher, name, None)
            if mod is None:
                continue
            weight = getattr(mod, "weight", None)
            if (
                isinstance(weight, torch.Tensor)
                and weight.ndim == 2
                and tuple(weight.shape) == tuple(self.lm_head.weight.shape)
            ):
                with torch.no_grad():
                    self.lm_head.weight.copy_(weight.detach())
                meta["copied"] = True
                meta["source"] = name
                break
        return meta


def estimate_mlp_head_ms(
    *,
    d_model: int = 768,
    vocab_size: int = 4097,
    hidden_mult: int = 2,
    gamma: int = 3,
    peak_tflops: float = 10.0,
) -> dict[str, float]:
    """Order-of-magnitude GPU time from FLOPs (not a measured claim)."""
    hid = d_model * hidden_mult
    # GEMMs per draft token: d→hid, hid→d, d→V (multiply-adds ≈ 2*m*n*k)
    flops_tok = 2.0 * (
        d_model * hid + hid * d_model + d_model * vocab_size
    )
    flops_chain = flops_tok * float(gamma)
    seconds = flops_chain / (peak_tflops * 1e12)
    ms_chain = seconds * 1e3
    return {
        "flops_per_token": flops_tok,
        "flops_gamma_chain": flops_chain,
        "ms_chain_est": ms_chain,
        "ms_per_token_est": ms_chain / float(gamma),
        "peak_tflops_assumed": peak_tflops,
        "gamma": float(gamma),
    }


def feature_shift_pairs(
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, slice]:
    """EAGLE uses h_{t-1} to predict token_t / logits_t.

    ``hidden_states``: [T, D] teacher-force decoder states aligned with logits[T,V].
    Returns features for positions 1..T-1 (predict from previous hidden).
    """
    if hidden_states.ndim != 2:
        raise ValueError("expected [T, D] hidden_states")
    t = int(hidden_states.shape[0])
    if t < 2:
        raise ValueError("need >=2 positions for EAGLE shift")
    return hidden_states[:-1], slice(1, t)
