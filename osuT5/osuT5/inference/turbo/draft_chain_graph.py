"""§58 tiger graphed-draft scaffold: γ q_len=1 steps on draft via CUDAGraphDecoder.

Ports the §49 contract (StaticCache draft + keep-KV rewind of the γ-th slot)
onto Tiger's uniform graph path — no ``inference.optimized`` imports.

Full in-graph sample+embed chain (Philox multinomial) is the next rung after
c_verify; this module lands the decoder graph + rewind surface used by that work.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch

from ..compiled_decode import CUDAGraphDecoder, CaptureError, _reset_cache
from ..cache_utils import get_cache
from .kv_rollback import rewind_self_cache
from .rejection import apply_temp_top_p

DRAFT_CHAIN_GRAPH_ENV = "MAPPERATORINATOR_TURBO_DRAFT_CHAIN_GRAPH"
DEFAULT_GAMMA = 3
DEFAULT_TOP_P = 0.9


def draft_chain_graph_enabled() -> bool:
    raw = os.environ.get(DRAFT_CHAIN_GRAPH_ENV, "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


@dataclass
class TigerDraftChainRunner:
    """Persistent draft q_len=1 CUDA graph + keep-KV rewind."""

    draft: Any
    gamma: int = DEFAULT_GAMMA
    temperature: float = 0.9
    top_p: float = DEFAULT_TOP_P
    cache_len: int | None = None
    decoder: CUDAGraphDecoder | None = None
    cache: Any = None
    enc_shape: tuple[int, ...] | None = None
    chain_replays: int = 0
    graph_captures: int = 0

    def ensure(self, *, enc_hidden: torch.Tensor, cache_len: int) -> None:
        need = (
            self.decoder is None
            or self.cache_len != int(cache_len)
            or self.enc_shape != tuple(enc_hidden.shape)
        )
        if not need:
            self.decoder.set_encoder_hidden(enc_hidden)
            return
        self.cache = get_cache(
            self.draft, batch_size=1, num_beams=1, cfg_scale=1.0, max_cache_len=cache_len
        )
        self.decoder = CUDAGraphDecoder(
            self.draft,
            batch_size=1,
            enc_shape=enc_hidden.shape,
            dtype=self.draft.dtype,
            q_len=1,
        )
        try:
            self.decoder.capture(self.cache)
        except Exception as exc:  # noqa: BLE001
            raise CaptureError(str(exc)) from exc
        self.cache_len = int(cache_len)
        self.enc_shape = tuple(enc_hidden.shape)
        self.graph_captures += 1
        self.decoder.set_encoder_hidden(enc_hidden)

    def rewind_keep_kv(self, keep_len: int) -> None:
        if self.cache is None:
            return
        # Occupied end is at most keep_len + gamma (last chain).
        rewind_self_cache(
            self.cache, int(keep_len), occupied_end=int(keep_len) + int(self.gamma) + 1
        )

    def reset_window(self) -> None:
        if self.cache is not None:
            _reset_cache(self.cache)

    @torch.no_grad()
    def propose_chain(
        self,
        *,
        seed_logits: torch.Tensor,
        past_length: int,
        rng: torch.Generator | None = None,
    ) -> dict[str, Any]:
        """Sample γ tokens with γ graph replays (eager host sample between)."""
        if self.decoder is None or self.cache is None:
            raise RuntimeError("call ensure() before propose_chain")
        device = seed_logits.device
        tokens: list[int] = []
        q_rows: list[torch.Tensor] = []
        logits_after: list[torch.Tensor] = []
        logits = seed_logits.float()
        cur = int(past_length)
        for _ in range(int(self.gamma)):
            probs = apply_temp_top_p(logits.view(1, -1), self.temperature, self.top_p)
            q_rows.append(probs.squeeze(0))
            tok = int(torch.multinomial(probs.view(-1), 1, generator=rng).item())
            tokens.append(tok)
            tok_t = torch.tensor([[tok]], device=device, dtype=torch.long)
            cache_pos = torch.tensor([cur], device=device, dtype=torch.long)
            logits = self.decoder.replay(tok_t, cache_pos)
            if logits.ndim == 2:
                step_logits = logits[0].float()
            else:
                step_logits = logits.float().view(-1)
            logits_after.append(step_logits)
            cur += 1
            self.chain_replays += 1
        return {
            "tokens": tokens,
            "q_probs": torch.stack(q_rows, dim=0),
            "logits_after": logits_after,
            "tail_logits": logits_after[-1] if logits_after else seed_logits.float(),
            "end_length": cur,
            "path": "tiger_draft_q1_chain",
        }

    def meta(self) -> dict[str, Any]:
        return {
            "tiger_draft_chain_gamma": self.gamma,
            "tiger_draft_chain_replays": self.chain_replays,
            "tiger_draft_graph_captures": self.graph_captures,
            "tiger_draft_cache_len": self.cache_len,
            "enabled": draft_chain_graph_enabled(),
        }


__all__ = [
    "TigerDraftChainRunner",
    "draft_chain_graph_enabled",
    "DEFAULT_GAMMA",
    "CaptureError",
]
