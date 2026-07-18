"""Teacher verify on tiger's uniform CUDAGraphDecoder at q_len=K (§58).

No fused-kernel forfeiture / §54 Stage B grind. Same HF forward as production
q_len=1 decode, captured at K for speculative verify.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ..compiled_decode import get_k_decoder
from .kv_rollback import rewind_self_cache


@dataclass
class TigerVerifySession:
    """Persistent q_len=K graph + StaticCache across turbo windows."""

    decoder: Any = None
    cache: Any = None
    cache_len: int = 0
    q_len: int = 0
    enc_shape: tuple[int, ...] | None = None
    forward_ms_total: float = 0.0
    forward_timed_calls: int = 0
    replays: int = 0
    hit_counters: dict[str, int] = field(
        default_factory=lambda: {
            "tiger_verify_graph_replay": 0,
            "tiger_verify_graph_capture": 0,
        }
    )


def ensure_tiger_verify(
    session: TigerVerifySession,
    model,
    *,
    enc_hidden: torch.Tensor,
    cache_len: int,
    q_len: int,
    cfg_scale: float = 1.0,
) -> TigerVerifySession:
    """Capture or reuse the K-token tiger decode graph."""
    enc_shape = tuple(enc_hidden.shape)
    need_new = (
        session.decoder is None
        or session.q_len != int(q_len)
        or session.cache_len != int(cache_len)
        or session.enc_shape != enc_shape
    )
    if need_new:
        decoder, cache = get_k_decoder(
            model,
            batch_size=1,
            cfg_scale=cfg_scale,
            enc_shape=enc_shape,
            dtype=model.dtype,
            cache_len=int(cache_len),
            q_len=int(q_len),
        )
        session.decoder = decoder
        session.cache = cache
        session.cache_len = int(cache_len)
        session.q_len = int(q_len)
        session.enc_shape = enc_shape
        session.hit_counters["tiger_verify_graph_capture"] += 1
    session.decoder.set_encoder_hidden(enc_hidden)
    return session


@torch.no_grad()
def tiger_verify_forward_k(
    session: TigerVerifySession,
    *,
    draft_tokens: torch.Tensor,
    prefix_len: int,
    time_it: bool = False,
    rewind_to_prefix: bool = False,
) -> torch.Tensor:
    """Replay K-token verify; returns logits ``(K, V)`` for batch-1.

    Caller must have prefilled ``session.cache`` through ``prefix_len``.
    Default leaves KV at ``prefix_len+K`` so keep-accepted-KV can rewind to
    the accepted length. Pass ``rewind_to_prefix=True`` for probe/microbench
    paths that need a clean prefix before the next replay.
    """
    k = int(session.q_len)
    if draft_tokens.ndim != 2 or draft_tokens.shape[0] != 1 or draft_tokens.shape[1] != k:
        raise ValueError(f"draft_tokens must be (1, {k}), got {tuple(draft_tokens.shape)}")
    device = draft_tokens.device
    cache_pos = torch.arange(
        int(prefix_len), int(prefix_len) + k, device=device, dtype=torch.long
    )

    if time_it:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        logits = session.decoder.replay(draft_tokens, cache_pos)
        end.record()
        torch.cuda.synchronize()
        session.forward_ms_total += float(start.elapsed_time(end))
        session.forward_timed_calls += 1
    else:
        logits = session.decoder.replay(draft_tokens, cache_pos)

    session.replays += 1
    session.hit_counters["tiger_verify_graph_replay"] += 1
    if rewind_to_prefix:
        rewind_self_cache(session.cache, int(prefix_len), occupied_end=int(prefix_len) + k)
    if logits.ndim == 3:
        return logits[0].float()
    return logits.float()


__all__ = [
    "TigerVerifySession",
    "ensure_tiger_verify",
    "tiger_verify_forward_k",
]
