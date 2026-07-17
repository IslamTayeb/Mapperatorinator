"""O(1) keep-accepted KV rollback for sampled turbo (§47 W-KV).

StaticCache.get_seq_length is nonzero-occupancy based, so rejected slots in
``[new_len, occupied_end)`` must be cleared. That band is ≤γ (fixed small), not
a full ``max_cache_len`` wipe. Stale capacity beyond ``occupied_end`` stays
untouched; attention uses ``cache_position`` / active-prefix masking.
"""
from __future__ import annotations

from typing import Any

import torch
from transformers import DynamicCache
from transformers.cache_utils import StaticCache

from ..cache_utils import MapperatorinatorCache


def _reset_side_cache(side_cache: Any) -> None:
    if isinstance(side_cache, DynamicCache):
        if hasattr(side_cache, "layers"):
            side_cache.layers.clear()
        if hasattr(side_cache, "_seen_tokens"):
            side_cache._seen_tokens = 0
        return
    if isinstance(side_cache, StaticCache):
        side_cache.reset()
        return
    raise TypeError(f"unsupported cache for reset: {type(side_cache).__name__}")


def reset_self_cache(cache: MapperatorinatorCache) -> None:
    """In-place clear for a new window; keeps StaticCache storage (graph-safe).

    Clears self + cross: encoder KV from the previous window must not linger.
    """
    _reset_side_cache(cache.self_attention_cache)
    _reset_side_cache(cache.cross_attention_cache)


def rewind_self_cache(
    cache: MapperatorinatorCache,
    new_len: int,
    *,
    occupied_end: int,
) -> None:
    """Roll logical KV length back to ``new_len`` without re-forwarding.

    For StaticCache: zero only ``[new_len, occupied_end)`` (O(γ)).
    For DynamicCache: HF crop (slice) to ``new_len``.
    """
    new_len = int(new_len)
    occupied_end = int(occupied_end)
    if new_len < 0:
        raise ValueError("new_len must be non-negative")
    if occupied_end < new_len:
        raise ValueError("occupied_end must be >= new_len")

    self_cache = cache.self_attention_cache
    if isinstance(self_cache, DynamicCache):
        cache.crop(new_len)
        return

    if not isinstance(self_cache, StaticCache):
        raise TypeError(
            f"unsupported self cache for rewind: {type(self_cache).__name__}"
        )

    if occupied_end > new_len:
        for layer in self_cache.layers:
            if not getattr(layer, "is_initialized", False):
                continue
            end = min(occupied_end, int(layer.keys.shape[2]))
            if new_len < end:
                layer.keys[:, :, new_len:end, :].zero_()
                layer.values[:, :, new_len:end, :].zero_()
            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length = new_len


def align_kwargs_after_rewind(
    model_kwargs: dict[str, Any],
    *,
    length: int,
    device: torch.device,
) -> None:
    """Rewind decoder mask + drop stale cache_position (recomputed on next prep)."""
    model_kwargs["decoder_attention_mask"] = torch.ones(
        (1, int(length)),
        device=device,
        dtype=torch.long,
    )
    model_kwargs.pop("cache_position", None)


__all__ = [
    "align_kwargs_after_rewind",
    "reset_self_cache",
    "rewind_self_cache",
]
