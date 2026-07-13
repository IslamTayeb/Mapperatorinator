"""Persistent shape-aware state for optimized decode windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from transformers.modeling_outputs import BaseModelOutput

from ...cache_utils import MapperatorinatorCache, get_cache


@dataclass
class ProductionDecodeSession:
    """State reused across windows of one unfinished output context.

    Sampling, logits processors, stopping criteria, and RNG deliberately remain
    per-window/global exactly as in the accepted runtime.
    """

    caches: dict[tuple[Any, ...], MapperatorinatorCache] = field(default_factory=dict)
    graph_cache: dict[tuple[Any, ...], dict[str, Any]] = field(default_factory=dict)
    stable_encoder_holders: dict[
        tuple[Any, ...],
        dict[str, BaseModelOutput],
    ] = field(default_factory=dict)
    active_state_signature: tuple[Any, ...] | None = None

    @staticmethod
    def _state_signature(
        model,
        *,
        batch_size: int,
        num_beams: int,
        cfg_scale: float,
    ) -> tuple[Any, ...]:
        if batch_size <= 0:
            raise ValueError("decode cache batch_size must be positive")
        if num_beams <= 0:
            raise ValueError("decode cache num_beams must be positive")
        if cfg_scale <= 0:
            raise ValueError("decode cache cfg_scale must be positive")
        dtype = getattr(model, "dtype", None)
        device = getattr(model, "device", None)
        if not isinstance(dtype, torch.dtype):
            raise TypeError("decode cache model must expose a torch dtype")
        if device is None:
            raise TypeError("decode cache model must expose a device")
        cfg_multiplier = 2 if cfg_scale > 1 else 1
        return (
            id(model),
            int(batch_size),
            int(num_beams),
            cfg_multiplier,
            str(dtype),
            str(torch.device(device)),
        )

    def cache_for_window(
        self,
        model,
        *,
        batch_size: int,
        num_beams: int,
        cfg_scale: float,
    ) -> MapperatorinatorCache:
        signature = self._state_signature(
            model,
            batch_size=batch_size,
            num_beams=num_beams,
            cfg_scale=cfg_scale,
        )
        cache = self.caches.get(signature)
        if cache is None:
            cache = get_cache(model, batch_size, num_beams, cfg_scale)
            self.caches[signature] = cache
        else:
            cache.self_attention_cache.reset()
            cache.cross_attention_cache.reset()
            for layer_idx in list(cache.is_updated):
                cache.is_updated[layer_idx] = False
        self.active_state_signature = signature
        self.stable_encoder_holders.setdefault(signature, {})
        return cache

    def active_prefix_decode_kwargs(self) -> dict[str, Any]:
        if self.active_state_signature is None:
            raise RuntimeError(
                "decode cache must be selected before requesting graph state"
            )
        return {
            "shared_graph_cache": self.graph_cache,
            "stable_encoder_holder": self.stable_encoder_holders[
                self.active_state_signature
            ],
        }

    @property
    def graph_count(self) -> int:
        return len(self.graph_cache)

    @property
    def graph_capture_seconds(self) -> float:
        return sum(
            float(entry.get("capture_seconds", 0.0) or 0.0)
            for entry in self.graph_cache.values()
        )

    @property
    def graph_decode_replays(self) -> int:
        return sum(
            int(entry.get("decode_replays", 0) or 0)
            for entry in self.graph_cache.values()
        )
