"""Persistent production state for optimized single-song decode windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from transformers.modeling_outputs import BaseModelOutput

from ...cache_utils import MapperatorinatorCache, get_cache


@dataclass
class ProductionDecodeSession:
    """State reused across windows of one unfinished output context.

    Sampling, logits processors, stopping criteria, and RNG deliberately remain
    per-window/global exactly as in the accepted runtime.
    """

    cache: MapperatorinatorCache | None = None
    graph_cache: dict[tuple[Any, ...], dict[str, Any]] = field(default_factory=dict)
    stable_encoder_holder: dict[str, BaseModelOutput] = field(default_factory=dict)

    def cache_for_window(
        self,
        model,
        *,
        batch_size: int,
        num_beams: int,
        cfg_scale: float,
    ) -> MapperatorinatorCache:
        if self.cache is None:
            self.cache = get_cache(model, batch_size, num_beams, cfg_scale)
        else:
            self.cache.self_attention_cache.reset()
            self.cache.cross_attention_cache.reset()
            for layer_idx in list(self.cache.is_updated):
                self.cache.is_updated[layer_idx] = False
        return self.cache

    def active_prefix_decode_kwargs(self) -> dict[str, Any]:
        return {
            "shared_graph_cache": self.graph_cache,
            "stable_encoder_holder": self.stable_encoder_holder,
        }

    @property
    def graph_count(self) -> int:
        return len(self.graph_cache)
