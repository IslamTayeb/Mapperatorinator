"""Persistent shape-aware state for optimized decode windows."""

from __future__ import annotations

from contextlib import nullcontext
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

    def window_lease(
        self,
        model,
        *,
        batch_size: int,
        num_beams: int,
        cfg_scale: float,
    ):
        """Return the default request-local no-op workspace lease."""

        del model, batch_size, num_beams, cfg_scale
        return nullcontext()

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

    def graph_profile_summary(self) -> dict[str, Any]:
        """Return address-free CUDA-graph evidence suitable for profiles."""
        buckets: dict[str, dict[str, int | float]] = {}
        for entry in self.graph_cache.values():
            prefix = int(entry.get("active_prefix_length", -1))
            if prefix < 0:
                raise RuntimeError("optimized graph entry is missing active_prefix_length")
            bucket = buckets.setdefault(
                str(prefix),
                {"graph_count": 0, "decode_replays": 0, "capture_seconds": 0.0},
            )
            bucket["graph_count"] = int(bucket["graph_count"]) + 1
            bucket["decode_replays"] = int(bucket["decode_replays"]) + int(
                entry.get("decode_replays", 0)
            )
            bucket["capture_seconds"] = float(bucket["capture_seconds"]) + float(
                entry.get("capture_seconds", 0.0)
            )
        result = {
            "graph_count": self.graph_count,
            "decode_replays": sum(
                int(bucket["decode_replays"]) for bucket in buckets.values()
            ),
            "capture_seconds": sum(
                float(bucket["capture_seconds"]) for bucket in buckets.values()
            ),
            "buckets": dict(sorted(buckets.items(), key=lambda item: int(item[0]))),
        }
        latest = None
        if self.active_state_signature is not None:
            holder = self.stable_encoder_holders.get(self.active_state_signature, {})
            candidate = holder.get("__k8_latest_stats__")
            if isinstance(candidate, dict) and candidate.get("k8_candidate") is True:
                latest = candidate
        if latest is not None:
            result["k8_candidate"] = {
                "block_size": int(latest.get("block_size", 8)),
                "prefill_steps": int(latest.get("prefill_steps", 0)),
                "eligible_steps": int(latest.get("eligible_steps", 0)),
                "block_replays": int(latest.get("block_replays", 0)),
                "remainder_steps": int(latest.get("remainder_steps", 0)),
                "remainder_backend": latest.get("remainder_backend", "eager"),
                "remainder_graph_replays": int(
                    latest.get("remainder_graph_replays", 0)
                ),
                "eager_remainder_steps": int(
                    latest.get("eager_remainder_steps", 0)
                ),
                "status_reads": int(latest.get("status_reads", 0)),
                "copy_calls": int(latest.get("copy_calls", 0)),
                "copy_bytes": int(latest.get("copy_bytes", 0)),
                "capture_seconds": float(latest.get("capture_seconds", 0.0)),
                "peak_vram_bytes": int(latest.get("peak_vram_bytes", 0)),
                "physical_steps": int(latest.get("physical_steps", 0)),
                "logical_steps": int(latest.get("logical_steps", 0)),
                "wasted_steps": int(latest.get("wasted_steps", 0)),
                "rng_policy": latest.get("rng_policy"),
                "rng_exact": bool(latest.get("rng_exact", False)),
                "rng_drift": latest.get("rng_drift"),
                "rng_request_seed": int(latest.get("rng_request_seed", 0)),
                "rng_window_identity": int(latest.get("rng_window_identity", 0)),
                "rng_early_eos_isolation": bool(
                    latest.get("rng_early_eos_isolation", False)
                ),
                "sampling_mode": latest.get("sampling_mode"),
                "prompt_seed_d2h_copy_calls": int(
                    latest.get("prompt_seed_d2h_copy_calls", 0)
                ),
                "prompt_seed_d2h_copy_bytes": int(
                    latest.get("prompt_seed_d2h_copy_bytes", 0)
                ),
                "prompt_seed_setup_seconds": float(
                    latest.get("prompt_seed_setup_seconds", 0.0)
                ),
                "processor_signature_d2h_copy_calls": int(
                    latest.get("processor_signature_d2h_copy_calls", 0)
                ),
                "processor_signature_d2h_copy_bytes": int(
                    latest.get("processor_signature_d2h_copy_bytes", 0)
                ),
                "processor_signature_setup_seconds": float(
                    latest.get("processor_signature_setup_seconds", 0.0)
                ),
                "parent_backend": latest.get("parent_backend"),
                "capture_state_restore_synchronized": bool(
                    latest.get("capture_state_restore_synchronized", False)
                ),
            }
            for key in (
                "shared_static_input_arena",
                "static_input_arena_refreshes",
                "static_input_arena_refresh_copy_calls",
                "static_input_arena_refresh_copy_bytes",
                "static_input_arena_graph_entries",
                "transition_timing",
            ):
                if key in latest:
                    result["k8_candidate"][key] = latest[key]
        return result
