"""§41 K-token teacher verify on optimized-style StaticCache machinery.

Turbo-only. Does not change bit-exact optimized engine presets or semantics.
Reuses StaticCache + active-prefix CUDA-graph patterns from optimized/single
for multi-token (K≈4–8) teacher verify; optional per-K graph capture.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers.cache_utils import StaticCache

from ..cache_utils import MapperatorinatorCache, get_cache


ACTIVE_PREFIX_BUCKET_SIZE = 64
DEFAULT_VERIFY_KS = (4, 5, 8)


def verify_fastpath_enabled() -> bool:
    raw = os.environ.get("MAPPERATORINATOR_TURBO_VERIFY_FASTPATH", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def use_verify_cuda_graphs() -> bool:
    # Default off: capture must be the live first forward (no warmup on StaticCache).
    # Microbench opts in via MAPPERATORINATOR_TURBO_VERIFY_CUDA_GRAPH=1.
    raw = os.environ.get("MAPPERATORINATOR_TURBO_VERIFY_CUDA_GRAPH", "0").strip().lower()
    return raw in {"1", "true", "on", "yes"}


def _bucketed_prefix_length(cur_len: int, bucket_size: int, max_cache_len: int) -> int:
    if bucket_size <= 0:
        raise ValueError("verify fastpath bucket size must be positive")
    bucketed = ((cur_len + bucket_size - 1) // bucket_size) * bucket_size
    return min(bucketed, max_cache_len)


def allocate_teacher_static_cache(model, *, cfg_scale: float = 1.0) -> MapperatorinatorCache:
    """Static KV for teacher verify (crop-capable; same allocator as optimized)."""
    if cfg_scale != 1.0:
        raise ValueError("turbo verify fastpath requires cfg_scale=1.0")
    return get_cache(model, batch_size=1, num_beams=1, cfg_scale=1.0)


@dataclass(slots=True)
class DeviceTokenBuffer:
    """Fixed-address token storage for multi-token verify/commit (device-side)."""

    sequence: torch.Tensor
    length: int
    max_length: int

    @classmethod
    def allocate(
        cls,
        prompt_ids: torch.Tensor,
        *,
        max_length: int,
    ) -> "DeviceTokenBuffer":
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
            raise ValueError("prompt_ids must have shape [1, T]")
        if prompt_ids.dtype != torch.long:
            raise TypeError("prompt_ids must be torch.long")
        prompt_len = int(prompt_ids.shape[1])
        if prompt_len <= 0 or prompt_len >= max_length:
            raise ValueError("prompt length must be in (0, max_length)")
        sequence = prompt_ids.new_empty((1, max_length))
        sequence[:, :prompt_len].copy_(prompt_ids)
        return cls(sequence=sequence, length=prompt_len, max_length=max_length)

    @property
    def active(self) -> torch.Tensor:
        return self.sequence[:, : self.length]

    def append_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError("token_ids must have shape [1, K]")
        k = int(token_ids.shape[1])
        if k <= 0:
            raise ValueError("token_ids must be non-empty")
        if self.length + k > self.max_length:
            raise RuntimeError("DeviceTokenBuffer overflow")
        end = self.length + k
        self.sequence[:, self.length : end].copy_(token_ids)
        self.length = end
        return self.active

    def truncate(self, length: int) -> None:
        if length < 0 or length > self.length:
            raise ValueError("truncate length out of range")
        self.length = int(length)


def _clone_static_graph_inputs(model_inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            value.clone(memory_format=torch.contiguous_format)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in model_inputs.items()
    }


def _copy_static_graph_inputs(
    static_inputs: dict[str, Any],
    model_inputs: dict[str, Any],
) -> None:
    for key, static_value in static_inputs.items():
        if not isinstance(static_value, torch.Tensor):
            continue
        value = model_inputs.get(key)
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(
                f"CUDA graph input {key} changed from tensor to {type(value).__name__}"
            )
        if (
            static_value.shape != value.shape
            or static_value.dtype != value.dtype
            or static_value.device != value.device
        ):
            raise RuntimeError(f"CUDA graph input {key} shape/dtype/device changed")
        static_value.copy_(value)


@dataclass
class VerifyGraphEntry:
    graph: torch.cuda.CUDAGraph
    outputs: Any
    static_inputs: dict[str, Any]
    prefix_length: int
    k: int
    capture_seconds: float
    replays: int = 0


@dataclass
class TeacherVerifyFastpath:
    """K-token teacher verify with StaticCache + optional per-K CUDA graphs."""

    model: Any
    bucket_size: int = ACTIVE_PREFIX_BUCKET_SIZE
    use_cuda_graph: bool = True
    graph_cache: dict[tuple[int, int], VerifyGraphEntry] = field(default_factory=dict)
    eager_calls: int = 0
    graph_replays: int = 0
    graph_captures: int = 0

    def _max_cache_len(self, model_kwargs: dict[str, Any]) -> int:
        cache = model_kwargs.get("past_key_values")
        if cache is not None and hasattr(cache, "get_max_cache_shape"):
            return int(cache.get_max_cache_shape())
        return int(self.model.config.max_target_positions)

    def _prepare_verify_inputs(
        self,
        *,
        decoder_input_ids: torch.LongTensor,
        model_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        mk = dict(model_kwargs)
        mk.pop("cache_position", None)
        length = int(decoder_input_ids.shape[1])
        mk["decoder_attention_mask"] = torch.ones(
            (1, length),
            device=decoder_input_ids.device,
            dtype=torch.long,
        )
        model_inputs = self.model.prepare_inputs_for_generation(
            decoder_input_ids,
            **mk,
        )
        if "cache_position" in model_inputs:
            mk["cache_position"] = model_inputs["cache_position"]
        return model_inputs

    def _capture_graph(
        self,
        *,
        model_inputs: dict[str, Any],
        prefix_length: int,
        k: int,
    ) -> VerifyGraphEntry:
        """Capture is the live first forward (StaticCache must not be warmed)."""
        from ..optimized.single.runtime_context import active_prefix_self_attention_context

        static_inputs = _clone_static_graph_inputs(model_inputs)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        t0 = time.perf_counter()
        with torch.cuda.graph(graph):
            with active_prefix_self_attention_context(prefix_length):
                outputs = self.model(**static_inputs, return_dict=True)
        torch.cuda.synchronize()
        capture_s = time.perf_counter() - t0
        self.graph_captures += 1
        return VerifyGraphEntry(
            graph=graph,
            outputs=outputs,
            static_inputs=static_inputs,
            prefix_length=prefix_length,
            k=k,
            capture_seconds=capture_s,
        )

    def forward_k(
        self,
        *,
        decoder_input_ids: torch.LongTensor,
        model_kwargs: dict[str, Any],
        k: int,
        allow_graph: bool | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """One multi-token teacher forward over the last K tokens (full ids)."""
        if k <= 0:
            raise ValueError("k must be positive")
        if int(decoder_input_ids.shape[1]) < k:
            raise ValueError("decoder_input_ids shorter than k")

        use_graph = self.use_cuda_graph if allow_graph is None else bool(allow_graph)
        model_inputs = self._prepare_verify_inputs(
            decoder_input_ids=decoder_input_ids,
            model_kwargs=model_kwargs,
        )
        # Sync cache_position into kwargs for HF update helper.
        if "cache_position" in model_inputs:
            model_kwargs["cache_position"] = model_inputs["cache_position"]

        past = model_kwargs.get("past_key_values")
        can_graph = (
            use_graph
            and torch.cuda.is_available()
            and isinstance(getattr(past, "self_attention_cache", None), StaticCache)
            and decoder_input_ids.device.type == "cuda"
        )

        from ..optimized.single.runtime_context import active_prefix_self_attention_context

        cur_len = int(decoder_input_ids.shape[1])
        prefix_length = _bucketed_prefix_length(
            cur_len,
            self.bucket_size,
            self._max_cache_len(model_kwargs),
        )

        if can_graph:
            key = (prefix_length, int(k))
            entry = self.graph_cache.get(key)
            if entry is None:
                entry = self._capture_graph(
                    model_inputs=model_inputs,
                    prefix_length=prefix_length,
                    k=int(k),
                )
                self.graph_cache[key] = entry
                # Capture advanced StaticCache; treat it as this call's forward.
                outputs = entry.outputs
            else:
                _copy_static_graph_inputs(entry.static_inputs, model_inputs)
                entry.graph.replay()
                entry.replays += 1
                self.graph_replays += 1
                outputs = entry.outputs
        else:
            with active_prefix_self_attention_context(prefix_length):
                outputs = self.model(**model_inputs, return_dict=True)
            self.eager_calls += 1

        model_kwargs = self.model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=True,
        )
        return outputs, model_kwargs

    def profile_metadata(self) -> dict[str, Any]:
        return {
            "turbo_verify_fastpath": True,
            "turbo_verify_bucket_size": self.bucket_size,
            "turbo_verify_cuda_graph": self.use_cuda_graph,
            "turbo_verify_eager_calls": self.eager_calls,
            "turbo_verify_graph_replays": self.graph_replays,
            "turbo_verify_graph_captures": self.graph_captures,
            "turbo_verify_graph_entries": len(self.graph_cache),
        }


def build_teacher_verify_fastpath(model) -> TeacherVerifyFastpath | None:
    if not verify_fastpath_enabled():
        return None
    return TeacherVerifyFastpath(
        model=model,
        bucket_size=ACTIVE_PREFIX_BUCKET_SIZE,
        use_cuda_graph=use_verify_cuda_graphs(),
    )


__all__ = [
    "ACTIVE_PREFIX_BUCKET_SIZE",
    "DEFAULT_VERIFY_KS",
    "DeviceTokenBuffer",
    "TeacherVerifyFastpath",
    "allocate_teacher_static_cache",
    "build_teacher_verify_fastpath",
    "verify_fastpath_enabled",
    "use_verify_cuda_graphs",
]
