"""§41 teacher verify: K-token fastpath + graph-aligned Q=1 (TIER1a).

Turbo-only. Does not change bit-exact optimized engine presets/semantics.
Reuses StaticCache, active-prefix, native q1/cross-mlp hooks, and optional
CUDA graphs from optimized machinery via the turbo package.

Two modes:
- **perf / K-token**: one multi-token teacher forward (c_verify ≤1.2× Q=1).
- **canary / greedy**: sequential Q=1 graph-aligned steps so teacher logits
  match optimized `active_prefix_cuda_graph` decode (§40 STOP_ESCALATE).
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import torch
from transformers.cache_utils import StaticCache

from ..cache_utils import MapperatorinatorCache, get_cache


ACTIVE_PREFIX_BUCKET_SIZE = 64
DEFAULT_VERIFY_KS = (4, 5, 8)


def verify_fastpath_enabled() -> bool:
    raw = os.environ.get("MAPPERATORINATOR_TURBO_VERIFY_FASTPATH", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def use_verify_cuda_graphs() -> bool:
    # Default ON for Q=1 canary alignment; K-token capture still opt-in via
    # allow_graph on forward_k (microbench sets True).
    raw = os.environ.get("MAPPERATORINATOR_TURBO_VERIFY_CUDA_GRAPH", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def graph_aligned_teacher_enabled() -> bool:
    """Sequential Q=1 teacher path for greedy / TIER1a canary."""
    raw = os.environ.get("MAPPERATORINATOR_TURBO_TEACHER_ALIGNED", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _bucketed_prefix_length(cur_len: int, bucket_size: int, max_cache_len: int) -> int:
    if bucket_size <= 0:
        raise ValueError("verify fastpath bucket size must be positive")
    bucketed = ((cur_len + bucket_size - 1) // bucket_size) * bucket_size
    return min(bucketed, max_cache_len)


def allocate_teacher_static_cache(model, *, cfg_scale: float = 1.0) -> MapperatorinatorCache:
    if cfg_scale != 1.0:
        raise ValueError("turbo verify fastpath requires cfg_scale=1.0")
    return get_cache(model, batch_size=1, num_beams=1, cfg_scale=1.0)


@contextmanager
def teacher_aligned_runtime_context(*, precision: str) -> Iterator[None]:
    """Arm the same native decode hooks optimized MAP generation uses."""
    from ..optimized.single.runtime_context import (
        attention_runtime_context,
        decoder_layer_runtime_context,
    )

    dtype = torch.float16 if precision == "fp16" else torch.float32
    with attention_runtime_context(
        q1_bmm_cross_attention=True,
        native_q1_self_attention=True,
        native_q1_rope_cache_self_attention=True,
        expected_dtype=dtype,
    ), decoder_layer_runtime_context(native_cross_mlp_tail=True):
        yield


@dataclass(slots=True)
class DeviceTokenBuffer:
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
    """Teacher verify: K-token perf path + Q=1 graph-aligned canary path."""

    model: Any
    bucket_size: int = ACTIVE_PREFIX_BUCKET_SIZE
    use_cuda_graph: bool = True
    graph_aligned: bool = True
    graph_cache: dict[tuple[int, int], VerifyGraphEntry] = field(default_factory=dict)
    eager_calls: int = 0
    graph_replays: int = 0
    graph_captures: int = 0
    q1_steps: int = 0

    def _max_cache_len(self, model_kwargs: dict[str, Any]) -> int:
        cache = model_kwargs.get("past_key_values")
        if cache is not None and hasattr(cache, "get_max_cache_shape"):
            return int(cache.get_max_cache_shape())
        return int(self.model.config.max_target_positions)

    def _prepare_inputs(
        self,
        *,
        decoder_input_ids: torch.LongTensor,
        model_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        mk = dict(model_kwargs)
        mk.pop("cache_position", None)
        # Never feed raw audio into decode/graph capture once encoder_outputs exist.
        if mk.get("encoder_outputs") is not None:
            for key in ("frames", "inputs", "input_features"):
                mk.pop(key, None)
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
        for key in ("frames", "inputs", "input_features"):
            model_inputs.pop(key, None)
        if "cache_position" in model_inputs:
            model_kwargs["cache_position"] = model_inputs["cache_position"]
        return model_inputs

    def _capture_graph(
        self,
        *,
        model_inputs: dict[str, Any],
        prefix_length: int,
        k: int,
    ) -> VerifyGraphEntry:
        from ..optimized.single.runtime_context import active_prefix_self_attention_context

        static_inputs = _clone_static_graph_inputs(model_inputs)
        # warmup=0 — match optimized production capture (StaticCache live write).
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        t0 = time.perf_counter()
        with torch.cuda.graph(graph):
            with active_prefix_self_attention_context(prefix_length):
                outputs = self.model(**static_inputs, return_dict=True)
        torch.cuda.synchronize()
        self.graph_captures += 1
        return VerifyGraphEntry(
            graph=graph,
            outputs=outputs,
            static_inputs=static_inputs,
            prefix_length=prefix_length,
            k=k,
            capture_seconds=time.perf_counter() - t0,
        )

    def _forward_shaped(
        self,
        *,
        decoder_input_ids: torch.LongTensor,
        model_kwargs: dict[str, Any],
        k: int,
        allow_graph: bool,
    ) -> tuple[Any, dict[str, Any]]:
        from ..optimized.single.runtime_context import active_prefix_self_attention_context

        model_inputs = self._prepare_inputs(
            decoder_input_ids=decoder_input_ids,
            model_kwargs=model_kwargs,
        )
        past = model_kwargs.get("past_key_values")
        can_graph = (
            allow_graph
            and self.use_cuda_graph
            and torch.cuda.is_available()
            and isinstance(getattr(past, "self_attention_cache", None), StaticCache)
            and decoder_input_ids.device.type == "cuda"
        )
        cur_len = int(decoder_input_ids.shape[1])
        prefix_length = _bucketed_prefix_length(
            cur_len,
            self.bucket_size,
            self._max_cache_len(model_kwargs),
        )
        if can_graph:
            dec = model_inputs.get("decoder_input_ids")
            dec_shape = tuple(dec.shape) if isinstance(dec, torch.Tensor) else ()
            key = (prefix_length, int(k), dec_shape)
            entry = self.graph_cache.get(key)
            if entry is None:
                entry = self._capture_graph(
                    model_inputs=model_inputs,
                    prefix_length=prefix_length,
                    k=int(k),
                )
                self.graph_cache[key] = entry
                outputs = entry.outputs
            else:
                try:
                    _copy_static_graph_inputs(entry.static_inputs, model_inputs)
                    entry.graph.replay()
                    entry.replays += 1
                    self.graph_replays += 1
                    outputs = entry.outputs
                except RuntimeError:
                    # Shape/address drift — recapture under same key.
                    entry = self._capture_graph(
                        model_inputs=model_inputs,
                        prefix_length=prefix_length,
                        k=int(k),
                    )
                    self.graph_cache[key] = entry
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

    def forward_k(
        self,
        *,
        decoder_input_ids: torch.LongTensor,
        model_kwargs: dict[str, Any],
        k: int,
        allow_graph: bool | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Multi-token teacher forward (perf path)."""
        if k <= 0:
            raise ValueError("k must be positive")
        if int(decoder_input_ids.shape[1]) < k:
            raise ValueError("decoder_input_ids shorter than k")
        # Default: no CUDA graph for K>1 (capture shape zoo); microbench opts in.
        use_graph = False if allow_graph is None else bool(allow_graph)
        if k == 1 and allow_graph is None:
            use_graph = self.use_cuda_graph
        return self._forward_shaped(
            decoder_input_ids=decoder_input_ids,
            model_kwargs=model_kwargs,
            k=k,
            allow_graph=use_graph,
        )

    def forward_q1(
        self,
        *,
        decoder_input_ids: torch.LongTensor,
        model_kwargs: dict[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        """One optimized-style Q=1 step (graph-aligned when CUDA graph on)."""
        self.q1_steps += 1
        return self._forward_shaped(
            decoder_input_ids=decoder_input_ids,
            model_kwargs=model_kwargs,
            k=1,
            allow_graph=self.use_cuda_graph,
        )

    def verify_sequential_q1(
        self,
        *,
        prompt_ids: torch.LongTensor,
        draft_ids: Sequence[int],
        model_kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Teacher-force draft via sequential Q=1 steps (TIER1a / greedy).

        Returns verify_logits[γ,V] (logits after each draft token), bonus_logits[V]
        (logits after last draft token — same as verify_logits[-1]), updated kwargs.
        """
        mk = model_kwargs
        ids = prompt_ids
        step_logits: list[torch.Tensor] = []
        for tok in draft_ids:
            tok_t = torch.tensor([[int(tok)]], device=ids.device, dtype=ids.dtype)
            ids = torch.cat([ids, tok_t], dim=-1)
            outputs, mk = self.forward_q1(decoder_input_ids=ids, model_kwargs=mk)
            step_logits.append(outputs.logits[:, -1, :].float().squeeze(0))
        if not step_logits:
            raise RuntimeError("sequential Q=1 verify produced no logits")
        verify = torch.stack(step_logits, dim=0)
        return verify, verify[-1].clone(), mk

    def profile_metadata(self) -> dict[str, Any]:
        return {
            "turbo_verify_fastpath": True,
            "turbo_verify_bucket_size": self.bucket_size,
            "turbo_verify_cuda_graph": self.use_cuda_graph,
            "turbo_teacher_graph_aligned": self.graph_aligned,
            "turbo_verify_eager_calls": self.eager_calls,
            "turbo_verify_graph_replays": self.graph_replays,
            "turbo_verify_graph_captures": self.graph_captures,
            "turbo_verify_graph_entries": len(self.graph_cache),
            "turbo_verify_q1_steps": self.q1_steps,
        }


def build_teacher_verify_fastpath(model) -> TeacherVerifyFastpath | None:
    if not verify_fastpath_enabled():
        return None
    return TeacherVerifyFastpath(
        model=model,
        bucket_size=ACTIVE_PREFIX_BUCKET_SIZE,
        use_cuda_graph=use_verify_cuda_graphs(),
        graph_aligned=graph_aligned_teacher_enabled(),
    )


__all__ = [
    "ACTIVE_PREFIX_BUCKET_SIZE",
    "DEFAULT_VERIFY_KS",
    "DeviceTokenBuffer",
    "TeacherVerifyFastpath",
    "allocate_teacher_static_cache",
    "build_teacher_verify_fastpath",
    "graph_aligned_teacher_enabled",
    "teacher_aligned_runtime_context",
    "verify_fastpath_enabled",
    "use_verify_cuda_graphs",
]
