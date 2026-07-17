"""§42 turbo draft fast path: 2-layer draft on optimized q1 kernels + own CUDA graph.

Turbo-only. Reuses optimized decode_loop / ProductionDecodeSession / q1 dispatch
machinery without mutating the bit-exact optimized default path.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from ...runtime_profiling import generation_profile_context
from transformers import StaticCache
from transformers.modeling_outputs import BaseModelOutput

from ..cache_utils import MapperatorinatorCache, get_cache
from ..optimized.single.decode_loop import (
    _bucketed_prefix_length,
    _capture_decode_cuda_graph,
    _copy_static_graph_inputs,
    _cuda_graph_signature,
    _clone_static_graph_inputs,
    _stable_encoder_outputs,
)
from ..optimized.single.runtime_context import active_prefix_self_attention_context
from ..optimized.single.state import ProductionDecodeSession
from .draft import get_decoder

ACTIVE_PREFIX_BUCKET_SIZE = 64
DRAFT_FASTPATH_ENV = "MAPPERATORINATOR_TURBO_DRAFT_FASTPATH"
_AUDIO_INPUT_KEYS = (
    "frames",
    "inputs",
    "input_features",
    "input_values",
    "inputs_embeds",
)


def draft_fastpath_enabled() -> bool:
    raw = os.environ.get(DRAFT_FASTPATH_ENV, "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def sync_draft_decoder_layer_count(draft) -> int:
    """Keep config layer counts aligned with the truncated 2-layer decoder."""
    n = len(get_decoder(draft).layers)
    configs = [
        getattr(draft, "config", None),
        getattr(getattr(draft, "transformer", None), "config", None),
        getattr(getattr(draft, "config", None), "backbone_config", None),
    ]
    for cfg in configs:
        if cfg is None:
            continue
        if hasattr(cfg, "decoder_layers"):
            cfg.decoder_layers = n
    return n


class _ExactDepthCacheConfig:
    """Minimal config so StaticCache allocates exactly ``num_layers`` slots."""

    def __init__(self, num_layers: int):
        self.num_hidden_layers = int(num_layers)
        self.layer_types = ["full_attention"] * int(num_layers)

    def get_text_config(self, decoder: bool = False):
        return self


def get_draft_cache(model, *, batch_size: int = 1, cfg_scale: float = 1.0) -> MapperatorinatorCache:
    """StaticCache sized to the model's decoder depth (not encoder depth).

    ``get_cache(model.config)`` uses MapperatorinatorConfig.num_hidden_layers
    (encoder depth, often 12). Draft/teacher runners pass the live decoder depth.
    """
    n_layers = sync_draft_decoder_layer_count(model)
    cache_config = _ExactDepthCacheConfig(n_layers)
    cache_kwargs = {
        "config": cache_config,
        "max_batch_size": batch_size * (2 if cfg_scale > 1 else 1),
        "max_cache_len": int(model.config.max_target_positions),
        "device": model.device,
        "dtype": model.dtype,
    }
    decoder_cache = StaticCache(**cache_kwargs)
    encoder_kwargs = dict(cache_kwargs)
    # Cross-attn cache depth follows decoder layers (one cross block per layer).
    encoder_kwargs["max_cache_len"] = int(model.config.max_source_positions)
    encoder_cache = StaticCache(**encoder_kwargs)
    self_layers = len(getattr(decoder_cache, "layers", []) or [])
    if self_layers != n_layers:
        raise RuntimeError(
            f"draft StaticCache has {self_layers} layers but decoder has {n_layers}"
        )
    return MapperatorinatorCache(decoder_cache, encoder_cache, cfg_scale)


def _strip_audio_inputs(model_inputs: dict[str, Any]) -> dict[str, Any]:
    """Drop audio tensors once encoder_outputs exist (keeps CUDA graphs decode-only)."""
    if model_inputs.get("encoder_outputs") is None:
        return model_inputs
    for key in _AUDIO_INPUT_KEYS:
        model_inputs.pop(key, None)
    return model_inputs


def _align_decoder_mask(
    model_kwargs: dict[str, Any],
    *,
    length: int,
    device: torch.device,
) -> None:
    model_kwargs["decoder_attention_mask"] = torch.ones(
        (1, int(length)),
        device=device,
        dtype=torch.long,
    )
    model_kwargs.pop("cache_position", None)


@dataclass
class DraftFastpathStats:
    prefill_seconds: float = 0.0
    decode_steps: int = 0
    graph_captures: int = 0
    graph_replays: int = 0
    capture_seconds: float = 0.0
    dispatch_counts: dict[str, int] = field(
        default_factory=lambda: {
            "native_q1_rope_cache_self_attention": 0,
            "native_q1_self_attention": 0,
            "q1_bmm_cross_attention": 0,
            "native_cross_mlp_tail": 0,
        }
    )


class DraftFastpathRunner:
    """One-token draft decode under optimized kernels + per-bucket CUDA graphs."""

    def __init__(
        self,
        model,
        *,
        dtype: torch.dtype,
        bucket_size: int = ACTIVE_PREFIX_BUCKET_SIZE,
        enable_native_kernels: bool = True,
        cuda_graph: bool = True,
        cuda_graph_warmup: int = 0,
    ):
        if bucket_size <= 0:
            raise ValueError("bucket_size must be positive")
        self.model = model
        self.dtype = dtype
        self.bucket_size = int(bucket_size)
        self.enable_native_kernels = bool(enable_native_kernels)
        self.cuda_graph = bool(cuda_graph)
        self.cuda_graph_warmup = int(cuda_graph_warmup)
        self.layer_count = sync_draft_decoder_layer_count(model)
        self.session = ProductionDecodeSession()
        self.stats = DraftFastpathStats()
        self._cache: MapperatorinatorCache | None = None
        self._model_kwargs: dict[str, Any] | None = None
        self._input_ids: torch.LongTensor | None = None
        self._cur_len = 0
        self._max_cache_len = int(model.config.max_target_positions)

    def _native_flags(self) -> dict[str, bool]:
        # RoPE+cache fused q1 needs tip shared-rope cos/sin layout [1,1,D].
        # Turbo draft sessions do not install that device-state path, so keep
        # q1 self (non-fused) + q1 cross BMM + cross-MLP and leave rope eager.
        on = self.enable_native_kernels
        return {
            "q1_bmm_cross_attention": on,
            "native_q1_self_attention": on,
            "native_q1_rope_cache_self_attention": False,
            "native_cross_mlp_tail": on,
        }

    def _profile_context(self):
        return generation_profile_context(
            **self._native_flags(),
            optimized_expected_dtype=self.dtype,
            optimized_dispatch_counts=self.stats.dispatch_counts,
        )

    def bind(
        self,
        *,
        model_kwargs: dict[str, Any],
        prompt_ids: torch.LongTensor,
    ) -> None:
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
            raise ValueError("draft fastpath requires batch_size=1 prompt_ids")
        # Use HF get_cache for storage init compatibility on this transformers
        # build. Extra unused self/cross slots (encoder-depth) are harmless for
        # the 2-layer draft; exact-depth shim remains available via get_draft_cache.
        signature = ProductionDecodeSession._state_signature(
            self.model, batch_size=1, num_beams=1, cfg_scale=1.0
        )
        cache = self.session.caches.get(signature)
        if cache is None:
            cache = get_cache(self.model, 1, 1, 1.0)
            self.session.caches[signature] = cache
        else:
            cache.self_attention_cache.reset()
            cache.cross_attention_cache.reset()
            for layer_idx in list(cache.is_updated):
                cache.is_updated[layer_idx] = False
        self.session.active_state_signature = signature
        self.session.stable_encoder_holders.setdefault(signature, {})
        self._cache = cache
        mk = {
            key: value.to(self.model.device) if isinstance(value, torch.Tensor) else value
            for key, value in model_kwargs.items()
        }
        mk.pop("decoder_input_ids", None)
        mk["past_key_values"] = self._cache
        mk["use_cache"] = True
        if "inputs" in mk and "frames" not in mk:
            mk["frames"] = mk.pop("inputs")
        self._model_kwargs = mk
        self._input_ids = prompt_ids.to(device=self.model.device)
        self._cur_len = int(self._input_ids.shape[1])
        _align_decoder_mask(
            self._model_kwargs,
            length=self._cur_len,
            device=self._input_ids.device,
        )

    def reset_cache(self) -> None:
        if self._cache is None:
            return
        self._cache.self_attention_cache.reset()
        self._cache.cross_attention_cache.reset()
        for layer_idx in list(self._cache.is_updated):
            self._cache.is_updated[layer_idx] = False

    def _forward_eager(self, input_ids: torch.LongTensor) -> torch.Tensor:
        assert self._model_kwargs is not None
        self._model_kwargs.pop("cache_position", None)
        _align_decoder_mask(
            self._model_kwargs,
            length=int(input_ids.shape[1]),
            device=input_ids.device,
        )
        model_inputs = self.model.prepare_inputs_for_generation(
            input_ids,
            **self._model_kwargs,
        )
        model_inputs = _strip_audio_inputs(model_inputs)
        if "cache_position" in model_inputs:
            self._model_kwargs["cache_position"] = model_inputs["cache_position"]
        prefix_length = _bucketed_prefix_length(
            int(input_ids.shape[1]),
            self.bucket_size,
            self._max_cache_len,
        )
        with active_prefix_self_attention_context(prefix_length):
            outputs = self.model(**model_inputs, return_dict=True)
        self._model_kwargs = self.model._update_model_kwargs_for_generation(
            outputs,
            self._model_kwargs,
            is_encoder_decoder=True,
        )
        holder = self.session.active_prefix_decode_kwargs()["stable_encoder_holder"]
        encoder_outputs = self._model_kwargs.get("encoder_outputs")
        if encoder_outputs is not None:
            self._model_kwargs["encoder_outputs"] = _stable_encoder_outputs(
                holder,
                encoder_outputs,
            )
        return outputs.logits[:, -1, :].float().squeeze(0)

    def prefill(self) -> torch.Tensor:
        if self._input_ids is None or self._model_kwargs is None:
            raise RuntimeError("draft fastpath bind() required before prefill")
        t0 = time.perf_counter()
        with self._profile_context():
            logits = self._forward_eager(self._input_ids)
        # After the first forward, keep encoder_outputs and drop audio tensors so
        # decode CUDA graphs never capture the spectrogram/encoder path.
        if self._model_kwargs.get("encoder_outputs") is not None:
            for key in _AUDIO_INPUT_KEYS:
                self._model_kwargs.pop(key, None)
        self.stats.prefill_seconds += time.perf_counter() - t0
        return logits

    def warm_native_kernels(self) -> None:
        """Eager one-token decode so CUDA extensions JIT before graph capture."""
        if self._input_ids is None or self._model_kwargs is None:
            raise RuntimeError("draft fastpath bind()/prefill() required")
        if self._input_ids.device.type != "cuda":
            return
        snap_ids = self._input_ids.clone()
        prev = self.cuda_graph
        self.cuda_graph = False
        try:
            _ = self.decode_token(1)
            torch.cuda.synchronize()
        finally:
            self.cuda_graph = prev
        # Restore exact prompt state after the probe forward dirtied the cache.
        self.rebuild_from_ids(snap_ids)
        torch.cuda.synchronize()

    def _decode_one_graph(self, input_ids: torch.LongTensor) -> torch.Tensor:
        assert self._model_kwargs is not None
        graph_cache = self.session.graph_cache
        self._model_kwargs.pop("cache_position", None)
        _align_decoder_mask(
            self._model_kwargs,
            length=int(input_ids.shape[1]),
            device=input_ids.device,
        )
        model_inputs = self.model.prepare_inputs_for_generation(
            input_ids,
            **self._model_kwargs,
        )
        model_inputs = _strip_audio_inputs(model_inputs)
        if "cache_position" in model_inputs:
            self._model_kwargs["cache_position"] = model_inputs["cache_position"]
        prefix_length = _bucketed_prefix_length(
            int(input_ids.shape[1]),
            self.bucket_size,
            self._max_cache_len,
        )
        graph_key = _cuda_graph_signature(prefix_length, model_inputs)
        entry = graph_cache.get(graph_key)
        if entry is None:
            static_inputs = _clone_static_graph_inputs(model_inputs)
            graph, outputs, capture_seconds = _capture_decode_cuda_graph(
                self.model,
                static_inputs,
                active_prefix_length=prefix_length,
                warmup=self.cuda_graph_warmup,
            )
            entry = {
                "graph": graph,
                "outputs": outputs,
                "static_inputs": static_inputs,
                "active_prefix_length": prefix_length,
                "capture_seconds": capture_seconds,
                "decode_replays": 0,
            }
            graph_cache[graph_key] = entry
            self.stats.graph_captures += 1
            self.stats.capture_seconds += float(capture_seconds)
        else:
            _copy_static_graph_inputs(entry["static_inputs"], model_inputs)
            entry["graph"].replay()
            self.stats.graph_replays += 1
        entry["decode_replays"] = int(entry.get("decode_replays", 0)) + 1
        outputs = entry["outputs"]
        self._model_kwargs = self.model._update_model_kwargs_for_generation(
            outputs,
            self._model_kwargs,
            is_encoder_decoder=True,
        )
        holder = self.session.active_prefix_decode_kwargs()["stable_encoder_holder"]
        encoder_outputs = self._model_kwargs.get("encoder_outputs")
        if encoder_outputs is not None:
            self._model_kwargs["encoder_outputs"] = _stable_encoder_outputs(
                holder,
                encoder_outputs,
            )
        return outputs.logits[:, -1, :].float().squeeze(0)

    def decode_token(self, token_id: int) -> torch.Tensor:
        if self._input_ids is None:
            raise RuntimeError("draft fastpath bind()/prefill() required")
        tok = torch.tensor(
            [[int(token_id)]],
            device=self._input_ids.device,
            dtype=self._input_ids.dtype,
        )
        self._input_ids = torch.cat([self._input_ids, tok], dim=-1)
        self._cur_len = int(self._input_ids.shape[1])
        with self._profile_context():
            if self.cuda_graph and self._input_ids.device.type == "cuda":
                logits = self._decode_one_graph(self._input_ids)
            else:
                logits = self._forward_eager(self._input_ids)
        self.stats.decode_steps += 1
        return logits

    def rebuild_from_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Reset StaticCache and eager-forward the full sequence (post-reject)."""
        if self._model_kwargs is None:
            raise RuntimeError("draft fastpath bind() required")
        self.reset_cache()
        # Drop stale encoder cache flags so cross KV is rewritten cleanly.
        for layer_idx in list(self._cache.is_updated):
            self._cache.is_updated[layer_idx] = False
        self._input_ids = input_ids.to(device=self.model.device)
        self._cur_len = int(self._input_ids.shape[1])
        with self._profile_context():
            logits = self._forward_eager(self._input_ids)
        return logits

    def profile_metadata(self) -> dict[str, Any]:
        return {
            "turbo_draft_fastpath": True,
            "turbo_draft_layers": self.layer_count,
            "turbo_draft_bucket_size": self.bucket_size,
            "turbo_draft_cuda_graph": self.cuda_graph,
            "turbo_draft_native_kernels": self.enable_native_kernels,
            "turbo_draft_graph_summary": self.session.graph_profile_summary(),
            "turbo_draft_dispatch_counts": dict(self.stats.dispatch_counts),
            "turbo_draft_decode_steps": self.stats.decode_steps,
            "turbo_draft_graph_captures": self.stats.graph_captures,
            "turbo_draft_graph_replays": self.stats.graph_replays,
        }


@torch.no_grad()
def time_graph_replay_ms(
    runner: DraftFastpathRunner,
    *,
    prefix_length: int,
    warmup: int = 10,
    iters: int = 50,
) -> dict[str, Any]:
    """Microbench one captured draft decode graph at a fixed bucket prefix."""
    if runner._input_ids is None or runner._model_kwargs is None:
        raise RuntimeError("runner must be bound and prefills before timing")
    if runner._cur_len > prefix_length:
        raise ValueError(
            f"runner cur_len={runner._cur_len} exceeds requested prefix={prefix_length}"
        )
    # Reach the target length with eager decode so graph capture stays clean.
    prev_graph = runner.cuda_graph
    runner.cuda_graph = False
    try:
        while runner._cur_len < prefix_length:
            runner.decode_token(1)
    finally:
        runner.cuda_graph = prev_graph
    torch.cuda.synchronize()

    assert runner._model_kwargs is not None
    with runner._profile_context():
        # Build the one-token decode inputs at the current length without
        # advancing the logical sequence; clone for static capture buffers.
        runner._model_kwargs.pop("cache_position", None)
        _align_decoder_mask(
            runner._model_kwargs,
            length=int(runner._input_ids.shape[1]),
            device=runner._input_ids.device,
        )
        # prepare_inputs typically emits the last token only when cache is hot.
        model_inputs = runner.model.prepare_inputs_for_generation(
            runner._input_ids,
            **runner._model_kwargs,
        )
        model_inputs = _strip_audio_inputs(model_inputs)
        if not isinstance(model_inputs.get("encoder_outputs"), BaseModelOutput):
            raise RuntimeError(
                "draft graph timing requires BaseModelOutput encoder_outputs"
            )
        bucket = _bucketed_prefix_length(
            int(runner._input_ids.shape[1]),
            runner.bucket_size,
            runner._max_cache_len,
        )
        static_inputs = _clone_static_graph_inputs(model_inputs)
        # Match optimized decode_loop capture: warmup+capture on static buffers.
        graph, _outputs, capture_seconds = _capture_decode_cuda_graph(
            runner.model,
            static_inputs,
            active_prefix_length=bucket,
            warmup=max(int(runner.cuda_graph_warmup), 2),
        )
        for _ in range(max(int(warmup), 0)):
            graph.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(int(iters)):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        ms = float(start.elapsed_time(end)) / float(iters)
    return {
        "prefix_length": int(prefix_length),
        "bucket_prefix_length": int(bucket),
        "ms_per_token": ms,
        "capture_seconds": float(capture_seconds),
        "warmup": int(warmup),
        "iters": int(iters),
        "dispatch_counts": dict(runner.stats.dispatch_counts),
    }
