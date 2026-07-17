"""§41/§48 teacher verify: graph-native K=γ + graph-aligned Q=1 (TIER1a).

Turbo-only. Does not change bit-exact optimized engine presets/semantics.

§48 (W-VG): lift k>1 graph gate; static replay inputs are only
``{decoder_input_ids[1,γ], cache_position[γ]}``; 4D causal mask is filled
in-graph; no HF ``prepare_inputs_for_generation`` on the hot path (avoids
device-scalar sync). Capture uses the production side-stream pattern with
warmup to avoid the §41 sequential Q=1 zero-logits failure mode.
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
# Side-stream warmups before CUDAGraph capture (§41 zero-logits class of bug).
GRAPH_CAPTURE_WARMUP = 1


def verify_fastpath_enabled() -> bool:
    raw = os.environ.get("MAPPERATORINATOR_TURBO_VERIFY_FASTPATH", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def use_verify_cuda_graphs() -> bool:
    # §48: default ON for K=γ graph-native verify. Sequential Q=1 still forces
    # allow_graph=False in forward_q1 (50148138 zero-logits).
    raw = os.environ.get("MAPPERATORINATOR_TURBO_VERIFY_CUDA_GRAPH", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def graph_aligned_teacher_enabled() -> bool:
    """Sequential Q=1 teacher path for greedy / TIER1a canary."""
    raw = os.environ.get("MAPPERATORINATOR_TURBO_TEACHER_ALIGNED", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def graph_native_verify_enabled() -> bool:
    raw = os.environ.get("MAPPERATORINATOR_TURBO_VERIFY_GRAPH_NATIVE", "1").strip().lower()
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


def _model_output_dtype(model) -> torch.dtype:
    """LM head dtype on Mapperatorinator or nested HF whisper backbone."""
    for obj in (model, getattr(model, "transformer", None)):
        if obj is None:
            continue
        proj = getattr(obj, "proj_out", None)
        if proj is not None and hasattr(proj, "weight"):
            return proj.weight.dtype
    param = next(model.parameters())
    return param.dtype


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


def _fill_causal_mask_4d(
    mask_4d: torch.Tensor,
    cache_position: torch.Tensor,
    key_arange: torch.Tensor,
    blocked_buf: torch.Tensor,
) -> torch.Tensor:
    """In-place 4D causal mask for StaticCache (all-valid 2D prefix).

    Equivalent to HF ``_prepare_4d_causal_attention_mask_with_cache_position``
    when ``cache_position[i] = past + i`` (triu is redundant). Uses only
    in-place ops + persistent buffers so the fill can live inside a CUDA graph
    without host realloc / device-scalar sync.
    """
    if mask_4d.ndim != 4:
        raise ValueError("mask_4d must be [B,1,K,target]")
    _batch, _heads, sequence_length, target_length = mask_4d.shape
    if int(cache_position.numel()) != int(sequence_length):
        raise ValueError("cache_position length must match mask query length")
    if int(key_arange.numel()) != int(target_length):
        raise ValueError("key_arange length must match mask target length")
    if tuple(blocked_buf.shape) != (sequence_length, target_length):
        raise ValueError("blocked_buf must be [K,target]")
    min_dtype = torch.finfo(mask_4d.dtype).min
    # Visible where key_pos <= cache_position[query]; else min_dtype.
    mask_4d.fill_(0)
    torch.gt(
        key_arange.view(1, -1),
        cache_position.view(-1, 1),
        out=blocked_buf,
    )
    mask_4d[0, 0].masked_fill_(blocked_buf, min_dtype)
    return mask_4d


@dataclass
class VerifyGraphEntry:
    graph: torch.cuda.CUDAGraph
    outputs: Any
    # Full model kwargs captured once; only ids + cache_position are copied/replayed.
    static_inputs: dict[str, Any]
    static_ids: torch.Tensor
    static_cache_position: torch.Tensor
    static_position_ids: torch.Tensor
    static_mask_4d: torch.Tensor
    key_arange: torch.Tensor
    blocked_buf: torch.Tensor
    prefix_length: int
    k: int
    max_cache_len: int
    capture_seconds: float
    replays: int = 0
    graph_native: bool = True


@dataclass
class TeacherVerifyFastpath:
    """Teacher verify: graph-native K-token + Q=1 graph-aligned canary path."""

    model: Any
    bucket_size: int = ACTIVE_PREFIX_BUCKET_SIZE
    use_cuda_graph: bool = True
    graph_aligned: bool = True
    graph_native: bool = True
    graph_cache: dict[tuple[Any, ...], VerifyGraphEntry] = field(default_factory=dict)
    eager_calls: int = 0
    graph_replays: int = 0
    graph_captures: int = 0
    q1_steps: int = 0
    prepare_inputs_calls: int = 0
    graph_native_replays: int = 0
    # In-loop forward-only timing (CUDA events); excludes host rejection glue.
    forward_ms_total: float = 0.0
    forward_timed_calls: int = 0
    time_forwards: bool = field(
        default_factory=lambda: os.environ.get(
            "MAPPERATORINATOR_TURBO_VERIFY_CUDA_TIMER", "1"
        )
        .strip()
        .lower()
        not in {"0", "false", "off", "no"}
    )

    def _max_cache_len(self, model_kwargs: dict[str, Any]) -> int:
        cache = model_kwargs.get("past_key_values")
        if cache is not None and hasattr(cache, "get_max_cache_shape"):
            return int(cache.get_max_cache_shape())
        return int(self.model.config.max_target_positions)

    def _prepare_inputs_legacy(
        self,
        *,
        decoder_input_ids: torch.LongTensor,
        model_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """HF prepare path — eager / fallback only (device-scalar sync)."""
        self.prepare_inputs_calls += 1
        mk = dict(model_kwargs)
        mk.pop("cache_position", None)
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

    def _build_graph_native_static(
        self,
        *,
        model_kwargs: dict[str, Any],
        k: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Allocate static decode inputs without HF prepare_inputs."""
        past = model_kwargs.get("past_key_values")
        if past is None:
            raise RuntimeError("graph-native verify requires past_key_values")
        encoder_outputs = model_kwargs.get("encoder_outputs")
        if encoder_outputs is None:
            raise RuntimeError("graph-native verify requires encoder_outputs")
        max_cache_len = self._max_cache_len(model_kwargs)
        static_ids = torch.zeros((1, k), device=device, dtype=torch.long)
        static_cache_position = torch.zeros((k,), device=device, dtype=torch.long)
        static_position_ids = torch.zeros((1, k), device=device, dtype=torch.long)
        static_mask_4d = torch.empty(
            (1, 1, k, max_cache_len),
            device=device,
            dtype=dtype,
        )
        key_arange = torch.arange(max_cache_len, device=device, dtype=torch.long)
        blocked_buf = torch.empty(
            (k, max_cache_len), device=device, dtype=torch.bool
        )
        _fill_causal_mask_4d(
            static_mask_4d, static_cache_position, key_arange, blocked_buf
        )
        static_inputs: dict[str, Any] = {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": static_ids,
            "use_cache": True,
            "decoder_attention_mask": static_mask_4d,
            "decoder_position_ids": static_position_ids,
            "cache_position": static_cache_position,
        }
        # Stable non-audio conds if present (do not touch per replay).
        for key in ("beatmap_idx", "difficulty", "mapper_idx", "song_position"):
            if key in model_kwargs and model_kwargs[key] is not None:
                static_inputs[key] = model_kwargs[key]
        return {
            "static_inputs": static_inputs,
            "static_ids": static_ids,
            "static_cache_position": static_cache_position,
            "static_position_ids": static_position_ids,
            "static_mask_4d": static_mask_4d,
            "key_arange": key_arange,
            "blocked_buf": blocked_buf,
            "max_cache_len": max_cache_len,
        }

    def _write_replay_inputs(
        self,
        entry: VerifyGraphEntry,
        *,
        token_ids_1k: torch.Tensor,
        past_length: int,
    ) -> None:
        """Copy only {ids[1,γ], cache_position[γ]}; mask filled in-graph."""
        if token_ids_1k.shape != entry.static_ids.shape:
            raise RuntimeError(
                f"verify id shape {tuple(token_ids_1k.shape)} != "
                f"static {tuple(entry.static_ids.shape)}"
            )
        entry.static_ids.copy_(token_ids_1k)
        # Host past_length (no .item() sync); write cache_position on device.
        torch.arange(
            int(past_length),
            int(past_length) + int(entry.k),
            device=entry.static_cache_position.device,
            dtype=entry.static_cache_position.dtype,
            out=entry.static_cache_position,
        )
        entry.static_position_ids.copy_(entry.static_cache_position.unsqueeze(0))

    def _capture_graph_native(
        self,
        *,
        model_kwargs: dict[str, Any],
        token_ids_1k: torch.Tensor,
        past_length: int,
        prefix_length: int,
        k: int,
    ) -> VerifyGraphEntry:
        from ..optimized.single.runtime_context import active_prefix_self_attention_context

        dtype = _model_output_dtype(self.model)
        built = self._build_graph_native_static(
            model_kwargs=model_kwargs,
            k=k,
            device=token_ids_1k.device,
            dtype=dtype,
        )
        static_inputs = built["static_inputs"]
        static_ids = built["static_ids"]
        static_cache_position = built["static_cache_position"]
        static_position_ids = built["static_position_ids"]
        static_mask_4d = built["static_mask_4d"]
        key_arange = built["key_arange"]
        blocked_buf = built["blocked_buf"]

        # Seed ids/positions before warmup+capture (StaticCache live write).
        static_ids.copy_(token_ids_1k)
        torch.arange(
            int(past_length),
            int(past_length) + k,
            device=static_cache_position.device,
            dtype=static_cache_position.dtype,
            out=static_cache_position,
        )
        static_position_ids.copy_(static_cache_position.unsqueeze(0))

        def _one_forward() -> Any:
            _fill_causal_mask_4d(
                static_mask_4d, static_cache_position, key_arange, blocked_buf
            )
            with active_prefix_self_attention_context(prefix_length):
                return self.model(**static_inputs, return_dict=True)

        # Production capture: side-stream warmup, then capture on default stream.
        # Warmup>0 avoids §41 zero-logits degeneracy; StaticCache slots are
        # rewritten by the capture forward at the same cache_position.
        t0 = time.perf_counter()
        if not torch.cuda.is_available():
            raise RuntimeError("graph-native verify requires CUDA")
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        graph_outputs = None
        with torch.cuda.stream(capture_stream):
            for _ in range(max(int(GRAPH_CAPTURE_WARMUP), 0)):
                graph_outputs = _one_forward()
        torch.cuda.current_stream().wait_stream(capture_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_outputs = _one_forward()
        graph.replay()
        torch.cuda.synchronize()
        self.graph_captures += 1
        return VerifyGraphEntry(
            graph=graph,
            outputs=graph_outputs,
            static_inputs=static_inputs,
            static_ids=static_ids,
            static_cache_position=static_cache_position,
            static_position_ids=static_position_ids,
            static_mask_4d=static_mask_4d,
            key_arange=key_arange,
            blocked_buf=blocked_buf,
            prefix_length=prefix_length,
            k=k,
            max_cache_len=int(built["max_cache_len"]),
            capture_seconds=time.perf_counter() - t0,
            graph_native=True,
        )

    def _capture_graph_legacy(
        self,
        *,
        model_inputs: dict[str, Any],
        prefix_length: int,
        k: int,
    ) -> VerifyGraphEntry:
        """Legacy full-input copy capture (non graph-native fallback)."""
        from ..optimized.single.runtime_context import active_prefix_self_attention_context

        static_inputs = _clone_static_graph_inputs(model_inputs)
        dec = static_inputs["decoder_input_ids"]
        cache_pos = static_inputs["cache_position"]
        if not isinstance(dec, torch.Tensor) or not isinstance(cache_pos, torch.Tensor):
            raise RuntimeError("legacy verify graph missing ids/cache_position")
        # Minimal mask placeholder for dataclass; legacy copies full inputs.
        dtype = _model_output_dtype(self.model)
        mask = static_inputs.get("decoder_attention_mask")
        if not isinstance(mask, torch.Tensor):
            mask = torch.empty((1, 1, k, k), device=dec.device, dtype=dtype)
        t0 = time.perf_counter()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        graph_outputs = None
        with torch.cuda.stream(capture_stream):
            for _ in range(max(int(GRAPH_CAPTURE_WARMUP), 0)):
                with active_prefix_self_attention_context(prefix_length):
                    graph_outputs = self.model(**static_inputs, return_dict=True)
        torch.cuda.current_stream().wait_stream(capture_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with active_prefix_self_attention_context(prefix_length):
                graph_outputs = self.model(**static_inputs, return_dict=True)
        graph.replay()
        torch.cuda.synchronize()
        self.graph_captures += 1
        tgt = int(mask.shape[-1]) if mask.ndim == 4 else int(k)
        key_arange = torch.arange(tgt, device=dec.device, dtype=torch.long)
        blocked_buf = torch.empty((int(k), tgt), device=dec.device, dtype=torch.bool)
        return VerifyGraphEntry(
            graph=graph,
            outputs=graph_outputs,
            static_inputs=static_inputs,
            static_ids=dec,
            static_cache_position=cache_pos,
            static_position_ids=static_inputs.get(
                "decoder_position_ids", cache_pos.unsqueeze(0)
            ),
            static_mask_4d=mask,
            key_arange=key_arange,
            blocked_buf=blocked_buf,
            prefix_length=prefix_length,
            k=k,
            max_cache_len=int(cache_pos.numel()),
            capture_seconds=time.perf_counter() - t0,
            graph_native=False,
        )

    def _forward_graph_native(
        self,
        *,
        decoder_input_ids: torch.LongTensor,
        model_kwargs: dict[str, Any],
        k: int,
    ) -> tuple[Any, dict[str, Any]]:
        past = model_kwargs.get("past_key_values")
        if not isinstance(getattr(past, "self_attention_cache", None), StaticCache):
            raise RuntimeError("graph-native verify requires StaticCache teacher")
        total_len = int(decoder_input_ids.shape[1])
        if total_len < k:
            raise ValueError("decoder_input_ids shorter than k")
        past_length = total_len - int(k)
        token_ids_1k = decoder_input_ids[:, past_length:]
        # Prefix bucket from the *post-verify* length (matches prior §41 keying).
        prefix_length = _bucketed_prefix_length(
            total_len,
            self.bucket_size,
            self._max_cache_len(model_kwargs),
        )
        # Encoder outputs identity is part of the key so multi-window reuse is safe
        # only when the same encoder holder is kept; otherwise recapture.
        enc = model_kwargs.get("encoder_outputs")
        enc_id = id(getattr(enc, "last_hidden_state", enc))
        key = (prefix_length, int(k), enc_id, "graph_native")
        entry = self.graph_cache.get(key)
        if entry is None:
            entry = self._capture_graph_native(
                model_kwargs=model_kwargs,
                token_ids_1k=token_ids_1k,
                past_length=past_length,
                prefix_length=prefix_length,
                k=int(k),
            )
            self.graph_cache[key] = entry
            outputs = entry.outputs
        else:
            # Refresh encoder/past pointers if the holder object was replaced.
            entry.static_inputs["encoder_outputs"] = enc
            entry.static_inputs["past_key_values"] = past
            self._write_replay_inputs(
                entry, token_ids_1k=token_ids_1k, past_length=past_length
            )
            entry.graph.replay()
            entry.replays += 1
            self.graph_replays += 1
            self.graph_native_replays += 1
            outputs = entry.outputs

        # HF update expects cache_position from the just-executed forward.
        model_kwargs["cache_position"] = entry.static_cache_position.clone()
        model_kwargs["decoder_attention_mask"] = torch.ones(
            (1, past_length + int(k)),
            device=entry.static_ids.device,
            dtype=torch.long,
        )
        model_kwargs = self.model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=True,
        )
        return outputs, model_kwargs

    def _forward_shaped(
        self,
        *,
        decoder_input_ids: torch.LongTensor,
        model_kwargs: dict[str, Any],
        k: int,
        allow_graph: bool,
    ) -> tuple[Any, dict[str, Any]]:
        from ..optimized.single.runtime_context import active_prefix_self_attention_context

        past = model_kwargs.get("past_key_values")
        can_graph = (
            allow_graph
            and self.use_cuda_graph
            and torch.cuda.is_available()
            and isinstance(getattr(past, "self_attention_cache", None), StaticCache)
            and decoder_input_ids.device.type == "cuda"
        )
        if can_graph and self.graph_native and int(k) >= 1:
            return self._forward_graph_native(
                decoder_input_ids=decoder_input_ids,
                model_kwargs=model_kwargs,
                k=int(k),
            )

        model_inputs = self._prepare_inputs_legacy(
            decoder_input_ids=decoder_input_ids,
            model_kwargs=model_kwargs,
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
            key = (prefix_length, int(k), dec_shape, "legacy")
            entry = self.graph_cache.get(key)
            if entry is None:
                entry = self._capture_graph_legacy(
                    model_inputs=model_inputs,
                    prefix_length=prefix_length,
                    k=int(k),
                )
                self.graph_cache[key] = entry
                outputs = entry.outputs
            else:
                for sk, sv in entry.static_inputs.items():
                    if isinstance(sv, torch.Tensor) and sk in model_inputs:
                        mv = model_inputs[sk]
                        if isinstance(mv, torch.Tensor) and sv.shape == mv.shape:
                            sv.copy_(mv)
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

    def _time_forward(self, fn):
        if not (
            self.time_forwards
            and torch.cuda.is_available()
            and getattr(self.model, "device", torch.device("cpu")).type == "cuda"
        ):
            return fn()
        captures_before = self.graph_captures
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        # Exclude capture setup (side-stream warmup + first record) from in-loop avg.
        if self.graph_captures == captures_before:
            self.forward_ms_total += float(start.elapsed_time(end))
            self.forward_timed_calls += 1
        return out

    def forward_k(
        self,
        *,
        decoder_input_ids: torch.LongTensor,
        model_kwargs: dict[str, Any],
        k: int,
        allow_graph: bool | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Multi-token teacher forward (perf path).

        §48: K>1 graphs are allowed by default (gate lifted). Pass
        ``allow_graph=False`` to force eager.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        if int(decoder_input_ids.shape[1]) < k:
            raise ValueError("decoder_input_ids shorter than k")
        use_graph = self.use_cuda_graph if allow_graph is None else bool(allow_graph)

        def _run():
            return self._forward_shaped(
                decoder_input_ids=decoder_input_ids,
                model_kwargs=model_kwargs,
                k=k,
                allow_graph=use_graph,
            )

        return self._time_forward(_run)

    def forward_q1(
        self,
        *,
        decoder_input_ids: torch.LongTensor,
        model_kwargs: dict[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        """One optimized-style Q=1 step (eager + native hooks).

        CUDA graphs disabled here until sequential Q=1 capture is proven
        non-degenerate (see canary 50148138 zero-logits).
        """
        self.q1_steps += 1
        return self._forward_shaped(
            decoder_input_ids=decoder_input_ids,
            model_kwargs=model_kwargs,
            k=1,
            allow_graph=False,
        )

    def verify_sequential_q1(
        self,
        *,
        prompt_ids: torch.LongTensor,
        draft_ids: Sequence[int],
        model_kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Teacher-force draft via sequential Q=1 steps (TIER1a / greedy)."""
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
        avg_ms = (
            self.forward_ms_total / self.forward_timed_calls
            if self.forward_timed_calls
            else None
        )
        return {
            "turbo_verify_fastpath": True,
            "turbo_verify_bucket_size": self.bucket_size,
            "turbo_verify_cuda_graph": self.use_cuda_graph,
            "turbo_verify_graph_native": self.graph_native,
            "turbo_teacher_graph_aligned": self.graph_aligned,
            "turbo_verify_eager_calls": self.eager_calls,
            "turbo_verify_graph_replays": self.graph_replays,
            "turbo_verify_graph_native_replays": self.graph_native_replays,
            "turbo_verify_graph_captures": self.graph_captures,
            "turbo_verify_graph_entries": len(self.graph_cache),
            "turbo_verify_prepare_inputs_calls": self.prepare_inputs_calls,
            "turbo_verify_q1_steps": self.q1_steps,
            "turbo_verify_forward_ms_total": self.forward_ms_total,
            "turbo_verify_forward_timed_calls": self.forward_timed_calls,
            "turbo_verify_forward_ms_avg": avg_ms,
        }


def build_teacher_verify_fastpath(model) -> TeacherVerifyFastpath | None:
    if not verify_fastpath_enabled():
        return None
    return TeacherVerifyFastpath(
        model=model,
        bucket_size=ACTIVE_PREFIX_BUCKET_SIZE,
        use_cuda_graph=use_verify_cuda_graphs(),
        graph_aligned=graph_aligned_teacher_enabled(),
        graph_native=graph_native_verify_enabled(),
    )


__all__ = [
    "ACTIVE_PREFIX_BUCKET_SIZE",
    "DEFAULT_VERIFY_KS",
    "DeviceTokenBuffer",
    "TeacherVerifyFastpath",
    "allocate_teacher_static_cache",
    "build_teacher_verify_fastpath",
    "graph_aligned_teacher_enabled",
    "graph_native_verify_enabled",
    "teacher_aligned_runtime_context",
    "verify_fastpath_enabled",
    "use_verify_cuda_graphs",
]
