import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

from ....runtime_profiling import profile_range
from .runtime_context import active_prefix_self_attention_context


def _bucketed_prefix_length(cur_len: int, bucket_size: int, max_cache_len: int) -> int:
    if bucket_size <= 0:
        raise ValueError("active-prefix decode bucket size must be positive")
    bucketed = ((cur_len + bucket_size - 1) // bucket_size) * bucket_size
    return min(bucketed, max_cache_len)


def _max_cache_shape(model_kwargs: dict, generation_config) -> int:
    cache = model_kwargs.get("past_key_values")
    if cache is not None and hasattr(cache, "get_max_cache_shape"):
        return int(cache.get_max_cache_shape())
    return int(generation_config.max_length)


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
                f"CUDA graph input {key} changed from tensor to "
                f"{type(value).__name__}"
            )
        if static_value.shape != value.shape:
            raise RuntimeError(
                f"CUDA graph input {key} shape changed from "
                f"{tuple(static_value.shape)} to {tuple(value.shape)}"
            )
        if static_value.dtype != value.dtype:
            raise RuntimeError(
                f"CUDA graph input {key} dtype changed from "
                f"{static_value.dtype} to {value.dtype}"
            )
        if static_value.device != value.device:
            raise RuntimeError(
                f"CUDA graph input {key} device changed from "
                f"{static_value.device} to {value.device}"
            )
        static_value.copy_(value)


def _static_graph_input_signature(
    model_inputs: dict[str, Any],
) -> tuple[Any, ...]:
    return tuple(
        (
            key,
            tuple(value.shape),
            str(value.dtype),
            str(value.device),
        )
        if isinstance(value, torch.Tensor)
        else (key, type(value).__name__, id(value))
        for key, value in sorted(model_inputs.items())
    )


def _tensor_copy_size(model_inputs: dict[str, Any]) -> tuple[int, int]:
    tensors = [
        value for value in model_inputs.values() if isinstance(value, torch.Tensor)
    ]
    return (
        len(tensors),
        sum(value.numel() * value.element_size() for value in tensors),
    )


@dataclass(slots=True)
class _SharedStaticInputArena:
    """One address-stable input owner shared by compatible K1 CUDA graphs."""

    static_inputs: dict[str, Any]
    signature: tuple[Any, ...]
    tensor_addresses: tuple[tuple[str, int], ...]
    owned_tensor_bytes: int
    refreshes: int = 0
    refresh_copy_calls: int = 0
    refresh_copy_bytes: int = 0
    graph_entries: int = 0

    @staticmethod
    def _tensor_addresses(
        values: dict[str, Any],
    ) -> tuple[tuple[str, int], ...]:
        return tuple(
            (name, value.data_ptr())
            for name, value in sorted(values.items())
            if isinstance(value, torch.Tensor)
        )

    @classmethod
    def create(cls, model_inputs: dict[str, Any]) -> "_SharedStaticInputArena":
        static_inputs = _clone_static_graph_inputs(model_inputs)
        _, owned_tensor_bytes = _tensor_copy_size(static_inputs)
        return cls(
            static_inputs=static_inputs,
            signature=_static_graph_input_signature(model_inputs),
            tensor_addresses=cls._tensor_addresses(static_inputs),
            owned_tensor_bytes=owned_tensor_bytes,
        )

    def validate_tensor_addresses(self) -> None:
        if self._tensor_addresses(self.static_inputs) != self.tensor_addresses:
            raise RuntimeError("shared static-input arena tensor address changed")

    def refresh(self, model_inputs: dict[str, Any]) -> None:
        self.validate_tensor_addresses()
        if _static_graph_input_signature(model_inputs) != self.signature:
            raise RuntimeError("shared static-input arena signature changed")
        copy_calls, copy_bytes = _tensor_copy_size(model_inputs)
        _copy_static_graph_inputs(self.static_inputs, model_inputs)
        self.validate_tensor_addresses()
        self.refreshes += 1
        self.refresh_copy_calls += copy_calls
        self.refresh_copy_bytes += copy_bytes

    def register_graph(self) -> None:
        self.validate_tensor_addresses()
        self.graph_entries += 1

    def profile_summary(self) -> dict[str, int | bool]:
        avoided_storage = self.owned_tensor_bytes * max(0, self.graph_entries - 1)
        return {
            "enabled": True,
            "graph_entries": self.graph_entries,
            "refreshes": self.refreshes,
            "refresh_copy_calls": self.refresh_copy_calls,
            "refresh_copy_bytes": self.refresh_copy_bytes,
            "owned_tensor_bytes": self.owned_tensor_bytes,
            "avoided_duplicate_graph_input_bytes": avoided_storage,
        }


def _resolve_shared_static_input_arena(
    graph_cache: dict[tuple[Any, ...], dict[str, Any]],
    signature: tuple[Any, ...],
) -> _SharedStaticInputArena | None:
    matches: dict[int, _SharedStaticInputArena] = {}
    for entry in graph_cache.values():
        arena = entry.get("_shared_static_input_arena")
        if arena is None:
            continue
        if not isinstance(arena, _SharedStaticInputArena):
            raise RuntimeError("graph cache contains an invalid static-input arena")
        if arena.signature == signature:
            matches[id(arena)] = arena
    if len(matches) > 1:
        raise RuntimeError(
            "graph cache contains multiple shared static-input arena owners"
        )
    return next(iter(matches.values()), None)


def _stable_encoder_outputs(
    holder: dict[str, BaseModelOutput],
    encoder_outputs: Any,
) -> BaseModelOutput:
    if not isinstance(encoder_outputs, BaseModelOutput):
        raise RuntimeError(
            "Expected BaseModelOutput encoder_outputs, got "
            f"{type(encoder_outputs).__name__}"
        )
    source = encoder_outputs.last_hidden_state
    if not isinstance(source, torch.Tensor):
        raise RuntimeError("encoder_outputs.last_hidden_state must be a tensor")
    current = holder.get("encoder_outputs")
    if (
        current is None
        or current.last_hidden_state.shape != source.shape
        or current.last_hidden_state.dtype != source.dtype
        or current.last_hidden_state.device != source.device
    ):
        holder["encoder_outputs"] = BaseModelOutput(
            last_hidden_state=source.clone(memory_format=torch.contiguous_format),
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    else:
        current.last_hidden_state.copy_(source)
    return holder["encoder_outputs"]


def _capture_decode_cuda_graph(
    model,
    static_inputs: dict[str, Any],
    *,
    active_prefix_length: int,
    warmup: int,
) -> tuple[torch.cuda.CUDAGraph, Any, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("active-prefix CUDA graph decode requires CUDA")

    started = time.perf_counter()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    graph_outputs = None
    with torch.cuda.stream(capture_stream):
        for _ in range(max(int(warmup), 0)):
            with active_prefix_self_attention_context(active_prefix_length):
                graph_outputs = model(**static_inputs, return_dict=True)
    torch.cuda.current_stream().wait_stream(capture_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with active_prefix_self_attention_context(active_prefix_length):
            graph_outputs = model(**static_inputs, return_dict=True)
    graph.replay()
    torch.cuda.synchronize()
    return graph, graph_outputs, time.perf_counter() - started


def _cuda_graph_signature(
    active_prefix_length: int,
    model_inputs: dict[str, Any],
) -> tuple[Any, ...]:
    return (active_prefix_length, *_static_graph_input_signature(model_inputs))


def active_prefix_decode_generate(
    model,
    input_ids: torch.LongTensor,
    logits_processor,
    stopping_criteria,
    generation_config,
    synced_gpus: bool = False,
    streamer=None,
    *,
    active_prefix_bucket_size: int = 128,
    cuda_graph_forward: bool = False,
    cuda_graph_warmup: int = 0,
    cuda_graph_min_decode_steps: int = 1,
    shared_graph_cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
    stable_encoder_holder: dict[str, BaseModelOutput] | None = None,
    shared_static_input_arena: bool = False,
    **model_kwargs,
) -> torch.LongTensor:
    """Fixed-batch HF loop with persistent active-prefix CUDA graphs."""

    if synced_gpus:
        raise ValueError("active_prefix_decode_generate does not support synced_gpus")
    if streamer is not None:
        raise ValueError("active_prefix_decode_generate does not support streamers")
    if generation_config.return_dict_in_generate:
        raise ValueError("active_prefix_decode_generate supports tensor outputs only")
    if (
        generation_config.output_attentions
        or generation_config.output_hidden_states
        or generation_config.output_scores
        or generation_config.output_logits
    ):
        raise ValueError(
            "active_prefix_decode_generate does not support auxiliary outputs"
        )
    if generation_config.prefill_chunk_size is not None:
        raise ValueError(
            "active_prefix_decode_generate does not support prefill_chunk_size"
        )

    pad_token_id = generation_config._pad_token_tensor
    has_eos = any(
        hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
    )
    batch_size, cur_len = input_ids.shape[:2]
    if batch_size > 1:
        decoder_attention_mask = model_kwargs.get("decoder_attention_mask")
        if not isinstance(decoder_attention_mask, torch.Tensor):
            raise ValueError(
                "batched active-prefix decode requires decoder_attention_mask"
            )
        if decoder_attention_mask.ndim != 2 or tuple(
            decoder_attention_mask.shape
        ) != (batch_size, cur_len):
            raise ValueError(
                "batched decoder_attention_mask must match input_ids shape"
            )
    if cuda_graph_forward and input_ids.device.type != "cuda":
        raise RuntimeError("active-prefix CUDA graph decode requires CUDA inputs")
    if cuda_graph_min_decode_steps <= 0:
        raise ValueError("cuda_graph_min_decode_steps must be positive")
    if not isinstance(shared_static_input_arena, bool):
        raise TypeError("shared_static_input_arena must be a bool")
    if shared_static_input_arena and not cuda_graph_forward:
        raise ValueError(
            "shared static-input arena requires CUDA graph forward"
        )

    model_kwargs = model._get_initial_cache_position(
        cur_len,
        input_ids.device,
        model_kwargs,
    )
    max_cache_len = _max_cache_shape(model_kwargs, generation_config)

    is_prefill = True
    scores = None
    decode_steps = 0
    graph_cache = shared_graph_cache if shared_graph_cache is not None else {}
    static_input_arena: _SharedStaticInputArena | None = None
    this_peer_finished = False
    unfinished = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

    while model._has_unfinished_sequences(
        this_peer_finished,
        synced_gpus,
        device=input_ids.device,
    ):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        if is_prefill:
            with profile_range("generation.encoder_prefill"):
                outputs = model(**model_inputs, return_dict=True)
            is_prefill = False
        else:
            decode_steps += 1
            prefix_length = _bucketed_prefix_length(
                cur_len,
                active_prefix_bucket_size,
                max_cache_len,
            )
            use_graph = (
                cuda_graph_forward
                and decode_steps >= cuda_graph_min_decode_steps
            )
            if cuda_graph_forward and not use_graph:
                with active_prefix_self_attention_context(prefix_length):
                    outputs = model(**model_inputs, return_dict=True)
            elif use_graph:
                graph_inputs = model_inputs
                if shared_static_input_arena:
                    input_signature = _static_graph_input_signature(model_inputs)
                    if static_input_arena is None:
                        static_input_arena = _resolve_shared_static_input_arena(
                            graph_cache,
                            input_signature,
                        )
                    if static_input_arena is None:
                        static_input_arena = _SharedStaticInputArena.create(
                            model_inputs
                        )
                    else:
                        static_input_arena.refresh(model_inputs)
                    graph_inputs = static_input_arena.static_inputs
                    graph_key = (
                        "shared_static_input_arena",
                        *_cuda_graph_signature(prefix_length, model_inputs),
                    )
                else:
                    graph_key = _cuda_graph_signature(prefix_length, model_inputs)
                graph_entry = graph_cache.get(graph_key)
                if graph_entry is None:
                    static_inputs = (
                        graph_inputs
                        if shared_static_input_arena
                        else _clone_static_graph_inputs(model_inputs)
                    )
                    with profile_range("generation.decode_graph_capture_setup"):
                        graph, outputs, capture_seconds = _capture_decode_cuda_graph(
                            model,
                            static_inputs,
                            active_prefix_length=prefix_length,
                            warmup=cuda_graph_warmup,
                        )
                    graph_entry = {
                        "graph": graph,
                        "outputs": outputs,
                        "static_inputs": static_inputs,
                        "active_prefix_length": prefix_length,
                        "capture_seconds": capture_seconds,
                        "decode_replays": 0,
                    }
                    if shared_static_input_arena:
                        if static_input_arena is None:
                            raise RuntimeError(
                                "shared static-input arena disappeared during capture"
                            )
                        static_input_arena.register_graph()
                        graph_entry["_shared_static_input_arena"] = static_input_arena
                    graph_cache[graph_key] = graph_entry
                else:
                    if shared_static_input_arena:
                        entry_arena = graph_entry.get("_shared_static_input_arena")
                        if entry_arena is not static_input_arena:
                            raise RuntimeError(
                                "CUDA graph resolved the wrong shared input arena"
                            )
                        if graph_entry["static_inputs"] is not graph_inputs:
                            raise RuntimeError(
                                "CUDA graph did not retain the shared input arena"
                            )
                        static_input_arena.validate_tensor_addresses()
                    else:
                        _copy_static_graph_inputs(
                            graph_entry["static_inputs"],
                            model_inputs,
                        )
                    with profile_range("generation.decode_graph_replay"):
                        graph_entry["graph"].replay()
                graph_entry["decode_replays"] += 1
                outputs = graph_entry["outputs"]
            else:
                with active_prefix_self_attention_context(prefix_length):
                    outputs = model(**model_inputs, return_dict=True)

        with profile_range("generation.cache_update"):
            model_kwargs = model._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=model.config.is_encoder_decoder,
            )
            if stable_encoder_holder is not None:
                encoder_outputs = model_kwargs.get("encoder_outputs")
                if encoder_outputs is not None:
                    model_kwargs["encoder_outputs"] = _stable_encoder_outputs(
                        stable_encoder_holder,
                        encoder_outputs,
                    )

        with profile_range("generation.logits_processors"):
            next_logits = outputs.logits[:, -1, :].to(
                copy=True,
                dtype=torch.float32,
                device=input_ids.device,
            )
            next_scores = logits_processor(input_ids, next_logits)
        with profile_range("generation.sampling"):
            if generation_config.do_sample:
                probabilities = nn.functional.softmax(next_scores, dim=-1)
                next_tokens = torch.multinomial(probabilities, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_scores, dim=-1)

        with profile_range("generation.token_append_and_stopping"):
            if has_eos:
                next_tokens = next_tokens * unfinished + pad_token_id * (1 - unfinished)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            stopped = stopping_criteria(input_ids, scores)
            if not isinstance(stopped, torch.Tensor) or tuple(stopped.shape) != (
                batch_size,
            ):
                raise RuntimeError(
                    "batched stopping criteria must return one value per row"
                )
            unfinished = unfinished & ~stopped.to(dtype=torch.bool)
            this_peer_finished = unfinished.max() == 0
            cur_len += 1
        del outputs

    return input_ids
