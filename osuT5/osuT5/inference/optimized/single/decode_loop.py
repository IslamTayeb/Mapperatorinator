import time
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
        from ..scout.decode_cast_copy_activation import decode_cast_copy_requested

        if (
            not decode_cast_copy_requested()
            or current.last_hidden_state.data_ptr() != source.data_ptr()
        ):
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
    signature: list[Any] = [active_prefix_length]
    for key in sorted(model_inputs):
        value = model_inputs[key]
        if isinstance(value, torch.Tensor):
            signature.append(
                (key, tuple(value.shape), str(value.dtype), str(value.device))
            )
        else:
            signature.append((key, type(value).__name__, id(value)))
    return tuple(signature)


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
    preallocated_batch1_state: bool = False,
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
    sequence_state = None
    if preallocated_batch1_state:
        if batch_size != 1:
            raise ValueError(
                "preallocated batch-one state requires batch_size=1"
            )
        from ..scout.device_sequence_state import (
            allocate_batch_one_sequence_state,
        )

        sequence_state = allocate_batch_one_sequence_state(
            input_ids,
            stopping_criteria,
        )
        input_ids = sequence_state.active
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
    this_peer_finished = False
    unfinished = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    logits_fp32_buffer: torch.Tensor | None = None

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
                graph_key = _cuda_graph_signature(prefix_length, model_inputs)
                graph_entry = graph_cache.get(graph_key)
                if graph_entry is None:
                    static_inputs = _clone_static_graph_inputs(model_inputs)
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
                    graph_cache[graph_key] = graph_entry
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
            from ..scout.decode_cast_copy_activation import decode_cast_copy_requested

            last_logits = outputs.logits[:, -1, :]
            if decode_cast_copy_requested():
                if (
                    logits_fp32_buffer is None
                    or logits_fp32_buffer.shape != last_logits.shape
                    or logits_fp32_buffer.device != input_ids.device
                    or logits_fp32_buffer.dtype != torch.float32
                ):
                    logits_fp32_buffer = torch.empty(
                        last_logits.shape,
                        dtype=torch.float32,
                        device=input_ids.device,
                    )
                logits_fp32_buffer.copy_(last_logits)
                next_logits = logits_fp32_buffer
            else:
                next_logits = last_logits.to(
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
            if sequence_state is not None:
                input_ids, this_peer_finished = sequence_state.append(next_tokens)
                cur_len = sequence_state.length
            else:
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
