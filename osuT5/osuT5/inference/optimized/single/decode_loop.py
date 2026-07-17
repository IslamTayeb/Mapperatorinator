import time
from typing import Any

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

from ....runtime_profiling import profile_range
from ..kernels.whole_token_step_cuda_graph import (
    record_whole_token_step_cuda_graph_hit,
    whole_token_step_cuda_graph_requested,
)
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


def _pad_sequence_tail_for_whole_token_graph(
    sequence_state,
    pad_token_id,
) -> None:
    """Fill unused preallocated tail so fixed-shape processor scans stay exact.

    Whole-token capture passes the full ``[1, max_length]`` sequence buffer into
    logits processors (CUDA-graph address stability). Uninitialized tail bytes
    would poison monotonic time-shift full scans; pad tokens are outside the
    time-shift / SOS ranges for this model family.
    """

    if pad_token_id is None:
        raise RuntimeError(
            "whole-token-step CUDA graph requires generation_config pad_token_id"
        )
    if isinstance(pad_token_id, torch.Tensor):
        if pad_token_id.numel() != 1:
            raise RuntimeError("pad_token_id must be a scalar tensor")
        pad_value = int(pad_token_id.item())
    else:
        pad_value = int(pad_token_id)
    max_length = int(sequence_state.sequence.shape[1])
    length = int(sequence_state.length)
    if length < 0 or length > max_length:
        raise RuntimeError("sequence_state.length outside preallocated buffer")
    if length < max_length:
        sequence_state.sequence[:, length:].fill_(pad_value)


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


def _capture_whole_token_sample_cuda_graph(
    *,
    scores_workspace: torch.Tensor,
    next_tokens: torch.Tensor,
    write_index: torch.Tensor,
    sequence: torch.Tensor,
    do_sample: bool,
) -> tuple[torch.cuda.CUDAGraph, float]:
    """Capture softmax/multinomial/append against fixed workspaces.

    Logits processors stay eager on the growing active prefix (exact). This
    graph only owns the fixed-shape sample + ``index_copy_`` tail.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("whole-token-step CUDA graph requires CUDA")
    if write_index.shape != (1,) or write_index.dtype != torch.long:
        raise ValueError("write_index must be int64 shape [1]")
    if next_tokens.shape != (1,) or next_tokens.dtype != torch.long:
        raise ValueError("next_tokens must be int64 shape [1]")
    if sequence.ndim != 2 or sequence.shape[0] != 1:
        raise ValueError("sequence must have shape [1, max_length]")
    if scores_workspace.ndim != 2 or scores_workspace.shape[0] != 1:
        raise ValueError("scores_workspace must have shape [1, vocab]")

    def _sample_step() -> None:
        if do_sample:
            probabilities = nn.functional.softmax(scores_workspace, dim=-1)
            sampled = torch.multinomial(probabilities, num_samples=1).squeeze(1)
            next_tokens.copy_(sampled)
        else:
            next_tokens.copy_(torch.argmax(scores_workspace, dim=-1))
        sequence.index_copy_(1, write_index, next_tokens.unsqueeze(0))

    started = time.perf_counter()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _sample_step()
    # Capture execution already sampled/appended. Do not replay here.
    torch.cuda.synchronize()
    return graph, time.perf_counter() - started


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
    this_peer_finished = False
    unfinished = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    whole_token_step = whole_token_step_cuda_graph_requested()
    if whole_token_step and sequence_state is None:
        raise RuntimeError(
            "whole-token-step CUDA graph requires preallocated batch-one sequence state"
        )
    if whole_token_step and not cuda_graph_forward:
        raise RuntimeError(
            "whole-token-step CUDA graph requires cuda_graph_forward=True"
        )
    # Whole-token graphs bind sequence/write_index addresses inside the capture.
    # Never reuse the session shared cache across windows — those tensors are
    # reallocated per generate_window and replaying a stale graph OOBs index_copy_.
    if whole_token_step:
        graph_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    else:
        graph_cache = shared_graph_cache if shared_graph_cache is not None else {}
    whole_token_buffers: dict[str, torch.Tensor] | None = None
    whole_token_hits = 0

    while model._has_unfinished_sequences(
        this_peer_finished,
        synced_gpus,
        device=input_ids.device,
    ):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        used_whole_token_graph = False
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
            elif use_graph and whole_token_step:
                assert sequence_state is not None
                vocab = int(model.config.vocab_size)
                if whole_token_buffers is None:
                    _pad_sequence_tail_for_whole_token_graph(
                        sequence_state,
                        pad_token_id,
                    )
                    whole_token_buffers = {
                        "logits_workspace": torch.empty(
                            (batch_size, vocab),
                            device=input_ids.device,
                            dtype=torch.float32,
                        ),
                        "scores_workspace": torch.empty(
                            (batch_size, vocab),
                            device=input_ids.device,
                            dtype=torch.float32,
                        ),
                        "next_tokens": torch.empty(
                            (batch_size,),
                            device=input_ids.device,
                            dtype=torch.long,
                        ),
                        "write_index": torch.zeros(
                            (1,),
                            device=input_ids.device,
                            dtype=torch.long,
                        ),
                    }
                # 1) Forward CUDA graph (same shape pinning as tip).
                forward_key = (
                    "whole_token_forward",
                    *_cuda_graph_signature(prefix_length, model_inputs),
                )
                forward_entry = graph_cache.get(forward_key)
                if forward_entry is None:
                    static_inputs = _clone_static_graph_inputs(model_inputs)
                    with profile_range("generation.decode_graph_capture_setup"):
                        graph, outputs, capture_seconds = _capture_decode_cuda_graph(
                            model,
                            static_inputs,
                            active_prefix_length=prefix_length,
                            warmup=cuda_graph_warmup,
                        )
                    forward_entry = {
                        "graph": graph,
                        "outputs": outputs,
                        "static_inputs": static_inputs,
                        "active_prefix_length": prefix_length,
                        "capture_seconds": capture_seconds,
                        "decode_replays": 0,
                    }
                    graph_cache[forward_key] = forward_entry
                else:
                    _copy_static_graph_inputs(
                        forward_entry["static_inputs"],
                        model_inputs,
                    )
                    with profile_range("generation.decode_graph_replay"):
                        forward_entry["graph"].replay()
                forward_entry["decode_replays"] += 1
                outputs = forward_entry["outputs"]

                # 2) Eager processors on the growing active prefix (exact).
                last = outputs.logits[:, -1, :]
                if last.dtype == torch.float32:
                    whole_token_buffers["logits_workspace"].copy_(last)
                else:
                    whole_token_buffers["logits_workspace"].copy_(
                        last.to(dtype=torch.float32)
                    )
                with profile_range("generation.logits_processors"):
                    scores = logits_processor(
                        sequence_state.active,
                        whole_token_buffers["logits_workspace"],
                    )
                whole_token_buffers["scores_workspace"].copy_(scores)

                # 3) Eager sample + index_copy_ (exact). CUDA-graph multinomial
                # diverged at the first decode token vs tip (smoke 50143356);
                # revisit sample-graph once Philox/capture semantics match eager.
                write_at = int(sequence_state.length)
                max_length = int(sequence_state.sequence.shape[1])
                if write_at < 0 or write_at >= max_length:
                    raise RuntimeError(
                        "whole-token-step write_index out of range: "
                        f"{write_at} not in [0, {max_length})"
                    )
                whole_token_buffers["write_index"].fill_(write_at)
                with profile_range("generation.sampling"):
                    if generation_config.do_sample:
                        probabilities = nn.functional.softmax(
                            whole_token_buffers["scores_workspace"],
                            dim=-1,
                        )
                        sampled = torch.multinomial(
                            probabilities, num_samples=1
                        ).squeeze(1)
                        whole_token_buffers["next_tokens"].copy_(sampled)
                    else:
                        whole_token_buffers["next_tokens"].copy_(
                            torch.argmax(
                                whole_token_buffers["scores_workspace"],
                                dim=-1,
                            )
                        )
                sequence_state.sequence.index_copy_(
                    1,
                    whole_token_buffers["write_index"],
                    whole_token_buffers["next_tokens"].unsqueeze(0),
                )
                whole_token_hits += 1
                record_whole_token_step_cuda_graph_hit()
                used_whole_token_graph = True
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

        if used_whole_token_graph:
            assert sequence_state is not None
            assert whole_token_buffers is not None
            # Append already happened inside the replayed graph via index_copy_.
            next_tokens = whole_token_buffers["next_tokens"]
            sequence_state.length = int(whole_token_buffers["write_index"].item()) + 1
            input_ids = sequence_state.active
            this_peer_finished = sequence_state.stopping_policy.stopped(
                next_tokens,
                next_length=sequence_state.length,
            )
            cur_len = sequence_state.length
        else:
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
                    next_tokens = torch.multinomial(
                        probabilities, num_samples=1
                    ).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_scores, dim=-1)

            with profile_range("generation.token_append_and_stopping"):
                if sequence_state is not None:
                    input_ids, this_peer_finished = sequence_state.append(next_tokens)
                    cur_len = sequence_state.length
                else:
                    if has_eos:
                        next_tokens = next_tokens * unfinished + pad_token_id * (
                            1 - unfinished
                        )
                    input_ids = torch.cat(
                        [input_ids, next_tokens[:, None]], dim=-1
                    )
                    stopped = stopping_criteria(input_ids, scores)
                    if not isinstance(stopped, torch.Tensor) or tuple(
                        stopped.shape
                    ) != (batch_size,):
                        raise RuntimeError(
                            "batched stopping criteria must return one value per row"
                        )
                    unfinished = unfinished & ~stopped.to(dtype=torch.bool)
                    this_peer_finished = unfinished.max() == 0
                    cur_len += 1
        del outputs

    # Windows that stop after the post-prefill non-graph token never enter
    # decode graph replay (decode_steps == 0). Require hits only when decode ran.
    if whole_token_step and decode_steps > 0 and whole_token_hits <= 0:
        raise RuntimeError(
            "whole-token-step CUDA graph requested but recorded zero replays"
        )
    return input_ids
