import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

from ..runtime_profiling import active_prefix_self_attention_context, profile_range


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


def _add_diagnostic_wall(diagnostics: dict[str, Any], key: str, seconds: float) -> None:
    diagnostics[key] = float(diagnostics.get(key, 0.0)) + float(seconds)


def _record_diagnostic_cuda_start(diagnostics: dict[str, Any]) -> tuple[Any, Any] | tuple[None, None]:
    if not diagnostics.get("_record_cuda_events", False):
        return None, None
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    return start_event, end_event


def _maybe_record_diagnostic_cuda_start(diagnostics: dict[str, Any] | None) -> tuple[Any, Any] | tuple[None, None]:
    if diagnostics is None:
        return None, None
    return _record_diagnostic_cuda_start(diagnostics)


def _add_diagnostic_cuda_event(
        diagnostics: dict[str, Any],
        name: str,
        start_event: Any,
        end_event: Any,
) -> None:
    if start_event is None or end_event is None:
        return
    end_event.record()
    diagnostics.setdefault("_cuda_events", []).append((name, start_event, end_event))


def _finalize_active_prefix_diagnostics(diagnostics: dict[str, Any] | None) -> None:
    if diagnostics is None:
        return
    events = diagnostics.pop("_cuda_events", [])
    if events:
        torch.cuda.synchronize()
    cuda_event_ms = diagnostics.setdefault("cuda_event_ms", {})
    cuda_event_calls = diagnostics.setdefault("cuda_event_calls", {})
    for name, start_event, end_event in events:
        cuda_event_ms[name] = float(cuda_event_ms.get(name, 0.0)) + float(start_event.elapsed_time(end_event))
        cuda_event_calls[name] = int(cuda_event_calls.get(name, 0)) + 1

    decode_steps = max(int(diagnostics.get("decode_steps", 0)), 1)
    per_decode_step_us = diagnostics.setdefault("cuda_event_per_decode_step_us", {})
    for name, milliseconds in cuda_event_ms.items():
        per_decode_step_us[name] = float(milliseconds) * 1000.0 / decode_steps
    child_wall_keys = (
        "prepare_inputs_wall_cpu_s",
        "prefill_forward_wall_cpu_s",
        "decode_forward_wall_cpu_s",
        "update_kwargs_wall_cpu_s",
        "logits_extract_wall_cpu_s",
        "logits_processor_wall_cpu_s",
        "sampling_wall_cpu_s",
        "eos_mask_wall_cpu_s",
        "append_token_wall_cpu_s",
        "stopping_criteria_wall_cpu_s",
        "finished_check_wall_cpu_s",
        "cache_position_wall_cpu_s",
        "max_cache_shape_wall_cpu_s",
        "compile_lookup_wall_cpu_s",
        "graph_signature_lookup_wall_cpu_s",
        "graph_capture_wall_cpu_s",
        "graph_input_copy_wall_cpu_s",
        "graph_replay_wall_cpu_s",
    )
    child_wall_total = sum(float(diagnostics.get(key, 0.0) or 0.0) for key in child_wall_keys)
    diagnostics["diagnostic_child_wall_cpu_s"] = float(child_wall_total)
    loop_wall = diagnostics.get("loop_total_wall_cpu_s")
    if isinstance(loop_wall, (float, int)):
        diagnostics["diagnostic_unattributed_loop_wall_cpu_s"] = float(loop_wall) - float(child_wall_total)
        diagnostics["diagnostic_child_wall_fraction_of_loop"] = (
            float(child_wall_total) / float(loop_wall)
            if float(loop_wall) > 0
            else None
        )
    loop_cuda_ms = cuda_event_ms.get("loop_total")
    decode_cuda_ms = cuda_event_ms.get("decode_forward.cuda_graph") or cuda_event_ms.get("decode_forward")
    if isinstance(loop_cuda_ms, (float, int)) and isinstance(decode_cuda_ms, (float, int)):
        diagnostics["diagnostic_decode_forward_fraction_of_loop_cuda"] = (
            float(decode_cuda_ms) / float(loop_cuda_ms)
            if float(loop_cuda_ms) > 0
            else None
        )
    graph_replay_ms = cuda_event_ms.get("graph.replay")
    if isinstance(loop_cuda_ms, (float, int)) and isinstance(graph_replay_ms, (float, int)):
        diagnostics["diagnostic_graph_replay_fraction_of_loop_cuda"] = (
            float(graph_replay_ms) / float(loop_cuda_ms)
            if float(loop_cuda_ms) > 0
            else None
        )
    diagnostics["cuda_event_note"] = (
        "Diagnostic-only CUDA event timings; nested ranges are non-exclusive and include queued GPU work "
        "between range entry/exit on the current stream."
    )


@contextmanager
def _diagnostic_range(
        diagnostics: dict[str, Any] | None,
        name: str,
        *,
        wall_key: str | None = None,
        aggregate_wall_key: str | None = None,
) -> Iterator[None]:
    if diagnostics is None:
        yield
        return

    start = time.perf_counter()
    start_event, end_event = _record_diagnostic_cuda_start(diagnostics)
    with profile_range(f"active_prefix.{name}"):
        try:
            yield
        finally:
            _add_diagnostic_cuda_event(diagnostics, name, start_event, end_event)
            elapsed = time.perf_counter() - start
            if wall_key is not None:
                _add_diagnostic_wall(diagnostics, wall_key, elapsed)
            if aggregate_wall_key is not None:
                _add_diagnostic_wall(diagnostics, aggregate_wall_key, elapsed)


def _record_decode_bucket(diagnostics: dict[str, Any] | None, prefix_length: int) -> None:
    if diagnostics is None:
        return

    diagnostics["decode_steps"] = int(diagnostics.get("decode_steps", 0)) + 1
    buckets = diagnostics.setdefault("bucket_lengths_seen", [])
    if not buckets or buckets[-1] != prefix_length:
        if buckets:
            diagnostics["bucket_transition_count"] = int(diagnostics.get("bucket_transition_count", 0)) + 1
        buckets.append(prefix_length)


def _apply_logits_processors_with_diagnostics(
        logits_processor,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        diagnostics: dict[str, Any],
) -> torch.FloatTensor:
    processor_times = diagnostics.setdefault("logits_processor_detail_wall_cpu_s", {})
    processor_calls = diagnostics.setdefault("logits_processor_detail_calls", {})
    for index, processor in enumerate(logits_processor):
        name = f"{index}:{processor.__class__.__name__}"
        start = time.perf_counter()
        with profile_range(f"active_prefix.logits_processor.{processor.__class__.__name__}"):
            scores = processor(input_ids, scores)
        elapsed = time.perf_counter() - start
        processor_times[name] = float(processor_times.get(name, 0.0)) + float(elapsed)
        processor_calls[name] = int(processor_calls.get(name, 0)) + 1
    return scores


def _clone_static_graph_inputs(model_inputs: dict[str, Any]) -> dict[str, Any]:
    static_inputs: dict[str, Any] = {}
    for key, value in model_inputs.items():
        if isinstance(value, torch.Tensor):
            static_inputs[key] = value.clone(memory_format=torch.contiguous_format)
        else:
            static_inputs[key] = value
    return static_inputs


def _copy_static_graph_inputs(static_inputs: dict[str, Any], model_inputs: dict[str, Any]) -> None:
    for key, static_value in static_inputs.items():
        value = model_inputs.get(key)
        if not isinstance(static_value, torch.Tensor):
            continue
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"CUDA graph input {key} changed from tensor to {type(value).__name__}")
        if tuple(static_value.shape) != tuple(value.shape):
            raise RuntimeError(
                f"CUDA graph input {key} shape changed from {tuple(static_value.shape)} "
                f"to {tuple(value.shape)}"
            )
        if static_value.dtype != value.dtype:
            raise RuntimeError(f"CUDA graph input {key} dtype changed from {static_value.dtype} to {value.dtype}")
        static_value.copy_(value)


def _record_static_input_copy_bytes(
        diagnostics: dict[str, Any] | None,
        static_inputs: dict[str, Any],
        model_inputs: dict[str, Any],
) -> None:
    if diagnostics is None:
        return
    bytes_by_key = diagnostics.setdefault("static_input_copy_bytes_by_key", {})
    calls_by_key = diagnostics.setdefault("static_input_copy_calls_by_key", {})
    for key, static_value in static_inputs.items():
        value = model_inputs.get(key)
        if not isinstance(static_value, torch.Tensor) or not isinstance(value, torch.Tensor):
            continue
        copied_bytes = int(value.numel() * value.element_size())
        bytes_by_key[key] = int(bytes_by_key.get(key, 0)) + copied_bytes
        calls_by_key[key] = int(calls_by_key.get(key, 0)) + 1


def _stable_encoder_outputs(
        holder: dict[str, BaseModelOutput],
        encoder_outputs: Any,
) -> BaseModelOutput:
    if not isinstance(encoder_outputs, BaseModelOutput):
        raise RuntimeError(f"Expected BaseModelOutput encoder_outputs, got {type(encoder_outputs).__name__}")
    source = encoder_outputs.last_hidden_state
    if not isinstance(source, torch.Tensor):
        raise RuntimeError("encoder_outputs.last_hidden_state must be a tensor")
    current = holder.get("encoder_outputs")
    if current is None or tuple(current.last_hidden_state.shape) != tuple(source.shape):
        holder["encoder_outputs"] = BaseModelOutput(
            last_hidden_state=source.clone(memory_format=torch.contiguous_format),
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    else:
        holder["encoder_outputs"].last_hidden_state.copy_(source)
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

    start = time.perf_counter()
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
    return graph, graph_outputs, time.perf_counter() - start


def _cuda_graph_signature(active_prefix_length: int, model_inputs: dict[str, Any]) -> tuple[Any, ...]:
    signature: list[Any] = [active_prefix_length]
    for key in sorted(model_inputs):
        value = model_inputs[key]
        if isinstance(value, torch.Tensor):
            signature.append((key, tuple(value.shape), str(value.dtype), str(value.device)))
        else:
            signature.append((key, type(value).__name__, id(value)))
    return tuple(signature)


def _update_cuda_graph_diagnostics(
        diagnostics: dict[str, Any] | None,
        *,
        warmup: int,
        graph_cache: dict[tuple[Any, ...], dict[str, Any]],
) -> None:
    if diagnostics is None:
        return

    graphs = []
    for entry in graph_cache.values():
        static_inputs = entry["static_inputs"]
        graphs.append({
            "active_prefix_length": entry["active_prefix_length"],
            "capture_seconds": float(entry["capture_seconds"]),
            "decode_replays": int(entry["decode_replays"]),
            "static_input_shapes": {
                key: list(value.shape)
                for key, value in static_inputs.items()
                if isinstance(value, torch.Tensor)
            },
        })

    cuda_graph_diagnostics = diagnostics.setdefault("cuda_graph", {})
    cuda_graph_diagnostics.update({
        "enabled": True,
        "warmup": int(warmup),
        "capture_count": len(graphs),
        "total_capture_seconds": float(sum(graph["capture_seconds"] for graph in graphs)),
        "decode_replays": int(sum(graph["decode_replays"] for graph in graphs)),
        "graphs": graphs,
    })


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
        active_prefix_decode_diagnostics: dict[str, Any] | None = None,
        shared_graph_cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
        stable_encoder_holder: dict[str, BaseModelOutput] | None = None,
        **model_kwargs,
) -> torch.LongTensor:
    """HF custom_generate loop with normal prefill and bucketed active-prefix one-token decode."""
    if synced_gpus:
        raise ValueError("active_prefix_decode_generate does not support synced_gpus")
    if streamer is not None:
        raise ValueError("active_prefix_decode_generate does not support streamers")
    if generation_config.return_dict_in_generate:
        raise ValueError("active_prefix_decode_generate currently supports tensor outputs only")
    if (
            generation_config.output_attentions
            or generation_config.output_hidden_states
            or generation_config.output_scores
            or generation_config.output_logits
    ):
        raise ValueError("active_prefix_decode_generate does not support auxiliary generation outputs")
    if generation_config.prefill_chunk_size is not None:
        raise ValueError("active_prefix_decode_generate does not support prefill_chunk_size")

    pad_token_id = generation_config._pad_token_tensor
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    batch_size, cur_len = input_ids.shape[:2]
    if batch_size != 1:
        raise ValueError("active_prefix_decode_generate currently supports batch_size=1 only")
    if cuda_graph_forward and input_ids.device.type != "cuda":
        raise RuntimeError("active-prefix CUDA graph decode requires CUDA input tensors")
    if cuda_graph_min_decode_steps <= 0:
        raise ValueError("cuda_graph_min_decode_steps must be positive")
    if active_prefix_decode_diagnostics is not None:
        active_prefix_decode_diagnostics["_record_cuda_events"] = (
            input_ids.device.type == "cuda" and torch.cuda.is_available()
        )
    loop_start_wall = time.perf_counter()
    loop_start_event, loop_end_event = _maybe_record_diagnostic_cuda_start(active_prefix_decode_diagnostics)

    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    if active_prefix_decode_diagnostics is not None:
        with _diagnostic_range(
                active_prefix_decode_diagnostics,
                "setup.cache_position",
                wall_key="cache_position_wall_cpu_s",
        ):
            model_kwargs = model._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)
        with _diagnostic_range(
                active_prefix_decode_diagnostics,
                "setup.max_cache_shape",
                wall_key="max_cache_shape_wall_cpu_s",
        ):
            max_cache_len = _max_cache_shape(model_kwargs, generation_config)
    else:
        model_kwargs = model._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)
        max_cache_len = _max_cache_shape(model_kwargs, generation_config)

    model_forward = model.__call__
    if active_prefix_decode_diagnostics is not None:
        with _diagnostic_range(
                active_prefix_decode_diagnostics,
                "setup.compile_lookup",
                wall_key="compile_lookup_wall_cpu_s",
        ):
            if model._valid_auto_compile_criteria(model_kwargs, generation_config):
                os.environ["TOKENIZERS_PARALLELISM"] = "0"
                if model.config._attn_implementation == "flash_attention_2":
                    compile_config = generation_config.compile_config
                    if compile_config is not None and compile_config.fullgraph:
                        compile_config.fullgraph = False
                model_forward = model.get_compiled_call(generation_config.compile_config)
    elif model._valid_auto_compile_criteria(model_kwargs, generation_config):
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        if model.config._attn_implementation == "flash_attention_2":
            compile_config = generation_config.compile_config
            if compile_config is not None and compile_config.fullgraph:
                compile_config.fullgraph = False
        model_forward = model.get_compiled_call(generation_config.compile_config)

    is_prefill = True
    scores = None
    decode_steps = 0
    graph_cache: dict[tuple[Any, ...], dict[str, Any]] = (
        shared_graph_cache if shared_graph_cache is not None else {}
    )
    diagnostic_graph_cache: dict[tuple[Any, ...], dict[str, Any]] = (
        graph_cache if shared_graph_cache is None else {}
    )
    try:
        while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            if active_prefix_decode_diagnostics is not None:
                with _diagnostic_range(
                        active_prefix_decode_diagnostics,
                        "prepare_inputs",
                        wall_key="prepare_inputs_wall_cpu_s",
                ):
                    model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
            else:
                model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            if is_prefill:
                if active_prefix_decode_diagnostics is not None:
                    with _diagnostic_range(
                            active_prefix_decode_diagnostics,
                            "prefill_forward",
                            wall_key="prefill_forward_wall_cpu_s",
                    ):
                        outputs = model(**model_inputs, return_dict=True)
                else:
                    outputs = model(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                decode_steps += 1
                prefix_length = _bucketed_prefix_length(cur_len, active_prefix_bucket_size, max_cache_len)
                _record_decode_bucket(active_prefix_decode_diagnostics, prefix_length)
                use_cuda_graph = cuda_graph_forward and decode_steps >= cuda_graph_min_decode_steps
                if cuda_graph_forward and not use_cuda_graph:
                    with active_prefix_self_attention_context(prefix_length):
                        outputs = model(**model_inputs, return_dict=True)
                elif cuda_graph_forward:
                    if active_prefix_decode_diagnostics is not None:
                        first_decode = int(active_prefix_decode_diagnostics.get("decode_steps", 0)) == 1
                        wall_key = (
                            "first_decode_forward_wall_cpu_s"
                            if first_decode
                            else "steady_decode_forward_wall_cpu_s"
                        )
                        decode_context = _diagnostic_range(
                            active_prefix_decode_diagnostics,
                            "decode_forward.cuda_graph",
                            wall_key=wall_key,
                            aggregate_wall_key="decode_forward_wall_cpu_s",
                        )
                    else:
                        decode_context = _diagnostic_range(None, "decode_forward.cuda_graph")
                    with decode_context:
                        with _diagnostic_range(
                                active_prefix_decode_diagnostics,
                                "graph.signature_lookup",
                                wall_key="graph_signature_lookup_wall_cpu_s",
                        ):
                            graph_key = _cuda_graph_signature(prefix_length, model_inputs)
                            graph_entry = graph_cache.get(graph_key)
                        captured_this_record = False
                        if graph_entry is None:
                            graph_static_inputs = _clone_static_graph_inputs(model_inputs)
                            with _diagnostic_range(
                                    active_prefix_decode_diagnostics,
                                    "graph.capture",
                                    wall_key="graph_capture_wall_cpu_s",
                            ):
                                graph, graph_outputs, capture_seconds = _capture_decode_cuda_graph(
                                    model,
                                    graph_static_inputs,
                                    active_prefix_length=prefix_length,
                                    warmup=cuda_graph_warmup,
                                )
                            graph_entry = {
                                "graph": graph,
                                "outputs": graph_outputs,
                                "static_inputs": graph_static_inputs,
                                "active_prefix_length": prefix_length,
                                "capture_seconds": capture_seconds,
                                "decode_replays": 0,
                            }
                            graph_cache[graph_key] = graph_entry
                            captured_this_record = True
                        else:
                            _record_static_input_copy_bytes(
                                active_prefix_decode_diagnostics,
                                graph_entry["static_inputs"],
                                model_inputs,
                            )
                            with _diagnostic_range(
                                    active_prefix_decode_diagnostics,
                                    "graph.input_copy",
                                    wall_key="graph_input_copy_wall_cpu_s",
                            ):
                                _copy_static_graph_inputs(graph_entry["static_inputs"], model_inputs)
                            with _diagnostic_range(
                                    active_prefix_decode_diagnostics,
                                    "graph.replay",
                                    wall_key="graph_replay_wall_cpu_s",
                            ):
                                graph_entry["graph"].replay()
                        graph_entry["decode_replays"] = int(graph_entry["decode_replays"]) + 1
                        diagnostic_graph_entry = diagnostic_graph_cache.get(graph_key)
                        if diagnostic_graph_entry is None:
                            diagnostic_graph_entry = {
                                "graph": graph_entry["graph"],
                                "outputs": graph_entry["outputs"],
                                "static_inputs": graph_entry["static_inputs"],
                                "active_prefix_length": graph_entry["active_prefix_length"],
                                "capture_seconds": (
                                    graph_entry["capture_seconds"]
                                    if captured_this_record
                                    else 0.0
                                ),
                                "decode_replays": 0,
                            }
                            diagnostic_graph_cache[graph_key] = diagnostic_graph_entry
                        diagnostic_graph_entry["decode_replays"] = int(diagnostic_graph_entry["decode_replays"]) + 1
                        _update_cuda_graph_diagnostics(
                            active_prefix_decode_diagnostics,
                            warmup=cuda_graph_warmup,
                            graph_cache=diagnostic_graph_cache,
                        )
                        outputs = graph_entry["outputs"]
                else:
                    with active_prefix_self_attention_context(prefix_length):
                        if active_prefix_decode_diagnostics is not None:
                            first_decode = int(active_prefix_decode_diagnostics.get("decode_steps", 0)) == 1
                            wall_key = (
                                "first_decode_forward_wall_cpu_s"
                                if first_decode
                                else "steady_decode_forward_wall_cpu_s"
                            )
                            with _diagnostic_range(
                                    active_prefix_decode_diagnostics,
                                    "decode_forward",
                                    wall_key=wall_key,
                                    aggregate_wall_key="decode_forward_wall_cpu_s",
                            ):
                                outputs = model_forward(**model_inputs, return_dict=True)
                        else:
                            outputs = model_forward(**model_inputs, return_dict=True)

            if active_prefix_decode_diagnostics is not None:
                with _diagnostic_range(
                        active_prefix_decode_diagnostics,
                        "update_model_kwargs",
                        wall_key="update_kwargs_wall_cpu_s",
                ):
                    model_kwargs = model._update_model_kwargs_for_generation(
                        outputs,
                        model_kwargs,
                        is_encoder_decoder=model.config.is_encoder_decoder,
                    )
            else:
                model_kwargs = model._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=model.config.is_encoder_decoder,
                )
            if stable_encoder_holder is not None:
                encoder_outputs = model_kwargs.get("encoder_outputs")
                if encoder_outputs is not None:
                    model_kwargs["encoder_outputs"] = _stable_encoder_outputs(stable_encoder_holder, encoder_outputs)

            if active_prefix_decode_diagnostics is not None:
                with _diagnostic_range(
                        active_prefix_decode_diagnostics,
                        "logits_extract",
                        wall_key="logits_extract_wall_cpu_s",
                ):
                    next_token_logits = outputs.logits[:, -1, :].to(
                        copy=True,
                        dtype=torch.float32,
                        device=input_ids.device,
                    )
                with _diagnostic_range(
                        active_prefix_decode_diagnostics,
                        "logits_processor",
                        wall_key="logits_processor_wall_cpu_s",
                ):
                    next_token_scores = _apply_logits_processors_with_diagnostics(
                        logits_processor,
                        input_ids,
                        next_token_logits,
                        active_prefix_decode_diagnostics,
                    )
            else:
                next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
                next_token_scores = logits_processor(input_ids, next_token_logits)

            if do_sample:
                if active_prefix_decode_diagnostics is not None:
                    with _diagnostic_range(
                            active_prefix_decode_diagnostics,
                            "sampling.softmax",
                            wall_key="softmax_wall_cpu_s",
                            aggregate_wall_key="sampling_wall_cpu_s",
                    ):
                        probs = nn.functional.softmax(next_token_scores, dim=-1)
                    with _diagnostic_range(
                            active_prefix_decode_diagnostics,
                            "sampling.multinomial",
                            wall_key="multinomial_wall_cpu_s",
                            aggregate_wall_key="sampling_wall_cpu_s",
                    ):
                        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                if active_prefix_decode_diagnostics is not None:
                    with _diagnostic_range(
                            active_prefix_decode_diagnostics,
                            "sampling.argmax",
                            wall_key="argmax_wall_cpu_s",
                            aggregate_wall_key="sampling_wall_cpu_s",
                    ):
                        next_tokens = torch.argmax(next_token_scores, dim=-1)
                else:
                    next_tokens = torch.argmax(next_token_scores, dim=-1)

            if has_eos_stopping_criteria:
                if active_prefix_decode_diagnostics is not None:
                    with _diagnostic_range(
                            active_prefix_decode_diagnostics,
                            "eos_mask",
                            wall_key="eos_mask_wall_cpu_s",
                    ):
                        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                else:
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            if active_prefix_decode_diagnostics is not None:
                with _diagnostic_range(
                        active_prefix_decode_diagnostics,
                        "append_token",
                        wall_key="append_token_wall_cpu_s",
                        aggregate_wall_key="token_append_stop_wall_cpu_s",
                ):
                    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                with _diagnostic_range(
                        active_prefix_decode_diagnostics,
                        "stopping_criteria",
                        wall_key="stopping_criteria_wall_cpu_s",
                        aggregate_wall_key="token_append_stop_wall_cpu_s",
                ):
                    unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
                with _diagnostic_range(
                        active_prefix_decode_diagnostics,
                        "finished_check",
                        wall_key="finished_check_wall_cpu_s",
                        aggregate_wall_key="token_append_stop_wall_cpu_s",
                ):
                    this_peer_finished = unfinished_sequences.max() == 0
            else:
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
                this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            del outputs
    finally:
        if active_prefix_decode_diagnostics is not None:
            active_prefix_decode_diagnostics["loop_total_wall_cpu_s"] = (
                time.perf_counter() - loop_start_wall
            )
            _add_diagnostic_cuda_event(
                active_prefix_decode_diagnostics,
                "loop_total",
                loop_start_event,
                loop_end_event,
            )
        _finalize_active_prefix_diagnostics(active_prefix_decode_diagnostics)

    return input_ids
