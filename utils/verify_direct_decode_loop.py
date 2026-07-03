from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import transformers
from torch import nn
from transformers import LogitsProcessor

from config import InferenceConfig
from inference import compile_args, load_model_with_server, setup_inference_environment
from osuT5.osuT5.inference.cache_utils import get_cache
from osuT5.osuT5.inference.decode_loop import _bucketed_prefix_length, _max_cache_shape
from osuT5.osuT5.inference.direct_decode import DecodeSession, SamplerState, prepare_one_token_decode_inputs_fast
from osuT5.osuT5.inference.server import get_eos_token_id
from osuT5.osuT5.runtime_profiling import active_prefix_self_attention_context, generation_profile_context
from osuT5.osuT5.tokenizer import ContextType
from utils.verify_one_token_decode import (
    _assert_supported_probe,
    _build_generation_logits_processors,
    _build_probe_inputs,
    _compare_logits,
    _condition_kwargs,
    _load_args,
    _move_kwargs_for_model,
)


class _CaptureRawLogitsProcessor(LogitsProcessor):
    def __init__(self, captures: list[torch.Tensor]):
        self.captures = captures

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        self.captures.append(scores.detach().clone().to(torch.float32))
        return scores


def _rng_state() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return cpu_state, cuda_states


def _set_rng_state(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    cpu_state, cuda_states = state
    torch.random.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


def _rng_states_equal(
        left: tuple[torch.Tensor, list[torch.Tensor] | None],
        right: tuple[torch.Tensor, list[torch.Tensor] | None],
) -> bool:
    left_cpu, left_cuda = left
    right_cpu, right_cuda = right
    if not torch.equal(left_cpu, right_cpu):
        return False
    if left_cuda is None or right_cuda is None:
        return left_cuda is None and right_cuda is None
    if len(left_cuda) != len(right_cuda):
        return False
    return all(torch.equal(left_item, right_item) for left_item, right_item in zip(left_cuda, right_cuda))


def _stop_reason(output: torch.Tensor, prompt_len: int, max_new_tokens: int, eos_token_ids: list[int]) -> str:
    generated_len = int(output.shape[-1]) - prompt_len
    if generated_len >= max_new_tokens:
        return "max_new_tokens"
    if generated_len > 0 and int(output[0, -1].item()) in eos_token_ids:
        return "eos"
    return "unknown"


def _active_prefix_length(
        cur_len: int,
        *,
        active_prefix_decode: bool,
        active_prefix_decode_length: int | None,
        active_prefix_bucket_size: int,
        max_cache_len: int,
) -> int | None:
    if not active_prefix_decode:
        return None
    if active_prefix_decode_length is not None:
        return min(int(active_prefix_decode_length), max_cache_len)
    return _bucketed_prefix_length(cur_len, active_prefix_bucket_size, max_cache_len)


def _prepare_sequence_buffer(input_ids: torch.LongTensor, max_length: int, pad_token_id: torch.Tensor | None):
    fill_value = int(pad_token_id.item()) if isinstance(pad_token_id, torch.Tensor) else 0
    sequence_buffer = torch.full(
        (input_ids.shape[0], max_length),
        fill_value,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    sequence_buffer[:, :input_ids.shape[-1]].copy_(input_ids)
    return sequence_buffer


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


def _capture_decode_cuda_graph(
        model,
        static_inputs: dict[str, Any],
        *,
        active_prefix_length: int | None,
        warmup: int,
) -> tuple[torch.cuda.CUDAGraph, Any, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("candidate CUDA graph forward requires CUDA")

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


def _cuda_graph_signature(active_prefix_length: int | None, model_inputs: dict[str, Any]) -> tuple[Any, ...]:
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

    total_capture_seconds = sum(graph["capture_seconds"] for graph in graphs)
    total_replays = sum(graph["decode_replays"] for graph in graphs)
    diagnostics.update({
        "enabled": True,
        "warmup": int(warmup),
        "capture_count": len(graphs),
        "total_capture_seconds": float(total_capture_seconds),
        "decode_replays": int(total_replays),
        "graphs": graphs,
    })
    if graphs:
        first_graph = graphs[0]
        diagnostics.update({
            "capture_seconds": first_graph["capture_seconds"],
            "active_prefix_length": first_graph["active_prefix_length"],
            "static_input_shapes": first_graph["static_input_shapes"],
        })


def _tail_begin(diagnostics: dict[str, Any] | None, name: str) -> tuple[str, float, Any, Any] | None:
    if diagnostics is None:
        return None
    start_event = end_event = None
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    return name, time.perf_counter(), start_event, end_event


def _tail_end(diagnostics: dict[str, Any] | None, handle: tuple[str, float, Any, Any] | None) -> None:
    if diagnostics is None or handle is None:
        return
    name, start_cpu, start_event, end_event = handle
    if end_event is not None:
        end_event.record()
        diagnostics.setdefault("_cuda_events", []).append((name, start_event, end_event))
    elapsed = time.perf_counter() - start_cpu
    cpu_totals = diagnostics.setdefault("cpu_wall_s", {})
    calls = diagnostics.setdefault("calls", {})
    cpu_totals[name] = float(cpu_totals.get(name, 0.0)) + float(elapsed)
    calls[name] = int(calls.get(name, 0)) + 1


def _record_tail_shape(
        diagnostics: dict[str, Any] | None,
        input_ids: torch.LongTensor,
        scores: torch.Tensor,
) -> None:
    if diagnostics is None:
        return
    shape_counts = diagnostics.setdefault("shape_counts", {})
    input_len = int(input_ids.shape[-1])
    shape_counts["vocab_size"] = int(scores.shape[-1])
    shape_counts["input_len_min"] = min(int(shape_counts.get("input_len_min", input_len)), input_len)
    shape_counts["input_len_max"] = max(int(shape_counts.get("input_len_max", input_len)), input_len)
    shape_counts["steps"] = int(shape_counts.get("steps", 0)) + 1


def _apply_logits_processor_with_tail_diagnostics(
        logits_processor,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        diagnostics: dict[str, Any],
) -> torch.FloatTensor:
    processor_classes = diagnostics.setdefault("processor_classes", [])
    for index, processor in enumerate(logits_processor):
        processor_name = f"{index}:{processor.__class__.__name__}"
        if len(processor_classes) <= index:
            processor_classes.append(processor_name)
        handle = _tail_begin(diagnostics, f"logits_processor.{processor_name}")
        scores = processor(input_ids, scores)
        _tail_end(diagnostics, handle)
    return scores


def _finalize_tail_diagnostics(diagnostics: dict[str, Any] | None, *, generated_tokens: int) -> None:
    if diagnostics is None:
        return
    events = diagnostics.pop("_cuda_events", [])
    if events:
        torch.cuda.synchronize()
    cuda_totals: dict[str, float] = diagnostics.setdefault("cuda_event_ms", {})
    for name, start_event, end_event in events:
        cuda_totals[name] = float(cuda_totals.get(name, 0.0)) + float(start_event.elapsed_time(end_event))

    diagnostics["generated_tokens"] = int(generated_tokens)
    per_token_us: dict[str, dict[str, float]] = {}
    for name, seconds in diagnostics.get("cpu_wall_s", {}).items():
        per_token_us.setdefault(name, {})["cpu_wall_us"] = (
            float(seconds) * 1_000_000.0 / max(int(generated_tokens), 1)
        )
    for name, milliseconds in diagnostics.get("cuda_event_ms", {}).items():
        per_token_us.setdefault(name, {})["cuda_event_us"] = (
            float(milliseconds) * 1_000.0 / max(int(generated_tokens), 1)
        )
    diagnostics["per_token_us"] = per_token_us


@torch.no_grad()
def direct_decode_loop_generate(
        model,
        input_ids: torch.LongTensor,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus: bool = False,
        streamer=None,
        *,
        buffered_input_ids: bool = False,
        active_prefix_decode: bool = False,
        active_prefix_decode_length: int | None = None,
        active_prefix_bucket_size: int = 512,
        cuda_graph_forward: bool = False,
        cuda_graph_warmup: int = 0,
        cuda_graph_diagnostics: dict[str, Any] | None = None,
        decode_session_runtime: bool = False,
        decode_session_metadata: dict[str, Any] | None = None,
        fast_prepare_inputs: bool = False,
        tail_diagnostics: dict[str, Any] | None = None,
        **model_kwargs,
) -> torch.LongTensor:
    """Verification-only direct cached decode loop used by utils/verify_direct_decode_loop.py."""
    if synced_gpus:
        raise ValueError("direct_decode_loop_generate does not support synced_gpus")
    if streamer is not None:
        raise ValueError("direct_decode_loop_generate does not support streamers")
    if generation_config.return_dict_in_generate:
        raise ValueError("direct_decode_loop_generate currently supports tensor outputs only")
    if (
            generation_config.output_attentions
            or generation_config.output_hidden_states
            or generation_config.output_scores
            or generation_config.output_logits
    ):
        raise ValueError("direct_decode_loop_generate does not support auxiliary generation outputs")
    if generation_config.prefill_chunk_size is not None:
        raise ValueError("direct_decode_loop_generate does not support prefill_chunk_size")

    batch_size, cur_len = input_ids.shape[:2]
    initial_len = int(cur_len)
    if batch_size != 1:
        raise ValueError("direct_decode_loop_generate currently supports batch_size=1 only")
    if cuda_graph_forward and not torch.cuda.is_available():
        raise RuntimeError("cuda_graph_forward requires CUDA")

    pad_token_id = generation_config._pad_token_tensor
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample
    max_length = int(generation_config.max_length)
    sequence_buffer = (
        _prepare_sequence_buffer(input_ids, max_length, pad_token_id)
        if buffered_input_ids
        else None
    )

    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = model._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)
    max_cache_len = _max_cache_shape(model_kwargs, generation_config)
    decode_session = (
        DecodeSession.for_generation_loop(
            model,
            prompt=input_ids,
            prompt_attention_mask=model_kwargs.get("decoder_attention_mask"),
            frames=model_kwargs.get("frames"),
            model_kwargs=model_kwargs,
            max_cache_len=max_cache_len,
            sequence_buffer=sequence_buffer,
            sampler_state=SamplerState(
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                rng_state=torch.random.get_rng_state(),
                cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            ),
        )
        if decode_session_runtime
        else None
    )

    model_forward = model.__call__
    if model._valid_auto_compile_criteria(model_kwargs, generation_config):
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        if model.config._attn_implementation == "flash_attention_2":
            compile_config = generation_config.compile_config
            if compile_config is not None and compile_config.fullgraph:
                compile_config.fullgraph = False
        model_forward = model.get_compiled_call(generation_config.compile_config)

    is_prefill = True
    scores = None
    graph_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        current_input_ids = sequence_buffer[:, :cur_len] if sequence_buffer is not None else input_ids
        model_inputs = (
            prepare_one_token_decode_inputs_fast(model, current_input_ids, model_kwargs)
            if fast_prepare_inputs and not is_prefill
            else model.prepare_inputs_for_generation(current_input_ids, **model_kwargs)
        )
        if is_prefill:
            outputs = model(**model_inputs, return_dict=True)
            is_prefill = False
        else:
            prefix_length = _active_prefix_length(
                cur_len,
                active_prefix_decode=active_prefix_decode,
                active_prefix_decode_length=active_prefix_decode_length,
                active_prefix_bucket_size=active_prefix_bucket_size,
                max_cache_len=max_cache_len,
            )
            if cuda_graph_forward:
                graph_key = _cuda_graph_signature(prefix_length, model_inputs)
                graph_entry = graph_cache.get(graph_key)
                if graph_entry is None:
                    graph_static_inputs = _clone_static_graph_inputs(model_inputs)
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
                else:
                    graph_static_inputs = graph_entry["static_inputs"]
                    _copy_static_graph_inputs(graph_static_inputs, model_inputs)
                    graph_entry["graph"].replay()
                graph_entry["decode_replays"] = int(graph_entry["decode_replays"]) + 1
                _update_cuda_graph_diagnostics(
                    cuda_graph_diagnostics,
                    warmup=cuda_graph_warmup,
                    graph_cache=graph_cache,
                )
                outputs = graph_entry["outputs"]
            else:
                with active_prefix_self_attention_context(prefix_length):
                    outputs = model_forward(**model_inputs, return_dict=True)

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        if decode_session is not None:
            decode_session.update_loop_model_kwargs(model_kwargs)

        _record_tail_shape(tail_diagnostics, current_input_ids, outputs.logits[:, -1, :])
        handle = _tail_begin(tail_diagnostics, "logits_extract")
        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
        _tail_end(tail_diagnostics, handle)

        if tail_diagnostics is not None:
            handle = _tail_begin(tail_diagnostics, "logits_processor.total")
            next_token_scores = _apply_logits_processor_with_tail_diagnostics(
                logits_processor,
                current_input_ids,
                next_token_logits,
                tail_diagnostics,
            )
            _tail_end(tail_diagnostics, handle)
        else:
            next_token_scores = logits_processor(current_input_ids, next_token_logits)

        if do_sample:
            handle = _tail_begin(tail_diagnostics, "sampling.softmax")
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            _tail_end(tail_diagnostics, handle)
            handle = _tail_begin(tail_diagnostics, "sampling.multinomial")
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            _tail_end(tail_diagnostics, handle)
        else:
            handle = _tail_begin(tail_diagnostics, "sampling.argmax")
            next_tokens = torch.argmax(next_token_scores, dim=-1)
            _tail_end(tail_diagnostics, handle)

        if has_eos_stopping_criteria:
            handle = _tail_begin(tail_diagnostics, "eos_mask")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            _tail_end(tail_diagnostics, handle)

        if sequence_buffer is None:
            handle = _tail_begin(tail_diagnostics, "append_token.cat")
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            _tail_end(tail_diagnostics, handle)
        else:
            handle = _tail_begin(tail_diagnostics, "append_token.buffer_copy")
            sequence_buffer[:, cur_len].copy_(next_tokens)
            input_ids = sequence_buffer[:, :cur_len + 1]
            _tail_end(tail_diagnostics, handle)
        if decode_session is not None:
            decode_session.record_generated_token(next_tokens, input_ids)

        handle = _tail_begin(tail_diagnostics, "stopping_criteria")
        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        _tail_end(tail_diagnostics, handle)
        handle = _tail_begin(tail_diagnostics, "finished_check")
        this_peer_finished = unfinished_sequences.max() == 0
        _tail_end(tail_diagnostics, handle)
        cur_len += 1

        del outputs

    generated_tokens = int(cur_len) - initial_len
    if sequence_buffer is not None:
        output = sequence_buffer[:, :cur_len].clone(memory_format=torch.contiguous_format)
        if decode_session is not None and decode_session_metadata is not None:
            decode_session_metadata.update(decode_session.metadata())
        _finalize_tail_diagnostics(tail_diagnostics, generated_tokens=generated_tokens)
        return output
    if decode_session is not None and decode_session_metadata is not None:
        decode_session_metadata.update(decode_session.metadata())
    _finalize_tail_diagnostics(tail_diagnostics, generated_tokens=generated_tokens)
    return input_ids


def _generated_token_ids(output: torch.Tensor, prompt_len: int) -> list[int]:
    return [int(item) for item in output[0, prompt_len:].tolist()]


@torch.no_grad()
def run_direct_decode_loop_gate(
        args: InferenceConfig,
        *,
        sequence_index: int,
        max_new_tokens: int,
        atol: float,
        rtol: float,
        top_k: int,
        buffered_input_ids: bool,
        candidate_active_prefix_decode: bool,
        candidate_active_prefix_decode_length: int | None,
        candidate_active_prefix_decode_bucket_size: int,
        candidate_cuda_graph_forward: bool,
        candidate_cuda_graph_warmup: int,
        candidate_q1_bmm_cross_attention: bool,
        candidate_native_q1_self_attention: bool,
        candidate_native_q1_rope_cache_self_attention: bool,
        candidate_decode_session: bool,
        candidate_fast_prepare: bool,
        tail_diagnostics: bool = False,
) -> dict[str, Any]:
    _assert_supported_probe(args)
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    compile_args(args, verbose=False)
    setup_inference_environment(args.seed)
    model, tokenizer = load_model_with_server(
        args.model_path,
        args.train,
        args.device,
        max_batch_size=args.max_batch_size,
        use_server=False,
        precision=args.precision,
        attn_implementation=args.attn_implementation,
        lora_path=args.lora_path,
        gamemode=args.gamemode,
        auto_select_gamemode_model=args.auto_select_gamemode_model,
        generation_compile=args.inference_generation_compile,
    )
    model_inputs = _move_kwargs_for_model(
        model,
        _build_probe_inputs(args, model, tokenizer, sequence_index=sequence_index),
    )
    metadata = {
        "config_name": None,
        "model_path": args.model_path,
        "precision": args.precision,
        "attn_implementation": args.attn_implementation,
        "profile_sdpa_backend": args.profile_sdpa_backend,
        "generation_compile_enabled": not bool(getattr(getattr(model, "generation_config", None), "disable_compile", True)),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "sequence_index": model_inputs.pop("sequence_index"),
        "frame_time_ms": model_inputs.pop("frame_time_ms"),
        "context_type": model_inputs.pop("context_type"),
        "lookback_time": model_inputs.pop("lookback_time"),
        "lookahead_time": model_inputs.pop("lookahead_time"),
        "do_sample": args.do_sample,
        "top_p": args.top_p,
        "top_k_sampling": args.top_k,
        "temperature": args.temperature,
        "timing_temperature": args.timing_temperature,
        "mania_column_temperature": args.mania_column_temperature,
        "taiko_hit_temperature": args.taiko_hit_temperature,
        "timeshift_bias": args.timeshift_bias,
        "cfg_scale": args.cfg_scale,
        "num_beams": args.num_beams,
        "max_new_tokens": max_new_tokens,
        "atol": atol,
        "rtol": rtol,
        "top_k": top_k,
        "buffered_input_ids": bool(buffered_input_ids),
        "candidate_active_prefix_decode": bool(candidate_active_prefix_decode),
        "candidate_active_prefix_decode_length": candidate_active_prefix_decode_length,
        "candidate_active_prefix_decode_bucket_size": candidate_active_prefix_decode_bucket_size,
        "candidate_cuda_graph_forward": bool(candidate_cuda_graph_forward),
        "candidate_cuda_graph_warmup": int(candidate_cuda_graph_warmup),
        "candidate_q1_bmm_cross_attention": bool(candidate_q1_bmm_cross_attention),
        "candidate_native_q1_self_attention": bool(candidate_native_q1_self_attention),
        "candidate_native_q1_rope_cache_self_attention": bool(candidate_native_q1_rope_cache_self_attention),
        "candidate_decode_session": bool(candidate_decode_session),
        "candidate_fast_prepare": bool(candidate_fast_prepare),
        "tail_diagnostics": bool(tail_diagnostics),
    }

    prompt = model_inputs["decoder_input_ids"]
    prompt_mask = model_inputs["decoder_attention_mask"]
    prompt_len = int(prompt.shape[-1])
    condition_kwargs = _condition_kwargs(model_inputs)
    context_type = ContextType(metadata["context_type"])
    eos_token_ids = get_eos_token_id(
        tokenizer,
        lookback_time=float(metadata["lookback_time"]),
        lookahead_time=float(metadata["lookahead_time"]),
        context_type=context_type,
    )
    metadata["eos_token_ids"] = [int(item) for item in eos_token_ids]

    rng_state = _rng_state()
    reference_captures: list[torch.Tensor] = []
    reference_processors = _build_generation_logits_processors(
        args,
        tokenizer,
        model.device,
        lookback_time=float(metadata["lookback_time"]),
    )
    reference_processors.insert(0, _CaptureRawLogitsProcessor(reference_captures))

    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=args.precision == "amp"), \
            generation_profile_context(sdpa_backend=args.profile_sdpa_backend):
        reference_cache = get_cache(model, batch_size=1, num_beams=1, cfg_scale=1.0)
        reference_output = model.generate(
            inputs=model_inputs["frames"],
            decoder_input_ids=prompt,
            decoder_attention_mask=prompt_mask,
            **condition_kwargs,
            do_sample=args.do_sample,
            num_beams=args.num_beams,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            past_key_values=reference_cache,
            logits_processor=reference_processors,
            eos_token_id=eos_token_ids,
        )
        reference_end_rng_state = _rng_state()

    _set_rng_state(rng_state)
    candidate_captures: list[torch.Tensor] = []
    candidate_processors = _build_generation_logits_processors(
        args,
        tokenizer,
        model.device,
        lookback_time=float(metadata["lookback_time"]),
    )
    candidate_processors.insert(0, _CaptureRawLogitsProcessor(candidate_captures))
    candidate_cuda_graph_diagnostics: dict[str, Any] = {}
    candidate_decode_session_metadata: dict[str, Any] = {}
    candidate_tail_diagnostics: dict[str, Any] | None = (
        {
            "enabled": True,
            "note": (
                "Verifier-only candidate-loop tail diagnostics; not an inference throughput claim. "
                "Processor 0 is the raw-logit capture hook inserted by this verifier."
            ),
        }
        if tail_diagnostics
        else None
    )

    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=args.precision == "amp"), \
            generation_profile_context(
                sdpa_backend=args.profile_sdpa_backend,
                q1_bmm_cross_attention=candidate_q1_bmm_cross_attention,
                native_q1_self_attention=candidate_native_q1_self_attention,
                native_q1_rope_cache_self_attention=candidate_native_q1_rope_cache_self_attention,
            ):
        candidate_cache = get_cache(model, batch_size=1, num_beams=1, cfg_scale=1.0)
        candidate_output = model.generate(
            inputs=model_inputs["frames"],
            decoder_input_ids=prompt,
            decoder_attention_mask=prompt_mask,
            **condition_kwargs,
            do_sample=args.do_sample,
            num_beams=args.num_beams,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            past_key_values=candidate_cache,
            logits_processor=candidate_processors,
            eos_token_id=eos_token_ids,
            custom_generate=partial(
                direct_decode_loop_generate,
                buffered_input_ids=buffered_input_ids,
                active_prefix_decode=candidate_active_prefix_decode,
                active_prefix_decode_length=candidate_active_prefix_decode_length,
                active_prefix_bucket_size=candidate_active_prefix_decode_bucket_size,
                cuda_graph_forward=candidate_cuda_graph_forward,
                cuda_graph_warmup=candidate_cuda_graph_warmup,
                cuda_graph_diagnostics=candidate_cuda_graph_diagnostics,
                decode_session_runtime=candidate_decode_session,
                decode_session_metadata=candidate_decode_session_metadata,
                fast_prepare_inputs=candidate_fast_prepare,
                tail_diagnostics=candidate_tail_diagnostics,
            ),
        )
        candidate_end_rng_state = _rng_state()

    if not torch.equal(reference_output[:, :prompt_len], prompt):
        raise RuntimeError("HF reference output prefix does not match the probe prompt")
    if not torch.equal(candidate_output[:, :prompt_len], prompt):
        raise RuntimeError("Candidate output prefix does not match the probe prompt")

    reference_stop_reason = _stop_reason(reference_output, prompt_len, max_new_tokens, metadata["eos_token_ids"])
    candidate_stop_reason = _stop_reason(candidate_output, prompt_len, max_new_tokens, metadata["eos_token_ids"])
    rng_match = _rng_states_equal(reference_end_rng_state, candidate_end_rng_state)
    token_match = bool(torch.equal(reference_output, candidate_output))
    logits_step_count = min(len(reference_captures), len(candidate_captures))
    logits_comparisons = [
        _compare_logits(reference_captures[index], candidate_captures[index], atol=atol, rtol=rtol, top_k=top_k)
        for index in range(logits_step_count)
    ]
    logits_allclose = all(item["allclose"] for item in logits_comparisons)
    logits_topk_match = all(item["topk_match"] for item in logits_comparisons)
    logits_pass = (
        len(reference_captures) == len(candidate_captures)
        and logits_step_count == len(reference_captures)
        and logits_allclose
        and logits_topk_match
    )

    return {
        "pass": token_match and logits_pass and rng_match,
        "token_match": token_match,
        "rng_match": rng_match,
        "logits_pass": logits_pass,
        "logits_allclose": logits_allclose,
        "logits_topk_match": logits_topk_match,
        "reference_stop_reason": reference_stop_reason,
        "candidate_stop_reason": candidate_stop_reason,
        "prompt_tokens": prompt_len,
        "reference_sequence_shape": list(reference_output.shape),
        "candidate_sequence_shape": list(candidate_output.shape),
        "reference_generated_token_ids": _generated_token_ids(reference_output, prompt_len),
        "candidate_generated_token_ids": _generated_token_ids(candidate_output, prompt_len),
        "reference_captured_raw_logit_steps": len(reference_captures),
        "candidate_captured_raw_logit_steps": len(candidate_captures),
        "logits_step_count": logits_step_count,
        "logits_comparisons": logits_comparisons,
        "candidate_cuda_graph_diagnostics": (
            candidate_cuda_graph_diagnostics
            if candidate_cuda_graph_forward
            else None
        ),
        "candidate_decode_session": (
            candidate_decode_session_metadata
            if candidate_decode_session
            else None
        ),
        "candidate_tail_diagnostics": candidate_tail_diagnostics,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify multi-token direct cached decode loop token/logit equivalence against HF generate."
    )
    parser.add_argument("--config-name", default="profile_salvalai_smoke15")
    parser.add_argument("--sequence-index", type=int, default=9)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--buffered-input-ids",
        action="store_true",
        help="Use a preallocated output-id buffer inside the candidate decode loop.",
    )
    parser.add_argument(
        "--candidate-active-prefix-decode",
        action="store_true",
        help="Enable active-prefix SDPA self-attention only for candidate decode steps.",
    )
    parser.add_argument(
        "--candidate-active-prefix-decode-length",
        type=int,
        default=None,
        help="Use this fixed active-prefix K/V length during candidate decode.",
    )
    parser.add_argument(
        "--candidate-active-prefix-decode-bucket-size",
        type=int,
        default=512,
        help="Bucket size for candidate active-prefix decode when no fixed length is provided.",
    )
    parser.add_argument(
        "--candidate-cuda-graph-forward",
        action="store_true",
        help="Capture and replay the candidate one-token decode forward with torch.cuda.CUDAGraph.",
    )
    parser.add_argument(
        "--candidate-cuda-graph-warmup",
        type=int,
        default=0,
        help="Warmup forwards before candidate CUDA graph capture.",
    )
    parser.add_argument(
        "--candidate-q1-bmm-cross-attention",
        action="store_true",
        help="Enable the experimental fp32 q_len=1 BMM cross-attention candidate for the direct loop.",
    )
    parser.add_argument(
        "--candidate-native-q1-self-attention",
        action="store_true",
        help="Enable the experimental native fp32 q_len=1 self-attention candidate for the direct loop.",
    )
    parser.add_argument(
        "--candidate-native-q1-rope-cache-self-attention",
        action="store_true",
        help="Enable the experimental fused RoPE/cache native q_len=1 self-attention candidate.",
    )
    parser.add_argument(
        "--candidate-decode-session",
        action="store_true",
        help="Wrap the candidate direct loop in the verifier-first DecodeSession state owner.",
    )
    parser.add_argument(
        "--candidate-fast-prepare",
        action="store_true",
        help="Use the verified fast post-prefill one-token input builder in the candidate direct loop.",
    )
    parser.add_argument(
        "--tail-diagnostics",
        action="store_true",
        help="Record verifier-only candidate-loop CPU/CUDA timing for logits processors, sampling, and stop/append tail.",
    )
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. model_path=/path/to/model")
    cli_args = parser.parse_args()

    start = time.perf_counter()
    args = _load_args(cli_args.config_name, cli_args.overrides)
    result = run_direct_decode_loop_gate(
        args,
        sequence_index=cli_args.sequence_index,
        max_new_tokens=cli_args.max_new_tokens,
        atol=cli_args.atol,
        rtol=cli_args.rtol,
        top_k=cli_args.top_k,
        buffered_input_ids=cli_args.buffered_input_ids,
        candidate_active_prefix_decode=(
            cli_args.candidate_active_prefix_decode
            or cli_args.candidate_active_prefix_decode_length is not None
        ),
        candidate_active_prefix_decode_length=cli_args.candidate_active_prefix_decode_length,
        candidate_active_prefix_decode_bucket_size=cli_args.candidate_active_prefix_decode_bucket_size,
        candidate_cuda_graph_forward=cli_args.candidate_cuda_graph_forward,
        candidate_cuda_graph_warmup=cli_args.candidate_cuda_graph_warmup,
        candidate_q1_bmm_cross_attention=cli_args.candidate_q1_bmm_cross_attention,
        candidate_native_q1_self_attention=cli_args.candidate_native_q1_self_attention,
        candidate_native_q1_rope_cache_self_attention=cli_args.candidate_native_q1_rope_cache_self_attention,
        candidate_decode_session=cli_args.candidate_decode_session,
        candidate_fast_prepare=cli_args.candidate_fast_prepare,
        tail_diagnostics=cli_args.tail_diagnostics,
    )
    result["metadata"]["config_name"] = cli_args.config_name
    result["wall_seconds"] = time.perf_counter() - start

    if cli_args.report_path is not None:
        cli_args.report_path.parent.mkdir(parents=True, exist_ok=True)
        cli_args.report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
