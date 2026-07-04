from __future__ import annotations

import argparse
import copy
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
from transformers import LogitsProcessor, LogitsProcessorList

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


class _ForceEosAtGeneratedStepProcessor(LogitsProcessor):
    def __init__(self, *, prompt_len: int, generated_step: int, eos_token_id: int):
        if generated_step <= 0:
            raise ValueError("generated_step must be positive")
        self.prompt_len = int(prompt_len)
        self.generated_step = int(generated_step)
        self.eos_token_id = int(eos_token_id)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        next_generated_step = int(input_ids.shape[-1]) - self.prompt_len + 1
        if next_generated_step != self.generated_step:
            return scores
        forced_scores = torch.full_like(scores, -torch.inf)
        forced_scores[:, self.eos_token_id] = 0.0
        return forced_scores


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


def _cuda_event_elapsed_ms(fn, *, warmup: int, iters: int) -> float:
    for _ in range(max(int(warmup), 0)):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(max(int(iters), 1)):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / max(int(iters), 1)


def _tail_graph_token_ids(tensor: torch.Tensor) -> list[int]:
    return [int(item) for item in tensor.detach().cpu().reshape(-1).tolist()]


def _clone_production_tail_processors(logits_processor) -> tuple[LogitsProcessorList, list[str]]:
    excluded_classes: list[str] = []
    cloned_processors = LogitsProcessorList()
    for processor in logits_processor:
        if isinstance(processor, _CaptureRawLogitsProcessor):
            excluded_classes.append(processor.__class__.__name__)
            continue
        cloned_processors.append(copy.deepcopy(processor))
    return cloned_processors, excluded_classes


def _run_one_step_tail_graph_ceiling(
        *,
        input_ids: torch.LongTensor,
        raw_logits: torch.FloatTensor,
        rng_state: tuple[torch.Tensor, list[torch.Tensor] | None],
        actual_next_tokens: torch.LongTensor | None,
        actual_output_ids: torch.LongTensor | None,
        actual_unfinished: torch.LongTensor | None,
        actual_finished: torch.Tensor | None,
        actual_final_rng_state: tuple[torch.Tensor, list[torch.Tensor] | None] | None,
        logits_processor,
        stopping_criteria,
        do_sample: bool,
        has_eos_stopping_criteria: bool,
        pad_token_id: torch.Tensor,
        unfinished_sequences: torch.LongTensor,
        warmup: int,
        iters: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available() or input_ids.device.type != "cuda":
        raise RuntimeError("tail graph ceiling requires CUDA tensors")
    if input_ids.shape[0] != 1 or raw_logits.shape[0] != 1:
        raise RuntimeError("tail graph ceiling currently supports batch=1")

    static_input_ids = input_ids.detach().clone(memory_format=torch.contiguous_format)
    static_raw_logits = raw_logits.detach().clone(memory_format=torch.contiguous_format)
    static_unfinished = unfinished_sequences.detach().clone(memory_format=torch.contiguous_format)
    static_pad = pad_token_id.detach().clone(memory_format=torch.contiguous_format)
    eager_processors, excluded_processor_classes = _clone_production_tail_processors(logits_processor)
    graph_processors, _ = _clone_production_tail_processors(logits_processor)
    eager_stopping = copy.deepcopy(stopping_criteria)
    graph_stopping = copy.deepcopy(stopping_criteria)

    def tail_fn(processors, criteria):
        scores = static_raw_logits.clone(memory_format=torch.contiguous_format)
        next_token_scores = processors(static_input_ids, scores)
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * static_unfinished + static_pad * (1 - static_unfinished)
        output_ids = torch.cat([static_input_ids, next_tokens[:, None]], dim=-1)
        next_unfinished = static_unfinished & ~criteria(output_ids, None)
        finished = next_unfinished.max() == 0
        return next_tokens, output_ids, next_unfinished, finished

    start_rng_state = rng_state
    _set_rng_state(start_rng_state)
    eager_next, eager_output, eager_unfinished, eager_finished = tail_fn(eager_processors, eager_stopping)
    eager_final_rng = _rng_state()
    eager_next_after = [
        int(torch.multinomial(torch.ones_like(static_raw_logits), num_samples=1).item())
        for _ in range(8)
    ]

    _set_rng_state(start_rng_state)
    capture_start_rng = _rng_state()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_next, graph_output, graph_unfinished, graph_finished = tail_fn(graph_processors, graph_stopping)
    _set_rng_state(capture_start_rng)
    graph.replay()
    torch.cuda.synchronize()
    graph_final_rng = _rng_state()
    graph_next_after = [
        int(torch.multinomial(torch.ones_like(static_raw_logits), num_samples=1).item())
        for _ in range(8)
    ]

    correctness = {
        "next_token_match": bool(torch.equal(eager_next, graph_next)),
        "output_ids_match": bool(torch.equal(eager_output, graph_output)),
        "unfinished_match": bool(torch.equal(eager_unfinished, graph_unfinished)),
        "finished_match": bool(torch.equal(eager_finished, graph_finished)),
        "final_rng_match": _rng_states_equal(eager_final_rng, graph_final_rng),
        "next_eager_sample_match": eager_next_after == graph_next_after,
        "actual_anchor_complete": (
            actual_next_tokens is not None
            and actual_output_ids is not None
            and actual_unfinished is not None
            and actual_finished is not None
            and actual_final_rng_state is not None
        ),
        "eager_next_tokens": _tail_graph_token_ids(eager_next),
        "graph_next_tokens": _tail_graph_token_ids(graph_next),
        "eager_next_samples_after": eager_next_after,
        "graph_next_samples_after": graph_next_after,
    }
    required_correctness = [
        "next_token_match",
        "output_ids_match",
        "unfinished_match",
        "finished_match",
        "final_rng_match",
        "next_eager_sample_match",
        "actual_anchor_complete",
    ]
    if actual_next_tokens is not None:
        actual_next_tokens = actual_next_tokens.detach().clone(memory_format=torch.contiguous_format)
        correctness["actual_next_token_match"] = bool(
            torch.equal(eager_next, actual_next_tokens)
            and torch.equal(graph_next, actual_next_tokens)
        )
        correctness["actual_next_tokens"] = _tail_graph_token_ids(actual_next_tokens)
        required_correctness.append("actual_next_token_match")
    if actual_output_ids is not None:
        actual_output_ids = actual_output_ids.detach().clone(memory_format=torch.contiguous_format)
        correctness["actual_output_ids_match"] = bool(
            torch.equal(eager_output, actual_output_ids)
            and torch.equal(graph_output, actual_output_ids)
        )
        required_correctness.append("actual_output_ids_match")
    if actual_unfinished is not None:
        actual_unfinished = actual_unfinished.detach().clone(memory_format=torch.contiguous_format)
        correctness["actual_unfinished_match"] = bool(
            torch.equal(eager_unfinished, actual_unfinished)
            and torch.equal(graph_unfinished, actual_unfinished)
        )
        required_correctness.append("actual_unfinished_match")
    if actual_finished is not None:
        actual_finished = actual_finished.detach().clone()
        correctness["actual_finished_match"] = bool(
            torch.equal(eager_finished, actual_finished)
            and torch.equal(graph_finished, actual_finished)
        )
        required_correctness.append("actual_finished_match")
    if actual_final_rng_state is not None:
        correctness["actual_final_rng_match"] = (
            _rng_states_equal(eager_final_rng, actual_final_rng_state)
            and _rng_states_equal(graph_final_rng, actual_final_rng_state)
        )
        required_correctness.append("actual_final_rng_match")

    _set_rng_state(start_rng_state)
    eager_timing_processors, _ = _clone_production_tail_processors(logits_processor)
    eager_timing_stopping = copy.deepcopy(stopping_criteria)
    eager_ms = _cuda_event_elapsed_ms(
        lambda: tail_fn(eager_timing_processors, eager_timing_stopping),
        warmup=warmup,
        iters=iters,
    )

    _set_rng_state(start_rng_state)
    graph_timing_processors, _ = _clone_production_tail_processors(logits_processor)
    graph_timing_stopping = copy.deepcopy(stopping_criteria)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_timing_outputs = tail_fn(graph_timing_processors, graph_timing_stopping)
    _set_rng_state(start_rng_state)
    graph.replay()
    torch.cuda.synchronize()
    graph_ms = _cuda_event_elapsed_ms(graph.replay, warmup=warmup, iters=iters)
    _ = graph_timing_outputs

    _set_rng_state(start_rng_state)
    return {
        "enabled": True,
        "pass": all(
            bool(correctness[key])
            for key in required_correctness
        ),
        "do_sample": bool(do_sample),
        "input_shape": list(static_input_ids.shape),
        "logits_shape": list(static_raw_logits.shape),
        "processor_classes": [processor.__class__.__name__ for processor in eager_processors],
        "excluded_verifier_processor_classes": excluded_processor_classes,
        "stopping_classes": [criteria.__class__.__name__ for criteria in stopping_criteria],
        "warmup": int(warmup),
        "iters": int(iters),
        "eager_ms_per_call": float(eager_ms),
        "graph_ms_per_replay": float(graph_ms),
        "speedup": float(eager_ms / graph_ms) if graph_ms > 0 else None,
        "saved_us_per_token": float((eager_ms - graph_ms) * 1000.0),
        "projected_full_song_saved_s_7639": float((eager_ms - graph_ms) * 7639.0 / 1000.0),
        "correctness": correctness,
        "note": (
            "Verifier-only one-step tail graph ceiling. The verifier raw-logit capture hook is "
            "excluded from timing. This does not solve multi-step EOS/control or production "
            "integration by itself."
        ),
    }


def _run_multistep_tail_graph_ceiling(
        *,
        input_ids: torch.LongTensor,
        raw_logits_steps: list[torch.FloatTensor],
        rng_state: tuple[torch.Tensor, list[torch.Tensor] | None],
        actual_next_tokens: list[torch.LongTensor] | None,
        actual_output_ids: torch.LongTensor | None,
        actual_unfinished: torch.LongTensor | None,
        actual_continue_values: list[torch.Tensor] | None,
        actual_final_rng_state: tuple[torch.Tensor, list[torch.Tensor] | None] | None,
        logits_processor,
        stopping_criteria,
        do_sample: bool,
        has_eos_stopping_criteria: bool,
        pad_token_id: torch.Tensor,
        unfinished_sequences: torch.LongTensor,
        warmup: int,
        iters: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available() or input_ids.device.type != "cuda":
        raise RuntimeError("multistep tail graph ceiling requires CUDA tensors")
    if input_ids.shape[0] != 1:
        raise RuntimeError("multistep tail graph ceiling currently supports batch=1")
    if not raw_logits_steps:
        raise RuntimeError("multistep tail graph ceiling requires at least one raw-logit step")
    if any(raw_logits.shape[0] != 1 for raw_logits in raw_logits_steps):
        raise RuntimeError("multistep tail graph ceiling currently supports batch=1 logits")

    steps = len(raw_logits_steps)
    start_len = int(input_ids.shape[-1])
    static_sequence_template = torch.full(
        (input_ids.shape[0], start_len + steps),
        int(pad_token_id.item()),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    static_sequence_template[:, :start_len].copy_(input_ids)
    static_raw_logits_steps = [
        raw_logits.detach().clone(memory_format=torch.contiguous_format)
        for raw_logits in raw_logits_steps
    ]
    static_unfinished = unfinished_sequences.detach().clone(memory_format=torch.contiguous_format)
    static_pad = pad_token_id.detach().clone(memory_format=torch.contiguous_format)
    eager_processors, excluded_processor_classes = _clone_production_tail_processors(logits_processor)
    graph_processors, _ = _clone_production_tail_processors(logits_processor)
    eager_stopping = copy.deepcopy(stopping_criteria)
    graph_stopping = copy.deepcopy(stopping_criteria)

    def tail_fn(processors, criteria):
        sequence_buffer = static_sequence_template.clone(memory_format=torch.contiguous_format)
        unfinished = static_unfinished.clone(memory_format=torch.contiguous_format)
        generated_tokens = torch.empty((steps, input_ids.shape[0]), dtype=input_ids.dtype, device=input_ids.device)
        continue_values = torch.empty((steps,), dtype=unfinished.dtype, device=input_ids.device)
        for step_index, raw_logits in enumerate(static_raw_logits_steps):
            active_len = start_len + step_index
            active_input_ids = sequence_buffer[:, :active_len]
            scores = raw_logits.clone(memory_format=torch.contiguous_format)
            next_token_scores = processors(active_input_ids, scores)
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished + static_pad * (1 - unfinished)
            sequence_buffer[:, active_len].copy_(next_tokens)
            output_ids = sequence_buffer[:, :active_len + 1]
            unfinished = unfinished & ~criteria(output_ids, None)
            generated_tokens[step_index].copy_(next_tokens)
            continue_values[step_index].copy_(unfinished.max())
        return sequence_buffer, generated_tokens, unfinished, continue_values

    start_rng_state = rng_state
    _set_rng_state(start_rng_state)
    eager_output, eager_next, eager_unfinished, eager_continue = tail_fn(eager_processors, eager_stopping)
    eager_final_rng = _rng_state()
    eager_next_after = [
        int(torch.multinomial(torch.ones_like(static_raw_logits_steps[-1]), num_samples=1).item())
        for _ in range(8)
    ]

    _set_rng_state(start_rng_state)
    capture_start_rng = _rng_state()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output, graph_next, graph_unfinished, graph_continue = tail_fn(graph_processors, graph_stopping)
    _set_rng_state(capture_start_rng)
    graph.replay()
    torch.cuda.synchronize()
    graph_final_rng = _rng_state()
    graph_next_after = [
        int(torch.multinomial(torch.ones_like(static_raw_logits_steps[-1]), num_samples=1).item())
        for _ in range(8)
    ]

    no_stop_before_block_end = (
        bool(torch.all(eager_continue[:-1] != 0).item())
        if eager_continue.numel() > 1
        else True
    )
    correctness = {
        "next_token_match": bool(torch.equal(eager_next, graph_next)),
        "output_buffer_match": bool(torch.equal(eager_output, graph_output)),
        "unfinished_match": bool(torch.equal(eager_unfinished, graph_unfinished)),
        "continue_values_match": bool(torch.equal(eager_continue, graph_continue)),
        "final_rng_match": _rng_states_equal(eager_final_rng, graph_final_rng),
        "next_eager_sample_match": eager_next_after == graph_next_after,
        "no_stop_before_block_end": no_stop_before_block_end,
        "actual_anchor_complete": (
            actual_next_tokens is not None
            and len(actual_next_tokens) == steps
            and actual_output_ids is not None
            and actual_unfinished is not None
            and actual_continue_values is not None
            and len(actual_continue_values) == steps
            and actual_final_rng_state is not None
        ),
        "eager_next_tokens": _tail_graph_token_ids(eager_next),
        "graph_next_tokens": _tail_graph_token_ids(graph_next),
        "eager_continue_values": _tail_graph_token_ids(eager_continue),
        "graph_continue_values": _tail_graph_token_ids(graph_continue),
        "eager_next_samples_after": eager_next_after,
        "graph_next_samples_after": graph_next_after,
    }
    required_correctness = [
        "next_token_match",
        "output_buffer_match",
        "unfinished_match",
        "continue_values_match",
        "final_rng_match",
        "next_eager_sample_match",
        "no_stop_before_block_end",
        "actual_anchor_complete",
    ]
    if actual_next_tokens is not None and len(actual_next_tokens) == steps:
        actual_next = torch.stack([
            token.detach().clone(memory_format=torch.contiguous_format)
            for token in actual_next_tokens
        ], dim=0)
        correctness["actual_next_token_match"] = bool(
            torch.equal(eager_next, actual_next)
            and torch.equal(graph_next, actual_next)
        )
        correctness["actual_next_tokens"] = _tail_graph_token_ids(actual_next)
        required_correctness.append("actual_next_token_match")
    elif actual_next_tokens is not None:
        correctness["actual_next_token_count"] = int(len(actual_next_tokens))
    if actual_output_ids is not None:
        actual_output_ids = actual_output_ids.detach().clone(memory_format=torch.contiguous_format)
        correctness["actual_output_buffer_match"] = bool(
            torch.equal(eager_output, actual_output_ids)
            and torch.equal(graph_output, actual_output_ids)
        )
        required_correctness.append("actual_output_buffer_match")
    if actual_unfinished is not None:
        actual_unfinished = actual_unfinished.detach().clone(memory_format=torch.contiguous_format)
        correctness["actual_unfinished_match"] = bool(
            torch.equal(eager_unfinished, actual_unfinished)
            and torch.equal(graph_unfinished, actual_unfinished)
        )
        required_correctness.append("actual_unfinished_match")
    if actual_continue_values is not None and len(actual_continue_values) == steps:
        actual_continue = torch.stack([
            value.detach().clone()
            for value in actual_continue_values
        ], dim=0)
        correctness["actual_continue_values_match"] = bool(
            torch.equal(eager_continue, actual_continue)
            and torch.equal(graph_continue, actual_continue)
        )
        correctness["actual_continue_values"] = _tail_graph_token_ids(actual_continue)
        required_correctness.append("actual_continue_values_match")
    elif actual_continue_values is not None:
        correctness["actual_continue_value_count"] = int(len(actual_continue_values))
    if actual_final_rng_state is not None:
        correctness["actual_final_rng_match"] = (
            _rng_states_equal(eager_final_rng, actual_final_rng_state)
            and _rng_states_equal(graph_final_rng, actual_final_rng_state)
        )
        required_correctness.append("actual_final_rng_match")

    _set_rng_state(start_rng_state)
    eager_timing_processors, _ = _clone_production_tail_processors(logits_processor)
    eager_timing_stopping = copy.deepcopy(stopping_criteria)
    eager_ms = _cuda_event_elapsed_ms(
        lambda: tail_fn(eager_timing_processors, eager_timing_stopping),
        warmup=warmup,
        iters=iters,
    )

    _set_rng_state(start_rng_state)
    graph_timing_processors, _ = _clone_production_tail_processors(logits_processor)
    graph_timing_stopping = copy.deepcopy(stopping_criteria)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_timing_outputs = tail_fn(graph_timing_processors, graph_timing_stopping)
    _set_rng_state(start_rng_state)
    graph.replay()
    torch.cuda.synchronize()
    graph_ms = _cuda_event_elapsed_ms(graph.replay, warmup=warmup, iters=iters)
    _ = graph_timing_outputs

    eager_ms_per_token = float(eager_ms) / steps
    graph_ms_per_token = float(graph_ms) / steps
    _set_rng_state(start_rng_state)
    return {
        "enabled": True,
        "pass": all(
            bool(correctness[key])
            for key in required_correctness
        ),
        "do_sample": bool(do_sample),
        "steps": int(steps),
        "start_len": int(start_len),
        "input_shape": list(input_ids.shape),
        "buffer_shape": list(static_sequence_template.shape),
        "logits_shape": list(static_raw_logits_steps[0].shape),
        "processor_classes": [processor.__class__.__name__ for processor in eager_processors],
        "excluded_verifier_processor_classes": excluded_processor_classes,
        "stopping_classes": [criteria.__class__.__name__ for criteria in stopping_criteria],
        "warmup": int(warmup),
        "iters": int(iters),
        "eager_ms_per_call": float(eager_ms),
        "graph_ms_per_replay": float(graph_ms),
        "eager_ms_per_token": eager_ms_per_token,
        "graph_ms_per_token": graph_ms_per_token,
        "speedup": float(eager_ms / graph_ms) if graph_ms > 0 else None,
        "saved_us_per_token": float((eager_ms_per_token - graph_ms_per_token) * 1000.0),
        "projected_full_song_saved_s_7639": float(
            (eager_ms_per_token - graph_ms_per_token) * 7639.0 / 1000.0
        ),
        "correctness": correctness,
        "note": (
            "Verifier-only fixed-start multi-step tail graph ceiling. This uses a preallocated "
            "sequence buffer and fails if the captured block stops before its final sampled step. "
            "It is not a production throughput claim."
        ),
    }


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
        tail_graph_ceiling: dict[str, Any] | None = None,
        tail_graph_multistep_ceiling: dict[str, Any] | None = None,
        tail_graph_step_index: int = 0,
        tail_graph_multistep_steps: int = 4,
        tail_graph_warmup: int = 50,
        tail_graph_iters: int = 500,
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
    tail_graph_capture: dict[str, Any] | None = None
    tail_graph_multistep_capture: dict[str, Any] | None = None
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
        generated_so_far = int(cur_len) - initial_len
        if (
                tail_graph_ceiling is not None
                and tail_graph_capture is None
                and generated_so_far == int(tail_graph_step_index)
        ):
            tail_graph_capture = {
                "input_ids": current_input_ids.detach().clone(memory_format=torch.contiguous_format),
                "raw_logits": next_token_logits.detach().clone(memory_format=torch.contiguous_format),
                "rng_state": _rng_state(),
                "actual_next_tokens": None,
                "actual_output_ids": None,
                "actual_unfinished": None,
                "actual_finished": None,
                "actual_final_rng_state": None,
                "logits_processor": copy.deepcopy(logits_processor),
                "stopping_criteria": copy.deepcopy(stopping_criteria),
                "do_sample": bool(do_sample),
                "has_eos_stopping_criteria": bool(has_eos_stopping_criteria),
                "pad_token_id": pad_token_id.detach().clone(memory_format=torch.contiguous_format),
                "unfinished_sequences": unfinished_sequences.detach().clone(memory_format=torch.contiguous_format),
                "generated_so_far": int(generated_so_far),
                "cur_len": int(cur_len),
            }
        if (
                tail_graph_multistep_ceiling is not None
                and tail_graph_multistep_capture is None
                and generated_so_far == int(tail_graph_step_index)
        ):
            tail_graph_multistep_capture = {
                "input_ids": current_input_ids.detach().clone(memory_format=torch.contiguous_format),
                "raw_logits_steps": [],
                "rng_state": _rng_state(),
                "actual_next_tokens": [],
                "actual_output_ids": None,
                "actual_unfinished": None,
                "actual_continue_values": [],
                "actual_final_rng_state": None,
                "logits_processor": copy.deepcopy(logits_processor),
                "stopping_criteria": copy.deepcopy(stopping_criteria),
                "do_sample": bool(do_sample),
                "has_eos_stopping_criteria": bool(has_eos_stopping_criteria),
                "pad_token_id": pad_token_id.detach().clone(memory_format=torch.contiguous_format),
                "unfinished_sequences": unfinished_sequences.detach().clone(memory_format=torch.contiguous_format),
                "generated_so_far": int(generated_so_far),
                "cur_len": int(cur_len),
                "requested_steps": int(tail_graph_multistep_steps),
            }
        if (
                tail_graph_multistep_capture is not None
                and len(tail_graph_multistep_capture["raw_logits_steps"]) < int(tail_graph_multistep_steps)
        ):
            tail_graph_multistep_capture["raw_logits_steps"].append(
                next_token_logits.detach().clone(memory_format=torch.contiguous_format)
            )

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
        if (
                tail_graph_capture is not None
                and tail_graph_capture["actual_next_tokens"] is None
                and generated_so_far == int(tail_graph_capture["generated_so_far"])
        ):
            tail_graph_capture["actual_next_tokens"] = next_tokens.detach().clone(
                memory_format=torch.contiguous_format
            )
            tail_graph_capture["actual_output_ids"] = input_ids.detach().clone(memory_format=torch.contiguous_format)
            tail_graph_capture["actual_unfinished"] = unfinished_sequences.detach().clone(
                memory_format=torch.contiguous_format
            )
            tail_graph_capture["actual_finished"] = this_peer_finished.detach().clone()
            tail_graph_capture["actual_final_rng_state"] = _rng_state()
        if tail_graph_multistep_capture is not None:
            actual_count = len(tail_graph_multistep_capture["actual_next_tokens"])
            raw_count = len(tail_graph_multistep_capture["raw_logits_steps"])
            if actual_count < raw_count and actual_count < int(tail_graph_multistep_steps):
                tail_graph_multistep_capture["actual_next_tokens"].append(
                    next_tokens.detach().clone(memory_format=torch.contiguous_format)
                )
                tail_graph_multistep_capture["actual_continue_values"].append(
                    unfinished_sequences.max().detach().clone()
                )
                if actual_count + 1 == int(tail_graph_multistep_steps):
                    tail_graph_multistep_capture["actual_output_ids"] = input_ids.detach().clone(
                        memory_format=torch.contiguous_format
                    )
                    tail_graph_multistep_capture["actual_unfinished"] = unfinished_sequences.detach().clone(
                        memory_format=torch.contiguous_format
                    )
                    tail_graph_multistep_capture["actual_final_rng_state"] = _rng_state()
        cur_len += 1

        del outputs

    generated_tokens = int(cur_len) - initial_len
    if sequence_buffer is not None:
        output = sequence_buffer[:, :cur_len].clone(memory_format=torch.contiguous_format)
        if decode_session is not None and decode_session_metadata is not None:
            decode_session_metadata.update(decode_session.metadata())
        _finalize_tail_diagnostics(tail_diagnostics, generated_tokens=generated_tokens)
        if tail_graph_ceiling is not None:
            if tail_graph_capture is None:
                tail_graph_ceiling.update({
                    "enabled": True,
                    "pass": False,
                    "error": f"did not reach tail graph step index {int(tail_graph_step_index)}",
                })
            else:
                final_rng_state = _rng_state()
                tail_graph_ceiling.update(tail_graph_capture)
                tail_graph_ceiling.update(_run_one_step_tail_graph_ceiling(
                    input_ids=tail_graph_capture["input_ids"],
                    raw_logits=tail_graph_capture["raw_logits"],
                    rng_state=tail_graph_capture["rng_state"],
                    actual_next_tokens=tail_graph_capture["actual_next_tokens"],
                    actual_output_ids=tail_graph_capture["actual_output_ids"],
                    actual_unfinished=tail_graph_capture["actual_unfinished"],
                    actual_finished=tail_graph_capture["actual_finished"],
                    actual_final_rng_state=tail_graph_capture["actual_final_rng_state"],
                    logits_processor=tail_graph_capture["logits_processor"],
                    stopping_criteria=tail_graph_capture["stopping_criteria"],
                    do_sample=tail_graph_capture["do_sample"],
                    has_eos_stopping_criteria=tail_graph_capture["has_eos_stopping_criteria"],
                    pad_token_id=tail_graph_capture["pad_token_id"],
                    unfinished_sequences=tail_graph_capture["unfinished_sequences"],
                    warmup=tail_graph_warmup,
                    iters=tail_graph_iters,
                ))
                _set_rng_state(final_rng_state)
                for key in ("input_ids", "raw_logits", "rng_state", "actual_next_tokens", "actual_output_ids",
                            "actual_unfinished", "actual_finished", "actual_final_rng_state", "logits_processor",
                            "stopping_criteria", "pad_token_id", "unfinished_sequences"):
                    tail_graph_ceiling.pop(key, None)
        if tail_graph_multistep_ceiling is not None:
            if tail_graph_multistep_capture is None:
                tail_graph_multistep_ceiling.update({
                    "enabled": True,
                    "pass": False,
                    "error": f"did not reach tail graph step index {int(tail_graph_step_index)}",
                })
            elif len(tail_graph_multistep_capture["raw_logits_steps"]) < int(tail_graph_multistep_steps):
                tail_graph_multistep_ceiling.update({
                    "enabled": True,
                    "pass": False,
                    "error": (
                        f"captured only {len(tail_graph_multistep_capture['raw_logits_steps'])} "
                        f"of {int(tail_graph_multistep_steps)} requested tail graph steps"
                    ),
                })
            else:
                final_rng_state = _rng_state()
                tail_graph_multistep_ceiling.update(tail_graph_multistep_capture)
                tail_graph_multistep_ceiling.update(_run_multistep_tail_graph_ceiling(
                    input_ids=tail_graph_multistep_capture["input_ids"],
                    raw_logits_steps=tail_graph_multistep_capture["raw_logits_steps"],
                    rng_state=tail_graph_multistep_capture["rng_state"],
                    actual_next_tokens=tail_graph_multistep_capture["actual_next_tokens"],
                    actual_output_ids=tail_graph_multistep_capture["actual_output_ids"],
                    actual_unfinished=tail_graph_multistep_capture["actual_unfinished"],
                    actual_continue_values=tail_graph_multistep_capture["actual_continue_values"],
                    actual_final_rng_state=tail_graph_multistep_capture["actual_final_rng_state"],
                    logits_processor=tail_graph_multistep_capture["logits_processor"],
                    stopping_criteria=tail_graph_multistep_capture["stopping_criteria"],
                    do_sample=tail_graph_multistep_capture["do_sample"],
                    has_eos_stopping_criteria=tail_graph_multistep_capture["has_eos_stopping_criteria"],
                    pad_token_id=tail_graph_multistep_capture["pad_token_id"],
                    unfinished_sequences=tail_graph_multistep_capture["unfinished_sequences"],
                    warmup=tail_graph_warmup,
                    iters=tail_graph_iters,
                ))
                _set_rng_state(final_rng_state)
                for key in ("input_ids", "raw_logits_steps", "rng_state", "actual_next_tokens",
                            "actual_output_ids", "actual_unfinished", "actual_continue_values",
                            "actual_final_rng_state", "logits_processor", "stopping_criteria", "pad_token_id",
                            "unfinished_sequences"):
                    tail_graph_multistep_ceiling.pop(key, None)
        return output
    if decode_session is not None and decode_session_metadata is not None:
        decode_session_metadata.update(decode_session.metadata())
    _finalize_tail_diagnostics(tail_diagnostics, generated_tokens=generated_tokens)
    if tail_graph_ceiling is not None:
        if tail_graph_capture is None:
            tail_graph_ceiling.update({
                "enabled": True,
                "pass": False,
                "error": f"did not reach tail graph step index {int(tail_graph_step_index)}",
            })
        else:
            final_rng_state = _rng_state()
            tail_graph_ceiling.update(tail_graph_capture)
            tail_graph_ceiling.update(_run_one_step_tail_graph_ceiling(
                input_ids=tail_graph_capture["input_ids"],
                raw_logits=tail_graph_capture["raw_logits"],
                rng_state=tail_graph_capture["rng_state"],
                actual_next_tokens=tail_graph_capture["actual_next_tokens"],
                actual_output_ids=tail_graph_capture["actual_output_ids"],
                actual_unfinished=tail_graph_capture["actual_unfinished"],
                actual_finished=tail_graph_capture["actual_finished"],
                actual_final_rng_state=tail_graph_capture["actual_final_rng_state"],
                logits_processor=tail_graph_capture["logits_processor"],
                stopping_criteria=tail_graph_capture["stopping_criteria"],
                do_sample=tail_graph_capture["do_sample"],
                has_eos_stopping_criteria=tail_graph_capture["has_eos_stopping_criteria"],
                pad_token_id=tail_graph_capture["pad_token_id"],
                unfinished_sequences=tail_graph_capture["unfinished_sequences"],
                warmup=tail_graph_warmup,
                iters=tail_graph_iters,
            ))
            _set_rng_state(final_rng_state)
            for key in ("input_ids", "raw_logits", "rng_state", "actual_next_tokens", "actual_output_ids",
                        "actual_unfinished", "actual_finished", "actual_final_rng_state", "logits_processor",
                        "stopping_criteria", "pad_token_id", "unfinished_sequences"):
                tail_graph_ceiling.pop(key, None)
    if tail_graph_multistep_ceiling is not None:
        if tail_graph_multistep_capture is None:
            tail_graph_multistep_ceiling.update({
                "enabled": True,
                "pass": False,
                "error": f"did not reach tail graph step index {int(tail_graph_step_index)}",
            })
        elif len(tail_graph_multistep_capture["raw_logits_steps"]) < int(tail_graph_multistep_steps):
            tail_graph_multistep_ceiling.update({
                "enabled": True,
                "pass": False,
                "error": (
                    f"captured only {len(tail_graph_multistep_capture['raw_logits_steps'])} "
                    f"of {int(tail_graph_multistep_steps)} requested tail graph steps"
                ),
            })
        else:
            final_rng_state = _rng_state()
            tail_graph_multistep_ceiling.update(tail_graph_multistep_capture)
            tail_graph_multistep_ceiling.update(_run_multistep_tail_graph_ceiling(
                input_ids=tail_graph_multistep_capture["input_ids"],
                raw_logits_steps=tail_graph_multistep_capture["raw_logits_steps"],
                rng_state=tail_graph_multistep_capture["rng_state"],
                actual_next_tokens=tail_graph_multistep_capture["actual_next_tokens"],
                actual_output_ids=tail_graph_multistep_capture["actual_output_ids"],
                actual_unfinished=tail_graph_multistep_capture["actual_unfinished"],
                actual_continue_values=tail_graph_multistep_capture["actual_continue_values"],
                actual_final_rng_state=tail_graph_multistep_capture["actual_final_rng_state"],
                logits_processor=tail_graph_multistep_capture["logits_processor"],
                stopping_criteria=tail_graph_multistep_capture["stopping_criteria"],
                do_sample=tail_graph_multistep_capture["do_sample"],
                has_eos_stopping_criteria=tail_graph_multistep_capture["has_eos_stopping_criteria"],
                pad_token_id=tail_graph_multistep_capture["pad_token_id"],
                unfinished_sequences=tail_graph_multistep_capture["unfinished_sequences"],
                warmup=tail_graph_warmup,
                iters=tail_graph_iters,
            ))
            _set_rng_state(final_rng_state)
            for key in ("input_ids", "raw_logits_steps", "rng_state", "actual_next_tokens",
                        "actual_output_ids", "actual_unfinished", "actual_continue_values",
                        "actual_final_rng_state", "logits_processor", "stopping_criteria", "pad_token_id",
                        "unfinished_sequences"):
                tail_graph_multistep_ceiling.pop(key, None)
    return input_ids


def _generated_token_ids(output: torch.Tensor, prompt_len: int) -> list[int]:
    return [int(item) for item in output[0, prompt_len:].tolist()]


def _sample_multinomial_sequence(
        probs_by_step: list[torch.Tensor],
        *,
        next_sample_count: int = 16,
) -> tuple[list[int], tuple[torch.Tensor, list[torch.Tensor] | None], list[int]]:
    samples = [
        int(torch.multinomial(probs, num_samples=1).item())
        for probs in probs_by_step
    ]
    final_state = _rng_state()
    next_samples = [
        int(torch.multinomial(probs_by_step[-1], num_samples=1).item())
        for _ in range(next_sample_count)
    ]
    return samples, final_state, next_samples


def _sample_multinomial_sequence_cuda_graph(
        probs_by_step: list[torch.Tensor],
        *,
        next_sample_count: int = 16,
) -> tuple[list[int], tuple[torch.Tensor, list[torch.Tensor] | None], list[int]]:
    if not torch.cuda.is_available():
        raise RuntimeError("fixed-K tail graph audit requires CUDA")
    if not probs_by_step:
        raise ValueError("probs_by_step must not be empty")
    if any(probs.device.type != "cuda" for probs in probs_by_step):
        raise ValueError("all probs tensors must be CUDA tensors")

    outputs = torch.empty((len(probs_by_step), 1), dtype=torch.long, device=probs_by_step[0].device)
    capture_start_state = _rng_state()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for index, probs in enumerate(probs_by_step):
            outputs[index].copy_(torch.multinomial(probs, num_samples=1).reshape(1))
    _set_rng_state(capture_start_state)
    graph.replay()
    torch.cuda.synchronize()
    samples = [int(item) for item in outputs.detach().cpu().view(-1).tolist()]
    final_state = _rng_state()
    next_samples = [
        int(torch.multinomial(probs_by_step[-1], num_samples=1).item())
        for _ in range(next_sample_count)
    ]
    return samples, final_state, next_samples


def _run_fixed_k_tail_graph_audit(
        *,
        device: torch.device,
        tail_graph_k: int,
        force_eos_at_generated_token: int,
        eos_token_id: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("fixed-K tail graph audit requires a CUDA device")
    if tail_graph_k <= 1:
        raise ValueError("tail_graph_k must be greater than 1")
    if force_eos_at_generated_token <= 0 or force_eos_at_generated_token >= tail_graph_k:
        raise ValueError("force_eos_at_generated_token must satisfy 1 <= N < tail_graph_k")

    vocab_size = max(int(eos_token_id) + 1, 512)
    logits = torch.linspace(-3.0, 3.0, steps=vocab_size, dtype=torch.float32, device=device).unsqueeze(0)
    probs = torch.softmax(logits, dim=-1)
    eos_probs = torch.zeros_like(probs)
    eos_probs[:, int(eos_token_id)] = 1.0
    next_sample_count = 16

    no_eos_steps = [probs for _ in range(tail_graph_k)]
    start_state = _rng_state()
    _set_rng_state(start_state)
    eager_samples, eager_state, eager_next = _sample_multinomial_sequence(
        no_eos_steps,
        next_sample_count=next_sample_count,
    )
    _set_rng_state(start_state)
    graph_samples, graph_state, graph_next = _sample_multinomial_sequence_cuda_graph(
        no_eos_steps,
        next_sample_count=next_sample_count,
    )
    no_eos_block = {
        "sequence_match": eager_samples == graph_samples,
        "final_rng_match": _rng_states_equal(eager_state, graph_state),
        "next_eager_sample_match": eager_next == graph_next,
        "graph_draw_count": int(tail_graph_k),
        "eager_draw_count": int(tail_graph_k),
        "eager_samples": eager_samples,
        "graph_samples": graph_samples,
        "eager_next_samples": eager_next,
        "graph_next_samples": graph_next,
    }

    forced_steps = [
        eos_probs if index == force_eos_at_generated_token - 1 else probs
        for index in range(tail_graph_k)
    ]
    eager_forced_steps = forced_steps[:force_eos_at_generated_token]
    start_state = _rng_state()
    _set_rng_state(start_state)
    eager_forced, eager_forced_state, eager_forced_next = _sample_multinomial_sequence(
        eager_forced_steps,
        next_sample_count=next_sample_count,
    )
    _set_rng_state(start_state)
    graph_forced, graph_forced_state, graph_forced_next = _sample_multinomial_sequence_cuda_graph(
        forced_steps,
        next_sample_count=next_sample_count,
    )
    graph_forced_prefix = graph_forced[:force_eos_at_generated_token]
    final_rng_match = _rng_states_equal(eager_forced_state, graph_forced_state)
    forced_eos_naive_block = {
        "output_prefix_through_eos_match": eager_forced == graph_forced_prefix,
        "expected_rng_divergence_detected": not final_rng_match,
        "final_rng_match": final_rng_match,
        "next_eager_sample_match": eager_forced_next == graph_forced_next,
        "graph_draw_count": int(tail_graph_k),
        "eager_draw_count": int(force_eos_at_generated_token),
        "forced_eos_token_id": int(eos_token_id),
        "eager_samples_through_eos": eager_forced,
        "graph_samples": graph_forced,
        "eager_next_samples": eager_forced_next,
        "graph_next_samples": graph_forced_next,
    }

    return {
        "enabled": True,
        "tail_graph_k": int(tail_graph_k),
        "force_eos_at_generated_token": int(force_eos_at_generated_token),
        "forced_eos_token_id": int(eos_token_id),
        "no_eos_block": no_eos_block,
        "forced_eos_naive_block": forced_eos_naive_block,
        "pass": (
            bool(no_eos_block["sequence_match"])
            and bool(no_eos_block["final_rng_match"])
            and bool(no_eos_block["next_eager_sample_match"])
            and bool(forced_eos_naive_block["output_prefix_through_eos_match"])
            and bool(forced_eos_naive_block["expected_rng_divergence_detected"])
        ),
        "note": (
            "Verifier-only audit. The forced-EOS naive fixed-K block is expected to diverge "
            "in final RNG state because it samples after eager generation would stop."
        ),
    }


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
        candidate_native_decoder_layer_mlp_tail: bool,
        candidate_native_decoder_layer_mlp_tail_outputs_per_block: int,
        candidate_decode_session: bool,
        candidate_fast_prepare: bool,
        force_eos_at_generated_token: int | None = None,
        fixed_k_tail_graph_audit: bool = False,
        tail_graph_k: int = 4,
        tail_diagnostics: bool = False,
        tail_graph_ceiling: bool = False,
        tail_graph_multistep_ceiling: bool = False,
        tail_graph_step_index: int = 0,
        tail_graph_multistep_steps: int = 4,
        tail_graph_warmup: int = 50,
        tail_graph_iters: int = 500,
) -> dict[str, Any]:
    _assert_supported_probe(args)
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if tail_graph_multistep_ceiling and tail_graph_multistep_steps <= 0:
        raise ValueError("--tail-graph-multistep-steps must be positive")

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
        "candidate_native_decoder_layer_mlp_tail": bool(candidate_native_decoder_layer_mlp_tail),
        "candidate_native_decoder_layer_mlp_tail_outputs_per_block": (
            int(candidate_native_decoder_layer_mlp_tail_outputs_per_block)
        ),
        "candidate_decode_session": bool(candidate_decode_session),
        "candidate_fast_prepare": bool(candidate_fast_prepare),
        "force_eos_at_generated_token": force_eos_at_generated_token,
        "fixed_k_tail_graph_audit": bool(fixed_k_tail_graph_audit),
        "tail_graph_k": int(tail_graph_k),
        "tail_diagnostics": bool(tail_diagnostics),
        "tail_graph_ceiling": bool(tail_graph_ceiling),
        "tail_graph_multistep_ceiling": bool(tail_graph_multistep_ceiling),
        "tail_graph_step_index": int(tail_graph_step_index),
        "tail_graph_multistep_steps": int(tail_graph_multistep_steps),
        "tail_graph_warmup": int(tail_graph_warmup),
        "tail_graph_iters": int(tail_graph_iters),
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
    forced_eos_token_id = int(metadata["eos_token_ids"][0])
    if fixed_k_tail_graph_audit and force_eos_at_generated_token is None:
        raise ValueError("--fixed-k-tail-graph-audit requires --force-eos-at-generated-token")
    if force_eos_at_generated_token is not None and force_eos_at_generated_token <= 0:
        raise ValueError("--force-eos-at-generated-token must be positive")

    rng_state = _rng_state()
    reference_captures: list[torch.Tensor] = []
    reference_processors = _build_generation_logits_processors(
        args,
        tokenizer,
        model.device,
        lookback_time=float(metadata["lookback_time"]),
    )
    reference_processors.insert(0, _CaptureRawLogitsProcessor(reference_captures))
    if force_eos_at_generated_token is not None:
        reference_processors.insert(
            1,
            _ForceEosAtGeneratedStepProcessor(
                prompt_len=prompt_len,
                generated_step=int(force_eos_at_generated_token),
                eos_token_id=forced_eos_token_id,
            ),
        )

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
    if force_eos_at_generated_token is not None:
        candidate_processors.insert(
            1,
            _ForceEosAtGeneratedStepProcessor(
                prompt_len=prompt_len,
                generated_step=int(force_eos_at_generated_token),
                eos_token_id=forced_eos_token_id,
            ),
        )
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
    candidate_tail_graph_ceiling: dict[str, Any] | None = (
        {
            "enabled": True,
            "note": "Verifier-only one-step tail CUDA graph ceiling; not an inference throughput claim.",
        }
        if tail_graph_ceiling
        else None
    )
    candidate_tail_graph_multistep_ceiling: dict[str, Any] | None = (
        {
            "enabled": True,
            "note": "Verifier-only fixed-start multi-step tail CUDA graph ceiling; not a throughput claim.",
        }
        if tail_graph_multistep_ceiling
        else None
    )

    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=args.precision == "amp"), \
            generation_profile_context(
                sdpa_backend=args.profile_sdpa_backend,
                q1_bmm_cross_attention=candidate_q1_bmm_cross_attention,
                native_q1_self_attention=candidate_native_q1_self_attention,
                native_q1_rope_cache_self_attention=candidate_native_q1_rope_cache_self_attention,
                native_decoder_layer_mlp_tail=candidate_native_decoder_layer_mlp_tail,
                native_decoder_layer_mlp_tail_outputs_per_block=(
                    candidate_native_decoder_layer_mlp_tail_outputs_per_block
                ),
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
                tail_graph_ceiling=candidate_tail_graph_ceiling,
                tail_graph_multistep_ceiling=candidate_tail_graph_multistep_ceiling,
                tail_graph_step_index=tail_graph_step_index,
                tail_graph_multistep_steps=tail_graph_multistep_steps,
                tail_graph_warmup=tail_graph_warmup,
                tail_graph_iters=tail_graph_iters,
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
    force_eos_direct_loop = {
        "enabled": force_eos_at_generated_token is not None,
        "force_eos_at_generated_token": force_eos_at_generated_token,
        "forced_eos_token_id": forced_eos_token_id if force_eos_at_generated_token is not None else None,
        "reference_stop_reason": reference_stop_reason,
        "candidate_stop_reason": candidate_stop_reason,
        "token_match": token_match,
        "logits_pass": logits_pass,
        "rng_match": rng_match,
        "pass": (
            token_match
            and logits_pass
            and rng_match
            and reference_stop_reason == "eos"
            and candidate_stop_reason == "eos"
        ) if force_eos_at_generated_token is not None else None,
    }
    fixed_k_tail_graph_report = (
        _run_fixed_k_tail_graph_audit(
            device=model.device,
            tail_graph_k=tail_graph_k,
            force_eos_at_generated_token=int(force_eos_at_generated_token),
            eos_token_id=forced_eos_token_id,
        )
        if fixed_k_tail_graph_audit
        else {"enabled": False}
    )
    pass_result = token_match and logits_pass and rng_match
    if force_eos_at_generated_token is not None:
        pass_result = pass_result and bool(force_eos_direct_loop["pass"])
    if fixed_k_tail_graph_audit:
        pass_result = pass_result and bool(fixed_k_tail_graph_report["pass"])
    if tail_graph_ceiling:
        pass_result = pass_result and bool(
            candidate_tail_graph_ceiling is not None
            and candidate_tail_graph_ceiling.get("pass")
        )
    if tail_graph_multistep_ceiling:
        pass_result = pass_result and bool(
            candidate_tail_graph_multistep_ceiling is not None
            and candidate_tail_graph_multistep_ceiling.get("pass")
        )

    return {
        "pass": pass_result,
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
        "candidate_tail_graph_ceiling": candidate_tail_graph_ceiling,
        "candidate_tail_graph_multistep_ceiling": candidate_tail_graph_multistep_ceiling,
        "force_eos_direct_loop": force_eos_direct_loop,
        "fixed_k_tail_graph_audit": fixed_k_tail_graph_report,
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
        "--candidate-native-decoder-layer-mlp-tail",
        action="store_true",
        help="Enable the experimental native fp32 decoder-layer MLP tail candidate.",
    )
    parser.add_argument(
        "--candidate-native-decoder-layer-mlp-tail-outputs-per-block",
        type=int,
        default=4,
        choices=(2, 4, 8),
        help="Warp-group width for the experimental native decoder-layer MLP tail.",
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
        "--force-eos-at-generated-token",
        type=int,
        default=None,
        help="Verifier-only: force EOS at the Nth generated token after raw-logit capture.",
    )
    parser.add_argument(
        "--fixed-k-tail-graph-audit",
        action="store_true",
        help=(
            "Verifier-only: audit fixed-K CUDA graph multinomial exactness and demonstrate "
            "naive RNG divergence when EOS occurs before K."
        ),
    )
    parser.add_argument(
        "--tail-graph-k",
        type=int,
        default=4,
        help="Fixed K for --fixed-k-tail-graph-audit.",
    )
    parser.add_argument(
        "--tail-graph-ceiling",
        action="store_true",
        help=(
            "Verifier-only: capture one exact logits-processor/sampling/stop/append tail step "
            "inside a CUDA graph and report the replay ceiling."
        ),
    )
    parser.add_argument(
        "--tail-graph-multistep-ceiling",
        action="store_true",
        help=(
            "Verifier-only: capture consecutive exact tail steps and replay them through a "
            "fixed-start preallocated-buffer CUDA graph."
        ),
    )
    parser.add_argument(
        "--tail-graph-step-index",
        type=int,
        default=0,
        help="Generated-token index to capture for --tail-graph-ceiling.",
    )
    parser.add_argument(
        "--tail-graph-multistep-steps",
        type=int,
        default=4,
        help="Number of consecutive generated tail steps for --tail-graph-multistep-ceiling.",
    )
    parser.add_argument(
        "--tail-graph-warmup",
        type=int,
        default=50,
        help="CUDA event warmup iterations for --tail-graph-ceiling timing.",
    )
    parser.add_argument(
        "--tail-graph-iters",
        type=int,
        default=500,
        help="CUDA event measured iterations for --tail-graph-ceiling timing.",
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
        candidate_native_decoder_layer_mlp_tail=cli_args.candidate_native_decoder_layer_mlp_tail,
        candidate_native_decoder_layer_mlp_tail_outputs_per_block=(
            cli_args.candidate_native_decoder_layer_mlp_tail_outputs_per_block
        ),
        candidate_decode_session=cli_args.candidate_decode_session,
        candidate_fast_prepare=cli_args.candidate_fast_prepare,
        force_eos_at_generated_token=cli_args.force_eos_at_generated_token,
        fixed_k_tail_graph_audit=cli_args.fixed_k_tail_graph_audit,
        tail_graph_k=cli_args.tail_graph_k,
        tail_graph_ceiling=cli_args.tail_graph_ceiling,
        tail_graph_multistep_ceiling=cli_args.tail_graph_multistep_ceiling,
        tail_graph_step_index=cli_args.tail_graph_step_index,
        tail_graph_multistep_steps=cli_args.tail_graph_multistep_steps,
        tail_graph_warmup=cli_args.tail_graph_warmup,
        tail_graph_iters=cli_args.tail_graph_iters,
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
