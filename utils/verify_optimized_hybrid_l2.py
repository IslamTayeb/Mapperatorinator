"""Verify one fixed-shape hybrid L2 model/processor/sampling pipeline.

This utility is a bounded GPU verifier, not runtime wiring.  It reuses the
reviewed two-lane setup, then captures a shared B2 full-scan base processor and
two private sampling tails.  The complete timing interval discards sampled
outputs and uses only preallocated storage plus graph/event replay.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import transformers
from transformers import LogitsProcessorList, TopKLogitsWarper, TopPLogitsWarper

from inference import compile_args, load_model_with_server, setup_inference_environment
from osuT5.osuT5.inference.decode_loop import _bucketed_prefix_length
from osuT5.osuT5.inference.optimized.graph_safe_logits import (
    make_graph_safe_logits_processor_list,
)
from osuT5.osuT5.inference.optimized.batch.hybrid_l2 import (
    FIXED_ACTIVE_PREFIX_BUCKET_SIZE,
    FIXED_ACTIVE_PREFIX_LENGTH,
    FIXED_ROW_SEEDS,
    FIXED_SEQUENCE_INDEX,
    FIXED_TIMING_REPEATS,
    HYBRID_EVENT_ORDER,
    HYBRID_L2_GATE,
    HYBRID_L2_SCHEMA_VERSION,
    REVIEWED_L2_REPORT_SHA256,
    REVIEWED_MODEL_PATH,
    REVIEWED_WORKLOAD_CONTRACT_SHA256,
    HybridBridgeResourceEvidence,
    HybridSamplingResourceEvidence,
    processor_class_names,
    summarize_hybrid_l2_performance,
    validate_hybrid_l2_report,
    validate_hybrid_processor_family,
    validate_hybrid_resource_ownership,
    validate_reviewed_l2_report,
)
from osuT5.osuT5.inference.optimized.batch.lane_one_token import (
    compare_lane_logits,
    normalized_one_token_workload_contract,
    reciprocal_lane_orders,
    tensor_sha256,
)
from osuT5.osuT5.inference.server import get_eos_token_id
from osuT5.osuT5.runtime_profiling import generation_profile_context
from osuT5.osuT5.tokenizer import ContextType
from utils.verify_one_token_decode import (
    _assert_supported_probe,
    _build_generation_logits_processors,
    _build_probe_inputs,
    _condition_kwargs,
    _load_args,
    _move_kwargs_for_model,
)
from utils.verify_optimized_b1_lane_capture import _new_generator, _tensor_storage_ptr
from utils.verify_optimized_l2_lane_capture import (
    _CapturedLaneRow,
    _ReferenceRow,
    _build_reference_row,
    _capture_lane_row,
    _json_sha256,
    _order_key,
)


ProcessorFactory = Callable[[], Any]


@dataclass
class _CapturedProcessorBridge:
    stream: torch.cuda.Stream
    graph: torch.cuda.CUDAGraph
    processor: Any
    prefix_batch: torch.LongTensor
    raw_logits_bridge: torch.Tensor
    raw_logits_rows: tuple[torch.Tensor, torch.Tensor]
    processed_scores: torch.Tensor
    processed_score_rows: tuple[torch.Tensor, torch.Tensor]
    warmup_seconds: float
    capture_seconds: float


@dataclass
class _CapturedSamplingTail:
    lane_id: int
    stream: torch.cuda.Stream
    graph: torch.cuda.CUDAGraph
    logits_warper: Any
    generator: torch.Generator
    processed_scores: torch.Tensor
    probabilities: torch.Tensor
    sampled_token: torch.LongTensor
    warmup_seconds: float
    capture_seconds: float


@dataclass
class _HybridEvents:
    lane_ready: tuple[torch.cuda.Event, torch.cuda.Event]
    source_released: tuple[torch.cuda.Event, torch.cuda.Event]
    processor_done: torch.cuda.Event
    sample_done: tuple[torch.cuda.Event, torch.cuda.Event]


@dataclass
class _HybridResources:
    lanes: tuple[_CapturedLaneRow, _CapturedLaneRow]
    source_logits: tuple[torch.Tensor, torch.Tensor]
    bridge: _CapturedProcessorBridge
    sampling: tuple[_CapturedSamplingTail, _CapturedSamplingTail]
    events: _HybridEvents


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object.")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_fixed_runtime_contract(
        args: Any,
        *,
        config_name: str,
        q1_bmm_cross_attention: bool,
        native_q1_self_attention: bool,
        native_q1_rope_cache_self_attention: bool,
) -> None:
    """Fail before model load when the direct CLI differs from reviewed L2."""

    if config_name != "profile_salvalai_smoke15":
        raise ValueError("hybrid L2 requires config_name=profile_salvalai_smoke15.")
    for name, enabled in (
            ("q1_bmm_cross_attention", q1_bmm_cross_attention),
            ("native_q1_self_attention", native_q1_self_attention),
            ("native_q1_rope_cache_self_attention", native_q1_rope_cache_self_attention),
    ):
        if enabled is not True:
            raise ValueError(f"hybrid L2 requires CLI {name}=true.")

    expected_strings_or_none = {
        "model_path": REVIEWED_MODEL_PATH,
        "device": "cuda",
        "precision": "fp32",
        "attn_implementation": "sdpa",
        "inference_engine": "v32",
        "optimized_inference_mode": "single",
        "profile_sdpa_backend": None,
        "lora_path": None,
    }
    for field, expected in expected_strings_or_none.items():
        if getattr(args, field, None) != expected:
            raise ValueError(f"hybrid L2 requires {field}={expected!r}.")

    expected_booleans = {
        "auto_select_gamemode_model": True,
        "use_server": False,
        "parallel": False,
        "do_sample": True,
        "inference_generation_compile": False,
        "inference_active_prefix_decode_loop": True,
        "inference_active_prefix_decode_cuda_graph": True,
        "inference_stateful_monotonic_logits_processor": True,
        "inference_q1_bmm_cross_attention": True,
        "inference_decode_session_runtime": True,
        "inference_decode_session_cuda_graph": True,
        "inference_native_decode_kernels": True,
        "inference_native_q1_self_attention": True,
        "inference_native_q1_rope_cache_self_attention": True,
    }
    for field, expected in expected_booleans.items():
        if getattr(args, field, None) is not expected:
            raise ValueError(f"hybrid L2 requires {field}={str(expected).lower()}.")

    expected_integers = {
        "seed": 12345,
        "gamemode": 0,
        "num_beams": 1,
        "top_k": 0,
        "inference_active_prefix_decode_bucket_size": 64,
        "inference_active_prefix_decode_cuda_graph_warmup": 0,
        "inference_active_prefix_decode_cuda_graph_min_decode_steps": 1,
        "inference_decode_session_chunk_size": 1,
    }
    for field, expected in expected_integers.items():
        value = getattr(args, field, None)
        if not isinstance(value, int) or isinstance(value, bool) or value != expected:
            raise ValueError(f"hybrid L2 requires {field}={expected}.")

    expected_numbers = {
        "cfg_scale": 1.0,
        "temperature": 0.9,
        "timeshift_bias": 0.0,
        "top_p": 0.9,
    }
    for field, expected in expected_numbers.items():
        value = getattr(args, field, None)
        if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) != expected
        ):
            raise ValueError(f"hybrid L2 requires {field}={expected}.")


def _new_event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=False, blocking=False, interprocess=False)


def _event_id(event: torch.cuda.Event) -> int:
    return id(event)


def _reset_sampling_generators(
        sampling: Sequence[_CapturedSamplingTail],
        seeds: Sequence[int],
) -> None:
    if len(sampling) != len(seeds):
        raise ValueError("sampling tails and seeds must have the same length.")
    for tail, seed in zip(sampling, seeds, strict=True):
        tail.generator.manual_seed(int(seed))


def _build_prefix_batch(lanes: Sequence[_CapturedLaneRow]) -> torch.LongTensor:
    if len(lanes) != 2:
        raise ValueError("hybrid L2 requires exactly two lanes.")
    first = lanes[0].full_prefix
    second = lanes[1].full_prefix
    if first.shape != second.shape or first.dtype != second.dtype or first.device != second.device:
        raise ValueError("hybrid L2 requires compatible fixed-shape lane prefixes.")
    prefix_batch = torch.empty(
        (2, int(first.shape[-1])),
        dtype=first.dtype,
        device=first.device,
    )
    prefix_batch[0:1].copy_(first)
    prefix_batch[1:2].copy_(second)
    return prefix_batch


@torch.no_grad()
def _capture_shared_processor_bridge(
        lanes: Sequence[_CapturedLaneRow],
        *,
        base_processor_factory: ProcessorFactory,
        capture_warmup_repeats: int,
) -> _CapturedProcessorBridge:
    if capture_warmup_repeats < 1:
        raise ValueError("shared processor capture requires a positive warmup count.")
    prefix_batch = _build_prefix_batch(lanes)
    source_logits = (lanes[0].lane.logits, lanes[1].lane.logits)
    raw_shape = (2, int(source_logits[0].shape[-1]))
    raw_logits_bridge = torch.empty(
        raw_shape,
        dtype=source_logits[0].dtype,
        device=source_logits[0].device,
    )
    raw_logits_rows = (raw_logits_bridge[0:1], raw_logits_bridge[1:2])
    raw_logits_rows[0].copy_(source_logits[0])
    raw_logits_rows[1].copy_(source_logits[1])
    processor = base_processor_factory()
    validate_hybrid_processor_family(
        base_processor_classes=processor_class_names(processor),
        private_warper_classes=("TopPLogitsWarper",),
    )

    device = prefix_batch.device
    current = torch.cuda.current_stream(device)
    control = torch.cuda.Stream(device=device)
    control.wait_stream(current)
    warmup_started = time.perf_counter()
    with torch.cuda.stream(control):
        for _ in range(capture_warmup_repeats):
            processor(prefix_batch, raw_logits_bridge)
    current.wait_stream(control)
    torch.cuda.synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_started

    graph = torch.cuda.CUDAGraph()
    capture_started = time.perf_counter()
    with torch.cuda.graph(graph, stream=control):
        processed_scores = processor(prefix_batch, raw_logits_bridge)
    capture_seconds = time.perf_counter() - capture_started
    control.synchronize()
    processed_score_rows = (processed_scores[0:1], processed_scores[1:2])
    return _CapturedProcessorBridge(
        stream=control,
        graph=graph,
        processor=processor,
        prefix_batch=prefix_batch,
        raw_logits_bridge=raw_logits_bridge,
        raw_logits_rows=raw_logits_rows,
        processed_scores=processed_scores,
        processed_score_rows=processed_score_rows,
        warmup_seconds=warmup_seconds,
        capture_seconds=capture_seconds,
    )


@torch.no_grad()
def _capture_private_sampling_tail(
        lane: _CapturedLaneRow,
        *,
        logits_warper_factory: ProcessorFactory,
        seed: int,
        capture_warmup_repeats: int,
) -> _CapturedSamplingTail:
    if capture_warmup_repeats < 1:
        raise ValueError("private sampling capture requires a positive warmup count.")
    stream = lane.lane.stream
    device = lane.full_prefix.device
    processed_scores = torch.empty_like(lane.lane.logits)
    processed_scores.copy_(lane.lane.logits)
    logits_warper = logits_warper_factory()
    validate_hybrid_processor_family(
        base_processor_classes=(
            "MonotonicTimeShiftLogitsProcessor",
            "TemperatureLogitsWarper",
        ),
        private_warper_classes=processor_class_names(logits_warper),
    )

    generator = _new_generator(torch.device(device), seed)
    warmup_generator = _new_generator(torch.device(device), seed + 900_000)
    stream.wait_stream(torch.cuda.current_stream(device))
    warmup_started = time.perf_counter()
    with torch.cuda.stream(stream):
        for _ in range(capture_warmup_repeats):
            warm_scores = logits_warper(lane.full_prefix, processed_scores)
            warm_probabilities = torch.nn.functional.softmax(warm_scores, dim=-1)
            torch.multinomial(
                warm_probabilities,
                num_samples=1,
                generator=warmup_generator,
            )
    stream.synchronize()
    warmup_seconds = time.perf_counter() - warmup_started

    graph = torch.cuda.CUDAGraph()
    graph.register_generator_state(generator)
    capture_started = time.perf_counter()
    with torch.cuda.graph(graph, stream=stream):
        warped_scores = logits_warper(lane.full_prefix, processed_scores)
        probabilities = torch.nn.functional.softmax(warped_scores, dim=-1)
        sampled_token = torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        ).squeeze(1)
    capture_seconds = time.perf_counter() - capture_started
    stream.synchronize()
    # The capture itself may reserve a Philox offset.  The registered generator
    # is reset before every measured/exact interval, matching the reviewed RNG
    # probe and the private eager reference transcript.
    generator.manual_seed(seed)
    return _CapturedSamplingTail(
        lane_id=lane.row,
        stream=stream,
        graph=graph,
        logits_warper=logits_warper,
        generator=generator,
        processed_scores=processed_scores,
        probabilities=probabilities,
        sampled_token=sampled_token,
        warmup_seconds=warmup_seconds,
        capture_seconds=capture_seconds,
    )


def _build_events() -> _HybridEvents:
    return _HybridEvents(
        lane_ready=(_new_event(), _new_event()),
        source_released=(_new_event(), _new_event()),
        processor_done=_new_event(),
        sample_done=(_new_event(), _new_event()),
    )


def _resource_evidence(resources: _HybridResources) -> HybridBridgeResourceEvidence:
    bridge = resources.bridge
    sampling = tuple(
        HybridSamplingResourceEvidence(
            lane_id=tail.lane_id,
            sampling_graph_id=id(tail.graph),
            sampling_graph_pool_id=repr(tail.graph.pool()),
            processed_scores_storage_ptr=_tensor_storage_ptr(tail.processed_scores),
            probabilities_storage_ptr=_tensor_storage_ptr(tail.probabilities),
            sampled_token_storage_ptr=_tensor_storage_ptr(tail.sampled_token),
            generator_id=id(tail.generator),
            logits_warper_id=id(tail.logits_warper),
        )
        for tail in resources.sampling
    )
    return HybridBridgeResourceEvidence(
        control_stream_id=int(bridge.stream.cuda_stream),
        processor_graph_id=id(bridge.graph),
        processor_graph_pool_id=repr(bridge.graph.pool()),
        prefix_batch_storage_ptr=_tensor_storage_ptr(bridge.prefix_batch),
        raw_logits_bridge_storage_ptr=_tensor_storage_ptr(bridge.raw_logits_bridge),
        processed_scores_storage_ptr=_tensor_storage_ptr(bridge.processed_scores),
        source_release_event_ids=tuple(
            _event_id(event) for event in resources.events.source_released
        ),
        lane_ready_event_ids=tuple(
            _event_id(event) for event in resources.events.lane_ready
        ),
        sample_done_event_ids=tuple(
            _event_id(event) for event in resources.events.sample_done
        ),
        processor_done_event_id=_event_id(resources.events.processor_done),
        sampling_lanes=sampling,
    )


def _enqueue_hybrid_steps(
        resources: _HybridResources,
        *,
        order: Sequence[int],
        repeats: int,
        token_archive_rows: Sequence[Sequence[torch.Tensor]] | None,
) -> tuple[torch.cuda.Event, torch.cuda.Event, float]:
    """Enqueue the audited event chain without a per-step host/device sync."""

    if tuple(sorted(order)) != (0, 1):
        raise ValueError("hybrid execution order must contain lanes 0 and 1 once.")
    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    if token_archive_rows is not None and (
            len(token_archive_rows) != repeats
            or any(len(rows) != 2 for rows in token_archive_rows)
    ):
        raise ValueError("token_archive_rows must contain two static views per repeat.")

    bridge = resources.bridge
    events = resources.events
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    start.record(bridge.stream)
    for lane in resources.lanes:
        lane.lane.stream.wait_event(start)

    for step in range(repeats):
        for row in order:
            lane = resources.lanes[row]
            with torch.cuda.stream(lane.lane.stream):
                if step > 0:
                    lane.lane.stream.wait_event(events.source_released[row])
                lane.lane.graph.replay()
                events.lane_ready[row].record(lane.lane.stream)
        with torch.cuda.stream(bridge.stream):
            for row in order:
                bridge.stream.wait_event(events.lane_ready[row])
            for row in order:
                bridge.raw_logits_rows[row].copy_(resources.source_logits[row])
            for row in order:
                events.source_released[row].record(bridge.stream)
            bridge.graph.replay()
            for row in order:
                resources.sampling[row].processed_scores.copy_(
                    bridge.processed_score_rows[row]
                )
            events.processor_done.record(bridge.stream)
        for row in order:
            tail = resources.sampling[row]
            with torch.cuda.stream(tail.stream):
                tail.stream.wait_event(events.processor_done)
                tail.graph.replay()
                if token_archive_rows is not None:
                    token_archive_rows[step][row].copy_(tail.sampled_token)
                events.sample_done[row].record(tail.stream)

    with torch.cuda.stream(bridge.stream):
        for row in order:
            bridge.stream.wait_event(events.sample_done[row])
        finish.record(bridge.stream)
    finish.synchronize()
    return start, finish, time.perf_counter() - wall_started


@torch.no_grad()
def _processed_score_gate(
        resources: _HybridResources,
        *,
        order: Sequence[int],
        base_processor_factory: ProcessorFactory,
        top_k: int,
) -> dict[str, Any]:
    bridge = resources.bridge
    events = resources.events
    for row in order:
        lane = resources.lanes[row]
        with torch.cuda.stream(lane.lane.stream):
            lane.lane.graph.replay()
            events.lane_ready[row].record(lane.lane.stream)
    with torch.cuda.stream(bridge.stream):
        for row in order:
            bridge.stream.wait_event(events.lane_ready[row])
        for row in order:
            bridge.raw_logits_rows[row].copy_(resources.source_logits[row])
            events.source_released[row].record(bridge.stream)
        bridge.graph.replay()
    bridge.stream.synchronize()

    rows: list[dict[str, Any]] = []
    for row in range(2):
        private_processor = base_processor_factory()
        private_scores = private_processor(
            resources.lanes[row].full_prefix,
            resources.source_logits[row].clone(memory_format=torch.contiguous_format),
        )
        shared_scores = bridge.processed_score_rows[row]
        comparison = compare_lane_logits(
            private_scores,
            shared_scores,
            atol=0.0,
            rtol=0.0,
            top_k=top_k,
        )
        bitwise = bool(torch.equal(private_scores, shared_scores))
        rows.append({
            "row": row,
            "bitwise_match": bitwise,
            "topk_match": bool(comparison["topk_match"]),
            "comparison": comparison,
            "private_sha256": tensor_sha256(private_scores),
            "shared_sha256": tensor_sha256(shared_scores),
            "pass": bool(bitwise and comparison["topk_match"]),
        })
    return {
        "rows": rows,
        "bitwise_match": all(row["bitwise_match"] for row in rows),
        "topk_match": all(row["topk_match"] for row in rows),
        "pass": all(row["pass"] for row in rows),
    }


@torch.no_grad()
def _run_exact_transcript(
        resources: _HybridResources,
        *,
        order: Sequence[int],
        seeds: Sequence[int],
        repeats: int,
) -> tuple[dict[str, list[int]], dict[str, str]]:
    _reset_sampling_generators(resources.sampling, seeds)
    archive = torch.empty(
        (repeats, 2),
        dtype=torch.long,
        device=resources.bridge.prefix_batch.device,
    )
    archive_rows = tuple(
        (archive[step:step + 1, 0], archive[step:step + 1, 1])
        for step in range(repeats)
    )
    _enqueue_hybrid_steps(
        resources,
        order=order,
        repeats=repeats,
        token_archive_rows=archive_rows,
    )
    tokens_cpu = archive.detach().cpu()
    transcripts = {
        f"row-{row}": [int(value) for value in tokens_cpu[:, row].tolist()]
        for row in range(2)
    }
    rng_hashes = {
        f"row-{row}": tensor_sha256(resources.sampling[row].generator.get_state())
        for row in range(2)
    }
    return transcripts, rng_hashes


@torch.no_grad()
def _run_complete_output_discard_timing(
        resources: _HybridResources,
        *,
        order: Sequence[int],
        seeds: Sequence[int],
        repeats: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Time complete graph/control work while retaining no per-step outputs."""

    _reset_sampling_generators(resources.sampling, seeds)
    device = resources.bridge.prefix_batch.device
    torch.cuda.reset_peak_memory_stats(device)
    memory_before_allocated = torch.cuda.memory_allocated(device)
    memory_before_reserved = torch.cuda.memory_reserved(device)
    start, finish, wall_seconds = _enqueue_hybrid_steps(
        resources,
        order=order,
        repeats=repeats,
        token_archive_rows=None,
    )
    cuda_seconds = start.elapsed_time(finish) / 1000.0
    generated_tokens = 2 * repeats
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    timing = {
        "measurement": "complete_output_discard",
        "outputs_discarded": True,
        "timing_repeats": repeats,
        "generated_tokens": generated_tokens,
        "wall_seconds": wall_seconds,
        "cuda_seconds": cuda_seconds,
        "wall_tokens_per_second": generated_tokens / wall_seconds,
        "cuda_tokens_per_second": generated_tokens / cuda_seconds,
        "memory_before_allocated_bytes": memory_before_allocated,
        "memory_before_reserved_bytes": memory_before_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_delta_bytes": peak_allocated - memory_before_allocated,
        "peak_reserved_delta_bytes": peak_reserved - memory_before_reserved,
        "per_step_global_synchronize": False,
        "per_step_stack_or_cat": False,
        "per_step_python_tensor_allocation": False,
        "static_tensor_storage": True,
        "sampling_tail_in_cuda_graph": True,
        "explicit_generators_registered_with_sampling_graphs": True,
    }
    rng_hashes = {
        f"row-{row}": tensor_sha256(resources.sampling[row].generator.get_state())
        for row in range(2)
    }
    return timing, rng_hashes


def _make_factories(
        args: Any,
        tokenizer: Any,
        device: torch.device,
        *,
        lookback_time: float,
) -> tuple[ProcessorFactory, ProcessorFactory, ProcessorFactory]:
    def base_processor_factory():
        return make_graph_safe_logits_processor_list(
            _build_generation_logits_processors(
                args,
                tokenizer,
                device,
                lookback_time=lookback_time,
            )
        )

    def logits_warper_factory():
        warpers = LogitsProcessorList()
        if args.do_sample:
            if args.top_k > 0:
                warpers.append(TopKLogitsWarper(top_k=int(args.top_k)))
            if 0.0 < args.top_p < 1.0:
                warpers.append(TopPLogitsWarper(top_p=float(args.top_p)))
        return warpers

    def combined_processor_factory():
        processors = base_processor_factory()
        processors.extend(logits_warper_factory())
        return processors

    validate_hybrid_processor_family(
        base_processor_classes=processor_class_names(base_processor_factory()),
        private_warper_classes=processor_class_names(logits_warper_factory()),
    )
    return base_processor_factory, logits_warper_factory, combined_processor_factory


@torch.no_grad()
def run_hybrid_l2_gate(
        args: Any,
        *,
        config_name: str,
        reviewed_l2_report: Mapping[str, Any],
        reviewed_l2_report_path: Path,
        sequence_index: int,
        seeds: tuple[int, int],
        timing_repeats: int,
        capture_warmup_repeats: int,
        comparison_top_k: int,
        active_prefix_bucket_size: int,
        q1_bmm_cross_attention: bool,
        native_q1_self_attention: bool,
        native_q1_rope_cache_self_attention: bool,
) -> dict[str, Any]:
    validate_reviewed_l2_report(reviewed_l2_report)
    if _file_sha256(reviewed_l2_report_path) != REVIEWED_L2_REPORT_SHA256:
        raise ValueError("reviewed L2 report SHA-256 does not match the audited artifact.")
    if sequence_index != FIXED_SEQUENCE_INDEX:
        raise ValueError("hybrid L2 is fixed to SALVALAI sequence 9.")
    if seeds != FIXED_ROW_SEEDS:
        raise ValueError(f"hybrid L2 row seeds must be {FIXED_ROW_SEEDS}.")
    if timing_repeats != FIXED_TIMING_REPEATS:
        raise ValueError("hybrid L2 requires exactly 50 repeats.")
    if active_prefix_bucket_size != FIXED_ACTIVE_PREFIX_BUCKET_SIZE:
        raise ValueError("hybrid L2 requires active-prefix bucket size 64.")
    _validate_fixed_runtime_contract(
        args,
        config_name=config_name,
        q1_bmm_cross_attention=q1_bmm_cross_attention,
        native_q1_self_attention=native_q1_self_attention,
        native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
    )
    _assert_supported_probe(args)
    if args.precision != "fp32":
        raise ValueError("hybrid L2 is FP32-only.")
    if args.inference_generation_compile:
        raise ValueError("hybrid L2 captures manual CUDA graphs and requires compile=false.")
    if not args.do_sample or float(args.top_p) != 0.9 or int(args.top_k) != 0:
        raise ValueError("hybrid L2 requires the reviewed sampling contract: sample, top_p=.9, top_k=0.")

    compile_args(args, verbose=False)
    setup_inference_environment(seeds[0])
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
        generation_compile=False,
    )
    model.eval()
    if model.device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("hybrid L2 requires a CUDA GPU allocation.")
    if model.dtype != torch.float32:
        raise ValueError(f"hybrid L2 requires FP32 model weights; got {model.dtype}.")

    model_inputs = _move_kwargs_for_model(
        model,
        _build_probe_inputs(args, model, tokenizer, sequence_index=sequence_index),
    )
    probe_metadata = {
        key: model_inputs.pop(key)
        for key in (
            "sequence_index",
            "frame_time_ms",
            "context_type",
            "lookback_time",
            "lookahead_time",
        )
    }
    if float(probe_metadata["lookback_time"]) != 0.0:
        raise ValueError("hybrid L2 processor family is pinned to seq9 lookback_time=0.")
    prompt = model_inputs["decoder_input_ids"]
    prompt_attention_mask = model_inputs["decoder_attention_mask"]
    frames = model_inputs["frames"]
    condition_kwargs = _condition_kwargs(model_inputs)
    active_prefix_length = _bucketed_prefix_length(
        int(prompt.shape[-1]) + 1,
        active_prefix_bucket_size,
        int(model.config.max_target_positions),
    )
    if active_prefix_length != FIXED_ACTIVE_PREFIX_LENGTH:
        raise ValueError(
            f"hybrid L2 expected active-prefix length 128; got {active_prefix_length}."
        )
    context_type = ContextType(probe_metadata["context_type"])
    eos_token_ids = [
        int(token_id)
        for token_id in get_eos_token_id(
            tokenizer,
            lookback_time=float(probe_metadata["lookback_time"]),
            lookahead_time=float(probe_metadata["lookahead_time"]),
            context_type=context_type,
        )
    ]
    condition_tensor_hashes = {
        key: tensor_sha256(value)
        for key, value in sorted(condition_kwargs.items())
        if isinstance(value, torch.Tensor)
    }
    workload_contract = normalized_one_token_workload_contract(
        batch_size=2,
        seeds=seeds,
        do_sample=bool(args.do_sample),
        prompt_sha256=tensor_sha256(prompt),
        prompt_attention_mask_sha256=tensor_sha256(prompt_attention_mask),
        frames_sha256=tensor_sha256(frames),
        condition_tensor_hashes=condition_tensor_hashes,
        active_prefix_prefill=False,
        active_prefix_decode=True,
        active_prefix_decode_length=active_prefix_length,
        runtime_contract={
            "config_name": config_name,
            "model_path": args.model_path,
            "precision": args.precision,
            "attn_implementation": args.attn_implementation,
            "global_seed": int(args.seed),
            "row_seeds": list(seeds),
            "do_sample": bool(args.do_sample),
            "top_p": float(args.top_p),
            "top_k_sampling": int(args.top_k),
            "temperature": float(args.temperature),
            "stateful_monotonic_logits_processor": bool(
                args.inference_stateful_monotonic_logits_processor
            ),
            "eos_token_ids": eos_token_ids,
            "probe": probe_metadata,
        },
    )
    if workload_contract != reviewed_l2_report.get("workload_contract"):
        raise ValueError(
            "hybrid L2 reconstructed workload differs from the reviewed seq9 L2 contract."
        )
    workload_contract_hash = _json_sha256(workload_contract)
    if workload_contract_hash != REVIEWED_WORKLOAD_CONTRACT_SHA256:
        raise ValueError("hybrid L2 workload contract hash differs from the pinned source.")
    base_factory, warper_factory, combined_factory = _make_factories(
        args,
        tokenizer,
        torch.device(model.device),
        lookback_time=float(probe_metadata["lookback_time"]),
    )

    native_context_started = time.perf_counter()
    with torch.autocast(device_type="cuda", enabled=False), generation_profile_context(
            sdpa_backend=args.profile_sdpa_backend,
            q1_bmm_cross_attention=q1_bmm_cross_attention,
            native_q1_self_attention=native_q1_self_attention,
            native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
    ):
        native_context_enter_seconds = time.perf_counter() - native_context_started
        references: list[_ReferenceRow] = [
            _build_reference_row(
                model,
                row=row,
                seed=seed,
                prompt=prompt,
                prompt_attention_mask=prompt_attention_mask,
                frames=frames,
                condition_kwargs=condition_kwargs,
                processor_factory=combined_factory,
                do_sample=True,
                active_prefix_length=active_prefix_length,
                timing_repeats=timing_repeats,
            )
            for row, seed in enumerate(seeds)
        ]
        lane_list: list[_CapturedLaneRow] = [
            _capture_lane_row(
                model,
                reference=reference,
                prompt=prompt,
                prompt_attention_mask=prompt_attention_mask,
                frames=frames,
                condition_kwargs=condition_kwargs,
                processor_factory=combined_factory,
                do_sample=True,
                active_prefix_length=active_prefix_length,
                capture_warmup_repeats=capture_warmup_repeats,
                atol=1e-4,
                rtol=1e-4,
                top_k=comparison_top_k,
            )
            for reference in references
        ]
        lanes = (lane_list[0], lane_list[1])
        bridge = _capture_shared_processor_bridge(
            lanes,
            base_processor_factory=base_factory,
            capture_warmup_repeats=capture_warmup_repeats,
        )
        sampling = tuple(
            _capture_private_sampling_tail(
                lanes[row],
                logits_warper_factory=warper_factory,
                seed=seeds[row],
                capture_warmup_repeats=capture_warmup_repeats,
            )
            for row in range(2)
        )
    resources = _HybridResources(
        lanes=lanes,
        source_logits=(lanes[0].lane.logits, lanes[1].lane.logits),
        bridge=bridge,
        sampling=(sampling[0], sampling[1]),
        events=_build_events(),
    )
    evidence = _resource_evidence(resources)
    validate_hybrid_resource_ownership(
        [lane.ownership for lane in resources.lanes],
        evidence,
    )
    resource_ownership_pass = True
    capture_pass = bool(
        len({repr(lane.lane.graph.pool()) for lane in resources.lanes}) == 2
        and len({repr(tail.graph.pool()) for tail in resources.sampling}) == 2
        and repr(resources.bridge.graph.pool())
        not in {
            *(repr(lane.lane.graph.pool()) for lane in resources.lanes),
            *(repr(tail.graph.pool()) for tail in resources.sampling),
        }
    )

    reference_tokens = {
        f"row-{row}": references[row].transcript_tokens
        for row in range(2)
    }
    reference_rng_hashes = {
        f"row-{row}": references[row].transcript_rng_hash
        for row in range(2)
    }
    order_reports: dict[str, Any] = {}
    order_tps: dict[str, float] = {}
    for order in reciprocal_lane_orders(2):
        key = _order_key(order)
        processed = _processed_score_gate(
            resources,
            order=order,
            base_processor_factory=base_factory,
            top_k=comparison_top_k,
        )
        transcripts, exact_rng_hashes = _run_exact_transcript(
            resources,
            order=order,
            seeds=seeds,
            repeats=timing_repeats,
        )
        timing, timed_rng_hashes = _run_complete_output_discard_timing(
            resources,
            order=order,
            seeds=seeds,
            repeats=timing_repeats,
        )
        transcript_match = transcripts == reference_tokens
        final_rng_match = exact_rng_hashes == reference_rng_hashes
        timed_rng_match = timed_rng_hashes == reference_rng_hashes
        exactness_pass = bool(
            processed["pass"]
            and transcript_match
            and final_rng_match
            and timed_rng_match
        )
        order_tps[key] = float(timing["wall_tokens_per_second"])
        order_reports[key] = {
            "graph_and_sampling_order": list(order),
            "processed_rows_bitwise_match": processed["bitwise_match"],
            "processed_rows_topk_match": processed["topk_match"],
            "processed_rows": processed["rows"],
            "transcript_match": transcript_match,
            "final_rng_state_match": final_rng_match,
            "timed_final_rng_state_match": timed_rng_match,
            "candidate_token_hashes": {
                request_id: _json_sha256(tokens)
                for request_id, tokens in transcripts.items()
            },
            "candidate_final_rng_state_hashes": exact_rng_hashes,
            "timed_final_rng_state_hashes": timed_rng_hashes,
            "complete_output_discard_timing": timing,
            "exactness_pass": exactness_pass,
        }

    performance = summarize_hybrid_l2_performance(order_tps)
    all_exact = all(report["exactness_pass"] for report in order_reports.values())
    promotion_pass = bool(
        all_exact
        and resource_ownership_pass
        and capture_pass
        and performance["five_percent_keep_bar_pass"]
    )
    graduation_pass = bool(promotion_pass and performance["target_500_bar_pass"])
    device_index = model.device.index if model.device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    report = {
        "schema_version": HYBRID_L2_SCHEMA_VERSION,
        "gate": HYBRID_L2_GATE,
        "lane_count": 2,
        "claim_scope": "fixed_shape_verifier_only_not_runtime_or_scheduler_throughput",
        "pass": graduation_pass,
        "exactness_pass": all_exact,
        "resource_ownership_pass": resource_ownership_pass,
        "capture_pass": capture_pass,
        "safe_event_order_pass": True,
        "promotion_gate_pass": promotion_pass,
        "graduation_to_next_shape_pass": graduation_pass,
        "base_processor_classes": list(processor_class_names(base_factory())),
        "private_warper_classes": list(processor_class_names(warper_factory())),
        "shared_processor_mode": "B2_forces_full_scan_even_when_stateful_batch1_is_configured",
        "event_order": list(HYBRID_EVENT_ORDER),
        "fixed_workload_contract": {
            "config_name": config_name,
            "sequence_index": sequence_index,
            "row_seeds": list(seeds),
            "timing_repeats": timing_repeats,
            "active_prefix_length": active_prefix_length,
            "active_prefix_bucket_size": active_prefix_bucket_size,
            "model_path": args.model_path,
            "device": args.device,
            "precision": args.precision,
            "attn_implementation": args.attn_implementation,
            "profile_sdpa_backend": args.profile_sdpa_backend,
            "inference_engine": args.inference_engine,
            "optimized_inference_mode": args.optimized_inference_mode,
            "do_sample": bool(args.do_sample),
            "temperature": float(args.temperature),
            "timeshift_bias": float(args.timeshift_bias),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "cfg_scale": float(args.cfg_scale),
            "num_beams": int(args.num_beams),
            "use_server": bool(args.use_server),
            "parallel": bool(args.parallel),
            "generation_compile": bool(args.inference_generation_compile),
            "active_prefix_decode_loop": bool(args.inference_active_prefix_decode_loop),
            "active_prefix_decode_cuda_graph": bool(
                args.inference_active_prefix_decode_cuda_graph
            ),
            "active_prefix_decode_cuda_graph_warmup": int(
                args.inference_active_prefix_decode_cuda_graph_warmup
            ),
            "active_prefix_decode_cuda_graph_min_decode_steps": int(
                args.inference_active_prefix_decode_cuda_graph_min_decode_steps
            ),
            "stateful_monotonic_logits_processor": bool(
                args.inference_stateful_monotonic_logits_processor
            ),
            "q1_bmm_cross_attention": q1_bmm_cross_attention,
            "native_q1_self_attention": native_q1_self_attention,
            "native_q1_rope_cache_self_attention": native_q1_rope_cache_self_attention,
            "decode_session_runtime": bool(args.inference_decode_session_runtime),
            "decode_session_cuda_graph": bool(args.inference_decode_session_cuda_graph),
            "decode_session_chunk_size": int(args.inference_decode_session_chunk_size),
            "native_decode_kernels": bool(args.inference_native_decode_kernels),
            "prompt_sha256": workload_contract["prompt_sha256"],
            "prompt_attention_mask_sha256": workload_contract[
                "prompt_attention_mask_sha256"
            ],
            "frames_sha256": workload_contract["frames_sha256"],
            "condition_tensor_hashes": condition_tensor_hashes,
            "probe": probe_metadata,
            "identical_base_prompt": True,
            "row_private_anchor_and_sampling_state": True,
        },
        "workload_contract": workload_contract,
        "workload_contract_sha256": workload_contract_hash,
        "reference_token_hashes": {
            request_id: _json_sha256(tokens)
            for request_id, tokens in reference_tokens.items()
        },
        "reference_final_rng_state_hashes": reference_rng_hashes,
        "orders": order_reports,
        "performance": performance,
        "resource_ownership": {
            "model_lanes": [lane.ownership.as_dict() for lane in resources.lanes],
            "hybrid_bridge": evidence.as_dict(),
        },
        "capture": {
            "model_graph_count": 2,
            "processor_graph_count": 1,
            "private_sampling_graph_count": 2,
            "total_graph_count": 5,
            "native_context_enter_seconds": native_context_enter_seconds,
            "processor_warmup_seconds": bridge.warmup_seconds,
            "processor_capture_seconds": bridge.capture_seconds,
            "sampling_warmup_seconds": [tail.warmup_seconds for tail in resources.sampling],
            "sampling_capture_seconds": [tail.capture_seconds for tail in resources.sampling],
            "explicit_generator_registration": True,
        },
        "reviewed_l2_source": {
            "report_path": str(reviewed_l2_report_path.resolve()),
            "report_sha256": REVIEWED_L2_REPORT_SHA256,
            "workload_contract_sha256": REVIEWED_WORKLOAD_CONTRACT_SHA256,
            "job_id": "49548733",
            "commit": reviewed_l2_report["runtime_metadata"]["git_commit"],
            "worst_order_tokens_per_second": reviewed_l2_report["performance"][
                "worst_order_tokens_per_second"
            ],
        },
        "stop_conditions": {
            "exactness_or_ownership_failure": "stop_family",
            "worst_order_at_or_below_five_percent_keep_bar": "reject_and_stop_family",
            "worst_order_above_keep_bar_but_at_or_below_500": (
                "retain_verifier_evidence_only; do_not_change_prefix_or_build_runtime"
            ),
            "worst_order_above_500": (
                "eligible_for_one separately reviewed changing-prefix exactness gate; "
                "still not runtime authorization"
            ),
        },
        "runtime_metadata": {
            "git_commit": _git_commit(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device_name": properties.name,
            "cuda_compute_capability": [properties.major, properties.minor],
            "cuda_total_memory_bytes": properties.total_memory,
            "model_path": args.model_path,
            "precision": args.precision,
            "attn_implementation": args.attn_implementation,
            "profile_sdpa_backend": args.profile_sdpa_backend,
            "probe": probe_metadata,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "torch_extensions_dir": os.environ.get("TORCH_EXTENSIONS_DIR"),
            "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "cuda_cache_path": os.environ.get("CUDA_CACHE_PATH"),
        },
    }
    validate_hybrid_l2_report(report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="profile_salvalai_smoke15")
    parser.add_argument("--sequence-index", type=int, default=FIXED_SEQUENCE_INDEX)
    parser.add_argument("--reviewed-l2-report", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--row-seed", action="append", type=int, default=[])
    parser.add_argument("--timing-repeats", type=int, default=FIXED_TIMING_REPEATS)
    parser.add_argument("--capture-warmup-repeats", type=int, default=1)
    parser.add_argument("--comparison-top-k", type=int, default=20)
    parser.add_argument(
        "--active-prefix-bucket-size",
        type=int,
        default=FIXED_ACTIVE_PREFIX_BUCKET_SIZE,
    )
    parser.add_argument("--q1-bmm-cross-attention", action="store_true")
    parser.add_argument("--native-q1-self-attention", action="store_true")
    parser.add_argument("--native-q1-rope-cache-self-attention", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides such as audio_path=/path/song.mp3")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    seeds = tuple(cli.row_seed) if cli.row_seed else FIXED_ROW_SEEDS
    if len(seeds) != 2:
        raise ValueError("hybrid L2 requires exactly two --row-seed values.")
    args = _load_args(cli.config_name, cli.overrides)
    total_started = time.perf_counter()
    reviewed = _load_json_object(cli.reviewed_l2_report)
    report = run_hybrid_l2_gate(
        args,
        config_name=cli.config_name,
        reviewed_l2_report=reviewed,
        reviewed_l2_report_path=cli.reviewed_l2_report,
        sequence_index=cli.sequence_index,
        seeds=(int(seeds[0]), int(seeds[1])),
        timing_repeats=cli.timing_repeats,
        capture_warmup_repeats=cli.capture_warmup_repeats,
        comparison_top_k=cli.comparison_top_k,
        active_prefix_bucket_size=cli.active_prefix_bucket_size,
        q1_bmm_cross_attention=cli.q1_bmm_cross_attention,
        native_q1_self_attention=cli.native_q1_self_attention,
        native_q1_rope_cache_self_attention=cli.native_q1_rope_cache_self_attention,
    )
    report["runtime_metadata"]["config_name"] = cli.config_name
    report["runtime_metadata"]["hydra_overrides"] = list(cli.overrides)
    report["total_wall_seconds"] = time.perf_counter() - total_started
    validate_hybrid_l2_report(report)
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
