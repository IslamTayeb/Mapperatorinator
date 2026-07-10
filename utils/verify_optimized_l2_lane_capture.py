"""Run the isolated reciprocal-order L=2 CUDA-graph lane physics gate.

This is verifier and fixed-shape measurement code only.  It creates two B1
sessions that share immutable model weights but own private streams, graphs,
graph pools, static buffers, encoder outputs, caches, generators, and logits
processors.  It does not implement a scheduler, server path, or L=3 runtime.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import transformers
from transformers import TopKLogitsWarper, TopPLogitsWarper

from inference import compile_args, load_model_with_server, setup_inference_environment
from osuT5.osuT5.inference.decode_loop import _bucketed_prefix_length
from osuT5.osuT5.inference.direct_decode import DecodeSession
from osuT5.osuT5.inference.optimized.batch.lane_one_token import (
    DEFAULT_SEPARATE_GRAPH_POOL,
    PYTORCH_PER_STREAM_CUBLAS_WORKSPACE,
    CapturedB1Lane,
    LaneResourceEvidence,
    capture_prepared_b1_lane,
    compare_lane_logits,
    normalized_one_token_request_ids,
    normalized_one_token_workload_contract,
    reciprocal_lane_orders,
    summarize_reciprocal_l2_performance,
    tensor_sha256,
    validate_l2_baseline_report,
    validate_lane_resource_ownership,
)
from osuT5.osuT5.inference.optimized.benchmark import (
    BatchPhysicsExecutionFamily,
    BatchPhysicsObservation,
    LANE_STATE_OWNERSHIP_CONTRACT,
    compare_batch_physics_observations,
)
from osuT5.osuT5.inference.optimized.exactness import ExactnessResultClass
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
from utils.verify_optimized_b1_lane_capture import (
    _cache_part_contract,
    _cache_position_is_overwritten,
    _cache_position_is_zero,
    _cache_tensor_ptrs,
    _compare_cache_parts,
    _json_sha256,
    _new_generator,
    _sample_next_token,
    _tensor_storage_ptr,
    _top_level_tensor_ptrs,
    _zero_cache_position,
)


REVIEWED_B1_COMPLETE_TPS = 414.9046973558729


@dataclass
class _ReferenceRow:
    row: int
    seed: int
    session: DecodeSession
    prefill_logits: torch.Tensor
    anchor_token: torch.LongTensor
    full_prefix: torch.LongTensor
    full_mask: torch.Tensor
    decode_logits: torch.Tensor
    next_token: torch.LongTensor
    one_step_rng_hash: str
    transcript_tokens: list[int]
    transcript_rng_hash: str


@dataclass
class _CapturedLaneRow:
    row: int
    seed: int
    session: DecodeSession
    lane: CapturedB1Lane
    full_prefix: torch.LongTensor
    full_mask: torch.Tensor
    parity_generator: torch.Generator
    parity_processor: Any
    ownership: LaneResourceEvidence
    parity_report: dict[str, Any]


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _load_previous_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("previous L1 gate report must contain a JSON object.")
    return payload


def _resolve_row_seeds(values: Sequence[int], default_seed: int) -> tuple[int, int]:
    if not values:
        return int(default_seed), int(default_seed) + 11111
    if len(values) != 2:
        raise ValueError("L2 requires exactly two --row-seed values when provided.")
    return int(values[0]), int(values[1])


def _processor_factory(args: Any, tokenizer: Any, device: torch.device, lookback_time: float):
    def factory():
        processors = _build_generation_logits_processors(
            args,
            tokenizer,
            device,
            lookback_time=lookback_time,
        )
        if args.do_sample:
            if args.top_k > 0:
                processors.append(TopKLogitsWarper(top_k=int(args.top_k)))
            if 0.0 < args.top_p < 1.0:
                processors.append(TopPLogitsWarper(top_p=float(args.top_p)))
        return processors

    return factory


@torch.no_grad()
def _build_reference_row(
        model: Any,
        *,
        row: int,
        seed: int,
        prompt: torch.LongTensor,
        prompt_attention_mask: torch.Tensor,
        frames: torch.Tensor,
        condition_kwargs: Mapping[str, Any],
        processor_factory: Any,
        do_sample: bool,
        active_prefix_length: int,
        timing_repeats: int,
) -> _ReferenceRow:
    processor = processor_factory()
    generator = _new_generator(torch.device(model.device), seed)
    session = DecodeSession.prefill(
        model,
        prompt=prompt,
        prompt_attention_mask=prompt_attention_mask,
        frames=frames,
        condition_kwargs=dict(condition_kwargs),
        active_prefix_self_attention=False,
    )
    prefill_logits = session.one_token_state().prefill_logits
    if prefill_logits is None:
        raise RuntimeError(f"reference row {row} did not retain prefill logits.")
    anchor = _sample_next_token(
        input_ids=prompt,
        raw_logits=prefill_logits,
        logits_processor=processor,
        generator=generator,
        do_sample=do_sample,
    )
    full_prefix = torch.cat([prompt, anchor[:, None]], dim=-1)
    full_mask = torch.cat(
        [
            prompt_attention_mask,
            torch.ones_like(anchor[:, None], dtype=prompt_attention_mask.dtype),
        ],
        dim=-1,
    )
    decode = session.decode_one_token_raw_logits(
        full_prefix=full_prefix,
        full_attention_mask=full_mask,
        active_prefix_self_attention=True,
        active_prefix_self_attention_length=active_prefix_length,
    )
    next_token = _sample_next_token(
        input_ids=full_prefix,
        raw_logits=decode.logits,
        logits_processor=processor,
        generator=generator,
        do_sample=do_sample,
    )
    one_step_rng_hash = tensor_sha256(generator.get_state())

    transcript_processor = processor_factory()
    transcript_generator = _new_generator(torch.device(model.device), seed)
    transcript_tensors: list[torch.Tensor] = []
    for _ in range(timing_repeats):
        result = session.decode_one_token_raw_logits(
            full_prefix=full_prefix,
            full_attention_mask=full_mask,
            active_prefix_self_attention=True,
            active_prefix_self_attention_length=active_prefix_length,
        )
        transcript_tensors.append(_sample_next_token(
            input_ids=full_prefix,
            raw_logits=result.logits,
            logits_processor=transcript_processor,
            generator=transcript_generator,
            do_sample=do_sample,
        ))
    torch.cuda.synchronize(model.device)
    transcript_tokens = [int(token.detach().cpu().item()) for token in transcript_tensors]
    transcript_rng_hash = tensor_sha256(transcript_generator.get_state())
    return _ReferenceRow(
        row=row,
        seed=seed,
        session=session,
        prefill_logits=prefill_logits.detach().to("cpu", dtype=torch.float32),
        anchor_token=anchor.detach().to("cpu"),
        full_prefix=full_prefix,
        full_mask=full_mask,
        decode_logits=decode.logits.detach().to("cpu", dtype=torch.float32),
        next_token=next_token.detach().to("cpu"),
        one_step_rng_hash=one_step_rng_hash,
        transcript_tokens=transcript_tokens,
        transcript_rng_hash=transcript_rng_hash,
    )


@torch.no_grad()
def _capture_lane_row(
        model: Any,
        *,
        reference: _ReferenceRow,
        prompt: torch.LongTensor,
        prompt_attention_mask: torch.Tensor,
        frames: torch.Tensor,
        condition_kwargs: Mapping[str, Any],
        processor_factory: Any,
        do_sample: bool,
        active_prefix_length: int,
        capture_warmup_repeats: int,
        atol: float,
        rtol: float,
        top_k: int,
) -> _CapturedLaneRow:
    processor = processor_factory()
    generator = _new_generator(torch.device(model.device), reference.seed)
    session = DecodeSession.prefill(
        model,
        prompt=prompt,
        prompt_attention_mask=prompt_attention_mask,
        frames=frames,
        condition_kwargs=dict(condition_kwargs),
        active_prefix_self_attention=False,
    )
    prefill_logits = session.one_token_state().prefill_logits
    if prefill_logits is None:
        raise RuntimeError(f"lane row {reference.row} did not retain prefill logits.")
    anchor = _sample_next_token(
        input_ids=prompt,
        raw_logits=prefill_logits,
        logits_processor=processor,
        generator=generator,
        do_sample=do_sample,
    )
    full_prefix = torch.cat([prompt, anchor[:, None]], dim=-1)
    full_mask = torch.cat(
        [
            prompt_attention_mask,
            torch.ones_like(anchor[:, None], dtype=prompt_attention_mask.dtype),
        ],
        dim=-1,
    )
    setup = session.decode_one_token_raw_logits(
        full_prefix=full_prefix,
        full_attention_mask=full_mask,
        active_prefix_self_attention=True,
        active_prefix_self_attention_length=active_prefix_length,
    )
    lane = capture_prepared_b1_lane(
        model,
        setup.prepared_inputs,
        active_prefix_length=active_prefix_length,
        capture_warmup_repeats=capture_warmup_repeats,
    )
    cache = session.cache_state.cache
    encoder_outputs = session.encoder_state.encoder_outputs
    reference_cache = reference.session.cache_state.cache
    if cache is None or reference_cache is None or encoder_outputs is None:
        raise RuntimeError(f"row {reference.row} lost cache or encoder ownership.")
    cache_position = int(setup.cache_position[-1].detach().cpu().item())
    with torch.cuda.stream(lane.stream):
        _zero_cache_position(cache.self_attention_cache, cache_position)
    lane.stream.synchronize()
    sentinel_zeroed = _cache_position_is_zero(cache.self_attention_cache, cache_position)
    lane.replay()
    lane.stream.synchronize()
    sentinel_overwritten = _cache_position_is_overwritten(
        cache.self_attention_cache,
        cache_position,
    )
    graph_logits = lane.logits
    with torch.cuda.stream(lane.stream):
        next_token = _sample_next_token(
            input_ids=full_prefix,
            raw_logits=graph_logits,
            logits_processor=processor,
            generator=generator,
            do_sample=do_sample,
        )
    lane.stream.synchronize()
    rng_hash = tensor_sha256(generator.get_state())

    prefill_comparison = compare_lane_logits(
        reference.prefill_logits,
        prefill_logits,
        atol=atol,
        rtol=rtol,
        top_k=top_k,
    )
    setup_comparison = compare_lane_logits(
        reference.decode_logits,
        setup.logits,
        atol=atol,
        rtol=rtol,
        top_k=top_k,
    )
    replay_comparison = compare_lane_logits(
        reference.decode_logits,
        graph_logits,
        atol=atol,
        rtol=rtol,
        top_k=top_k,
    )
    self_cache = _compare_cache_parts(
        reference_cache.self_attention_cache,
        cache.self_attention_cache,
        atol=atol,
        rtol=rtol,
    )
    cross_cache = _compare_cache_parts(
        reference_cache.cross_attention_cache,
        cache.cross_attention_cache,
        atol=atol,
        rtol=rtol,
    )
    cache_position_match = bool(torch.equal(setup.cache_position, torch.arange(
        int(prompt.shape[-1]),
        int(prompt.shape[-1]) + 1,
        device=setup.cache_position.device,
    )))
    cache_contract_match = (
        _cache_part_contract(reference_cache.self_attention_cache)
        == _cache_part_contract(cache.self_attention_cache)
        and _cache_part_contract(reference_cache.cross_attention_cache)
        == _cache_part_contract(cache.cross_attention_cache)
    )
    stream_id = int(lane.stream.cuda_stream)
    ownership = LaneResourceEvidence(
        lane_id=reference.row,
        shared_model_id=id(model),
        stream_id=stream_id,
        graph_id=id(lane.graph),
        graph_pool_id=repr(lane.graph.pool()),
        graph_pool_policy=DEFAULT_SEPARATE_GRAPH_POOL,
        graph_pool_sharing_requested=False,
        session_id=id(session),
        cache_id=id(cache),
        encoder_output_storage_ptr=_tensor_storage_ptr(encoder_outputs.last_hidden_state),
        generator_id=id(generator),
        logits_processor_id=id(processor),
        static_input_storage_ptrs=_top_level_tensor_ptrs(lane.static_inputs),
        self_cache_storage_ptrs=_cache_tensor_ptrs(cache.self_attention_cache),
        cross_cache_storage_ptrs=_cache_tensor_ptrs(cache.cross_attention_cache),
        cublas_workspace_policy=PYTORCH_PER_STREAM_CUBLAS_WORKSPACE,
        cublas_workspace_owner_stream_id=stream_id,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )
    parity_pass = bool(
        prefill_comparison["allclose"]
        and prefill_comparison["topk_match"]
        and setup_comparison["allclose"]
        and setup_comparison["topk_match"]
        and replay_comparison["allclose"]
        and replay_comparison["topk_match"]
        and torch.equal(reference.anchor_token, anchor.detach().cpu())
        and torch.equal(reference.next_token, next_token.detach().cpu())
        and reference.one_step_rng_hash == rng_hash
        and sentinel_zeroed
        and sentinel_overwritten
        and cache_position_match
        and cache_contract_match
        and self_cache["pass"]
        and cross_cache["pass"]
    )
    return _CapturedLaneRow(
        row=reference.row,
        seed=reference.seed,
        session=session,
        lane=lane,
        full_prefix=full_prefix,
        full_mask=full_mask,
        parity_generator=generator,
        parity_processor=processor,
        ownership=ownership,
        parity_report={
            "pass": parity_pass,
            "row": reference.row,
            "seed": reference.seed,
            "anchor_token_match": bool(torch.equal(reference.anchor_token, anchor.detach().cpu())),
            "sampled_token_match": bool(torch.equal(reference.next_token, next_token.detach().cpu())),
            "rng_state_match": reference.one_step_rng_hash == rng_hash,
            "prefill_logits": prefill_comparison,
            "setup_logits": setup_comparison,
            "graph_logits": replay_comparison,
            "cache_position": cache_position,
            "cache_position_match": cache_position_match,
            "cache_contract_match": cache_contract_match,
            "cache_sentinel_zeroed": sentinel_zeroed,
            "cache_sentinel_overwritten": sentinel_overwritten,
            "self_cache": self_cache,
            "cross_cache": cross_cache,
            "graph_pool_id": ownership.graph_pool_id,
            "stream_id": stream_id,
        },
    )


def _order_key(order: Sequence[int]) -> str:
    return ",".join(str(row) for row in order)


def _synchronize_lanes(lanes: Sequence[_CapturedLaneRow]) -> None:
    for item in lanes:
        item.lane.stream.synchronize()


@torch.no_grad()
def _run_exact_graph_transcript(
        lanes: Sequence[_CapturedLaneRow],
        *,
        order: Sequence[int],
        processor_factory: Any,
        do_sample: bool,
        timing_repeats: int,
) -> tuple[dict[str, list[int]], dict[str, str]]:
    processors = [processor_factory() for _ in lanes]
    generators = [_new_generator(torch.device(lanes[0].full_prefix.device), lane.seed) for lane in lanes]
    token_tensors: list[list[torch.Tensor]] = [[] for _ in lanes]
    for _ in range(timing_repeats):
        for row in order:
            item = lanes[row]
            with torch.cuda.stream(item.lane.stream):
                item.lane.graph.replay()
                token_tensors[row].append(_sample_next_token(
                    input_ids=item.full_prefix,
                    raw_logits=item.lane.logits,
                    logits_processor=processors[row],
                    generator=generators[row],
                    do_sample=do_sample,
                ))
    for item in lanes:
        item.lane.replay_count += timing_repeats
    _synchronize_lanes(lanes)
    request_ids = normalized_one_token_request_ids(len(lanes))
    tokens = {
        request_ids[row]: [int(token.detach().cpu().item()) for token in token_tensors[row]]
        for row in range(len(lanes))
    }
    rng_hashes = {
        request_ids[row]: tensor_sha256(generators[row].get_state())
        for row in range(len(lanes))
    }
    return tokens, rng_hashes


@torch.no_grad()
def _run_serial_exact_graph_transcript(
        lanes: Sequence[_CapturedLaneRow],
        *,
        processor_factory: Any,
        do_sample: bool,
        timing_repeats: int,
) -> tuple[dict[str, list[int]], dict[str, str]]:
    """Observe the serial graph transcript used by the same-work baseline."""

    processors = [processor_factory() for _ in lanes]
    generators = [_new_generator(torch.device(lanes[0].full_prefix.device), lane.seed) for lane in lanes]
    token_tensors: list[list[torch.Tensor]] = [[] for _ in lanes]
    for row, item in enumerate(lanes):
        with torch.cuda.stream(item.lane.stream):
            for _ in range(timing_repeats):
                item.lane.graph.replay()
                token_tensors[row].append(_sample_next_token(
                    input_ids=item.full_prefix,
                    raw_logits=item.lane.logits,
                    logits_processor=processors[row],
                    generator=generators[row],
                    do_sample=do_sample,
                ))
        item.lane.replay_count += timing_repeats
        item.lane.stream.synchronize()
    request_ids = normalized_one_token_request_ids(len(lanes))
    tokens = {
        request_ids[row]: [int(token.detach().cpu().item()) for token in token_tensors[row]]
        for row in range(len(lanes))
    }
    rng_hashes = {
        request_ids[row]: tensor_sha256(generators[row].get_state())
        for row in range(len(lanes))
    }
    return tokens, rng_hashes


@torch.no_grad()
def _time_serial_model_only(
        lanes: Sequence[_CapturedLaneRow],
        *,
        timing_repeats: int,
) -> dict[str, float]:
    wall_started = time.perf_counter()
    cuda_seconds = 0.0
    for item in lanes:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(item.lane.stream):
            start.record(item.lane.stream)
            for _ in range(timing_repeats):
                item.lane.graph.replay()
            end.record(item.lane.stream)
        item.lane.replay_count += timing_repeats
        end.synchronize()
        cuda_seconds += start.elapsed_time(end) / 1000.0
    wall_seconds = time.perf_counter() - wall_started
    tokens = len(lanes) * timing_repeats
    return {
        "wall_seconds": wall_seconds,
        "cuda_seconds": cuda_seconds,
        "wall_tokens_per_second": tokens / wall_seconds,
        "cuda_tokens_per_second": tokens / cuda_seconds,
    }


@torch.no_grad()
def _time_serial_complete(
        lanes: Sequence[_CapturedLaneRow],
        *,
        processor_factory: Any,
        do_sample: bool,
        timing_repeats: int,
) -> tuple[dict[str, float], dict[str, str]]:
    processors = [processor_factory() for _ in lanes]
    generators = [_new_generator(torch.device(lanes[0].full_prefix.device), lane.seed) for lane in lanes]
    torch.cuda.reset_peak_memory_stats(lanes[0].full_prefix.device)
    memory_before_allocated = torch.cuda.memory_allocated(lanes[0].full_prefix.device)
    memory_before_reserved = torch.cuda.memory_reserved(lanes[0].full_prefix.device)
    wall_started = time.perf_counter()
    cuda_seconds = 0.0
    for row, item in enumerate(lanes):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(item.lane.stream):
            start.record(item.lane.stream)
            for _ in range(timing_repeats):
                item.lane.graph.replay()
                _sample_next_token(
                    input_ids=item.full_prefix,
                    raw_logits=item.lane.logits,
                    logits_processor=processors[row],
                    generator=generators[row],
                    do_sample=do_sample,
                )
            end.record(item.lane.stream)
        item.lane.replay_count += timing_repeats
        end.synchronize()
        cuda_seconds += start.elapsed_time(end) / 1000.0
    wall_seconds = time.perf_counter() - wall_started
    tokens = len(lanes) * timing_repeats
    timing = {
        "wall_seconds": wall_seconds,
        "cuda_seconds": cuda_seconds,
        "wall_tokens_per_second": tokens / wall_seconds,
        "cuda_tokens_per_second": tokens / cuda_seconds,
        "memory_before_allocated_bytes": memory_before_allocated,
        "memory_before_reserved_bytes": memory_before_reserved,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(lanes[0].full_prefix.device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(lanes[0].full_prefix.device),
    }
    request_ids = normalized_one_token_request_ids(len(lanes))
    rng_hashes = {
        request_ids[row]: tensor_sha256(generators[row].get_state())
        for row in range(len(lanes))
    }
    return timing, rng_hashes


def _concurrent_interval(
        lanes: Sequence[_CapturedLaneRow],
        *,
        order: Sequence[int],
        timing_repeats: int,
        step: Any,
) -> tuple[float, float]:
    device = lanes[0].full_prefix.device
    coordinator = torch.cuda.current_stream(device)
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    lane_ends = [torch.cuda.Event(enable_timing=True) for _ in lanes]
    wall_started = time.perf_counter()
    start.record(coordinator)
    for item in lanes:
        item.lane.stream.wait_event(start)
    for _ in range(timing_repeats):
        for row in order:
            step(row)
    for row, item in enumerate(lanes):
        with torch.cuda.stream(item.lane.stream):
            lane_ends[row].record(item.lane.stream)
        coordinator.wait_event(lane_ends[row])
    finish.record(coordinator)
    finish.synchronize()
    wall_seconds = time.perf_counter() - wall_started
    cuda_seconds = start.elapsed_time(finish) / 1000.0
    for item in lanes:
        item.lane.replay_count += timing_repeats
    return wall_seconds, cuda_seconds


@torch.no_grad()
def _time_concurrent_model_only(
        lanes: Sequence[_CapturedLaneRow],
        *,
        order: Sequence[int],
        timing_repeats: int,
) -> dict[str, float]:
    def step(row: int) -> None:
        item = lanes[row]
        with torch.cuda.stream(item.lane.stream):
            item.lane.graph.replay()

    wall_seconds, cuda_seconds = _concurrent_interval(
        lanes,
        order=order,
        timing_repeats=timing_repeats,
        step=step,
    )
    tokens = len(lanes) * timing_repeats
    return {
        "wall_seconds": wall_seconds,
        "cuda_seconds": cuda_seconds,
        "wall_tokens_per_second": tokens / wall_seconds,
        "cuda_tokens_per_second": tokens / cuda_seconds,
    }


@torch.no_grad()
def _time_concurrent_complete(
        lanes: Sequence[_CapturedLaneRow],
        *,
        order: Sequence[int],
        processor_factory: Any,
        do_sample: bool,
        timing_repeats: int,
) -> tuple[dict[str, float], dict[str, str]]:
    processors = [processor_factory() for _ in lanes]
    generators = [_new_generator(torch.device(lanes[0].full_prefix.device), lane.seed) for lane in lanes]
    device = lanes[0].full_prefix.device
    torch.cuda.reset_peak_memory_stats(device)
    memory_before_allocated = torch.cuda.memory_allocated(device)
    memory_before_reserved = torch.cuda.memory_reserved(device)

    def step(row: int) -> None:
        item = lanes[row]
        with torch.cuda.stream(item.lane.stream):
            item.lane.graph.replay()
            _sample_next_token(
                input_ids=item.full_prefix,
                raw_logits=item.lane.logits,
                logits_processor=processors[row],
                generator=generators[row],
                do_sample=do_sample,
            )

    wall_seconds, cuda_seconds = _concurrent_interval(
        lanes,
        order=order,
        timing_repeats=timing_repeats,
        step=step,
    )
    tokens = len(lanes) * timing_repeats
    timing = {
        "wall_seconds": wall_seconds,
        "cuda_seconds": cuda_seconds,
        "wall_tokens_per_second": tokens / wall_seconds,
        "cuda_tokens_per_second": tokens / cuda_seconds,
        "memory_before_allocated_bytes": memory_before_allocated,
        "memory_before_reserved_bytes": memory_before_reserved,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    request_ids = normalized_one_token_request_ids(len(lanes))
    rng_hashes = {
        request_ids[row]: tensor_sha256(generators[row].get_state())
        for row in range(len(lanes))
    }
    return timing, rng_hashes


def _observation(
        *,
        parallelism: int,
        seeds: Sequence[int],
        timing_repeats: int,
        workload_contract_hash: str,
        token_hashes: Mapping[str, str],
        rng_hashes: Mapping[str, str],
        model_timing: Mapping[str, float],
        complete_timing: Mapping[str, float],
) -> BatchPhysicsObservation:
    request_ids = normalized_one_token_request_ids(len(seeds))
    return BatchPhysicsObservation(
        execution_family=BatchPhysicsExecutionFamily.B1_LANE_POOL,
        parallelism=parallelism,
        state_ownership_contract=LANE_STATE_OWNERSHIP_CONTRACT,
        workload_contract_hash=workload_contract_hash,
        result_class=ExactnessResultClass.EXACT_OUTPUT,
        seeds={request_ids[row]: seed for row, seed in enumerate(seeds)},
        generated_tokens=len(seeds) * timing_repeats,
        scheduler_wall_seconds=float(complete_timing["wall_seconds"]),
        model_seconds=float(model_timing["cuda_seconds"]),
        cuda_seconds=float(complete_timing["cuda_seconds"]),
        peak_memory_bytes=int(complete_timing["peak_allocated_bytes"]),
        graph_capture_count=len(seeds),
        graph_replay_count=len(seeds) * timing_repeats,
        active_batch_size_histogram={
            parallelism: len(seeds) * timing_repeats // parallelism
        },
        token_hashes=dict(token_hashes),
        final_rng_state_hashes=dict(rng_hashes),
        stop_reasons={request_id: "fixed_shape_repeats_complete" for request_id in request_ids},
    )


@torch.no_grad()
def run_l2_gate(
        args: Any,
        *,
        config_name: str,
        previous_report: Mapping[str, Any],
        expected_b1_complete_tps: float,
        sequence_index: int,
        seeds: tuple[int, int],
        atol: float,
        rtol: float,
        comparison_top_k: int,
        capture_warmup_repeats: int,
        timing_repeats: int,
        active_prefix_bucket_size: int,
        q1_bmm_cross_attention: bool,
        native_q1_self_attention: bool,
        native_q1_rope_cache_self_attention: bool,
) -> dict[str, Any]:
    validate_l2_baseline_report(
        previous_report,
        expected_complete_tokens_per_second=expected_b1_complete_tps,
    )
    _assert_supported_probe(args)
    if args.precision != "fp32":
        raise ValueError("the L2 lane gate is FP32-only.")
    if args.inference_generation_compile:
        raise ValueError("the L2 manual graph gate requires inference_generation_compile=false.")
    if native_q1_rope_cache_self_attention and not native_q1_self_attention:
        raise ValueError("RoPE/cache native self-attention requires native q1 self-attention.")
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
        raise RuntimeError("the L2 lane gate requires a CUDA GPU allocation.")
    if model.dtype != torch.float32:
        raise ValueError(f"the L2 lane gate requires FP32 model weights; got {model.dtype}.")

    model_inputs = _move_kwargs_for_model(
        model,
        _build_probe_inputs(args, model, tokenizer, sequence_index=sequence_index),
    )
    probe_metadata = {
        key: model_inputs.pop(key)
        for key in ("sequence_index", "frame_time_ms", "context_type", "lookback_time", "lookahead_time")
    }
    prompt = model_inputs["decoder_input_ids"]
    prompt_attention_mask = model_inputs["decoder_attention_mask"]
    frames = model_inputs["frames"]
    condition_kwargs = _condition_kwargs(model_inputs)
    active_prefix_length = _bucketed_prefix_length(
        int(prompt.shape[-1]) + 1,
        int(active_prefix_bucket_size),
        int(model.config.max_target_positions),
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
    processor_factory = _processor_factory(
        args,
        tokenizer,
        torch.device(model.device),
        float(probe_metadata["lookback_time"]),
    )

    native_context_started = time.perf_counter()
    with torch.autocast(device_type="cuda", enabled=False), generation_profile_context(
            sdpa_backend=args.profile_sdpa_backend,
            q1_bmm_cross_attention=q1_bmm_cross_attention,
            native_q1_self_attention=native_q1_self_attention,
            native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
    ):
        native_context_enter_seconds = time.perf_counter() - native_context_started
        references = [
            _build_reference_row(
                model,
                row=row,
                seed=seed,
                prompt=prompt,
                prompt_attention_mask=prompt_attention_mask,
                frames=frames,
                condition_kwargs=condition_kwargs,
                processor_factory=processor_factory,
                do_sample=bool(args.do_sample),
                active_prefix_length=active_prefix_length,
                timing_repeats=timing_repeats,
            )
            for row, seed in enumerate(seeds)
        ]
        lanes = [
            _capture_lane_row(
                model,
                reference=reference,
                prompt=prompt,
                prompt_attention_mask=prompt_attention_mask,
                frames=frames,
                condition_kwargs=condition_kwargs,
                processor_factory=processor_factory,
                do_sample=bool(args.do_sample),
                active_prefix_length=active_prefix_length,
                capture_warmup_repeats=capture_warmup_repeats,
                atol=atol,
                rtol=rtol,
                top_k=comparison_top_k,
            )
            for reference in references
        ]

    validate_lane_resource_ownership(
        [lane.ownership for lane in lanes],
        expected_lane_count=2,
    )
    reference_cache_pointers = [
        set(_cache_tensor_ptrs(reference.session.cache_state.cache.self_attention_cache))
        | set(_cache_tensor_ptrs(reference.session.cache_state.cache.cross_attention_cache))
        for reference in references
    ]
    lane_cache_pointers = [
        set(lane.ownership.self_cache_storage_ptrs)
        | set(lane.ownership.cross_cache_storage_ptrs)
        for lane in lanes
    ]
    reference_storage_disjoint = not bool(
        reference_cache_pointers[0] & reference_cache_pointers[1]
    )
    reference_lane_storage_disjoint = all(
        not bool(reference_cache_pointers[reference_row] & lane_cache_pointers[lane_row])
        for reference_row in range(2)
        for lane_row in range(2)
    )
    resource_ownership_pass = (
        reference_storage_disjoint
        and reference_lane_storage_disjoint
    )
    parity_pass = all(lane.parity_report["pass"] for lane in lanes)

    request_ids = normalized_one_token_request_ids(2)
    reference_tokens = {
        request_ids[row]: references[row].transcript_tokens
        for row in range(2)
    }
    reference_rng_hashes = {
        request_ids[row]: references[row].transcript_rng_hash
        for row in range(2)
    }
    token_hashes = {
        request_id: _json_sha256(tokens)
        for request_id, tokens in reference_tokens.items()
    }
    workload_contract = normalized_one_token_workload_contract(
        batch_size=2,
        seeds=seeds,
        do_sample=bool(args.do_sample),
        prompt_sha256=tensor_sha256(prompt),
        prompt_attention_mask_sha256=tensor_sha256(prompt_attention_mask),
        frames_sha256=tensor_sha256(frames),
        condition_tensor_hashes={
            key: tensor_sha256(value)
            for key, value in sorted(condition_kwargs.items())
            if isinstance(value, torch.Tensor)
        },
        active_prefix_prefill=False,
        active_prefix_decode=True,
        active_prefix_decode_length=active_prefix_length,
        runtime_contract={
            "config_name": config_name,
            "model_path": args.model_path,
            "precision": args.precision,
            "attn_implementation": args.attn_implementation,
            "global_seed": seeds[0],
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
    workload_contract_hash = _json_sha256(workload_contract)

    serial_exact_tokens, serial_exact_rng_hashes = _run_serial_exact_graph_transcript(
        lanes,
        processor_factory=processor_factory,
        do_sample=bool(args.do_sample),
        timing_repeats=timing_repeats,
    )
    serial_token_hashes = {
        request_id: _json_sha256(tokens)
        for request_id, tokens in serial_exact_tokens.items()
    }
    serial_model = _time_serial_model_only(lanes, timing_repeats=timing_repeats)
    serial_complete, serial_rng_hashes = _time_serial_complete(
        lanes,
        processor_factory=processor_factory,
        do_sample=bool(args.do_sample),
        timing_repeats=timing_repeats,
    )
    serial_exactness_pass = bool(
        serial_exact_tokens == reference_tokens
        and serial_exact_rng_hashes == reference_rng_hashes
        and serial_rng_hashes == reference_rng_hashes
    )
    serial_observation = _observation(
        parallelism=1,
        seeds=seeds,
        timing_repeats=timing_repeats,
        workload_contract_hash=workload_contract_hash,
        token_hashes=serial_token_hashes,
        rng_hashes=serial_rng_hashes,
        model_timing=serial_model,
        complete_timing=serial_complete,
    )

    order_reports: dict[str, Any] = {}
    order_tps: dict[str, float] = {}
    all_order_exact = True
    all_same_job_compare = True
    for order in reciprocal_lane_orders(2):
        order_key = _order_key(order)
        exact_tokens, exact_rng_hashes = _run_exact_graph_transcript(
            lanes,
            order=order,
            processor_factory=processor_factory,
            do_sample=bool(args.do_sample),
            timing_repeats=timing_repeats,
        )
        transcript_match = exact_tokens == reference_tokens
        transcript_rng_match = exact_rng_hashes == reference_rng_hashes
        model_timing = _time_concurrent_model_only(
            lanes,
            order=order,
            timing_repeats=timing_repeats,
        )
        complete_timing, timed_rng_hashes = _time_concurrent_complete(
            lanes,
            order=order,
            processor_factory=processor_factory,
            do_sample=bool(args.do_sample),
            timing_repeats=timing_repeats,
        )
        timed_rng_match = timed_rng_hashes == reference_rng_hashes
        order_exact = transcript_match and transcript_rng_match and timed_rng_match
        all_order_exact = all_order_exact and order_exact
        candidate_observation = _observation(
            parallelism=2,
            seeds=seeds,
            timing_repeats=timing_repeats,
            workload_contract_hash=workload_contract_hash,
            token_hashes={
                request_id: _json_sha256(tokens)
                for request_id, tokens in exact_tokens.items()
            },
            rng_hashes=timed_rng_hashes,
            model_timing=model_timing,
            complete_timing=complete_timing,
        )
        compare_result = None
        compare_error = None
        if order_exact and serial_exactness_pass:
            try:
                compare_result = compare_batch_physics_observations(
                    serial_observation,
                    candidate_observation,
                )
            except ValueError as exc:
                compare_error = str(exc)
                all_same_job_compare = False
        else:
            all_same_job_compare = False
        order_tps[order_key] = float(complete_timing["wall_tokens_per_second"])
        order_reports[order_key] = {
            "order": list(order),
            "exactness_pass": order_exact,
            "transcript_match": transcript_match,
            "transcript_rng_match": transcript_rng_match,
            "timed_rng_match": timed_rng_match,
            "token_hashes": {
                request_id: _json_sha256(tokens)
                for request_id, tokens in exact_tokens.items()
            },
            "exact_rng_hashes": exact_rng_hashes,
            "timed_rng_hashes": timed_rng_hashes,
            "model_timing": model_timing,
            "complete_timing": complete_timing,
            "observation": candidate_observation.as_dict(),
            "same_job_serial_compare": compare_result,
            "same_job_serial_compare_error": compare_error,
        }

    performance = summarize_reciprocal_l2_performance(
        b1_tokens_per_second=expected_b1_complete_tps,
        order_tokens_per_second=order_tps,
    )
    performance_gate_pass = bool(performance["clears_five_percent_b1_keep_bar"])
    exactness_pass = bool(parity_pass and serial_exactness_pass and all_order_exact)
    capture_pass = bool(
        all(lane.lane.replay_count > 0 for lane in lanes)
        and len({lane.ownership.graph_pool_id for lane in lanes}) == 2
    )
    pass_gate = bool(
        exactness_pass
        and resource_ownership_pass
        and capture_pass
        and all_same_job_compare
        and performance_gate_pass
    )

    device_index = model.device.index if model.device.index is not None else torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(device_index)
    return {
        "pass": pass_gate,
        "lane_count": 2,
        "gate": "reciprocal_two_lane_identical_prompt_fixed_shape",
        "claim_scope": "verifier_and_fixed_shape_scout_not_runtime_throughput",
        "exactness_pass": exactness_pass,
        "resource_ownership_pass": resource_ownership_pass,
        "capture_pass": capture_pass,
        "performance_gate_pass": performance_gate_pass,
        "same_job_serial_compare_pass": all_same_job_compare,
        "reviewed_b1_complete_tokens_per_second": expected_b1_complete_tps,
        "required_five_percent_tokens_per_second": expected_b1_complete_tps * 1.05,
        "performance": performance,
        "parity_rows": [lane.parity_report for lane in lanes],
        "resource_ownership": [lane.ownership.as_dict() for lane in lanes],
        "reference_storage_disjoint": reference_storage_disjoint,
        "reference_lane_storage_disjoint": reference_lane_storage_disjoint,
        "serial_baseline": {
            "exactness_pass": serial_exactness_pass,
            "transcript_match": serial_exact_tokens == reference_tokens,
            "transcript_rng_match": serial_exact_rng_hashes == reference_rng_hashes,
            "timed_rng_match": serial_rng_hashes == reference_rng_hashes,
            "token_hashes": serial_token_hashes,
            "exact_rng_hashes": serial_exact_rng_hashes,
            "model_timing": serial_model,
            "complete_timing": serial_complete,
            "timed_rng_hashes": serial_rng_hashes,
            "observation": serial_observation.as_dict(),
        },
        "orders": order_reports,
        "workload_contract": workload_contract,
        "reference_token_hashes": token_hashes,
        "reference_rng_hashes": reference_rng_hashes,
        "graph": {
            "native_context_enter_seconds": native_context_enter_seconds,
            "capture_count": 2,
            "graph_pool_ids": [lane.ownership.graph_pool_id for lane in lanes],
            "stream_ids": [lane.ownership.stream_id for lane in lanes],
            "lane_warmup_seconds": [lane.lane.warmup_seconds for lane in lanes],
            "lane_capture_seconds": [lane.lane.capture_seconds for lane in lanes],
            "lane_total_replay_counts": [lane.lane.replay_count for lane in lanes],
            "pool_sharing_requested": False,
        },
        "runtime_metadata": {
            "git_commit": _git_commit(),
            "model_path": args.model_path,
            "precision": args.precision,
            "attn_implementation": args.attn_implementation,
            "profile_sdpa_backend": args.profile_sdpa_backend,
            "generation_compile_requested": bool(args.inference_generation_compile),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device_name": device_properties.name,
            "cuda_compute_capability": [device_properties.major, device_properties.minor],
            "cuda_total_memory_bytes": device_properties.total_memory,
            "seeds": list(seeds),
            "do_sample": bool(args.do_sample),
            "top_p": float(args.top_p),
            "top_k_sampling": int(args.top_k),
            "active_prefix_bucket_size": active_prefix_bucket_size,
            "active_prefix_length": active_prefix_length,
            "q1_bmm_cross_attention": q1_bmm_cross_attention,
            "native_q1_self_attention": native_q1_self_attention,
            "native_q1_rope_cache_self_attention": native_q1_rope_cache_self_attention,
            "probe": probe_metadata,
            "eos_token_ids": eos_token_ids,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cublas_workspace_policy": PYTORCH_PER_STREAM_CUBLAS_WORKSPACE,
            "torch_extensions_dir": os.environ.get("TORCH_EXTENSIONS_DIR"),
            "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "cuda_cache_path": os.environ.get("CUDA_CACHE_PATH"),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="profile_salvalai_smoke15")
    parser.add_argument("--sequence-index", type=int, default=9)
    parser.add_argument("--previous-gate-report", type=Path, required=True)
    parser.add_argument("--expected-b1-complete-tps", type=float, default=REVIEWED_B1_COMPLETE_TPS)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--row-seed", action="append", type=int, default=[])
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--comparison-top-k", type=int, default=20)
    parser.add_argument("--capture-warmup-repeats", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=50)
    parser.add_argument("--active-prefix-bucket-size", type=int, default=64)
    parser.add_argument("--q1-bmm-cross-attention", action="store_true")
    parser.add_argument("--native-q1-self-attention", action="store_true")
    parser.add_argument("--native-q1-rope-cache-self-attention", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides such as audio_path=/path/song.mp3")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    total_started = time.perf_counter()
    previous_report = _load_previous_report(cli.previous_gate_report)
    args = _load_args(cli.config_name, cli.overrides)
    seeds = _resolve_row_seeds(cli.row_seed, int(args.seed))
    report = run_l2_gate(
        args,
        config_name=cli.config_name,
        previous_report=previous_report,
        expected_b1_complete_tps=cli.expected_b1_complete_tps,
        sequence_index=cli.sequence_index,
        seeds=seeds,
        atol=cli.atol,
        rtol=cli.rtol,
        comparison_top_k=cli.comparison_top_k,
        capture_warmup_repeats=cli.capture_warmup_repeats,
        timing_repeats=cli.timing_repeats,
        active_prefix_bucket_size=cli.active_prefix_bucket_size,
        q1_bmm_cross_attention=cli.q1_bmm_cross_attention,
        native_q1_self_attention=cli.native_q1_self_attention,
        native_q1_rope_cache_self_attention=cli.native_q1_rope_cache_self_attention,
    )
    report["runtime_metadata"]["config_name"] = cli.config_name
    report["runtime_metadata"]["hydra_overrides"] = list(cli.overrides)
    report["previous_gate_report"] = str(cli.previous_gate_report.resolve())
    report["total_wall_seconds"] = time.perf_counter() - total_started
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
