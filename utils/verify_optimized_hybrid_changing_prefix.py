"""Run the bounded changing-prefix hybrid B2 verifier.

This utility is verifier/measurement infrastructure only.  It proves Phase A
at eight closed-loop bucket-128 steps before it may run Phase B at sixteen
steps.  It does not wire a scheduler, server, or production runtime.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
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
from transformers import LogitsProcessorList, TopPLogitsWarper

from inference import compile_args, load_model_with_server, setup_inference_environment
from osuT5.osuT5.inference.decode_loop import _bucketed_prefix_length
from osuT5.osuT5.inference.direct_decode import DecodeSession
from osuT5.osuT5.inference.optimized.batch.hybrid_changing_prefix import (
    ACTIVE_PREFIX_LENGTH,
    CHANGING_PREFIX_EVENT_ORDER,
    CHANGING_PREFIX_GATE,
    CHANGING_PREFIX_SCHEMA_VERSION,
    FIXED_HYBRID_SOURCE_COMMIT,
    FIXED_HYBRID_SOURCE_JOB,
    FIXED_HYBRID_SOURCE_REPORT_SHA256,
    FIXED_HYBRID_SOURCE_WORST_TPS,
    FIXED_PROMPT_LENGTH,
    NEUTRAL_PAD_TOKEN_ID,
    PHASE_A_HORIZON,
    PHASE_B_HORIZON,
    PHASE_B_TIMING_TRIALS,
    PRODUCTION_BUCKET128_DECODE_TOKENS,
    PRODUCTION_TOTAL_DECODE_TOKENS,
    summarize_changing_prefix_performance,
    validate_changing_prefix_report,
)
from osuT5.osuT5.inference.optimized.batch.hybrid_l2 import (
    HybridBridgeResourceEvidence,
    HybridSamplingResourceEvidence,
    REVIEWED_FRAMES_SHA256,
    REVIEWED_PROMPT_ATTENTION_MASK_SHA256,
    REVIEWED_PROMPT_SHA256,
    REVIEWED_WORKLOAD_CONTRACT_SHA256,
    processor_class_names,
    validate_hybrid_l2_report,
    validate_hybrid_processor_family,
    validate_hybrid_resource_ownership,
)
from osuT5.osuT5.inference.optimized.batch.lane_one_token import (
    DEFAULT_SEPARATE_GRAPH_POOL,
    PYTORCH_PER_STREAM_CUBLAS_WORKSPACE,
    CapturedB1Lane,
    LaneResourceEvidence,
    capture_prepared_b1_lane,
    compare_lane_logits,
    reciprocal_lane_orders,
    tensor_sha256,
)
from osuT5.osuT5.inference.server import get_eos_token_id
from osuT5.osuT5.runtime_profiling import generation_profile_context
from osuT5.osuT5.tokenizer import ContextType
from utils.verify_one_token_decode import (
    _assert_supported_probe,
    _build_probe_inputs,
    _condition_kwargs,
    _load_args,
    _move_kwargs_for_model,
)
from utils.verify_optimized_b1_lane_capture import (
    _cache_tensor_ptrs,
    _cache_tensor_values,
    _new_generator,
    _sample_next_token,
    _tensor_storage_ptr,
    _top_level_tensor_ptrs,
)
from utils.verify_optimized_hybrid_l2 import (
    _file_sha256,
    _make_factories,
    _validate_fixed_runtime_contract,
)


ProcessorFactory = Callable[[], Any]
STATIC_KEYS = (
    "decoder_input_ids",
    "cache_position",
    "decoder_position_ids",
    "decoder_attention_mask",
)


@dataclass
class _CandidateLane:
    row: int
    seed: int
    session: DecodeSession
    lane: CapturedB1Lane
    source_logits: torch.Tensor
    anchor_token: torch.LongTensor
    full_prefix: torch.LongTensor
    full_mask: torch.Tensor
    initial_seed_rng_state: torch.Tensor
    post_anchor_rng_state: torch.Tensor
    anchor_generator: torch.Generator
    anchor_processor: Any
    self_snapshot: tuple[torch.Tensor, ...]
    cross_snapshot: tuple[torch.Tensor, ...]
    ownership: LaneResourceEvidence
    prefill_wall_seconds: float
    prefill_cuda_seconds: float
    setup_wall_seconds: float
    setup_cuda_seconds: float


@dataclass
class _ReferenceLane:
    row: int
    seed: int
    session: DecodeSession
    prefix: torch.LongTensor
    mask: torch.Tensor
    anchor_token: torch.LongTensor
    initial_seed_rng_state: torch.Tensor
    post_anchor_rng_state: torch.Tensor
    generator: torch.Generator
    base_processor: Any
    warper: Any
    stopped: bool = False


@dataclass
class _ProcessorBridge:
    stream: torch.cuda.Stream
    graph: torch.cuda.CUDAGraph
    processor: Any
    prefix_batch: torch.LongTensor
    initial_prefix_batch: torch.LongTensor
    raw_logits_bridge: torch.Tensor
    raw_rows: tuple[torch.Tensor, torch.Tensor]
    processed_scores: torch.Tensor
    processed_rows: tuple[torch.Tensor, torch.Tensor]
    warmup_seconds: float
    capture_seconds: float


@dataclass
class _SamplingTail:
    row: int
    stream: torch.cuda.Stream
    graph: torch.cuda.CUDAGraph
    warper: Any
    generator: torch.Generator
    processed_scores: torch.Tensor
    probabilities: torch.Tensor
    sampled_token: torch.LongTensor
    sampled_token_2d: torch.LongTensor
    stop_flag: torch.Tensor
    eos_token_ids: torch.LongTensor
    warmup_seconds: float
    capture_seconds: float


@dataclass
class _Events:
    state_ready: tuple[torch.cuda.Event, torch.cuda.Event]
    lane_ready: tuple[torch.cuda.Event, torch.cuda.Event]
    source_released: tuple[torch.cuda.Event, torch.cuda.Event]
    processor_done: torch.cuda.Event
    sample_done: tuple[torch.cuda.Event, torch.cuda.Event]


@dataclass
class _Resources:
    lanes: tuple[_CandidateLane, _CandidateLane]
    bridge: _ProcessorBridge
    sampling: tuple[_SamplingTail, _SamplingTail]
    events: _Events
    stop_flags: tuple[torch.Tensor, torch.Tensor]
    prefix_write_views: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _json_sha256(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _extension_cache_state() -> dict[str, Any]:
    raw_path = os.environ.get("TORCH_EXTENSIONS_DIR")
    if not raw_path:
        return {"path": None, "exists": False, "entry_count": 0, "file_count": 0}
    path = Path(raw_path)
    if not path.exists():
        return {"path": str(path), "exists": False, "entry_count": 0, "file_count": 0}
    return {
        "path": str(path),
        "exists": True,
        "entry_count": sum(1 for _ in path.iterdir()),
        "file_count": sum(1 for value in path.rglob("*") if value.is_file()),
    }


def _new_event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=False, blocking=False, interprocess=False)


def _cache_hash(cache_part: Any, *, start: int | None = None, stop: int | None = None) -> str:
    digest = hashlib.sha256()
    for tensor in _cache_tensor_values(cache_part):
        value = tensor if start is None else tensor[..., start:stop, :]
        digest.update(tensor_sha256(value).encode("ascii"))
        digest.update(str(list(value.shape)).encode("ascii"))
    return digest.hexdigest()


def _snapshot_cache(cache_part: Any) -> tuple[torch.Tensor, ...]:
    return tuple(
        tensor.clone(memory_format=torch.contiguous_format)
        for tensor in _cache_tensor_values(cache_part)
    )


def _restore_cache(cache_part: Any, snapshot: Sequence[torch.Tensor]) -> None:
    tensors = _cache_tensor_values(cache_part)
    if len(tensors) != len(snapshot):
        raise RuntimeError("cache snapshot topology changed before in-place restore.")
    for target, source in zip(tensors, snapshot, strict=True):
        target.copy_(source)


def _cache_comparison(
        reference_part: Any,
        candidate_part: Any,
        *,
        start: int | None = None,
        stop: int | None = None,
) -> dict[str, Any]:
    reference = _cache_tensor_values(reference_part)
    candidate = _cache_tensor_values(candidate_part)
    shape_match = len(reference) == len(candidate)
    allclose = shape_match
    max_abs = 0.0
    reference_shapes: list[list[int]] = []
    candidate_shapes: list[list[int]] = []
    if shape_match:
        for left, right in zip(reference, candidate, strict=True):
            left_view = left if start is None else left[..., start:stop, :]
            right_view = right if start is None else right[..., start:stop, :]
            reference_shapes.append(list(left_view.shape))
            candidate_shapes.append(list(right_view.shape))
            if left_view.shape != right_view.shape:
                shape_match = False
                allclose = False
                continue
            allclose = bool(allclose and torch.allclose(left_view, right_view, atol=1e-4, rtol=1e-4))
            if left_view.numel():
                max_abs = max(max_abs, float(torch.max(torch.abs(left_view - right_view)).item()))
    return {
        "shape_match": shape_match,
        "allclose": allclose,
        "topk_match": shape_match,
        "reference_shape": reference_shapes,
        "candidate_shape": candidate_shapes,
        "reference_topk": [],
        "candidate_topk": [],
        "max_abs": max_abs,
        "reference_sha256": _cache_hash(reference_part, start=start, stop=stop),
        "candidate_sha256": _cache_hash(candidate_part, start=start, stop=stop),
    }


def _logit_comparison(reference: torch.Tensor, candidate: torch.Tensor, *, top_k: int) -> dict[str, Any]:
    comparison = compare_lane_logits(
        reference,
        candidate,
        atol=1e-4,
        rtol=1e-4,
        top_k=top_k,
    )
    return {
        "shape_match": bool(comparison.get("shape_match", False)),
        "allclose": bool(comparison.get("allclose", False)),
        "topk_match": bool(comparison.get("topk_match", False)),
        "reference_shape": comparison.get("reference_shape", []),
        "candidate_shape": comparison.get("candidate_shape", []),
        "reference_topk": comparison.get("reference_topk", []),
        "candidate_topk": comparison.get("candidate_topk", []),
        "max_abs": float(comparison.get("max_abs", math.inf)),
        "reference_sha256": tensor_sha256(reference),
        "candidate_sha256": tensor_sha256(candidate),
    }


def _hash_pair(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference_hash = tensor_sha256(reference)
    candidate_hash = tensor_sha256(candidate)
    return {
        "reference_sha256": reference_hash,
        "candidate_sha256": candidate_hash,
        "bitwise_match": reference_hash == candidate_hash,
    }


def _copy_static_inputs(
        lane: _CandidateLane,
        source: Mapping[str, torch.Tensor],
        *,
        copy_decoder_token: bool,
) -> None:
    for key in STATIC_KEYS:
        if key == "decoder_input_ids" and not copy_decoder_token:
            continue
        target = lane.lane.static_inputs.get(key)
        value = source.get(key)
        if not isinstance(target, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise RuntimeError(f"changing-prefix static tensor {key} is missing.")
        if target.shape != value.shape or target.dtype != value.dtype:
            raise RuntimeError(f"changing-prefix static tensor {key} shape/dtype changed.")
        target.copy_(value)


@torch.no_grad()
def _prepare_static_source(
        lane: _CandidateLane,
        *,
        prefix: torch.LongTensor,
        mask: torch.Tensor,
        cache_position: int,
) -> dict[str, torch.Tensor]:
    state = lane.session.one_token_state()
    prepared = lane.session.model.prepare_inputs_for_generation(
        prefix,
        past_key_values=state.cache,
        use_cache=True,
        encoder_outputs=state.encoder_outputs,
        decoder_attention_mask=mask,
        cache_position=torch.arange(
            cache_position,
            cache_position + 1,
            device=prefix.device,
        ),
        **lane.session.condition_kwargs,
    )
    result: dict[str, torch.Tensor] = {}
    for key in STATIC_KEYS:
        value = prepared.get(key)
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"normal preparation did not produce tensor {key}.")
        result[key] = value.clone(memory_format=torch.contiguous_format)
    return result


@torch.no_grad()
def _capture_candidate_lane(
        model: Any,
        *,
        row: int,
        seed: int,
        prompt: torch.LongTensor,
        prompt_mask: torch.Tensor,
        frames: torch.Tensor,
        condition_kwargs: Mapping[str, Any],
        combined_factory: ProcessorFactory,
        capture_warmup_repeats: int,
) -> _CandidateLane:
    torch.cuda.synchronize(model.device)
    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_finish = torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    prefill_start.record()
    session = DecodeSession.prefill(
        model,
        prompt=prompt,
        prompt_attention_mask=prompt_mask,
        frames=frames,
        condition_kwargs=dict(condition_kwargs),
        active_prefix_self_attention=False,
    )
    prefill_finish.record()
    prefill_finish.synchronize()
    prefill_wall_seconds = time.perf_counter() - started
    prefill_cuda_seconds = prefill_start.elapsed_time(prefill_finish) / 1000.0
    cache = session.cache_state.cache
    encoder = session.encoder_state.encoder_outputs
    prefill_logits = session.one_token_state().prefill_logits
    if cache is None or encoder is None or prefill_logits is None:
        raise RuntimeError(f"candidate row {row} lost prefill cache/encoder/logits.")
    self_snapshot = _snapshot_cache(cache.self_attention_cache)
    cross_snapshot = _snapshot_cache(cache.cross_attention_cache)

    generator = _new_generator(torch.device(model.device), seed)
    anchor_processor = combined_factory()
    initial_rng = generator.get_state().clone()
    anchor = _sample_next_token(
        input_ids=prompt,
        raw_logits=prefill_logits,
        logits_processor=anchor_processor,
        generator=generator,
        do_sample=True,
    )
    post_anchor_rng = generator.get_state().clone()
    full_prefix = torch.cat([prompt, anchor[:, None]], dim=-1)
    full_mask = torch.cat(
        [prompt_mask, torch.ones_like(anchor[:, None], dtype=prompt_mask.dtype)],
        dim=-1,
    )
    setup_start = torch.cuda.Event(enable_timing=True)
    setup_finish = torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    setup_start.record()
    setup = session.decode_one_token_raw_logits(
        full_prefix=full_prefix,
        full_attention_mask=full_mask,
        cache_position=torch.arange(FIXED_PROMPT_LENGTH, FIXED_PROMPT_LENGTH + 1, device=prompt.device),
        active_prefix_self_attention=True,
        active_prefix_self_attention_length=ACTIVE_PREFIX_LENGTH,
    )
    setup_finish.record()
    setup_finish.synchronize()
    setup_wall_seconds = time.perf_counter() - started
    setup_cuda_seconds = setup_start.elapsed_time(setup_finish) / 1000.0
    lane = capture_prepared_b1_lane(
        model,
        setup.prepared_inputs,
        active_prefix_length=ACTIVE_PREFIX_LENGTH,
        capture_warmup_repeats=capture_warmup_repeats,
    )
    # Never use CapturedB1Lane.logits here: that property returns a clone.
    source_logits = lane.graph_outputs.logits[:, -1, :]
    if source_logits.dtype != torch.float32:
        raise ValueError(f"candidate source logits must be FP32; got {source_logits.dtype}.")
    if _tensor_storage_ptr(source_logits) != _tensor_storage_ptr(lane.graph_outputs.logits):
        raise RuntimeError("candidate source logits are not a zero-copy graph-output view.")

    with torch.cuda.stream(lane.stream):
        _restore_cache(cache.self_attention_cache, self_snapshot)
        _restore_cache(cache.cross_attention_cache, cross_snapshot)
    lane.stream.synchronize()
    stream_id = int(lane.stream.cuda_stream)
    ownership = LaneResourceEvidence(
        lane_id=row,
        shared_model_id=id(model),
        stream_id=stream_id,
        graph_id=id(lane.graph),
        graph_pool_id=repr(lane.graph.pool()),
        graph_pool_policy=DEFAULT_SEPARATE_GRAPH_POOL,
        graph_pool_sharing_requested=False,
        session_id=id(session),
        cache_id=id(cache),
        encoder_output_storage_ptr=_tensor_storage_ptr(encoder.last_hidden_state),
        generator_id=id(generator),
        logits_processor_id=id(anchor_processor),
        static_input_storage_ptrs=_top_level_tensor_ptrs(lane.static_inputs),
        self_cache_storage_ptrs=_cache_tensor_ptrs(cache.self_attention_cache),
        cross_cache_storage_ptrs=_cache_tensor_ptrs(cache.cross_attention_cache),
        cublas_workspace_policy=PYTORCH_PER_STREAM_CUBLAS_WORKSPACE,
        cublas_workspace_owner_stream_id=stream_id,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )
    return _CandidateLane(
        row=row,
        seed=seed,
        session=session,
        lane=lane,
        source_logits=source_logits,
        anchor_token=anchor,
        full_prefix=full_prefix,
        full_mask=full_mask,
        initial_seed_rng_state=initial_rng,
        post_anchor_rng_state=post_anchor_rng,
        anchor_generator=generator,
        anchor_processor=anchor_processor,
        self_snapshot=self_snapshot,
        cross_snapshot=cross_snapshot,
        ownership=ownership,
        prefill_wall_seconds=prefill_wall_seconds,
        prefill_cuda_seconds=prefill_cuda_seconds,
        setup_wall_seconds=setup_wall_seconds,
        setup_cuda_seconds=setup_cuda_seconds,
    )


@torch.no_grad()
def _build_reference_lane(
        model: Any,
        *,
        row: int,
        seed: int,
        prompt: torch.LongTensor,
        prompt_mask: torch.Tensor,
        frames: torch.Tensor,
        condition_kwargs: Mapping[str, Any],
        base_factory: ProcessorFactory,
        warper_factory: ProcessorFactory,
        combined_factory: ProcessorFactory,
) -> _ReferenceLane:
    session = DecodeSession.prefill(
        model,
        prompt=prompt,
        prompt_attention_mask=prompt_mask,
        frames=frames,
        condition_kwargs=dict(condition_kwargs),
        active_prefix_self_attention=False,
    )
    prefill_logits = session.one_token_state().prefill_logits
    if prefill_logits is None:
        raise RuntimeError(f"reference row {row} lost prefill logits.")
    generator = _new_generator(torch.device(model.device), seed)
    initial_rng = generator.get_state().clone()
    anchor = _sample_next_token(
        input_ids=prompt,
        raw_logits=prefill_logits,
        logits_processor=combined_factory(),
        generator=generator,
        do_sample=True,
    )
    post_anchor_rng = generator.get_state().clone()
    return _ReferenceLane(
        row=row,
        seed=seed,
        session=session,
        prefix=torch.cat([prompt, anchor[:, None]], dim=-1),
        mask=torch.cat(
            [prompt_mask, torch.ones_like(anchor[:, None], dtype=prompt_mask.dtype)],
            dim=-1,
        ),
        anchor_token=anchor,
        initial_seed_rng_state=initial_rng,
        post_anchor_rng_state=post_anchor_rng,
        generator=generator,
        base_processor=base_factory(),
        warper=warper_factory(),
    )


@torch.no_grad()
def _capture_processor_bridge(
        lanes: Sequence[_CandidateLane],
        *,
        base_factory: ProcessorFactory,
        capture_warmup_repeats: int,
) -> _ProcessorBridge:
    device = lanes[0].full_prefix.device
    prefix_batch = torch.full(
        (2, ACTIVE_PREFIX_LENGTH),
        NEUTRAL_PAD_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )
    for row, lane in enumerate(lanes):
        prefix_batch[row:row + 1, :lane.full_prefix.shape[-1]].copy_(lane.full_prefix)
    initial_prefix_batch = prefix_batch.clone(memory_format=torch.contiguous_format)
    raw_bridge = torch.empty(
        (2, int(lanes[0].source_logits.shape[-1])),
        dtype=torch.float32,
        device=device,
    )
    raw_rows = (raw_bridge[0:1], raw_bridge[1:2])
    for row in range(2):
        raw_rows[row].copy_(lanes[row].source_logits)
    processor = base_factory()
    validate_hybrid_processor_family(
        base_processor_classes=processor_class_names(processor),
        private_warper_classes=("TopPLogitsWarper",),
    )
    current = torch.cuda.current_stream(device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(current)
    started = time.perf_counter()
    with torch.cuda.stream(stream):
        for _ in range(capture_warmup_repeats):
            processor(prefix_batch, raw_bridge)
    stream.synchronize()
    warmup_seconds = time.perf_counter() - started
    graph = torch.cuda.CUDAGraph()
    started = time.perf_counter()
    with torch.cuda.graph(graph, stream=stream):
        processed = processor(prefix_batch, raw_bridge)
    capture_seconds = time.perf_counter() - started
    stream.synchronize()
    return _ProcessorBridge(
        stream=stream,
        graph=graph,
        processor=processor,
        prefix_batch=prefix_batch,
        initial_prefix_batch=initial_prefix_batch,
        raw_logits_bridge=raw_bridge,
        raw_rows=raw_rows,
        processed_scores=processed,
        processed_rows=(processed[0:1], processed[1:2]),
        warmup_seconds=warmup_seconds,
        capture_seconds=capture_seconds,
    )


@torch.no_grad()
def _capture_sampling_tail(
        lane: _CandidateLane,
        *,
        warper_factory: ProcessorFactory,
        eos_ids: Sequence[int],
        capture_warmup_repeats: int,
) -> _SamplingTail:
    stream = lane.lane.stream
    device = lane.full_prefix.device
    processed = torch.empty_like(lane.source_logits)
    processed.copy_(lane.source_logits)
    warper = warper_factory()
    generator = _new_generator(torch.device(device), lane.seed)
    generator.set_state(lane.post_anchor_rng_state)
    warm_generator = _new_generator(torch.device(device), lane.seed + 900_000)
    eos_tensor = torch.tensor(list(eos_ids), dtype=torch.long, device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    started = time.perf_counter()
    with torch.cuda.stream(stream):
        for _ in range(capture_warmup_repeats):
            scores = warper(lane.full_prefix, processed)
            probabilities = torch.nn.functional.softmax(scores, dim=-1)
            token = torch.multinomial(probabilities, 1, generator=warm_generator).squeeze(1)
            torch.isin(token, eos_tensor)
    stream.synchronize()
    warmup_seconds = time.perf_counter() - started
    graph = torch.cuda.CUDAGraph()
    graph.register_generator_state(generator)
    started = time.perf_counter()
    with torch.cuda.graph(graph, stream=stream):
        warped = warper(lane.full_prefix, processed)
        probabilities = torch.nn.functional.softmax(warped, dim=-1)
        sampled = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
        stop_flag = torch.isin(sampled, eos_tensor).any()
    capture_seconds = time.perf_counter() - started
    stream.synchronize()
    generator.set_state(lane.post_anchor_rng_state)
    return _SamplingTail(
        row=lane.row,
        stream=stream,
        graph=graph,
        warper=warper,
        generator=generator,
        processed_scores=processed,
        probabilities=probabilities,
        sampled_token=sampled,
        sampled_token_2d=sampled.reshape(1, 1),
        stop_flag=stop_flag,
        eos_token_ids=eos_tensor,
        warmup_seconds=warmup_seconds,
        capture_seconds=capture_seconds,
    )


def _build_resources(
        lanes: tuple[_CandidateLane, _CandidateLane],
        bridge: _ProcessorBridge,
        sampling: tuple[_SamplingTail, _SamplingTail],
) -> _Resources:
    device = bridge.prefix_batch.device
    with torch.cuda.stream(bridge.stream):
        stop_flags = (
            torch.zeros((), dtype=torch.bool, device=device),
            torch.zeros((), dtype=torch.bool, device=device),
        )
    bridge.stream.synchronize()
    return _Resources(
        lanes=lanes,
        bridge=bridge,
        sampling=sampling,
        events=_Events(
            state_ready=(_new_event(), _new_event()),
            lane_ready=(_new_event(), _new_event()),
            source_released=(_new_event(), _new_event()),
            processor_done=_new_event(),
            sample_done=(_new_event(), _new_event()),
        ),
        stop_flags=stop_flags,
        prefix_write_views=tuple(
            tuple(
                bridge.prefix_batch[row:row + 1, FIXED_PROMPT_LENGTH + 1 + step]
                for step in range(PHASE_B_HORIZON)
            )
            for row in range(2)
        ),
    )


@torch.no_grad()
def _restore_resources(resources: _Resources) -> dict[str, Any]:
    for row, lane in enumerate(resources.lanes):
        cache = lane.session.cache_state.cache
        if cache is None:
            raise RuntimeError(f"candidate row {row} lost cache before restore.")
        with torch.cuda.stream(lane.lane.stream):
            _restore_cache(cache.self_attention_cache, lane.self_snapshot)
            _restore_cache(cache.cross_attention_cache, lane.cross_snapshot)
        resources.sampling[row].generator.set_state(lane.post_anchor_rng_state)
    for lane in resources.lanes:
        lane.lane.stream.synchronize()
    return {
        "self_cache_snapshot_hashes": {
            f"row-{row}": _snapshot_tuple_hash(lane.self_snapshot)
            for row, lane in enumerate(resources.lanes)
        },
        "post_restore_self_cache_hashes": {
            f"row-{row}": _cache_hash(lane.session.cache_state.cache.self_attention_cache)
            for row, lane in enumerate(resources.lanes)
        },
        "cross_cache_snapshot_hashes": {
            f"row-{row}": _snapshot_tuple_hash(lane.cross_snapshot)
            for row, lane in enumerate(resources.lanes)
        },
        "post_restore_cross_cache_hashes": {
            f"row-{row}": _cache_hash(lane.session.cache_state.cache.cross_attention_cache)
            for row, lane in enumerate(resources.lanes)
        },
    }


def _snapshot_tuple_hash(values: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(tensor_sha256(value).encode("ascii"))
        digest.update(str(list(value.shape)).encode("ascii"))
    return digest.hexdigest()


@torch.no_grad()
def _initialize_candidate_state(
        resources: _Resources,
        sources: Sequence[Sequence[Mapping[str, torch.Tensor]]],
) -> None:
    bridge = resources.bridge
    with torch.cuda.stream(bridge.stream):
        bridge.prefix_batch.copy_(bridge.initial_prefix_batch)
        for stop_flag in resources.stop_flags:
            stop_flag.zero_()
        for row, lane in enumerate(resources.lanes):
            _copy_static_inputs(lane, sources[row][0], copy_decoder_token=True)
            resources.events.state_ready[row].record(bridge.stream)


@torch.no_grad()
def _enqueue_one_exact_step(
        resources: _Resources,
        *,
        order: Sequence[int],
        step: int,
        next_sources: Sequence[Mapping[str, torch.Tensor]],
) -> None:
    events = resources.events
    bridge = resources.bridge
    for row in order:
        lane = resources.lanes[row]
        with torch.cuda.stream(lane.lane.stream):
            lane.lane.stream.wait_event(events.state_ready[row])
            if step > 0:
                lane.lane.stream.wait_event(events.source_released[row])
            lane.lane.graph.replay()
            events.lane_ready[row].record(lane.lane.stream)
    with torch.cuda.stream(bridge.stream):
        for row in order:
            bridge.stream.wait_event(events.lane_ready[row])
        for row in order:
            bridge.raw_rows[row].copy_(resources.lanes[row].source_logits)
            events.source_released[row].record(bridge.stream)
        bridge.graph.replay()
        for row in order:
            resources.sampling[row].processed_scores.copy_(bridge.processed_rows[row])
        events.processor_done.record(bridge.stream)
    for row in order:
        tail = resources.sampling[row]
        with torch.cuda.stream(tail.stream):
            tail.stream.wait_event(events.processor_done)
            tail.graph.replay()
            events.sample_done[row].record(tail.stream)
    with torch.cuda.stream(bridge.stream):
        for row in order:
            bridge.stream.wait_event(events.sample_done[row])
        for row in order:
            tail = resources.sampling[row]
            lane = resources.lanes[row]
            resources.stop_flags[row].logical_or_(tail.stop_flag)
            resources.prefix_write_views[row][step].copy_(tail.sampled_token)
            lane.lane.static_inputs["decoder_input_ids"].copy_(tail.sampled_token_2d)
            _copy_static_inputs(lane, next_sources[row], copy_decoder_token=False)
            events.state_ready[row].record(bridge.stream)
    bridge.stream.synchronize()


@torch.no_grad()
def _enqueue_timed_trial(
        resources: _Resources,
        *,
        order: Sequence[int],
        sources: Sequence[Sequence[Mapping[str, torch.Tensor]]],
        horizon: int,
        start: torch.cuda.Event,
        finish: torch.cuda.Event,
) -> tuple[float, float]:
    """Complete no-sync changing-prefix interval using only static storage."""

    bridge = resources.bridge
    events = resources.events
    wall_started = time.perf_counter()
    start.record(bridge.stream)
    with torch.cuda.stream(bridge.stream):
        bridge.prefix_batch.copy_(bridge.initial_prefix_batch)
        for stop_flag in resources.stop_flags:
            stop_flag.zero_()
        for row in order:
            _copy_static_inputs(resources.lanes[row], sources[row][0], copy_decoder_token=True)
            events.state_ready[row].record(bridge.stream)
    for step in range(horizon):
        for row in order:
            lane = resources.lanes[row]
            with torch.cuda.stream(lane.lane.stream):
                lane.lane.stream.wait_event(events.state_ready[row])
                if step > 0:
                    lane.lane.stream.wait_event(events.source_released[row])
                lane.lane.graph.replay()
                events.lane_ready[row].record(lane.lane.stream)
        with torch.cuda.stream(bridge.stream):
            for row in order:
                bridge.stream.wait_event(events.lane_ready[row])
            for row in order:
                bridge.raw_rows[row].copy_(resources.lanes[row].source_logits)
                events.source_released[row].record(bridge.stream)
            bridge.graph.replay()
            for row in order:
                resources.sampling[row].processed_scores.copy_(bridge.processed_rows[row])
            events.processor_done.record(bridge.stream)
        for row in order:
            tail = resources.sampling[row]
            with torch.cuda.stream(tail.stream):
                tail.stream.wait_event(events.processor_done)
                tail.graph.replay()
                events.sample_done[row].record(tail.stream)
        with torch.cuda.stream(bridge.stream):
            for row in order:
                bridge.stream.wait_event(events.sample_done[row])
            for row in order:
                tail = resources.sampling[row]
                lane = resources.lanes[row]
                resources.stop_flags[row].logical_or_(tail.stop_flag)
                resources.prefix_write_views[row][step].copy_(tail.sampled_token)
                lane.lane.static_inputs["decoder_input_ids"].copy_(tail.sampled_token_2d)
                _copy_static_inputs(lane, sources[row][step + 1], copy_decoder_token=False)
                events.state_ready[row].record(bridge.stream)
    with torch.cuda.stream(bridge.stream):
        finish.record(bridge.stream)
    finish.synchronize()
    return time.perf_counter() - wall_started, start.elapsed_time(finish) / 1000.0


def _snapshot_slice_hash(
        snapshot: Sequence[torch.Tensor],
        *,
        start: int,
        stop: int | None,
) -> str:
    digest = hashlib.sha256()
    for value in snapshot:
        view = value[..., start:stop, :]
        digest.update(tensor_sha256(view).encode("ascii"))
        digest.update(str(list(view.shape)).encode("ascii"))
    return digest.hexdigest()


def _cache_slot_nonzero(cache_part: Any, position: int) -> bool:
    return all(
        bool(torch.count_nonzero(value[..., position, :]).item() > 0)
        for value in _cache_tensor_values(cache_part)
    )


def _normal_static_evidence(
        source: Mapping[str, torch.Tensor],
        lane: _CandidateLane,
) -> dict[str, Any]:
    return {
        key: _hash_pair(source[key], lane.lane.static_inputs[key])
        for key in STATIC_KEYS
    }


def _padded_reference(prefix: torch.LongTensor) -> torch.LongTensor:
    result = torch.full(
        (1, ACTIVE_PREFIX_LENGTH),
        NEUTRAL_PAD_TOKEN_ID,
        dtype=prefix.dtype,
        device=prefix.device,
    )
    result[:, :prefix.shape[-1]].copy_(prefix)
    return result


def _post_step_state(
        *,
        reference: _ReferenceLane,
        candidate: _CandidateLane,
        candidate_prefix: torch.Tensor,
        next_source: Mapping[str, torch.Tensor],
        candidate_generator: torch.Generator,
        step: int,
) -> dict[str, Any]:
    reference_cache = reference.session.cache_state.cache
    candidate_cache = candidate.session.cache_state.cache
    if reference_cache is None or candidate_cache is None:
        raise RuntimeError("post-step state lost reference/candidate cache.")
    reference_padded = _padded_reference(reference.prefix)
    reference_rng = reference.generator.get_state()
    candidate_rng = candidate_generator.get_state()
    return {
        "rng_state": _hash_pair(reference_rng, candidate_rng),
        "prefix": _hash_pair(reference_padded, candidate_prefix),
        "next_static_inputs": {
            key: _hash_pair(next_source[key], candidate.lane.static_inputs[key])
            for key in STATIC_KEYS
        },
        "self_cache": _cache_comparison(
            reference_cache.self_attention_cache,
            candidate_cache.self_attention_cache,
            start=0,
            stop=FIXED_PROMPT_LENGTH + step + 1,
        ),
        "cross_cache": _cache_comparison(
            reference_cache.cross_attention_cache,
            candidate_cache.cross_attention_cache,
        ),
        "executed_cache_position": FIXED_PROMPT_LENGTH + step,
        "next_static_cache_position": FIXED_PROMPT_LENGTH + step + 1,
        "prefix_real_length": FIXED_PROMPT_LENGTH + step + 2,
    }


@torch.no_grad()
def _run_exact_order(
        model: Any,
        resources: _Resources,
        *,
        order: Sequence[int],
        horizon: int,
        prompt: torch.LongTensor,
        prompt_mask: torch.Tensor,
        frames: torch.Tensor,
        condition_kwargs: Mapping[str, Any],
        base_factory: ProcessorFactory,
        warper_factory: ProcessorFactory,
        combined_factory: ProcessorFactory,
        comparison_top_k: int,
        eos_ids: Sequence[int],
) -> tuple[dict[str, Any], list[list[dict[str, torch.Tensor]]], tuple[_ReferenceLane, _ReferenceLane]]:
    references_list = [
        _build_reference_lane(
            model,
            row=row,
            seed=resources.lanes[row].seed,
            prompt=prompt,
            prompt_mask=prompt_mask,
            frames=frames,
            condition_kwargs=condition_kwargs,
            base_factory=base_factory,
            warper_factory=warper_factory,
            combined_factory=combined_factory,
        )
        for row in range(2)
    ]
    references = (references_list[0], references_list[1])
    for row in range(2):
        if not torch.equal(references[row].anchor_token, resources.lanes[row].anchor_token):
            raise RuntimeError(f"row {row} independent anchor token differs from candidate.")
        if tensor_sha256(references[row].post_anchor_rng_state) != tensor_sha256(
                resources.lanes[row].post_anchor_rng_state
        ):
            raise RuntimeError(f"row {row} independent post-anchor RNG differs from candidate.")

    _restore_resources(resources)
    sources: list[list[dict[str, torch.Tensor]]] = [[], []]
    for row in range(2):
        sources[row].append(_prepare_static_source(
            resources.lanes[row],
            prefix=references[row].prefix,
            mask=references[row].mask,
            cache_position=FIXED_PROMPT_LENGTH,
        ))
    _initialize_candidate_state(resources, sources)
    resources.bridge.stream.synchronize()

    previous_raw_hash: list[str | None] = [None, None]
    steps: list[dict[str, Any]] = []
    final_rng_hashes: dict[str, str] = {}
    final_states: dict[str, Any] = {}
    aggregates = {
        "raw_logits_allclose": True,
        "raw_logits_topk_match": True,
        "raw_logits_changed_each_step": True,
        "processed_rows_bitwise_match": True,
        "processed_rows_topk_match": True,
        "sampled_tokens_match": True,
        "per_step_rng_state_match": True,
        "final_rng_state_match": True,
        "stop_behavior_match": True,
        "static_inputs_bitwise_match": True,
        "cache_active_prefix_allclose": True,
        "cache_current_slot_written": True,
        "cache_future_suffix_unchanged": True,
        "cross_cache_unchanged_bitwise": True,
        "cross_cache_reference_allclose": True,
        "neutral_padding_proof_pass": True,
        "no_dummy_or_stopped_row_sampled": True,
        "no_eos_in_horizon": True,
    }
    for step in range(horizon):
        static_before = [
            _normal_static_evidence(sources[row][step], resources.lanes[row])
            for row in range(2)
        ]
        prefix_before = [
            resources.bridge.prefix_batch[row:row + 1].clone(memory_format=torch.contiguous_format)
            for row in range(2)
        ]
        reference_raw: list[torch.Tensor] = [torch.empty(0), torch.empty(0)]
        reference_processed: list[torch.Tensor] = [torch.empty(0), torch.empty(0)]
        reference_tokens: list[torch.Tensor] = [torch.empty(0), torch.empty(0)]
        reference_stops: list[bool] = [False, False]
        next_sources: list[dict[str, torch.Tensor]] = [{}, {}]
        for row in order:
            reference = references[row]
            result = reference.session.decode_one_token_raw_logits(
                full_prefix=reference.prefix,
                full_attention_mask=reference.mask,
                cache_position=torch.arange(
                    FIXED_PROMPT_LENGTH + step,
                    FIXED_PROMPT_LENGTH + step + 1,
                    device=prompt.device,
                ),
                active_prefix_self_attention=True,
                active_prefix_self_attention_length=ACTIVE_PREFIX_LENGTH,
            )
            raw = result.logits
            processed = reference.base_processor(
                reference.prefix,
                raw.clone(memory_format=torch.contiguous_format),
            )
            warped = reference.warper(reference.prefix, processed.clone(memory_format=torch.contiguous_format))
            token = torch.multinomial(
                torch.nn.functional.softmax(warped, dim=-1),
                1,
                generator=reference.generator,
            ).squeeze(1)
            stopped = bool(torch.isin(
                token,
                torch.tensor(list(eos_ids), device=token.device, dtype=torch.long),
            ).any().item())
            reference_raw[row] = raw
            reference_processed[row] = processed
            reference_tokens[row] = token
            reference_stops[row] = stopped
            reference.prefix = torch.cat([reference.prefix, token[:, None]], dim=-1)
            reference.mask = torch.cat(
                [reference.mask, torch.ones_like(token[:, None], dtype=reference.mask.dtype)],
                dim=-1,
            )
            next_sources[row] = _prepare_static_source(
                resources.lanes[row],
                prefix=reference.prefix,
                mask=reference.mask,
                cache_position=FIXED_PROMPT_LENGTH + step + 1,
            )
            sources[row].append(next_sources[row])
        _enqueue_one_exact_step(
            resources,
            order=order,
            step=step,
            next_sources=next_sources,
        )

        row_reports: list[dict[str, Any]] = []
        for row in range(2):
            lane = resources.lanes[row]
            reference = references[row]
            cache = lane.session.cache_state.cache
            reference_cache = reference.session.cache_state.cache
            if cache is None or reference_cache is None:
                raise RuntimeError("exact changing-prefix step lost cache.")
            candidate_raw = lane.source_logits
            raw_comparison = _logit_comparison(
                reference_raw[row], candidate_raw, top_k=comparison_top_k
            )
            candidate_raw_hash = raw_comparison["candidate_sha256"]
            raw_comparison["changed_from_previous"] = (
                None
                if previous_raw_hash[row] is None
                else candidate_raw_hash != previous_raw_hash[row]
            )
            previous_raw_hash[row] = candidate_raw_hash
            processed_comparison = _logit_comparison(
                reference_processed[row],
                resources.bridge.processed_rows[row],
                top_k=comparison_top_k,
            )
            processed_comparison.update(_hash_pair(
                reference_processed[row], resources.bridge.processed_rows[row]
            ))
            candidate_token = resources.sampling[row].sampled_token
            token_match = bool(torch.equal(reference_tokens[row], candidate_token))
            reference_rng = reference.generator.get_state()
            candidate_rng = resources.sampling[row].generator.get_state()
            candidate_stop = bool(resources.sampling[row].stop_flag.item())
            real_length_before = FIXED_PROMPT_LENGTH + 1 + step
            reference_padded_before = _padded_reference(
                reference.prefix[:, :real_length_before]
            )
            suffix_reference = reference_padded_before[:, real_length_before:]
            suffix_candidate = prefix_before[row][:, real_length_before:]
            active_cache = _cache_comparison(
                reference_cache.self_attention_cache,
                cache.self_attention_cache,
                start=0,
                stop=FIXED_PROMPT_LENGTH + step + 1,
            )
            current_cache = _cache_comparison(
                reference_cache.self_attention_cache,
                cache.self_attention_cache,
                start=FIXED_PROMPT_LENGTH + step,
                stop=FIXED_PROMPT_LENGTH + step + 1,
            )
            current_cache["candidate_nonzero"] = _cache_slot_nonzero(
                cache.self_attention_cache, FIXED_PROMPT_LENGTH + step
            )
            current_cache["changed_from_reset"] = (
                current_cache["candidate_sha256"]
                != _snapshot_slice_hash(
                    lane.self_snapshot,
                    start=FIXED_PROMPT_LENGTH + step,
                    stop=FIXED_PROMPT_LENGTH + step + 1,
                )
            )
            future_hash = _cache_hash(
                cache.self_attention_cache,
                start=FIXED_PROMPT_LENGTH + step + 1,
                stop=None,
            )
            future_reference_hash = _snapshot_slice_hash(
                lane.self_snapshot,
                start=FIXED_PROMPT_LENGTH + step + 1,
                stop=None,
            )
            future = {
                "reference_sha256": future_reference_hash,
                "candidate_sha256": future_hash,
                "bitwise_match": future_reference_hash == future_hash,
            }
            cross_hash = _cache_hash(cache.cross_attention_cache)
            cross_snapshot_hash = _snapshot_tuple_hash(lane.cross_snapshot)
            cross = {
                "reference_sha256": cross_snapshot_hash,
                "candidate_sha256": cross_hash,
                "bitwise_match": cross_snapshot_hash == cross_hash,
                "reference_comparison": _cache_comparison(
                    reference_cache.cross_attention_cache,
                    cache.cross_attention_cache,
                ),
            }
            post_state = _post_step_state(
                reference=reference,
                candidate=lane,
                candidate_prefix=resources.bridge.prefix_batch[row:row + 1],
                next_source=next_sources[row],
                candidate_generator=resources.sampling[row].generator,
                step=step,
            )
            row_report = {
                "row": row,
                "raw_logits": raw_comparison,
                "processed_scores": processed_comparison,
                "sampled_token": {
                    "reference": int(reference_tokens[row].item()),
                    "candidate": int(candidate_token.item()),
                    "match": token_match,
                },
                "rng_state": _hash_pair(reference_rng, candidate_rng),
                "stop": {
                    "reference": reference_stops[row],
                    "candidate": candidate_stop,
                    "match": reference_stops[row] == candidate_stop,
                    "sampled_after_stop": False,
                },
                "padded_prefix": {
                    **_hash_pair(reference_padded_before, prefix_before[row]),
                    "real_length": real_length_before,
                    "full_length": ACTIVE_PREFIX_LENGTH,
                    "model_cache_position": FIXED_PROMPT_LENGTH + step,
                    "pad_token_id": NEUTRAL_PAD_TOKEN_ID,
                    "suffix_all_pad": bool(torch.count_nonzero(suffix_candidate).item() == 0),
                    "pad_outside_sos_ids": True,
                    "pad_outside_time_shift_range": True,
                    "suffix": _hash_pair(suffix_reference, suffix_candidate),
                },
                "static_inputs": static_before[row],
                "cache": {
                    "active_prefix": active_cache,
                    "current_slot": current_cache,
                    "future_suffix": future,
                    "cross": cross,
                },
                "post_step_state": post_state,
            }
            row_reports.append(row_report)
            checks = {
                "raw_logits_allclose": raw_comparison["allclose"],
                "raw_logits_topk_match": raw_comparison["topk_match"],
                "raw_logits_changed_each_step": (
                    step == 0 or raw_comparison["changed_from_previous"]
                ),
                "processed_rows_bitwise_match": processed_comparison["bitwise_match"],
                "processed_rows_topk_match": processed_comparison["topk_match"],
                "sampled_tokens_match": token_match,
                "per_step_rng_state_match": bool(torch.equal(reference_rng, candidate_rng)),
                "stop_behavior_match": reference_stops[row] == candidate_stop,
                "static_inputs_bitwise_match": all(
                    value["bitwise_match"] for value in static_before[row].values()
                ),
                "cache_active_prefix_allclose": active_cache["allclose"],
                "cache_current_slot_written": bool(
                    current_cache["allclose"]
                    and current_cache["candidate_nonzero"]
                    and current_cache["changed_from_reset"]
                ),
                "cache_future_suffix_unchanged": future["bitwise_match"],
                "cross_cache_unchanged_bitwise": cross["bitwise_match"],
                "cross_cache_reference_allclose": cross["reference_comparison"]["allclose"],
                "neutral_padding_proof_pass": bool(
                    processed_comparison["bitwise_match"]
                    and row_report["padded_prefix"]["bitwise_match"]
                    and row_report["padded_prefix"]["suffix"]["bitwise_match"]
                ),
                "no_dummy_or_stopped_row_sampled": not reference.stopped,
                "no_eos_in_horizon": not candidate_stop,
            }
            for field, passed in checks.items():
                aggregates[field] = bool(aggregates[field] and passed)
            reference.stopped = bool(reference.stopped or reference_stops[row])
            final_rng_hashes[f"row-{row}"] = tensor_sha256(candidate_rng)
            final_states[f"row-{row}"] = post_state
        steps.append({"step": step, "rows": row_reports})
        if any(reference_stops) or any(bool(flag.item()) for flag in resources.stop_flags):
            raise RuntimeError("seq9 changing-prefix exact reference reached EOS inside fixed horizon.")

    aggregates["final_rng_state_match"] = all(
        final_rng_hashes[f"row-{row}"]
        == tensor_sha256(references[row].generator.get_state())
        for row in range(2)
    )
    exactness_pass = all(aggregates.values())
    return ({
        "graph_and_sampling_order": list(order),
        "committed_steps": horizon,
        "steps": steps,
        "final_rng_state_hashes": final_rng_hashes,
        "final_state_fingerprints": final_states,
        **aggregates,
        "exactness_pass": exactness_pass,
    }, sources, references)


def _verify_restore(resources: _Resources, reset_contract: Mapping[str, Any]) -> None:
    observed = _restore_resources(resources)
    for field in (
            "self_cache_snapshot_hashes",
            "post_restore_self_cache_hashes",
            "cross_cache_snapshot_hashes",
            "post_restore_cross_cache_hashes",
    ):
        if observed[field] != reset_contract[field]:
            raise RuntimeError(f"in-place reset field {field} differs from contract.")


def _candidate_final_state_matches(
        resources: _Resources,
        *,
        row: int,
        horizon: int,
        expected: Mapping[str, Any],
) -> bool:
    lane = resources.lanes[row]
    cache = lane.session.cache_state.cache
    if cache is None:
        return False
    observed = {
        "rng": tensor_sha256(resources.sampling[row].generator.get_state()),
        "prefix": tensor_sha256(resources.bridge.prefix_batch[row:row + 1]),
        "static": {
            key: tensor_sha256(lane.lane.static_inputs[key])
            for key in STATIC_KEYS
        },
        "self": _cache_hash(
            cache.self_attention_cache,
            start=0,
            stop=FIXED_PROMPT_LENGTH + horizon,
        ),
        "cross": _cache_hash(cache.cross_attention_cache),
    }
    return bool(
        observed["rng"] == expected["rng_state"]["candidate_sha256"]
        and observed["prefix"] == expected["prefix"]["candidate_sha256"]
        and all(
            observed["static"][key]
            == expected["next_static_inputs"][key]["candidate_sha256"]
            for key in STATIC_KEYS
        )
        and observed["self"] == expected["self_cache"]["candidate_sha256"]
        and observed["cross"] == expected["cross_cache"]["candidate_sha256"]
        and expected["executed_cache_position"] == FIXED_PROMPT_LENGTH + horizon - 1
        and expected["next_static_cache_position"] == FIXED_PROMPT_LENGTH + horizon
    )


@torch.no_grad()
def _run_candidate_timing(
        resources: _Resources,
        *,
        order: Sequence[int],
        sources: Sequence[Sequence[Mapping[str, torch.Tensor]]],
        horizon: int,
        trial_count: int,
        reset_contract: Mapping[str, Any],
        final_states: Mapping[str, Any],
) -> dict[str, Any]:
    device = resources.bridge.prefix_batch.device
    trials: list[dict[str, Any]] = []
    total_wall = 0.0
    total_cuda = 0.0
    before_allocated_values: list[int] = []
    before_reserved_values: list[int] = []
    peak_allocated_values: list[int] = []
    peak_reserved_values: list[int] = []
    for trial_index in range(trial_count):
        _verify_restore(resources, reset_contract)
        start = torch.cuda.Event(enable_timing=True)
        finish = torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats(device)
        before_allocated = torch.cuda.memory_allocated(device)
        before_reserved = torch.cuda.memory_reserved(device)
        wall_seconds, cuda_seconds = _enqueue_timed_trial(
            resources,
            order=order,
            sources=sources,
            horizon=horizon,
            start=start,
            finish=finish,
        )
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        if peak_allocated != before_allocated or peak_reserved != before_reserved:
            raise RuntimeError(
                "changing-prefix timed trial allocated or reserved tensor memory."
            )
        final_stop = {
            f"row-{row}": bool(resources.stop_flags[row].item())
            for row in range(2)
        }
        if any(final_stop.values()):
            raise RuntimeError("timed changing-prefix rollout observed EOS.")
        if not all(
                _candidate_final_state_matches(
                    resources,
                    row=row,
                    horizon=horizon,
                    expected=final_states[f"row-{row}"],
                )
                for row in range(2)
        ):
            raise RuntimeError("timed changing-prefix final state differs from exact decode.")
        tokens = 2 * horizon
        trials.append({
            "trial": trial_index,
            "generated_tokens": tokens,
            "wall_seconds": wall_seconds,
            "cuda_seconds": cuda_seconds,
            "wall_tokens_per_second": tokens / wall_seconds,
            "cuda_tokens_per_second": tokens / cuda_seconds,
            "reset_evidence": copy.deepcopy(reset_contract),
            "initial_rng_state_hashes": copy.deepcopy(
                reset_contract["post_anchor_rng_state_hashes"]
            ),
            "final_stop_flags": final_stop,
            "final_state_fingerprints": copy.deepcopy(final_states),
        })
        total_wall += wall_seconds
        total_cuda += cuda_seconds
        before_allocated_values.append(before_allocated)
        before_reserved_values.append(before_reserved)
        peak_allocated_values.append(peak_allocated)
        peak_reserved_values.append(peak_reserved)
    before_allocated = max(before_allocated_values)
    before_reserved = max(before_reserved_values)
    peak_allocated = max(peak_allocated_values)
    peak_reserved = max(peak_reserved_values)
    generated_tokens = 2 * horizon * trial_count
    return {
        "measurement": "complete_changing_prefix_hybrid_B2",
        "horizon": horizon,
        "trial_count": trial_count,
        "generated_tokens": generated_tokens,
        "wall_seconds": total_wall,
        "cuda_seconds": total_cuda,
        "wall_tokens_per_second": generated_tokens / total_wall,
        "cuda_tokens_per_second": generated_tokens / total_cuda,
        "trials": trials,
        "memory_before_allocated_bytes": before_allocated,
        "memory_before_reserved_bytes": before_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_delta_bytes": peak_allocated - before_allocated,
        "peak_reserved_delta_bytes": peak_reserved - before_reserved,
        "reset_outside_timing": True,
        "initial_state_copies_inside_timing": True,
        "per_step_updates_inside_timing": True,
        "device_stop_flags": True,
        "static_tensor_storage": True,
        "sampling_tail_in_cuda_graph": True,
        "explicit_generators_registered_with_sampling_graphs": True,
        "no_eos_in_timed_horizon": True,
        "per_step_global_synchronize": False,
        "per_step_stack_or_cat": False,
        "per_step_python_tensor_allocation": False,
        "graph_recapture_between_trials": False,
        "dummy_or_stopped_row_sampled": False,
    }


@torch.no_grad()
def _run_private_b1_control(
        resources: _Resources,
        *,
        order: Sequence[int],
        sources: Sequence[Sequence[Mapping[str, torch.Tensor]]],
        horizon: int,
        trial_count: int,
        reset_contract: Mapping[str, Any],
        final_states: Mapping[str, Any],
        base_factory: ProcessorFactory,
        warper_factory: ProcessorFactory,
) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    total_wall = 0.0
    total_cuda = 0.0
    for trial_index in range(trial_count):
        _verify_restore(resources, reset_contract)
        generators = [
            _new_generator(resources.lanes[row].full_prefix.device, resources.lanes[row].seed)
            for row in range(2)
        ]
        for row in range(2):
            generators[row].set_state(resources.lanes[row].post_anchor_rng_state)
        processors = [base_factory(), base_factory()]
        warpers = [warper_factory(), warper_factory()]
        private_prefixes = [
            resources.bridge.initial_prefix_batch[row:row + 1].clone(
                memory_format=torch.contiguous_format
            )
            for row in range(2)
        ]
        cuda_seconds = 0.0
        wall_started = time.perf_counter()
        for row in order:
            lane = resources.lanes[row]
            start = torch.cuda.Event(enable_timing=True)
            finish = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(lane.lane.stream):
                _copy_static_inputs(lane, sources[row][0], copy_decoder_token=True)
                start.record(lane.lane.stream)
                for step in range(horizon):
                    lane.lane.graph.replay()
                    real_length = FIXED_PROMPT_LENGTH + 1 + step
                    scores = processors[row](
                        private_prefixes[row][:, :real_length],
                        lane.source_logits.clone(memory_format=torch.contiguous_format),
                    )
                    warped = warpers[row](private_prefixes[row][:, :real_length], scores)
                    token = torch.multinomial(
                        torch.nn.functional.softmax(warped, dim=-1),
                        1,
                        generator=generators[row],
                    ).squeeze(1)
                    private_prefixes[row][:, real_length].copy_(token)
                    lane.lane.static_inputs["decoder_input_ids"].copy_(token.reshape(1, 1))
                    _copy_static_inputs(
                        lane,
                        sources[row][step + 1],
                        copy_decoder_token=False,
                    )
                finish.record(lane.lane.stream)
            finish.synchronize()
            cuda_seconds += start.elapsed_time(finish) / 1000.0
        wall_seconds = time.perf_counter() - wall_started
        for row in range(2):
            lane = resources.lanes[row]
            cache = lane.session.cache_state.cache
            expected = final_states[f"row-{row}"]
            if cache is None:
                raise RuntimeError("same-job B1 control lost candidate cache.")
            matches = bool(
                tensor_sha256(generators[row].get_state())
                == expected["rng_state"]["candidate_sha256"]
                and tensor_sha256(private_prefixes[row])
                == expected["prefix"]["candidate_sha256"]
                and all(
                    tensor_sha256(lane.lane.static_inputs[key])
                    == expected["next_static_inputs"][key]["candidate_sha256"]
                    for key in STATIC_KEYS
                )
                and _cache_hash(
                    cache.self_attention_cache,
                    start=0,
                    stop=FIXED_PROMPT_LENGTH + horizon,
                ) == expected["self_cache"]["candidate_sha256"]
                and _cache_hash(cache.cross_attention_cache)
                == expected["cross_cache"]["candidate_sha256"]
            )
            if not matches:
                raise RuntimeError("same-job private B1 control final state differs from exact decode.")
        tokens = 2 * horizon
        trials.append({
            "trial": trial_index,
            "generated_tokens": tokens,
            "wall_seconds": wall_seconds,
            "cuda_seconds": cuda_seconds,
            "wall_tokens_per_second": tokens / wall_seconds,
            "cuda_tokens_per_second": tokens / cuda_seconds,
            "reset_evidence": copy.deepcopy(reset_contract),
            "initial_rng_state_hashes": copy.deepcopy(
                reset_contract["post_anchor_rng_state_hashes"]
            ),
            "final_state_fingerprints": copy.deepcopy(final_states),
        })
        total_wall += wall_seconds
        total_cuda += cuda_seconds
    generated_tokens = 2 * horizon * trial_count
    return {
        "measurement": "ordinary_private_B1_changing_prefix_control",
        "promotion_gate_uses_control": False,
        "horizon": horizon,
        "trial_count": trial_count,
        "generated_tokens": generated_tokens,
        "wall_seconds": total_wall,
        "cuda_seconds": total_cuda,
        "wall_tokens_per_second": generated_tokens / total_wall,
        "cuda_tokens_per_second": generated_tokens / total_cuda,
        "trials": trials,
    }


def _bridge_evidence(resources: _Resources) -> HybridBridgeResourceEvidence:
    sampling = tuple(
        HybridSamplingResourceEvidence(
            lane_id=row,
            sampling_graph_id=id(tail.graph),
            sampling_graph_pool_id=repr(tail.graph.pool()),
            processed_scores_storage_ptr=_tensor_storage_ptr(tail.processed_scores),
            probabilities_storage_ptr=_tensor_storage_ptr(tail.probabilities),
            sampled_token_storage_ptr=_tensor_storage_ptr(tail.sampled_token),
            generator_id=id(tail.generator),
            logits_warper_id=id(tail.warper),
        )
        for row, tail in enumerate(resources.sampling)
    )
    return HybridBridgeResourceEvidence(
        control_stream_id=int(resources.bridge.stream.cuda_stream),
        processor_graph_id=id(resources.bridge.graph),
        processor_graph_pool_id=repr(resources.bridge.graph.pool()),
        prefix_batch_storage_ptr=_tensor_storage_ptr(resources.bridge.prefix_batch),
        raw_logits_bridge_storage_ptr=_tensor_storage_ptr(resources.bridge.raw_logits_bridge),
        processed_scores_storage_ptr=_tensor_storage_ptr(resources.bridge.processed_scores),
        source_release_event_ids=tuple(id(value) for value in resources.events.source_released),
        lane_ready_event_ids=tuple(id(value) for value in resources.events.lane_ready),
        sample_done_event_ids=tuple(id(value) for value in resources.events.sample_done),
        processor_done_event_id=id(resources.events.processor_done),
        sampling_lanes=sampling,
    )


def _resource_report(resources: _Resources) -> dict[str, Any]:
    bridge = _bridge_evidence(resources)
    validate_hybrid_resource_ownership(
        [lane.ownership for lane in resources.lanes],
        bridge,
    )
    if any(
            id(resources.lanes[row].anchor_generator) == id(resources.sampling[row].generator)
            for row in range(2)
    ):
        raise RuntimeError("anchor and captured sampling generators must be distinct retained objects.")
    return {
        "model_lanes": [lane.ownership.as_dict() for lane in resources.lanes],
        "hybrid_bridge": bridge.as_dict(),
        "source_logit_views": [
            {
                "row": row,
                "graph_output_storage_ptr": _tensor_storage_ptr(lane.lane.graph_outputs.logits),
                "view_storage_ptr": _tensor_storage_ptr(lane.source_logits),
                "zero_copy_alias": (
                    _tensor_storage_ptr(lane.lane.graph_outputs.logits)
                    == _tensor_storage_ptr(lane.source_logits)
                ),
                "captured_lane_logits_property_used": False,
            }
            for row, lane in enumerate(resources.lanes)
        ],
        "state_ready_event_ids": [id(value) for value in resources.events.state_ready],
        "stop_flag_storage_ptrs": [
            _tensor_storage_ptr(value) for value in resources.stop_flags
        ],
        "sampling_eos_token_storage_ptrs": [
            _tensor_storage_ptr(value.eos_token_ids) for value in resources.sampling
        ],
        "sampling_stop_output_storage_ptrs": [
            _tensor_storage_ptr(value.stop_flag) for value in resources.sampling
        ],
        "capture": {
            "model_graph_count": 2,
            "processor_graph_count": 1,
            "private_sampling_and_stop_graph_count": 2,
            "total_graph_count": 5,
            "explicit_generator_registration": True,
            "graph_recapture_between_orders_or_trials": False,
        },
    }


def _build_reset_contract(
        resources: _Resources,
        references: Sequence[_ReferenceLane],
) -> dict[str, Any]:
    restored = _restore_resources(resources)
    anchor_tokens: dict[str, Any] = {}
    anchor_rng: dict[str, Any] = {}
    for row in range(2):
        request_id = f"row-{row}"
        reference_anchor = int(references[row].anchor_token.item())
        candidate_anchor = int(resources.lanes[row].anchor_token.item())
        anchor_tokens[request_id] = {
            "reference": reference_anchor,
            "candidate": candidate_anchor,
            "match": reference_anchor == candidate_anchor,
        }
        anchor_rng[request_id] = _hash_pair(
            references[row].post_anchor_rng_state,
            resources.lanes[row].post_anchor_rng_state,
        )
    return {
        **restored,
        "initial_seed_rng_state_hashes": {
            f"row-{row}": tensor_sha256(lane.initial_seed_rng_state)
            for row, lane in enumerate(resources.lanes)
        },
        "post_anchor_rng_state_hashes": {
            f"row-{row}": tensor_sha256(lane.post_anchor_rng_state)
            for row, lane in enumerate(resources.lanes)
        },
        "anchor_tokens": anchor_tokens,
        "post_anchor_rng_states": anchor_rng,
        "static_input_storage_ptrs": {
            f"row-{row}": dict(lane.ownership.static_input_storage_ptrs)
            for row, lane in enumerate(resources.lanes)
        },
        "self_cache_storage_ptrs": {
            f"row-{row}": list(lane.ownership.self_cache_storage_ptrs)
            for row, lane in enumerate(resources.lanes)
        },
        "cross_cache_storage_ptrs": {
            f"row-{row}": list(lane.ownership.cross_cache_storage_ptrs)
            for row, lane in enumerate(resources.lanes)
        },
        "in_place_restore": True,
        "captured_pointer_swap": False,
        "cache_snapshot_stage": "post_prefill_before_anchor_or_capture",
        "restore_verified_after_graph_capture": True,
    }


def _exact_order_signature(values: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(values))
    payload.pop("graph_and_sampling_order", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


@torch.no_grad()
def _run_phase(
        model: Any,
        resources: _Resources,
        *,
        horizon: int,
        trial_count: int,
        prompt: torch.LongTensor,
        prompt_mask: torch.Tensor,
        frames: torch.Tensor,
        condition_kwargs: Mapping[str, Any],
        base_factory: ProcessorFactory,
        warper_factory: ProcessorFactory,
        combined_factory: ProcessorFactory,
        comparison_top_k: int,
        eos_ids: Sequence[int],
        reset_contract: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    order_payloads: dict[str, Any] = {}
    order_sources: dict[str, Any] = {}
    phase_references: tuple[_ReferenceLane, _ReferenceLane] | None = None
    for order in reciprocal_lane_orders(2):
        key = ",".join(str(row) for row in order)
        exact, sources, references = _run_exact_order(
            model,
            resources,
            order=order,
            horizon=horizon,
            prompt=prompt,
            prompt_mask=prompt_mask,
            frames=frames,
            condition_kwargs=condition_kwargs,
            base_factory=base_factory,
            warper_factory=warper_factory,
            combined_factory=combined_factory,
            comparison_top_k=comparison_top_k,
            eos_ids=eos_ids,
        )
        phase_references = references
        order_payloads[key] = exact
        order_sources[key] = sources
    if phase_references is None:
        raise RuntimeError("changing-prefix phase produced no independent references.")
    if _exact_order_signature(order_payloads["0,1"]) != _exact_order_signature(
            order_payloads["1,0"]
    ):
        raise RuntimeError(
            "changing-prefix reciprocal exact evidence drifted before timing; refusing timing."
        )
    if reset_contract is None:
        reset_contract = _build_reset_contract(resources, phase_references)
    exactness_pass = all(values["exactness_pass"] for values in order_payloads.values())
    if not exactness_pass:
        for values in order_payloads.values():
            values["reset_evidence"] = copy.deepcopy(reset_contract)
            values["initial_rng_state_hashes"] = copy.deepcopy(
                reset_contract["post_anchor_rng_state_hashes"]
            )
            values["complete_timing"] = None
            values["same_job_private_b1_control"] = None
        return ({
            "horizon": horizon,
            "timing_trials_per_order": trial_count,
            "orders": order_payloads,
            "performance": None,
            "exactness_pass": False,
            "target_500_gate_pass": False,
        }, reset_contract)

    timing_inputs: dict[str, Any] = {}
    for order in reciprocal_lane_orders(2):
        key = ",".join(str(row) for row in order)
        values = order_payloads[key]
        values["reset_evidence"] = copy.deepcopy(reset_contract)
        values["initial_rng_state_hashes"] = copy.deepcopy(
            reset_contract["post_anchor_rng_state_hashes"]
        )
        timing = _run_candidate_timing(
            resources,
            order=order,
            sources=order_sources[key],
            horizon=horizon,
            trial_count=trial_count,
            reset_contract=reset_contract,
            final_states=values["final_state_fingerprints"],
        )
        control = _run_private_b1_control(
            resources,
            order=order,
            sources=order_sources[key],
            horizon=horizon,
            trial_count=trial_count,
            reset_contract=reset_contract,
            final_states=values["final_state_fingerprints"],
            base_factory=base_factory,
            warper_factory=warper_factory,
        )
        values["complete_timing"] = timing
        values["same_job_private_b1_control"] = control
        relative = (
            float(timing["wall_tokens_per_second"])
            / float(control["wall_tokens_per_second"])
            - 1.0
        )
        values["relative_gain_vs_same_job_control"] = relative
        values["five_percent_vs_same_job_control"] = relative > 0.05
        timing_inputs[key] = timing
    performance = summarize_changing_prefix_performance(timing_inputs)
    return ({
        "horizon": horizon,
        "timing_trials_per_order": trial_count,
        "orders": order_payloads,
        "performance": performance,
        "exactness_pass": True,
        "target_500_gate_pass": bool(performance["target_500_bar_pass"]),
    }, reset_contract)


@torch.no_grad()
def run_changing_prefix_gate(
        args: Any,
        *,
        config_name: str,
        fixed_hybrid_report: Mapping[str, Any],
        fixed_hybrid_report_path: Path,
        sequence_index: int,
        seeds: tuple[int, int],
        capture_warmup_repeats: int,
        comparison_top_k: int,
        q1_bmm_cross_attention: bool,
        native_q1_self_attention: bool,
        native_q1_rope_cache_self_attention: bool,
) -> dict[str, Any]:
    gate_started = time.perf_counter()
    extension_cache_before = _extension_cache_state()
    validate_hybrid_l2_report(fixed_hybrid_report)
    if _file_sha256(fixed_hybrid_report_path) != FIXED_HYBRID_SOURCE_REPORT_SHA256:
        raise ValueError("fixed hybrid report SHA-256 differs from job 49552768.")
    source_runtime = fixed_hybrid_report.get("runtime_metadata") or {}
    if source_runtime.get("git_commit") != FIXED_HYBRID_SOURCE_COMMIT:
        raise ValueError("fixed hybrid report commit differs from audited source.")
    if sequence_index != 9 or seeds != (12345, 23456):
        raise ValueError("changing-prefix gate is fixed to seq9 and seeds 12345/23456.")
    _validate_fixed_runtime_contract(
        args,
        config_name=config_name,
        q1_bmm_cross_attention=q1_bmm_cross_attention,
        native_q1_self_attention=native_q1_self_attention,
        native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
    )
    _assert_supported_probe(args)
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
    if model.device.type != "cuda" or model.dtype != torch.float32:
        raise RuntimeError("changing-prefix hybrid gate requires a CUDA FP32 model.")
    torch.cuda.reset_peak_memory_stats(model.device)

    model_inputs = _move_kwargs_for_model(
        model,
        _build_probe_inputs(args, model, tokenizer, sequence_index=sequence_index),
    )
    probe = {
        key: model_inputs.pop(key)
        for key in (
            "sequence_index",
            "frame_time_ms",
            "context_type",
            "lookback_time",
            "lookahead_time",
        )
    }
    prompt = model_inputs["decoder_input_ids"]
    prompt_mask = model_inputs["decoder_attention_mask"]
    frames = model_inputs["frames"]
    condition_kwargs = _condition_kwargs(model_inputs)
    condition_hashes = {
        key: tensor_sha256(value)
        for key, value in sorted(condition_kwargs.items())
        if isinstance(value, torch.Tensor)
    }
    observed_probe_hashes = {
        "prompt_sha256": tensor_sha256(prompt),
        "prompt_attention_mask_sha256": tensor_sha256(prompt_mask),
        "frames_sha256": tensor_sha256(frames),
        "condition_tensor_hashes": condition_hashes,
    }
    expected_probe_hashes = {
        "prompt_sha256": REVIEWED_PROMPT_SHA256,
        "prompt_attention_mask_sha256": REVIEWED_PROMPT_ATTENTION_MASK_SHA256,
        "frames_sha256": REVIEWED_FRAMES_SHA256,
        "condition_tensor_hashes": {},
    }
    if observed_probe_hashes != expected_probe_hashes:
        raise ValueError("changing-prefix probe tensors differ from audited fixed hybrid job.")
    if fixed_hybrid_report.get("workload_contract_sha256") != REVIEWED_WORKLOAD_CONTRACT_SHA256:
        raise ValueError("fixed hybrid workload contract hash differs from audited source.")
    if int(prompt.shape[-1]) != FIXED_PROMPT_LENGTH:
        raise ValueError(
            f"changing-prefix gate expected prompt length {FIXED_PROMPT_LENGTH}; "
            f"got {int(prompt.shape[-1])}."
        )
    active_prefix = _bucketed_prefix_length(
        FIXED_PROMPT_LENGTH + 1,
        64,
        int(model.config.max_target_positions),
    )
    if active_prefix != ACTIVE_PREFIX_LENGTH:
        raise ValueError("changing-prefix seq9 did not map to active-prefix bucket 128.")
    context_type = ContextType(probe["context_type"])
    eos_ids = [
        int(value)
        for value in get_eos_token_id(
            tokenizer,
            lookback_time=float(probe["lookback_time"]),
            lookahead_time=float(probe["lookahead_time"]),
            context_type=context_type,
        )
    ]
    base_factory, warper_factory, combined_factory = _make_factories(
        args,
        tokenizer,
        torch.device(model.device),
        lookback_time=float(probe["lookback_time"]),
    )
    native_started = time.perf_counter()
    with torch.autocast(device_type="cuda", enabled=False), generation_profile_context(
            sdpa_backend=args.profile_sdpa_backend,
            q1_bmm_cross_attention=q1_bmm_cross_attention,
            native_q1_self_attention=native_q1_self_attention,
            native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
    ):
        native_context_enter_seconds = time.perf_counter() - native_started
        lane_list = [
            _capture_candidate_lane(
                model,
                row=row,
                seed=seed,
                prompt=prompt,
                prompt_mask=prompt_mask,
                frames=frames,
                condition_kwargs=condition_kwargs,
                combined_factory=combined_factory,
                capture_warmup_repeats=capture_warmup_repeats,
            )
            for row, seed in enumerate(seeds)
        ]
        lanes = (lane_list[0], lane_list[1])
        bridge = _capture_processor_bridge(
            lanes,
            base_factory=base_factory,
            capture_warmup_repeats=capture_warmup_repeats,
        )
        sampling_list = [
            _capture_sampling_tail(
                lanes[row],
                warper_factory=warper_factory,
                eos_ids=eos_ids,
                capture_warmup_repeats=capture_warmup_repeats,
            )
            for row in range(2)
        ]
        resources = _build_resources(
            lanes,
            bridge,
            (sampling_list[0], sampling_list[1]),
        )
        setup_peak_allocated = torch.cuda.max_memory_allocated(model.device)
        setup_peak_reserved = torch.cuda.max_memory_reserved(model.device)
        phase_a_started = time.perf_counter()
        phase_a, reset_contract = _run_phase(
            model,
            resources,
            horizon=PHASE_A_HORIZON,
            trial_count=1,
            prompt=prompt,
            prompt_mask=prompt_mask,
            frames=frames,
            condition_kwargs=condition_kwargs,
            base_factory=base_factory,
            warper_factory=warper_factory,
            combined_factory=combined_factory,
            comparison_top_k=comparison_top_k,
            eos_ids=eos_ids,
            reset_contract=None,
        )
        phase_a_wall_seconds = time.perf_counter() - phase_a_started
        if phase_a["exactness_pass"] and phase_a["target_500_gate_pass"]:
            phase_b_started = time.perf_counter()
            phase_b, _ = _run_phase(
                model,
                resources,
                horizon=PHASE_B_HORIZON,
                trial_count=PHASE_B_TIMING_TRIALS,
                prompt=prompt,
                prompt_mask=prompt_mask,
                frames=frames,
                condition_kwargs=condition_kwargs,
                base_factory=base_factory,
                warper_factory=warper_factory,
                combined_factory=combined_factory,
                comparison_top_k=comparison_top_k,
                eos_ids=eos_ids,
                reset_contract=reset_contract,
            )
            phase_b_wall_seconds = time.perf_counter() - phase_b_started
        else:
            phase_b = None
            phase_b_wall_seconds = None

    resource_report = _resource_report(resources)
    base_processor = base_factory()
    monotonic = base_processor[0]
    processor_contract = {
        "shared_base_processor_classes": list(processor_class_names(base_processor)),
        "private_warper_classes": list(processor_class_names(warper_factory())),
        "neutral_pad_token_id": int(tokenizer.pad_id),
        "sos_token_ids": [int(value) for value in monotonic.sos_ids.tolist()],
        "time_shift_start": int(monotonic.time_shift_start),
        "time_shift_end": int(monotonic.time_shift_end),
    }
    pass_gate = bool(
        phase_a["exactness_pass"]
        and phase_a["target_500_gate_pass"]
        and isinstance(phase_b, Mapping)
        and phase_b["exactness_pass"]
        and phase_b["target_500_gate_pass"]
    )
    extension_cache_after = _extension_cache_state()
    current_allocated = torch.cuda.memory_allocated(model.device)
    current_reserved = torch.cuda.memory_reserved(model.device)
    final_peak_allocated = torch.cuda.max_memory_allocated(model.device)
    final_peak_reserved = torch.cuda.max_memory_reserved(model.device)
    setup_metadata = {
        "excluded_from_complete_decode_timing": True,
        "lane_setup": [
            {
                "row": row,
                "prefill_wall_seconds": lane.prefill_wall_seconds,
                "prefill_cuda_seconds": lane.prefill_cuda_seconds,
                "one_token_setup_wall_seconds": lane.setup_wall_seconds,
                "one_token_setup_cuda_seconds": lane.setup_cuda_seconds,
                "model_graph_warmup_seconds": lane.lane.warmup_seconds,
                "model_graph_capture_seconds": lane.lane.capture_seconds,
                "model_graph_capture_allocated_delta_bytes": (
                    lane.lane.capture_memory_allocated_delta_bytes
                ),
                "model_graph_capture_reserved_delta_bytes": (
                    lane.lane.capture_memory_reserved_delta_bytes
                ),
            }
            for row, lane in enumerate(resources.lanes)
        ],
        "processor_graph_warmup_seconds": resources.bridge.warmup_seconds,
        "processor_graph_capture_seconds": resources.bridge.capture_seconds,
        "sampling_graphs": [
            {
                "row": row,
                "warmup_seconds": tail.warmup_seconds,
                "capture_seconds": tail.capture_seconds,
            }
            for row, tail in enumerate(resources.sampling)
        ],
        "graph_count": 5,
        "phase_a_total_wall_seconds": phase_a_wall_seconds,
        "phase_b_total_wall_seconds": phase_b_wall_seconds,
        "total_gate_wall_seconds": time.perf_counter() - gate_started,
        "memory_current_allocated_bytes": current_allocated,
        "memory_current_reserved_bytes": current_reserved,
        "memory_peak_allocated_bytes": max(setup_peak_allocated, final_peak_allocated),
        "memory_peak_reserved_bytes": max(setup_peak_reserved, final_peak_reserved),
        "torch_extensions_cache_before": extension_cache_before,
        "torch_extensions_cache_after": extension_cache_after,
    }
    report = {
        "schema_version": CHANGING_PREFIX_SCHEMA_VERSION,
        "gate": CHANGING_PREFIX_GATE,
        "lane_count": 2,
        "pass": pass_gate,
        "fixed_hybrid_source": {
            "job_id": FIXED_HYBRID_SOURCE_JOB,
            "commit": FIXED_HYBRID_SOURCE_COMMIT,
            "report_sha256": FIXED_HYBRID_SOURCE_REPORT_SHA256,
            "worst_order_tokens_per_second": FIXED_HYBRID_SOURCE_WORST_TPS,
        },
        "fixed_workload_contract": {
            "phase_a_horizon": PHASE_A_HORIZON,
            "phase_b_horizon": PHASE_B_HORIZON,
            "phase_b_timing_trials": PHASE_B_TIMING_TRIALS,
            "active_prefix_length": ACTIVE_PREFIX_LENGTH,
            "prompt_length": FIXED_PROMPT_LENGTH,
            "neutral_pad_token_id": NEUTRAL_PAD_TOKEN_ID,
            "model_path": "OliBomby/Mapperatorinator-v32",
            "precision": "fp32",
            "attn_implementation": "sdpa",
            "sequence_index": 9,
            "row_seeds": list(seeds),
            "processor_prefix_shape": [2, ACTIVE_PREFIX_LENGTH],
            "reference_prefix_policy": "private_unpadded_B1",
            "candidate_prefix_policy": "shared_neutral_padded_B2",
            **observed_probe_hashes,
            "source_workload_contract_sha256": REVIEWED_WORKLOAD_CONTRACT_SHA256,
        },
        "event_order": list(CHANGING_PREFIX_EVENT_ORDER),
        "resource_ownership_pass": True,
        "capture_pass": True,
        "pointer_stability_pass": True,
        "reset_in_place_pass": True,
        "processor_family_pass": True,
        "processor_contract": processor_contract,
        "resource_ownership": resource_report,
        "reset_contract": reset_contract,
        "setup_metadata": setup_metadata,
        "phase_a": phase_a,
        "phase_b": phase_b,
        "graduation_scope": {
            "production_bucket128_decode_tokens": PRODUCTION_BUCKET128_DECODE_TOKENS,
            "production_total_decode_tokens": PRODUCTION_TOTAL_DECODE_TOKENS,
            "production_bucket128_fraction": (
                PRODUCTION_BUCKET128_DECODE_TOKENS / PRODUCTION_TOTAL_DECODE_TOKENS
            ),
            "authorized_next_step": "weighted_active_prefix_bucket_sweep",
            "scheduler_or_runtime_wiring_authorized": False,
        },
        "runtime_metadata": {
            "git_commit": _git_commit(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(model.device),
            "gpu_capability": list(torch.cuda.get_device_capability(model.device)),
            "native_context_enter_seconds": native_context_enter_seconds,
            "probe": probe,
            "eos_token_ids": eos_ids,
            "torch_extensions_dir": os.environ.get("TORCH_EXTENSIONS_DIR"),
        },
    }
    validate_changing_prefix_report(report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="profile_salvalai_smoke15")
    parser.add_argument("--sequence-index", type=int, default=9)
    parser.add_argument("--fixed-hybrid-report", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--row-seed", action="append", type=int, default=[])
    parser.add_argument("--capture-warmup-repeats", type=int, default=1)
    parser.add_argument("--comparison-top-k", type=int, default=20)
    parser.add_argument("--q1-bmm-cross-attention", action="store_true")
    parser.add_argument("--native-q1-self-attention", action="store_true")
    parser.add_argument("--native-q1-rope-cache-self-attention", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    seeds = tuple(cli.row_seed) if cli.row_seed else (12345, 23456)
    if len(seeds) != 2:
        raise ValueError("changing-prefix gate requires exactly two row seeds.")
    fixed_report = json.loads(cli.fixed_hybrid_report.read_text(encoding="utf-8"))
    args = _load_args(cli.config_name, cli.overrides)
    started = time.perf_counter()
    report = run_changing_prefix_gate(
        args,
        config_name=cli.config_name,
        fixed_hybrid_report=fixed_report,
        fixed_hybrid_report_path=cli.fixed_hybrid_report,
        sequence_index=cli.sequence_index,
        seeds=(int(seeds[0]), int(seeds[1])),
        capture_warmup_repeats=cli.capture_warmup_repeats,
        comparison_top_k=cli.comparison_top_k,
        q1_bmm_cross_attention=cli.q1_bmm_cross_attention,
        native_q1_self_attention=cli.native_q1_self_attention,
        native_q1_rope_cache_self_attention=cli.native_q1_rope_cache_self_attention,
    )
    report["runtime_metadata"]["config_name"] = cli.config_name
    report["runtime_metadata"]["hydra_overrides"] = list(cli.overrides)
    report["total_wall_seconds"] = time.perf_counter() - started
    validate_changing_prefix_report(report)
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
