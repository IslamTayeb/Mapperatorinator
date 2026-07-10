"""Run the verifier-only real-prefix bucket-576 Hybrid-B2 H8 scout."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import hashlib
import json
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
from transformers import LogitsProcessorList

from inference import compile_args, load_model_with_server, setup_inference_environment
from osuT5.osuT5.inference.decode_loop import _bucketed_prefix_length
from osuT5.osuT5.inference.direct_decode import DecodeSession
from osuT5.osuT5.inference.optimized.batch.hybrid_l2 import (
    HybridBridgeResourceEvidence,
    HybridSamplingResourceEvidence,
    validate_hybrid_resource_ownership,
)
from osuT5.osuT5.inference.optimized.batch.lane_one_token import (
    DEFAULT_SEPARATE_GRAPH_POOL,
    PYTORCH_PER_STREAM_CUBLAS_WORKSPACE,
    CapturedB1Lane,
    LaneResourceEvidence,
    capture_prepared_b1_lane,
    reciprocal_lane_orders,
    tensor_sha256,
)
from osuT5.osuT5.inference.optimized.batch.weighted_bucket import (
    BASE_TIMING_TOKENIZER_SCOPE,
    BASE_TIMING_VOCAB_SIZE,
    MAIN_TOKENIZER_SCOPE,
    MAIN_VOCAB_SIZE,
    PRE_TARGET_MAIN_TRANSCRIPT_TOKENS,
    REVIEWED_SOURCE_CAPTURE_COMMIT,
    REVIEWED_SOURCE_CAPTURE_JOB_ID,
    REVIEWED_SOURCE_CONTRACT_SHA256,
    REVIEWED_SOURCE_REPORT_FILE_SHA256,
    SOURCE_PREFIX_REPLAY_TOKENS,
    SOURCE_PROMPT_LENGTH,
    TIMING_TRANSCRIPT_TOKENS,
    load_reviewed_source_capture_report,
    validate_reviewed_context_evidence,
    validate_model_free_ceiling_report,
)
from osuT5.osuT5.inference.optimized.batch.weighted_h8 import (
    ACTIVE_PREFIX_LENGTH,
    EVENT_ORDER,
    HORIZON,
    NEUTRAL_PAD_TOKEN_ID,
    REQUIRED_WEIGHTED_COMPLETE_TPS,
    ROW_FIRST_OUTPUT_OFFSETS,
    ROW_INITIAL_CACHE_POSITIONS,
    ROW_INITIAL_PREFIX_LENGTHS,
    ROW_SEED,
    ROW_SOURCE_OFFSETS,
    TIMING_TRIALS_PER_ORDER,
    WEIGHTED_CEILING_REPORT_SHA256,
    WEIGHTED_H8_GATE,
    WEIGHTED_H8_SCHEMA_VERSION,
    summarize_weighted_h8_performance,
    validate_weighted_h8_report,
)
from osuT5.osuT5.inference.server import get_eos_token_id
from osuT5.osuT5.runtime_profiling import generation_profile_context
from osuT5.osuT5.tokenizer import ContextType, Tokenizer
from utils.verify_one_token_decode import (
    _assert_supported_probe,
    _condition_kwargs,
    _load_args,
)
from utils.verify_optimized_b1_lane_capture import (
    _cache_tensor_ptrs,
    _cache_tensor_values,
    _new_generator,
    _sample_next_token,
    _tensor_storage_ptr,
    _top_level_tensor_ptrs,
)
from utils.verify_optimized_hybrid_changing_prefix import (
    STATIC_KEYS,
    _cache_comparison,
    _cache_hash,
    _cache_slot_nonzero,
    _hash_pair,
    _logit_comparison,
    _restore_cache,
    _snapshot_cache,
    _snapshot_slice_hash,
    _snapshot_tuple_hash,
    _capture_sampling_tail,
)
from utils.verify_optimized_hybrid_l2 import (
    _file_sha256,
    _make_factories,
    _validate_fixed_runtime_contract,
)
from utils.verify_optimized_weighted_prefix_source import (
    _advance_generator_segment,
    _build_context_evidence,
    _extension_cache_state,
    _load_source_evidence,
    _reconstruct_target_inputs,
    _validate_tokenizer_scopes,
)


ProcessorFactory = Callable[[], Any]


@dataclass
class _ReachableState:
    row: int
    source_offset: int
    session: DecodeSession
    prefix: torch.LongTensor
    mask: torch.Tensor
    generator: torch.Generator
    initial_rng_state: torch.Tensor
    base_processor: Any
    warper: Any
    capture_inputs: dict[str, Any]
    prepared_inputs: dict[str, torch.Tensor]
    last_sample_raw_logits: torch.Tensor
    self_snapshot: tuple[torch.Tensor, ...]
    cross_snapshot: tuple[torch.Tensor, ...]
    starting_rng_state: torch.Tensor
    first_output_offset: int
    initial_cache_position: int
    source_handoff_match: bool


@dataclass
class _CandidateLane:
    row: int
    seed: int
    source_offset: int
    session: DecodeSession
    lane: CapturedB1Lane
    source_logits: torch.Tensor
    full_prefix: torch.LongTensor
    full_mask: torch.Tensor
    initial_rng_state: torch.Tensor
    post_anchor_rng_state: torch.Tensor
    anchor_generator: torch.Generator
    anchor_processor: Any
    self_snapshot: tuple[torch.Tensor, ...]
    cross_snapshot: tuple[torch.Tensor, ...]
    ownership: LaneResourceEvidence
    first_output_offset: int
    initial_cache_position: int
    source_handoff_match: bool


@dataclass
class _ProcessorBridge:
    stream: torch.cuda.Stream
    graph: torch.cuda.CUDAGraph
    processor: Any
    prefix_batch: torch.LongTensor
    initial_prefix_batch: torch.LongTensor
    raw_logits: torch.Tensor
    raw_rows: tuple[torch.Tensor, torch.Tensor]
    processed_scores: torch.Tensor
    processed_rows: tuple[torch.Tensor, torch.Tensor]
    warmup_seconds: float
    capture_seconds: float


@dataclass
class _PrivateProcessor:
    row: int
    stream: torch.cuda.Stream
    graph: torch.cuda.CUDAGraph
    processor: Any
    output_prefix: torch.LongTensor
    initial_output_prefix: torch.LongTensor
    processor_input: torch.LongTensor
    raw_logits: torch.Tensor
    processed_scores: torch.Tensor
    monotonic_input_has_time_shift: torch.Tensor
    monotonic_input_last_time_shift_value: torch.Tensor
    monotonic_output_has_time_shift: torch.Tensor
    monotonic_output_last_time_shift_value: torch.Tensor
    reset_has_time_shift: torch.Tensor
    reset_last_time_shift_value: torch.Tensor
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
    shared: _ProcessorBridge
    private: tuple[_PrivateProcessor, _PrivateProcessor]
    sampling: tuple[Any, Any]
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


def _new_event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=False, blocking=False, interprocess=False)


def _padded_prefix(prefix: torch.LongTensor) -> torch.LongTensor:
    result = torch.full(
        (1, ACTIVE_PREFIX_LENGTH),
        NEUTRAL_PAD_TOKEN_ID,
        dtype=prefix.dtype,
        device=prefix.device,
    )
    result[:, :prefix.shape[-1]].copy_(prefix)
    return result


def _prepare_full_inputs(state: _ReachableState) -> dict[str, Any]:
    one = state.session.one_token_state()
    return state.session.model.prepare_inputs_for_generation(
        state.prefix,
        past_key_values=one.cache,
        use_cache=True,
        encoder_outputs=one.encoder_outputs,
        decoder_attention_mask=state.mask,
        cache_position=torch.arange(
            state.initial_cache_position,
            state.initial_cache_position + 1,
            device=state.prefix.device,
        ),
        **state.session.condition_kwargs,
    )


def _static_inputs(prepared: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key in STATIC_KEYS:
        value = prepared.get(key)
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"weighted H8 preparation lost {key}.")
        result[key] = value.clone(memory_format=torch.contiguous_format)
    return result


def _prepare_static(state: _ReachableState) -> dict[str, torch.Tensor]:
    return _static_inputs(_prepare_full_inputs(state))


def _reachable_state(
        model: Any,
        model_inputs: Mapping[str, Any],
        target_ids: Sequence[int],
        *,
        row: int,
        source_offset: int,
        base_factory: ProcessorFactory,
        warper_factory: ProcessorFactory,
        combined_factory: ProcessorFactory,
        eos_ids: Sequence[int],
        reviewed_source: Mapping[str, Any],
) -> _ReachableState:
    prompt = model_inputs["decoder_input_ids"]
    prompt_mask = model_inputs["decoder_attention_mask"]
    frames = model_inputs["frames"]
    condition_kwargs = _condition_kwargs(dict(model_inputs))
    session = DecodeSession.prefill(
        model,
        prompt=prompt,
        prompt_attention_mask=prompt_mask,
        frames=frames,
        condition_kwargs=condition_kwargs,
        active_prefix_self_attention=False,
    )
    raw = session.one_token_state().prefill_logits
    if raw is None:
        raise RuntimeError("weighted H8 reachable state lost prefill logits.")
    generator = _new_generator(torch.device(model.device), ROW_SEED)
    initial_rng = generator.get_state().clone()
    _advance_generator_segment(
        generator,
        device=torch.device(model.device),
        scope=BASE_TIMING_TOKENIZER_SCOPE,
        vocab_size=BASE_TIMING_VOCAB_SIZE,
        draw_calls=TIMING_TRANSCRIPT_TOKENS,
    )
    _advance_generator_segment(
        generator,
        device=torch.device(model.device),
        scope=MAIN_TOKENIZER_SCOPE,
        vocab_size=MAIN_VOCAB_SIZE,
        draw_calls=PRE_TARGET_MAIN_TRANSCRIPT_TOKENS,
    )
    combined = combined_factory()
    base_count = len(base_factory())
    base_processor = LogitsProcessorList(list(combined[:base_count]))
    warper = LogitsProcessorList(list(combined[base_count:]))
    if len(warper) != len(warper_factory()):
        raise RuntimeError("weighted H8 processor split changed.")
    prefix = prompt.clone(memory_format=torch.contiguous_format)
    mask = prompt_mask.clone(memory_format=torch.contiguous_format)
    replay_count = SOURCE_PREFIX_REPLAY_TOKENS + source_offset
    eos = set(int(value) for value in eos_ids)
    for step in range(replay_count):
        token = _sample_next_token(
            input_ids=prefix,
            raw_logits=raw,
            logits_processor=combined,
            generator=generator,
            do_sample=True,
        )
        observed = int(token.item())
        if observed != int(target_ids[step]) or observed in eos:
            raise RuntimeError(
                f"weighted H8 row {row} source replay diverged at token {step}."
            )
        prefix = torch.cat([prefix, token[:, None]], dim=-1)
        mask = torch.cat(
            [mask, torch.ones_like(token[:, None], dtype=mask.dtype)], dim=-1
        )
        if step < replay_count - 1:
            position = SOURCE_PROMPT_LENGTH + step
            bucket = _bucketed_prefix_length(
                position + 1,
                64,
                int(model.config.max_target_positions),
            )
            result = session.decode_one_token_raw_logits(
                full_prefix=prefix,
                full_attention_mask=mask,
                cache_position=torch.arange(
                    position, position + 1, device=prefix.device
                ),
                active_prefix_self_attention=True,
                active_prefix_self_attention_length=bucket,
            )
            raw = result.logits
    cache = session.cache_state.cache
    if cache is None:
        raise RuntimeError("weighted H8 reachable state lost its cache.")
    state = _ReachableState(
        row=row,
        source_offset=source_offset,
        session=session,
        prefix=prefix,
        mask=mask,
        generator=generator,
        initial_rng_state=initial_rng,
        base_processor=base_processor,
        warper=warper,
        capture_inputs={},
        prepared_inputs={},
        last_sample_raw_logits=raw,
        self_snapshot=_snapshot_cache(cache.self_attention_cache),
        cross_snapshot=_snapshot_cache(cache.cross_attention_cache),
        starting_rng_state=generator.get_state().clone(),
        first_output_offset=SOURCE_PREFIX_REPLAY_TOKENS + source_offset,
        initial_cache_position=SOURCE_PROMPT_LENGTH + replay_count - 1,
        source_handoff_match=False,
    )
    state.capture_inputs = _prepare_full_inputs(state)
    state.prepared_inputs = _static_inputs(state.capture_inputs)
    if int(prefix.shape[-1]) != ROW_INITIAL_PREFIX_LENGTHS[row]:
        raise RuntimeError(f"weighted H8 row {row} prefix length changed.")
    if state.initial_cache_position != ROW_INITIAL_CACHE_POSITIONS[row]:
        raise RuntimeError(f"weighted H8 row {row} cache position changed.")
    if source_offset == 0:
        reconstruction = reviewed_source["reconstruction"]
        observed = {
            "capture_prompt": tensor_sha256(prefix),
            "capture_prompt_attention_mask": tensor_sha256(mask),
            "post_replay_rng_state": tensor_sha256(state.starting_rng_state),
            "pre_last_sample_raw_logits": tensor_sha256(raw),
            "self_cache": _cache_hash(cache.self_attention_cache),
            "cross_cache": _cache_hash(cache.cross_attention_cache),
            "prepared_static_inputs": {
                key: tensor_sha256(value)
                for key, value in sorted(state.prepared_inputs.items())
            },
        }
        expected = {
            "capture_prompt": reconstruction["capture_prompt"]["sha256"],
            "capture_prompt_attention_mask": reconstruction[
                "capture_prompt_attention_mask"
            ]["sha256"],
            "post_replay_rng_state": reconstruction["post_replay_rng_state"][
                "sha256"
            ],
            "pre_last_sample_raw_logits": reconstruction[
                "pre_last_sample_raw_logits"
            ]["sha256"],
            "self_cache": reconstruction["pre_next_forward_cache"]["self_sha256"],
            "cross_cache": reconstruction["pre_next_forward_cache"]["cross_sha256"],
            "prepared_static_inputs": {
                key: value["sha256"]
                for key, value in sorted(
                    reconstruction["prepared_static_inputs"].items()
                )
            },
        }
        state.source_handoff_match = observed == expected
        if not state.source_handoff_match:
            raise RuntimeError(
                f"weighted H8 live source handoff differs: {observed} != {expected}."
            )
    else:
        expected_tail = torch.tensor(
            list(target_ids[:replay_count]),
            dtype=prefix.dtype,
            device=prefix.device,
        ).unsqueeze(0)
        prepared = state.prepared_inputs
        state.source_handoff_match = bool(
            torch.equal(prefix[:, -replay_count:], expected_tail)
            and int(prepared["cache_position"].item())
            == ROW_INITIAL_CACHE_POSITIONS[row]
            and int(prepared["decoder_input_ids"].item())
            == int(target_ids[replay_count - 1])
            and _cache_slot_nonzero(
                cache.self_attention_cache,
                ROW_INITIAL_CACHE_POSITIONS[row] - 1,
            )
            and all(
                not bool(torch.count_nonzero(
                    value[..., ROW_INITIAL_CACHE_POSITIONS[row], :]
                ).item())
                for value in _cache_tensor_values(cache.self_attention_cache)
            )
        )
        if not state.source_handoff_match:
            raise RuntimeError("weighted H8 staggered row derivation changed.")
    return state


def _capture_candidate(model: Any, state: _ReachableState, warmup: int) -> _CandidateLane:
    if (
            state.capture_inputs.get("encoder_outputs") is None
            or state.capture_inputs.get("past_key_values") is None
            or state.capture_inputs.get("decoder_input_ids") is None
    ):
        raise RuntimeError(
            "weighted H8 capture requires complete prepared decoder inputs "
            "including encoder_outputs and past_key_values."
        )
    lane = capture_prepared_b1_lane(
        model,
        state.capture_inputs,
        active_prefix_length=ACTIVE_PREFIX_LENGTH,
        capture_warmup_repeats=warmup,
    )
    cache = state.session.cache_state.cache
    encoder = state.session.encoder_state.encoder_outputs
    if cache is None or encoder is None:
        raise RuntimeError("weighted H8 candidate lost cache or encoder output.")
    source_logits = lane.graph_outputs.logits[:, -1, :]
    with torch.cuda.stream(lane.stream):
        _restore_cache(cache.self_attention_cache, state.self_snapshot)
        _restore_cache(cache.cross_attention_cache, state.cross_snapshot)
    lane.stream.synchronize()
    stream_id = int(lane.stream.cuda_stream)
    ownership = LaneResourceEvidence(
        lane_id=state.row,
        shared_model_id=id(model),
        stream_id=stream_id,
        graph_id=id(lane.graph),
        graph_pool_id=repr(lane.graph.pool()),
        graph_pool_policy=DEFAULT_SEPARATE_GRAPH_POOL,
        graph_pool_sharing_requested=False,
        session_id=id(state.session),
        cache_id=id(cache),
        encoder_output_storage_ptr=_tensor_storage_ptr(encoder.last_hidden_state),
        generator_id=id(state.generator),
        logits_processor_id=id(state.base_processor),
        static_input_storage_ptrs=_top_level_tensor_ptrs(lane.static_inputs),
        self_cache_storage_ptrs=_cache_tensor_ptrs(cache.self_attention_cache),
        cross_cache_storage_ptrs=_cache_tensor_ptrs(cache.cross_attention_cache),
        cublas_workspace_policy=PYTORCH_PER_STREAM_CUBLAS_WORKSPACE,
        cublas_workspace_owner_stream_id=stream_id,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )
    return _CandidateLane(
        row=state.row,
        seed=ROW_SEED,
        source_offset=state.source_offset,
        session=state.session,
        lane=lane,
        source_logits=source_logits,
        full_prefix=state.prefix,
        full_mask=state.mask,
        initial_rng_state=state.initial_rng_state,
        post_anchor_rng_state=state.starting_rng_state,
        anchor_generator=state.generator,
        anchor_processor=state.base_processor,
        self_snapshot=state.self_snapshot,
        cross_snapshot=state.cross_snapshot,
        ownership=ownership,
        first_output_offset=state.first_output_offset,
        initial_cache_position=state.initial_cache_position,
        source_handoff_match=state.source_handoff_match,
    )


def _capture_shared(
        lanes: Sequence[_CandidateLane],
        base_factory: ProcessorFactory,
        warmup: int,
) -> _ProcessorBridge:
    device = lanes[0].full_prefix.device
    prefix = torch.full(
        (2, ACTIVE_PREFIX_LENGTH), NEUTRAL_PAD_TOKEN_ID,
        dtype=torch.long, device=device,
    )
    for row, lane in enumerate(lanes):
        prefix[row:row + 1, :lane.full_prefix.shape[-1]].copy_(lane.full_prefix)
    initial = prefix.clone(memory_format=torch.contiguous_format)
    raw = torch.empty((2, int(lanes[0].source_logits.shape[-1])), device=device)
    raw_rows = (raw[0:1], raw[1:2])
    for row in range(2):
        raw_rows[row].copy_(lanes[row].source_logits)
    processor = base_factory()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    started = time.perf_counter()
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            processor(prefix, raw)
    stream.synchronize()
    warmup_seconds = time.perf_counter() - started
    graph = torch.cuda.CUDAGraph()
    started = time.perf_counter()
    with torch.cuda.graph(graph, stream=stream):
        processed = processor(prefix, raw)
    capture_seconds = time.perf_counter() - started
    stream.synchronize()
    return _ProcessorBridge(
        stream, graph, processor, prefix, initial, raw, raw_rows, processed,
        (processed[0:1], processed[1:2]), warmup_seconds, capture_seconds,
    )


def _capture_private_processor(
        lane: _CandidateLane,
        base_factory: ProcessorFactory,
        warmup: int,
) -> _PrivateProcessor:
    stream = lane.lane.stream
    device = lane.full_prefix.device
    current_stream = torch.cuda.current_stream(device)
    output_prefix = _padded_prefix(lane.full_prefix)
    initial_output = output_prefix.clone(memory_format=torch.contiguous_format)
    processor_input = torch.full_like(output_prefix, NEUTRAL_PAD_TOKEN_ID)
    # Prime the stateful B1 processor on a neutral-padded length-575 history.
    # The captured length-576 graph then executes only the one-token update
    # path, with the live current token transported in the final slot.
    history_length = int(lane.full_prefix.shape[-1]) - 1
    processor_input[:, :history_length].copy_(
        lane.full_prefix[:, :history_length]
    )
    processor_input[:, -1].copy_(lane.full_prefix[:, -1])
    raw = torch.empty_like(lane.source_logits)
    raw.copy_(lane.source_logits)
    warm_processor = base_factory()
    warm_scores = torch.zeros_like(raw)
    # These tensors were produced on the caller's current stream.  The
    # private graph reuses the lane's non-default stream, so bind the first
    # consumer to those writes explicitly.
    stream.wait_stream(current_stream)
    with torch.cuda.stream(stream):
        started = time.perf_counter()
        for _ in range(warmup):
            warm_processor(processor_input[:, :-1], warm_scores)
            warm_processor(processor_input, warm_scores)
    stream.synchronize()
    warmup_seconds = time.perf_counter() - started
    processor = base_factory()
    prime_scores = torch.zeros_like(raw)
    stream.wait_stream(current_stream)
    with torch.cuda.stream(stream):
        processor(processor_input[:, :-1], prime_scores)
    stream.synchronize()
    monotonic = processor[0]
    input_has = monotonic._state_has_time_shift
    input_last = monotonic._state_last_time_shift_value
    if input_has is None or input_last is None:
        raise RuntimeError("weighted H8 private processor priming lost state.")
    reset_has = input_has.clone()
    reset_last = input_last.clone()
    # Reset snapshots are cloned on the current stream and consumed by the
    # lane stream after capture.
    stream.wait_stream(current_stream)
    graph = torch.cuda.CUDAGraph()
    started = time.perf_counter()
    with torch.cuda.graph(graph, stream=stream):
        processed = processor(processor_input, raw)
    capture_seconds = time.perf_counter() - started
    stream.synchronize()
    if (
            monotonic._state_has_time_shift is None
            or monotonic._state_last_time_shift_value is None
    ):
        raise RuntimeError("weighted H8 private processor lost monotonic state.")
    with torch.cuda.stream(stream):
        input_has.copy_(reset_has)
        input_last.copy_(reset_last)
        monotonic._state_has_time_shift.copy_(reset_has)
        monotonic._state_last_time_shift_value.copy_(reset_last)
    stream.synchronize()
    return _PrivateProcessor(
        lane.row, stream, graph, processor, output_prefix, initial_output,
        processor_input, raw, processed,
        input_has, input_last,
        monotonic._state_has_time_shift,
        monotonic._state_last_time_shift_value,
        reset_has, reset_last, warmup_seconds, capture_seconds,
    )


def _build_resources(
        lanes: tuple[_CandidateLane, _CandidateLane],
        shared: _ProcessorBridge,
        private: tuple[_PrivateProcessor, _PrivateProcessor],
        sampling: tuple[Any, Any],
) -> _Resources:
    device = shared.prefix_batch.device
    stop_flags = (
        torch.zeros((), dtype=torch.bool, device=device),
        torch.zeros((), dtype=torch.bool, device=device),
    )
    return _Resources(
        lanes=lanes,
        shared=shared,
        private=private,
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
                shared.prefix_batch[
                    row:row + 1,
                    ROW_INITIAL_PREFIX_LENGTHS[row] + step,
                ]
                for step in range(HORIZON)
            )
            for row in range(2)
        ),
    )


def _copy_static(lane: _CandidateLane, source: Mapping[str, torch.Tensor], token: bool) -> None:
    for key in STATIC_KEYS:
        if key == "decoder_input_ids" and not token:
            continue
        target = lane.lane.static_inputs[key]
        value = source[key]
        if target.shape != value.shape or target.dtype != value.dtype:
            raise RuntimeError(f"weighted H8 static input {key} changed.")
        target.copy_(value)


def _restore(resources: _Resources) -> None:
    for row, lane in enumerate(resources.lanes):
        cache = lane.session.cache_state.cache
        if cache is None:
            raise RuntimeError("weighted H8 candidate lost cache before reset.")
        with torch.cuda.stream(lane.lane.stream):
            _restore_cache(cache.self_attention_cache, lane.self_snapshot)
            _restore_cache(cache.cross_attention_cache, lane.cross_snapshot)
            private = resources.private[row]
            private.monotonic_input_has_time_shift.copy_(private.reset_has_time_shift)
            private.monotonic_input_last_time_shift_value.copy_(
                private.reset_last_time_shift_value
            )
            private.monotonic_output_has_time_shift.copy_(private.reset_has_time_shift)
            private.monotonic_output_last_time_shift_value.copy_(
                private.reset_last_time_shift_value
            )
        resources.sampling[row].generator.set_state(lane.post_anchor_rng_state)
    for lane in resources.lanes:
        lane.lane.stream.synchronize()


def _initialize_candidate(
        resources: _Resources,
        sources: Sequence[Sequence[Mapping[str, torch.Tensor]]],
) -> None:
    with torch.cuda.stream(resources.shared.stream):
        resources.shared.prefix_batch.copy_(resources.shared.initial_prefix_batch)
        for flag in resources.stop_flags:
            flag.zero_()
        for row, lane in enumerate(resources.lanes):
            _copy_static(lane, sources[row][0], True)
            resources.events.state_ready[row].record(resources.shared.stream)


def _enqueue_candidate_step(
        resources: _Resources,
        *,
        order: Sequence[int],
        step: int,
        next_sources: Sequence[Mapping[str, torch.Tensor]],
        synchronize: bool,
) -> None:
    events = resources.events
    shared = resources.shared
    for row in order:
        lane = resources.lanes[row]
        with torch.cuda.stream(lane.lane.stream):
            lane.lane.stream.wait_event(events.state_ready[row])
            if step > 0:
                lane.lane.stream.wait_event(events.source_released[row])
            lane.lane.graph.replay()
            events.lane_ready[row].record(lane.lane.stream)
    with torch.cuda.stream(shared.stream):
        for row in order:
            shared.stream.wait_event(events.lane_ready[row])
        for row in order:
            shared.raw_rows[row].copy_(resources.lanes[row].source_logits)
            events.source_released[row].record(shared.stream)
        shared.graph.replay()
        for row in order:
            resources.sampling[row].processed_scores.copy_(shared.processed_rows[row])
        events.processor_done.record(shared.stream)
    for row in order:
        tail = resources.sampling[row]
        with torch.cuda.stream(tail.stream):
            tail.stream.wait_event(events.processor_done)
            tail.graph.replay()
            events.sample_done[row].record(tail.stream)
    with torch.cuda.stream(shared.stream):
        for row in order:
            shared.stream.wait_event(events.sample_done[row])
        for row in order:
            tail = resources.sampling[row]
            lane = resources.lanes[row]
            resources.stop_flags[row].logical_or_(tail.stop_flag)
            resources.prefix_write_views[row][step].copy_(tail.sampled_token)
            lane.lane.static_inputs["decoder_input_ids"].copy_(tail.sampled_token_2d)
            _copy_static(lane, next_sources[row], False)
            events.state_ready[row].record(shared.stream)
    if synchronize:
        shared.stream.synchronize()


def _next_static(reference: _ReachableState) -> dict[str, torch.Tensor]:
    return _prepare_static(reference)


def _processor_monotonic_state(processor: Any) -> tuple[torch.Tensor, torch.Tensor]:
    monotonic = processor[0]
    has_time_shift = monotonic._state_has_time_shift
    last_time_shift = monotonic._state_last_time_shift_value
    if has_time_shift is None or last_time_shift is None:
        raise RuntimeError("weighted H8 monotonic processor state is missing.")
    return (
        has_time_shift.clone(memory_format=torch.contiguous_format),
        last_time_shift.clone(memory_format=torch.contiguous_format),
    )


def _reference_step(
        state: _ReachableState,
        target_ids: Sequence[int],
        eos_ids: Sequence[int],
        step: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, dict[str, torch.Tensor]]:
    result = state.session.decode_one_token_raw_logits(
        full_prefix=state.prefix,
        full_attention_mask=state.mask,
        cache_position=torch.arange(
            state.initial_cache_position,
            state.initial_cache_position + 1,
            device=state.prefix.device,
        ),
        active_prefix_self_attention=True,
        active_prefix_self_attention_length=ACTIVE_PREFIX_LENGTH,
    )
    raw = result.logits
    processed = state.base_processor(
        state.prefix, raw.clone(memory_format=torch.contiguous_format)
    )
    warped = state.warper(
        state.prefix, processed.clone(memory_format=torch.contiguous_format)
    )
    probabilities = torch.nn.functional.softmax(warped, dim=-1)
    token = torch.multinomial(
        probabilities, 1, generator=state.generator
    ).squeeze(1)
    stopped = bool(torch.isin(
        token,
        torch.tensor(list(eos_ids), dtype=torch.long, device=token.device),
    ).any().item())
    expected_offset = state.first_output_offset + step
    if int(token.item()) != int(target_ids[expected_offset]) or stopped:
        raise RuntimeError(
            f"weighted H8 B1 row {state.row} diverged at output {expected_offset}."
        )
    state.prefix = torch.cat([state.prefix, token[:, None]], dim=-1)
    state.mask = torch.cat(
        [state.mask, torch.ones_like(token[:, None], dtype=state.mask.dtype)], dim=-1
    )
    state.initial_cache_position += 1
    return raw, processed, probabilities, stopped, _next_static(state)


def _exact_order(
        model: Any,
        resources: _Resources,
        *,
        order: Sequence[int],
        model_inputs: Mapping[str, Any],
        target_ids: Sequence[int],
        base_factory: ProcessorFactory,
        warper_factory: ProcessorFactory,
        combined_factory: ProcessorFactory,
        eos_ids: Sequence[int],
        reviewed_source: Mapping[str, Any],
        comparison_top_k: int,
) -> tuple[
    dict[str, Any],
    list[list[dict[str, torch.Tensor]]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    references = [
        _reachable_state(
            model, model_inputs, target_ids,
            row=row, source_offset=ROW_SOURCE_OFFSETS[row],
            base_factory=base_factory, warper_factory=warper_factory,
            combined_factory=combined_factory, eos_ids=eos_ids,
            reviewed_source=reviewed_source,
        )
        for row in range(2)
    ]
    for row in range(2):
        if tensor_sha256(references[row].starting_rng_state) != tensor_sha256(
                resources.lanes[row].post_anchor_rng_state
        ):
            raise RuntimeError(f"weighted H8 row {row} starting RNG changed.")
    sources: list[list[dict[str, torch.Tensor]]] = [
        [copy.deepcopy(references[0].prepared_inputs)],
        [copy.deepcopy(references[1].prepared_inputs)],
    ]
    private_reference: list[dict[str, Any]] = []
    for row, reference in enumerate(references):
        starting_has, starting_last = _processor_monotonic_state(
            reference.base_processor
        )
        private_reference.append({
            "row": row,
            "starting_has_time_shift": starting_has,
            "starting_last_time_shift_value": starting_last,
            "steps": [],
        })
    _restore(resources)
    _initialize_candidate(resources, sources)
    resources.shared.stream.synchronize()
    steps: list[dict[str, Any]] = []
    final_states: dict[str, Any] = {}
    aggregate_fields = (
        "accepted_tokens_match", "candidate_tokens_match", "stop_behavior_match",
        "per_step_rng_match", "raw_logits_allclose", "raw_logits_topk_match",
        "processed_scores_bitwise_match", "processed_scores_topk_match",
        "sampling_probabilities_bitwise_match", "static_inputs_bitwise_match",
        "padded_prefixes_bitwise_match", "active_cache_allclose",
        "current_cache_written", "future_cache_unchanged",
        "cross_cache_unchanged",
    )
    aggregates = {field: True for field in aggregate_fields}
    calculation_parity = {
        "raw_logits_bitwise_match": True,
        "active_cache_bitwise_match": True,
        "cross_cache_bitwise_match": True,
    }
    for step in range(HORIZON):
        static_before = [
            {
                key: _hash_pair(value, resources.lanes[row].lane.static_inputs[key])
                for key, value in sources[row][step].items()
            }
            for row in range(2)
        ]
        reference_values: list[Any] = [None, None]
        next_sources: list[dict[str, torch.Tensor]] = [{}, {}]
        for row in order:
            raw, processed, probabilities, stopped, next_source = _reference_step(
                references[row], target_ids, eos_ids, step
            )
            reference_values[row] = (raw, processed, probabilities, stopped)
            state_has, state_last = _processor_monotonic_state(
                references[row].base_processor
            )
            private_reference[row]["steps"].append({
                "processed_scores": processed.clone(
                    memory_format=torch.contiguous_format
                ),
                "sampling_probabilities": probabilities.clone(
                    memory_format=torch.contiguous_format
                ),
                "sampled_token": references[row].prefix[:, -1].clone(
                    memory_format=torch.contiguous_format
                ),
                "rng_state": references[row].generator.get_state().clone(),
                "has_time_shift": state_has,
                "last_time_shift_value": state_last,
            })
            next_sources[row] = next_source
            sources[row].append(next_source)
        _enqueue_candidate_step(
            resources, order=order, step=step,
            next_sources=next_sources, synchronize=True,
        )
        row_reports: list[dict[str, Any]] = []
        for row in range(2):
            lane = resources.lanes[row]
            reference = references[row]
            raw, processed, probabilities, reference_stop = reference_values[row]
            candidate_token = resources.sampling[row].sampled_token
            accepted_offset = lane.first_output_offset + step
            accepted_token = int(target_ids[accepted_offset])
            raw_comparison = _logit_comparison(
                raw, lane.source_logits, top_k=comparison_top_k
            )
            raw_bitwise = (
                raw_comparison["reference_sha256"]
                == raw_comparison["candidate_sha256"]
            )
            processed_hash = _hash_pair(processed, resources.shared.processed_rows[row])
            processed_topk = _logit_comparison(
                processed, resources.shared.processed_rows[row],
                top_k=comparison_top_k,
            )["topk_match"]
            probability_hash = _hash_pair(
                probabilities, resources.sampling[row].probabilities
            )
            rng_hash = _hash_pair(
                reference.generator.get_state(),
                resources.sampling[row].generator.get_state(),
            )
            prefix_hash = _hash_pair(
                _padded_prefix(reference.prefix),
                resources.shared.prefix_batch[row:row + 1],
            )
            next_static = {
                key: _hash_pair(value, lane.lane.static_inputs[key])
                for key, value in next_sources[row].items()
            }
            cache = lane.session.cache_state.cache
            reference_cache = reference.session.cache_state.cache
            if cache is None or reference_cache is None:
                raise RuntimeError("weighted H8 exact step lost cache.")
            executed = lane.initial_cache_position + step
            active = _cache_comparison(
                reference_cache.self_attention_cache,
                cache.self_attention_cache,
                start=0, stop=executed + 1,
            )
            active_bitwise = (
                active["reference_sha256"] == active["candidate_sha256"]
            )
            current = _cache_comparison(
                reference_cache.self_attention_cache,
                cache.self_attention_cache,
                start=executed, stop=executed + 1,
            )
            current_nonzero = _cache_slot_nonzero(
                cache.self_attention_cache, executed
            )
            current_written = bool(
                current["allclose"] and current_nonzero
            )
            future_hash = _cache_hash(
                cache.self_attention_cache, start=executed + 1, stop=None
            )
            future_expected = _snapshot_slice_hash(
                lane.self_snapshot, start=executed + 1, stop=None
            )
            future_unchanged = future_hash == future_expected
            cross_reference = _cache_comparison(
                reference_cache.cross_attention_cache,
                cache.cross_attention_cache,
            )
            cross_expected = _snapshot_tuple_hash(lane.cross_snapshot)
            cross_unchanged = bool(
                cross_reference["allclose"]
                and _cache_hash(cache.cross_attention_cache)
                == cross_expected
            )
            cross_bitwise = (
                cross_reference["reference_sha256"]
                == cross_reference["candidate_sha256"]
            )
            candidate_stop = bool(resources.sampling[row].stop_flag.item())
            flags = {
                "accepted_token_match": int(candidate_token.item()) == accepted_token,
                "candidate_token_match": int(candidate_token.item()) == int(
                    reference.prefix[0, -1].item()
                ),
                "stop_match": (
                    candidate_stop == reference_stop and not reference_stop
                ),
                "rng_match": rng_hash["bitwise_match"],
                "raw_allclose": raw_comparison["allclose"],
                "raw_topk_match": raw_comparison["topk_match"],
                "raw_bitwise_match": raw_bitwise,
                "processed_bitwise_match": processed_hash["bitwise_match"],
                "processed_topk_match": processed_topk,
                "probabilities_bitwise_match": probability_hash["bitwise_match"],
                "static_inputs_bitwise_match": all(
                    value["bitwise_match"]
                    for value in (*static_before[row].values(), *next_static.values())
                ),
                "padded_prefix_bitwise_match": prefix_hash["bitwise_match"],
                "active_cache_allclose": active["allclose"],
                "active_cache_bitwise_match": active_bitwise,
                "current_cache_written": current_written,
                "future_cache_unchanged": future_unchanged,
                "cross_cache_unchanged": cross_unchanged,
                "cross_cache_bitwise_match": cross_bitwise,
            }
            for field, value in flags.items():
                if field == "raw_bitwise_match":
                    calculation_parity["raw_logits_bitwise_match"] = bool(
                        calculation_parity["raw_logits_bitwise_match"] and value
                    )
                    continue
                if field == "active_cache_bitwise_match":
                    calculation_parity["active_cache_bitwise_match"] = bool(
                        calculation_parity["active_cache_bitwise_match"] and value
                    )
                    continue
                if field == "cross_cache_bitwise_match":
                    calculation_parity["cross_cache_bitwise_match"] = bool(
                        calculation_parity["cross_cache_bitwise_match"] and value
                    )
                    continue
                aggregate = {
                    "accepted_token_match": "accepted_tokens_match",
                    "candidate_token_match": "candidate_tokens_match",
                    "stop_match": "stop_behavior_match",
                    "rng_match": "per_step_rng_match",
                    "raw_allclose": "raw_logits_allclose",
                    "raw_topk_match": "raw_logits_topk_match",
                    "processed_bitwise_match": "processed_scores_bitwise_match",
                    "processed_topk_match": "processed_scores_topk_match",
                    "probabilities_bitwise_match": "sampling_probabilities_bitwise_match",
                    "static_inputs_bitwise_match": "static_inputs_bitwise_match",
                    "padded_prefix_bitwise_match": "padded_prefixes_bitwise_match",
                    "active_cache_allclose": "active_cache_allclose",
                    "current_cache_written": "current_cache_written",
                    "future_cache_unchanged": "future_cache_unchanged",
                    "cross_cache_unchanged": "cross_cache_unchanged",
                }[field]
                aggregates[aggregate] = bool(aggregates[aggregate] and value)
            row_reports.append({
                "row": row,
                "accepted_output_offset": accepted_offset,
                "accepted_token_id": accepted_token,
                "reference_token_id": int(reference.prefix[0, -1].item()),
                "candidate_token_id": int(candidate_token.item()),
                "reference_stop": reference_stop,
                "candidate_stop": candidate_stop,
                **flags,
                "raw_logits": raw_comparison,
                "processed_scores": {**processed_hash, "topk_match": processed_topk},
                "sampling_probabilities": probability_hash,
                "rng_state": rng_hash,
                "prefix": prefix_hash,
                "static_inputs": {"before": static_before[row], "after": next_static},
                "cache": {
                    "active": active,
                    "current": current,
                    "future_sha256": future_hash,
                    "future_expected_sha256": future_expected,
                    "current_nonzero": current_nonzero,
                    "cross": cross_reference,
                    "cross_expected_sha256": cross_expected,
                },
            })
            final_states[f"row-{row}"] = {
                "rng_sha256": rng_hash["candidate_sha256"],
                "prefix_sha256": prefix_hash["candidate_sha256"],
                "self_sha256": _cache_hash(
                    cache.self_attention_cache,
                    start=0,
                    stop=lane.initial_cache_position + HORIZON,
                ),
                "cross_sha256": _cache_hash(cache.cross_attention_cache),
                "static_sha256": {
                    key: value["candidate_sha256"]
                    for key, value in next_static.items()
                },
            }
        steps.append({"step": step, "rows": row_reports})
    final_rng_match = all(
        final_states[f"row-{row}"]["rng_sha256"]
        == tensor_sha256(references[row].generator.get_state())
        for row in range(2)
    )
    exact_pass = bool(all(aggregates.values()) and final_rng_match)
    if not exact_pass:
        raise RuntimeError("weighted H8 reciprocal exactness failed before timing.")
    signature = _json_sha256(steps)
    return ({
        "steps": steps,
        **aggregates,
        **calculation_parity,
        "final_rng_match": final_rng_match,
        "source_handoff_match": all(
            lane.source_handoff_match for lane in resources.lanes
        ),
        "exactness_pass": exact_pass,
        "signature": signature,
    }, sources, final_states, private_reference)


def _final_matches(
        resources: _Resources,
        prefixes: Sequence[torch.Tensor],
        expected: Mapping[str, Any],
) -> bool:
    for row, lane in enumerate(resources.lanes):
        cache = lane.session.cache_state.cache
        if cache is None:
            return False
        value = expected[f"row-{row}"]
        if (
                tensor_sha256(resources.sampling[row].generator.get_state())
                != value["rng_sha256"]
                or tensor_sha256(prefixes[row]) != value["prefix_sha256"]
                or _cache_hash(
                    cache.self_attention_cache,
                    start=0,
                    stop=lane.initial_cache_position + HORIZON,
                ) != value["self_sha256"]
                or _cache_hash(cache.cross_attention_cache) != value["cross_sha256"]
                or any(
                    tensor_sha256(lane.lane.static_inputs[key]) != digest
                    for key, digest in value["static_sha256"].items()
                )
        ):
            return False
    return True


def _private_control_exactness(
        resources: _Resources,
        *,
        order: Sequence[int],
        sources: Sequence[Sequence[Mapping[str, torch.Tensor]]],
        final_states: Mapping[str, Any],
        private_reference: Sequence[Mapping[str, Any]],
        target_ids: Sequence[int],
        comparison_top_k: int,
) -> dict[str, Any]:
    """Prove the recurrent private-B1 control before collecting timings."""

    _restore(resources)
    starting_states: list[dict[str, Any]] = []
    aggregates = {
        "starting_monotonic_state_match": True,
        "offset41_time_shift_retention_match": True,
        "processed_scores_bitwise_match": True,
        "processed_scores_topk_match": True,
        "sampling_probabilities_bitwise_match": True,
        "candidate_tokens_match": True,
        "per_step_rng_match": True,
        "monotonic_state_match": True,
        "state_feedback_match": True,
    }
    for row in range(2):
        private = resources.private[row]
        reference = private_reference[row]
        has_pair = _hash_pair(
            reference["starting_has_time_shift"],
            private.monotonic_input_has_time_shift,
        )
        last_pair = _hash_pair(
            reference["starting_last_time_shift_value"],
            private.monotonic_input_last_time_shift_value,
        )
        state_match = bool(has_pair["bitwise_match"] and last_pair["bitwise_match"])
        current_offset = ROW_FIRST_OUTPUT_OFFSETS[row] - 1
        token_match = int(resources.lanes[row].full_prefix[0, -1].item()) == int(
            target_ids[current_offset]
        )
        aggregates["starting_monotonic_state_match"] = bool(
            aggregates["starting_monotonic_state_match"] and state_match
        )
        aggregates["offset41_time_shift_retention_match"] = bool(
            aggregates["offset41_time_shift_retention_match"]
            and state_match
            and token_match
        )
        starting_states.append({
            "row": row,
            "current_output_offset": current_offset,
            "current_token_id": int(target_ids[current_offset]),
            "has_time_shift": has_pair,
            "last_time_shift_value": last_pair,
            "state_match": state_match,
        })
    steps_by_index: list[list[dict[str, Any] | None]] = [
        [None, None] for _ in range(HORIZON)
    ]
    for row in order:
        lane = resources.lanes[row]
        private = resources.private[row]
        tail = resources.sampling[row]
        reference_steps = private_reference[row]["steps"]
        with torch.cuda.stream(lane.lane.stream):
            private.output_prefix.copy_(private.initial_output_prefix)
            private.processor_input[:, -1].copy_(lane.full_prefix[:, -1])
            _copy_static(lane, sources[row][0], True)
        lane.lane.stream.synchronize()
        for step in range(HORIZON):
            expected = reference_steps[step]
            with torch.cuda.stream(lane.lane.stream):
                lane.lane.graph.replay()
                private.raw_logits.copy_(lane.source_logits)
                private.graph.replay()
                private.monotonic_input_has_time_shift.copy_(
                    private.monotonic_output_has_time_shift
                )
                private.monotonic_input_last_time_shift_value.copy_(
                    private.monotonic_output_last_time_shift_value
                )
                tail.processed_scores.copy_(private.processed_scores)
                tail.graph.replay()
                private.output_prefix[
                    :, ROW_INITIAL_PREFIX_LENGTHS[row] + step
                ].copy_(tail.sampled_token)
                private.processor_input[:, -1].copy_(tail.sampled_token)
                lane.lane.static_inputs["decoder_input_ids"].copy_(
                    tail.sampled_token_2d
                )
                _copy_static(lane, sources[row][step + 1], False)
            lane.lane.stream.synchronize()
            processed = _hash_pair(
                expected["processed_scores"], private.processed_scores
            )
            processed_topk = _logit_comparison(
                expected["processed_scores"],
                private.processed_scores,
                top_k=comparison_top_k,
            )["topk_match"]
            probabilities = _hash_pair(
                expected["sampling_probabilities"], tail.probabilities
            )
            rng = _hash_pair(expected["rng_state"], tail.generator.get_state())
            has_state = _hash_pair(
                expected["has_time_shift"],
                private.monotonic_output_has_time_shift,
            )
            last_state = _hash_pair(
                expected["last_time_shift_value"],
                private.monotonic_output_last_time_shift_value,
            )
            feedback_has = _hash_pair(
                private.monotonic_output_has_time_shift,
                private.monotonic_input_has_time_shift,
            )
            feedback_last = _hash_pair(
                private.monotonic_output_last_time_shift_value,
                private.monotonic_input_last_time_shift_value,
            )
            token_id = int(tail.sampled_token.item())
            accepted_offset = ROW_FIRST_OUTPUT_OFFSETS[row] + step
            flags = {
                "processed_bitwise_match": processed["bitwise_match"],
                "processed_topk_match": processed_topk,
                "probabilities_bitwise_match": probabilities["bitwise_match"],
                "candidate_token_match": (
                    token_id == int(expected["sampled_token"].item())
                    == int(target_ids[accepted_offset])
                ),
                "rng_match": rng["bitwise_match"],
                "monotonic_state_match": bool(
                    has_state["bitwise_match"] and last_state["bitwise_match"]
                ),
                "state_feedback_match": bool(
                    feedback_has["bitwise_match"]
                    and feedback_last["bitwise_match"]
                ),
            }
            aggregate_map = {
                "processed_bitwise_match": "processed_scores_bitwise_match",
                "processed_topk_match": "processed_scores_topk_match",
                "probabilities_bitwise_match": (
                    "sampling_probabilities_bitwise_match"
                ),
                "candidate_token_match": "candidate_tokens_match",
                "rng_match": "per_step_rng_match",
                "monotonic_state_match": "monotonic_state_match",
                "state_feedback_match": "state_feedback_match",
            }
            for field, value in flags.items():
                aggregate = aggregate_map[field]
                aggregates[aggregate] = bool(aggregates[aggregate] and value)
            steps_by_index[step][row] = {
                "row": row,
                "accepted_output_offset": accepted_offset,
                "accepted_token_id": int(target_ids[accepted_offset]),
                "candidate_token_id": token_id,
                "processed_scores": {**processed, "topk_match": processed_topk},
                "sampling_probabilities": probabilities,
                "rng_state": rng,
                "has_time_shift": has_state,
                "last_time_shift_value": last_state,
                "feedback_has_time_shift": feedback_has,
                "feedback_last_time_shift_value": feedback_last,
                **flags,
            }
    steps = [
        {"step": step, "rows": rows}
        for step, rows in enumerate(steps_by_index)
    ]
    prefixes = [value.output_prefix for value in resources.private]
    final_match = bool(
        _final_matches(resources, prefixes, final_states)
        and not any(bool(tail.stop_flag.item()) for tail in resources.sampling)
    )
    exact_pass = bool(all(aggregates.values()) and final_match)
    if not exact_pass:
        raise RuntimeError("weighted H8 private-B1 exactness failed before timing.")
    return {
        "starting_states": starting_states,
        "steps": steps,
        **aggregates,
        "final_state_matches_exact": final_match,
        "exactness_pass": exact_pass,
        "signature": _json_sha256([*starting_states, *steps]),
    }


def _timing_payload(
        measurement: str,
        trials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    wall = sum(float(value["wall_seconds"]) for value in trials)
    cuda = sum(float(value["cuda_seconds"]) for value in trials)
    tokens = 2 * HORIZON * len(trials)
    return {
        "measurement": measurement,
        "horizon": HORIZON,
        "trial_count": len(trials),
        "generated_tokens": tokens,
        "wall_seconds": wall,
        "cuda_seconds": cuda,
        "wall_tokens_per_second": tokens / wall,
        "cuda_tokens_per_second": tokens / cuda,
        "trials": list(trials),
        "no_per_step_synchronize": True,
        "no_per_step_allocation": True,
        "no_graph_recapture": True,
        "reset_outside_timing": True,
        "initial_state_copy_inside_timing": True,
        "final_state_matches_exact": True,
    }


def _candidate_timing(
        resources: _Resources,
        *,
        order: Sequence[int],
        sources: Sequence[Sequence[Mapping[str, torch.Tensor]]],
        final_states: Mapping[str, Any],
) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    device = resources.shared.prefix_batch.device
    for trial in range(TIMING_TRIALS_PER_ORDER):
        _restore(resources)
        torch.cuda.reset_peak_memory_stats(device)
        before_allocated = torch.cuda.memory_allocated(device)
        before_reserved = torch.cuda.memory_reserved(device)
        start = torch.cuda.Event(enable_timing=True)
        finish = torch.cuda.Event(enable_timing=True)
        wall_started = time.perf_counter()
        start.record(resources.shared.stream)
        _initialize_candidate(resources, sources)
        for step in range(HORIZON):
            _enqueue_candidate_step(
                resources, order=order, step=step,
                next_sources=[sources[row][step + 1] for row in range(2)],
                synchronize=False,
            )
        with torch.cuda.stream(resources.shared.stream):
            finish.record(resources.shared.stream)
        finish.synchronize()
        wall = time.perf_counter() - wall_started
        cuda = start.elapsed_time(finish) / 1000.0
        if (
                torch.cuda.max_memory_allocated(device) != before_allocated
                or torch.cuda.max_memory_reserved(device) != before_reserved
        ):
            raise RuntimeError("weighted H8 candidate allocated during timing.")
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        prefixes = [
            resources.shared.prefix_batch[row:row + 1] for row in range(2)
        ]
        matches = _final_matches(resources, prefixes, final_states)
        if not matches or any(bool(flag.item()) for flag in resources.stop_flags):
            raise RuntimeError("weighted H8 candidate timed state differs from exact.")
        tokens = 2 * HORIZON
        trials.append({
            "trial": trial,
            "generated_tokens": tokens,
            "wall_seconds": wall,
            "cuda_seconds": cuda,
            "wall_tokens_per_second": tokens / wall,
            "cuda_tokens_per_second": tokens / cuda,
            "final_state_matches_exact": True,
            "memory_before_allocated_bytes": before_allocated,
            "memory_before_reserved_bytes": before_reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "peak_allocated_delta_bytes": peak_allocated - before_allocated,
            "peak_reserved_delta_bytes": peak_reserved - before_reserved,
        })
    return _timing_payload("complete_hybrid_b2_h8", trials)


def _control_timing(
        resources: _Resources,
        *,
        order: Sequence[int],
        sources: Sequence[Sequence[Mapping[str, torch.Tensor]]],
        final_states: Mapping[str, Any],
) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    device = resources.shared.prefix_batch.device
    for trial in range(TIMING_TRIALS_PER_ORDER):
        _restore(resources)
        torch.cuda.reset_peak_memory_stats(device)
        before_allocated = torch.cuda.memory_allocated(device)
        before_reserved = torch.cuda.memory_reserved(device)
        timing_events = {
            row: (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for row in order
        }
        wall_started = time.perf_counter()
        cuda = 0.0
        for row in order:
            lane = resources.lanes[row]
            private = resources.private[row]
            tail = resources.sampling[row]
            start, finish = timing_events[row]
            with torch.cuda.stream(lane.lane.stream):
                start.record(lane.lane.stream)
                private.output_prefix.copy_(private.initial_output_prefix)
                private.processor_input[:, -1].copy_(lane.full_prefix[:, -1])
                _copy_static(lane, sources[row][0], True)
                for step in range(HORIZON):
                    lane.lane.graph.replay()
                    private.raw_logits.copy_(lane.source_logits)
                    private.graph.replay()
                    private.monotonic_input_has_time_shift.copy_(
                        private.monotonic_output_has_time_shift
                    )
                    private.monotonic_input_last_time_shift_value.copy_(
                        private.monotonic_output_last_time_shift_value
                    )
                    tail.processed_scores.copy_(private.processed_scores)
                    tail.graph.replay()
                    private.output_prefix[
                        :,
                        ROW_INITIAL_PREFIX_LENGTHS[row] + step,
                    ].copy_(tail.sampled_token)
                    private.processor_input[:, -1].copy_(tail.sampled_token)
                    lane.lane.static_inputs["decoder_input_ids"].copy_(
                        tail.sampled_token_2d
                    )
                    _copy_static(lane, sources[row][step + 1], False)
                finish.record(lane.lane.stream)
            finish.synchronize()
            cuda += start.elapsed_time(finish) / 1000.0
        wall = time.perf_counter() - wall_started
        if (
                torch.cuda.max_memory_allocated(device) != before_allocated
                or torch.cuda.max_memory_reserved(device) != before_reserved
        ):
            raise RuntimeError("weighted H8 private-B1 control allocated during timing.")
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        prefixes = [private.output_prefix for private in resources.private]
        matches = _final_matches(resources, prefixes, final_states)
        if not matches or any(bool(tail.stop_flag.item()) for tail in resources.sampling):
            raise RuntimeError("weighted H8 private-B1 control differs from exact.")
        tokens = 2 * HORIZON
        trials.append({
            "trial": trial,
            "generated_tokens": tokens,
            "wall_seconds": wall,
            "cuda_seconds": cuda,
            "wall_tokens_per_second": tokens / wall,
            "cuda_tokens_per_second": tokens / cuda,
            "final_state_matches_exact": True,
            "memory_before_allocated_bytes": before_allocated,
            "memory_before_reserved_bytes": before_reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "peak_allocated_delta_bytes": peak_allocated - before_allocated,
            "peak_reserved_delta_bytes": peak_reserved - before_reserved,
        })
    return _timing_payload("serial_private_b1_graph_h8", trials)


def _resource_evidence(resources: _Resources) -> dict[str, Any]:
    bridge = HybridBridgeResourceEvidence(
        control_stream_id=int(resources.shared.stream.cuda_stream),
        processor_graph_id=id(resources.shared.graph),
        processor_graph_pool_id=repr(resources.shared.graph.pool()),
        prefix_batch_storage_ptr=_tensor_storage_ptr(resources.shared.prefix_batch),
        raw_logits_bridge_storage_ptr=_tensor_storage_ptr(resources.shared.raw_logits),
        processed_scores_storage_ptr=_tensor_storage_ptr(
            resources.shared.processed_scores
        ),
        source_release_event_ids=tuple(id(value) for value in resources.events.source_released),
        lane_ready_event_ids=tuple(id(value) for value in resources.events.lane_ready),
        sample_done_event_ids=tuple(id(value) for value in resources.events.sample_done),
        processor_done_event_id=id(resources.events.processor_done),
        sampling_lanes=tuple(
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
        ),
    )
    validate_hybrid_resource_ownership(
        [lane.ownership for lane in resources.lanes], bridge
    )
    graph_pools = {
        *(repr(lane.lane.graph.pool()) for lane in resources.lanes),
        repr(resources.shared.graph.pool()),
        *(repr(value.graph.pool()) for value in resources.private),
        *(repr(value.graph.pool()) for value in resources.sampling),
    }
    if len(graph_pools) != 7:
        raise RuntimeError("weighted H8 graph pools alias.")
    private_evidence = [
        {
            "row": row,
            "graph_id": id(value.graph),
            "graph_pool_id": repr(value.graph.pool()),
            "processor_id": id(value.processor),
            "processor_input_storage_ptr": _tensor_storage_ptr(
                value.processor_input
            ),
            "output_prefix_storage_ptr": _tensor_storage_ptr(value.output_prefix),
            "raw_logits_storage_ptr": _tensor_storage_ptr(value.raw_logits),
            "processed_scores_storage_ptr": _tensor_storage_ptr(
                value.processed_scores
            ),
            "monotonic_input_has_time_shift_storage_ptr": _tensor_storage_ptr(
                value.monotonic_input_has_time_shift
            ),
            "monotonic_input_last_time_shift_storage_ptr": _tensor_storage_ptr(
                value.monotonic_input_last_time_shift_value
            ),
            "monotonic_output_has_time_shift_storage_ptr": _tensor_storage_ptr(
                value.monotonic_output_has_time_shift
            ),
            "monotonic_output_last_time_shift_storage_ptr": _tensor_storage_ptr(
                value.monotonic_output_last_time_shift_value
            ),
        }
        for row, value in enumerate(resources.private)
    ]
    cross_row_fields = (
        "graph_id", "graph_pool_id", "processor_id",
        "processor_input_storage_ptr", "output_prefix_storage_ptr",
        "raw_logits_storage_ptr", "processed_scores_storage_ptr",
        "monotonic_input_has_time_shift_storage_ptr",
        "monotonic_input_last_time_shift_storage_ptr",
        "monotonic_output_has_time_shift_storage_ptr",
        "monotonic_output_last_time_shift_storage_ptr",
    )
    private_distinct = all(
        private_evidence[0][field] != private_evidence[1][field]
        for field in cross_row_fields
    )
    shared_pointers = {
        _tensor_storage_ptr(resources.shared.prefix_batch),
        _tensor_storage_ptr(resources.shared.raw_logits),
        _tensor_storage_ptr(resources.shared.processed_scores),
    }
    private_shared_non_alias = all(
        evidence[field] not in shared_pointers
        for evidence in private_evidence
        for field in (
            "processor_input_storage_ptr", "output_prefix_storage_ptr",
            "raw_logits_storage_ptr", "processed_scores_storage_ptr",
            "monotonic_input_has_time_shift_storage_ptr",
            "monotonic_input_last_time_shift_storage_ptr",
            "monotonic_output_has_time_shift_storage_ptr",
            "monotonic_output_last_time_shift_storage_ptr",
        )
    )
    state_input_output_distinct = all(
        evidence["monotonic_input_has_time_shift_storage_ptr"]
        != evidence["monotonic_output_has_time_shift_storage_ptr"]
        and evidence["monotonic_input_last_time_shift_storage_ptr"]
        != evidence["monotonic_output_last_time_shift_storage_ptr"]
        for evidence in private_evidence
    )
    if (
            not private_distinct
            or not private_shared_non_alias
            or not state_input_output_distinct
    ):
        raise RuntimeError("weighted H8 private-B1 processor resources alias.")
    return {
        "model_graph_count": 2,
        "shared_b2_processor_graph_count": 1,
        "private_b1_processor_graph_count": 2,
        "private_sampling_stop_graph_count": 2,
        "total_graph_count": 7,
        "model_lanes": [lane.ownership.as_dict() for lane in resources.lanes],
        "hybrid_bridge": bridge.as_dict(),
        "private_processor_graph_pool_ids": [
            repr(value.graph.pool()) for value in resources.private
        ],
        "private_b1_processors": private_evidence,
        "private_b1_ownership_pass": private_distinct,
        "private_b1_shared_bridge_non_alias": private_shared_non_alias,
        "private_b1_state_feedback_pass": state_input_output_distinct,
        "private_b1_state_feedback_inside_timed_interval": True,
        "private_b1_state_feedback_policy": (
            "captured_output_copied_to_captured_input_after_every_private_processor_replay"
        ),
        "private_b1_processor_policy": (
            "stateful_incremental_one_token_graph_with_neutral_padded_history_priming"
        ),
    }


@torch.no_grad()
def run_weighted_h8_gate(
        args: Any,
        *,
        config_name: str,
        source_report_path: Path,
        source_profile_path: Path,
        source_audio_path: Path,
        ceiling_report_path: Path,
        capture_warmup_repeats: int,
        comparison_top_k: int,
        q1_bmm_cross_attention: bool,
        native_q1_self_attention: bool,
        native_q1_rope_cache_self_attention: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    extension_cache_before = _extension_cache_state()
    reviewed_report = load_reviewed_source_capture_report(source_report_path)
    reviewed_contract = {
        key: value for key, value in reviewed_report.items()
        if key in {
            "schema_version", "gate", "source", "transcript", "reconstruction",
            "accepted_prefix_replay", "maintainer_boundary",
        }
    }
    if reviewed_report["source_contract_sha256"] != REVIEWED_SOURCE_CONTRACT_SHA256:
        raise ValueError("weighted H8 source contract digest changed.")
    if _file_sha256(ceiling_report_path) != WEIGHTED_CEILING_REPORT_SHA256:
        raise ValueError("weighted H8 ceiling report file changed.")
    ceiling = json.loads(ceiling_report_path.read_text(encoding="utf-8"))
    validate_model_free_ceiling_report(ceiling)
    required = float(
        ceiling["duplicated_ten_request_b2"][
            "required_weighted_complete_tokens_per_second"
        ]
    )
    if required != REQUIRED_WEIGHTED_COMPLETE_TPS:
        raise ValueError("weighted H8 required TPS changed.")
    _validate_fixed_runtime_contract(
        args,
        config_name=config_name,
        q1_bmm_cross_attention=q1_bmm_cross_attention,
        native_q1_self_attention=native_q1_self_attention,
        native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
    )
    _assert_supported_probe(args)
    args.audio_path = str(source_audio_path)
    compile_args(args, verbose=False)
    setup_inference_environment(ROW_SEED)
    source_evidence = _load_source_evidence(source_profile_path, source_audio_path)
    timing_tokenizer = Tokenizer.from_pretrained(args.model_path)
    model, tokenizer = load_model_with_server(
        args.model_path, args.train, args.device,
        max_batch_size=args.max_batch_size, use_server=False,
        precision=args.precision, attn_implementation=args.attn_implementation,
        lora_path=args.lora_path, gamemode=args.gamemode,
        auto_select_gamemode_model=args.auto_select_gamemode_model,
        generation_compile=False,
    )
    model.eval()
    if (
            model.device.type != "cuda"
            or model.dtype != torch.float32
            or int(model.config.max_target_positions) != 2560
    ):
        raise RuntimeError("weighted H8 requires CUDA FP32 max_target_positions=2560.")
    gpu_name = torch.cuda.get_device_name(model.device)
    gpu_capability = list(torch.cuda.get_device_capability(model.device))
    if "RTX 2080" not in gpu_name or gpu_capability != [7, 5]:
        raise RuntimeError(
            f"weighted H8 requires RTX 2080/2080 Ti SM75; got {gpu_name} "
            f"capability={gpu_capability}."
        )
    torch.cuda.reset_peak_memory_stats(model.device)
    tokenizer_contract = _validate_tokenizer_scopes(timing_tokenizer, tokenizer)
    model_inputs, probe = _reconstruct_target_inputs(
        args, model, tokenizer, timing_tokenizer=timing_tokenizer,
        source_evidence=source_evidence,
    )
    validate_reviewed_context_evidence(_build_context_evidence(
        tokenizer_contract=tokenizer_contract,
        model_inputs=model_inputs,
        probe=probe,
    ))
    target_ids = source_evidence["target_token_ids"]
    context_type = ContextType(probe["context_type"])
    eos_ids = [
        int(value) for value in get_eos_token_id(
            tokenizer,
            lookback_time=float(probe["lookback_time"]),
            lookahead_time=float(probe["lookahead_time"]),
            context_type=context_type,
        )
    ]
    base_factory, warper_factory, combined_factory = _make_factories(
        args, tokenizer, torch.device(model.device), lookback_time=0.0
    )
    with torch.autocast(device_type="cuda", enabled=False), generation_profile_context(
            sdpa_backend=args.profile_sdpa_backend,
            q1_bmm_cross_attention=q1_bmm_cross_attention,
            native_q1_self_attention=native_q1_self_attention,
            native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
    ):
        candidate_states = [
            _reachable_state(
                model, model_inputs, target_ids,
                row=row, source_offset=ROW_SOURCE_OFFSETS[row],
                base_factory=base_factory, warper_factory=warper_factory,
                combined_factory=combined_factory, eos_ids=eos_ids,
                reviewed_source=reviewed_contract,
            )
            for row in range(2)
        ]
        lanes_list = [
            _capture_candidate(model, state, capture_warmup_repeats)
            for state in candidate_states
        ]
        lanes = (lanes_list[0], lanes_list[1])
        shared = _capture_shared(lanes, base_factory, capture_warmup_repeats)
        private_list = [
            _capture_private_processor(lane, base_factory, capture_warmup_repeats)
            for lane in lanes
        ]
        sampling_list = [
            _capture_sampling_tail(
                lane,
                warper_factory=warper_factory,
                eos_ids=eos_ids,
                capture_warmup_repeats=capture_warmup_repeats,
            )
            for lane in lanes
        ]
        resources = _build_resources(
            lanes, shared, (private_list[0], private_list[1]),
            (sampling_list[0], sampling_list[1]),
        )
        resource_report = _resource_evidence(resources)
        order_exact: dict[str, Any] = {}
        order_private_exact: dict[str, Any] = {}
        order_sources: dict[str, Any] = {}
        order_final: dict[str, Any] = {}
        for order in reciprocal_lane_orders(2):
            key = ",".join(str(value) for value in order)
            exact, sources, final_states, private_reference = _exact_order(
                model, resources, order=order,
                model_inputs=model_inputs, target_ids=target_ids,
                base_factory=base_factory, warper_factory=warper_factory,
                combined_factory=combined_factory, eos_ids=eos_ids,
                reviewed_source=reviewed_contract,
                comparison_top_k=comparison_top_k,
            )
            order_exact[key] = exact
            order_sources[key] = sources
            order_final[key] = final_states
            order_private_exact[key] = _private_control_exactness(
                resources,
                order=order,
                sources=sources,
                final_states=final_states,
                private_reference=private_reference,
                target_ids=target_ids,
                comparison_top_k=comparison_top_k,
            )
        if order_exact["0,1"]["signature"] != order_exact["1,0"]["signature"]:
            raise RuntimeError("weighted H8 exact evidence is order-dependent.")
        candidate_timings: dict[str, Any] = {}
        control_timings: dict[str, Any] = {}
        for order in reciprocal_lane_orders(2):
            key = ",".join(str(value) for value in order)
            candidate_timings[key] = _candidate_timing(
                resources, order=order, sources=order_sources[key],
                final_states=order_final[key],
            )
            control_timings[key] = _control_timing(
                resources, order=order, sources=order_sources[key],
                final_states=order_final[key],
            )
    performance = summarize_weighted_h8_performance(
        candidate_timings, control_timings
    )
    raw_bitwise = all(
        bool(values["raw_logits_bitwise_match"])
        for values in order_exact.values()
    )
    cache_bitwise = all(
        bool(values["active_cache_bitwise_match"])
        for values in order_exact.values()
    )
    cross_cache_bitwise = all(
        bool(values["cross_cache_bitwise_match"])
        for values in order_exact.values()
    )
    result_classification = (
        "bitwise-calculation-exact"
        if raw_bitwise and cache_bitwise and cross_cache_bitwise
        else "exact-sampled-prefix"
    )
    pass_gate = bool(performance["performance_pass"])
    extension_cache_after = _extension_cache_state()
    memory_current_allocated = torch.cuda.memory_allocated(model.device)
    memory_current_reserved = torch.cuda.memory_reserved(model.device)
    memory_peak_allocated = torch.cuda.max_memory_allocated(model.device)
    memory_peak_reserved = torch.cuda.max_memory_reserved(model.device)
    report = {
        "schema_version": WEIGHTED_H8_SCHEMA_VERSION,
        "gate": WEIGHTED_H8_GATE,
        "claim_scope": "verifier_only_real_prefix_bucket576_phase_a_not_runtime",
        "pass": pass_gate,
        "exactness_pass": True,
        "resource_ownership_pass": True,
        "result": {
            "scope": "h8_sampled_prefix_scout_not_full_exact_output_or_osu",
            "classification": result_classification,
            "raw_logits_bitwise_match": raw_bitwise,
            "active_cache_bitwise_match": cache_bitwise,
            "cross_cache_bitwise_match": cross_cache_bitwise,
            "full_exact_output_claimed": False,
        },
        "source_contract": {
            "job_id": REVIEWED_SOURCE_CAPTURE_JOB_ID,
            "commit": REVIEWED_SOURCE_CAPTURE_COMMIT,
            "contract_sha256": REVIEWED_SOURCE_CONTRACT_SHA256,
            "report_file_sha256": REVIEWED_SOURCE_REPORT_FILE_SHA256,
            "digest_override_allowed": False,
        },
        "workload": {
            "active_prefix_length": ACTIVE_PREFIX_LENGTH,
            "horizon": HORIZON,
            "timing_trials_per_order": TIMING_TRIALS_PER_ORDER,
            "row_seeds": [ROW_SEED, ROW_SEED],
            "row_source_offsets": list(ROW_SOURCE_OFFSETS),
            "row_initial_prefix_lengths": list(ROW_INITIAL_PREFIX_LENGTHS),
            "row_initial_cache_positions": list(ROW_INITIAL_CACHE_POSITIONS),
            "row_first_output_offsets": list(ROW_FIRST_OUTPUT_OFFSETS),
            "processor_prefix_shape": [2, ACTIVE_PREFIX_LENGTH],
            "neutral_pad_token_id": NEUTRAL_PAD_TOKEN_ID,
            "request_shape": "same_seed_staggered_reachable_production_states",
            "distinct_seed_rejected_reason": (
                "source contract authorizes only seed12345; resetting a future row to "
                "seed23456 would fabricate an unreachable request state"
            ),
            "weighted_ceiling_report_sha256": WEIGHTED_CEILING_REPORT_SHA256,
            "required_weighted_complete_tokens_per_second": (
                REQUIRED_WEIGHTED_COMPLETE_TPS
            ),
        },
        "event_order": list(EVENT_ORDER),
        "resources": resource_report,
        "reset_contract": {
            "in_place_cache_restore": True,
            "exact_generator_state_restore": True,
            "static_pointer_stability": True,
            "restore_verified_after_capture": True,
        },
        "orders": {
            order: {
                "execution_order": [int(value) for value in order.split(",")],
                "exactness": order_exact[order],
                "private_control_exactness": order_private_exact[order],
                "candidate_timing": candidate_timings[order],
                "control_timing": control_timings[order],
                "same_order_gain_fraction": performance[
                    "same_order_gain_fraction"
                ][order],
            }
            for order in ("0,1", "1,0")
        },
        "performance": performance,
        "setup": {
            "excluded_from_performance": True,
            "excluded_components": [
                "source_reconstruction", "prefill", "graph_warmup", "graph_capture"
            ],
            "total_gate_wall_seconds": time.perf_counter() - started,
            "graph_capture_count": 7,
            "model_graphs": [
                {
                    "warmup_seconds": lane.lane.warmup_seconds,
                    "capture_seconds": lane.lane.capture_seconds,
                }
                for lane in resources.lanes
            ],
            "shared_b2_processor_graph": {
                "warmup_seconds": resources.shared.warmup_seconds,
                "capture_seconds": resources.shared.capture_seconds,
            },
            "private_b1_processor_graphs": [
                {
                    "warmup_seconds": value.warmup_seconds,
                    "capture_seconds": value.capture_seconds,
                }
                for value in resources.private
            ],
            "private_sampling_graphs": [
                {
                    "warmup_seconds": value.warmup_seconds,
                    "capture_seconds": value.capture_seconds,
                }
                for value in resources.sampling
            ],
            "extension_cache_before": extension_cache_before,
            "extension_cache_after": extension_cache_after,
            "memory_current_allocated_bytes": memory_current_allocated,
            "memory_current_reserved_bytes": memory_current_reserved,
            "memory_peak_allocated_bytes": memory_peak_allocated,
            "memory_peak_reserved_bytes": memory_peak_reserved,
        },
        "decision": {
            "phase_a_only": True,
            "phase_b_authorized": False,
            "weighted_bucket_sweep_authorized": pass_gate,
            "scheduler_or_runtime_wiring_authorized": False,
            "server_authorized": False,
        },
        "runtime_metadata": {
            "git_commit": _git_commit(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "manual"),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "device": "cuda",
            "precision": "fp32",
            "gpu_name": gpu_name,
            "gpu_capability": gpu_capability,
            "model_max_target_positions": int(model.config.max_target_positions),
            "h8_executed": True,
            "scheduler_or_runtime_wiring_authorized": False,
        },
    }
    validate_weighted_h8_report(report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="profile_salvalai_smoke15")
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--source-profile", type=Path, required=True)
    parser.add_argument("--source-audio", type=Path, required=True)
    parser.add_argument("--weighted-ceiling-report", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--capture-warmup-repeats", type=int, default=1)
    parser.add_argument("--comparison-top-k", type=int, default=20)
    parser.add_argument("--q1-bmm-cross-attention", action="store_true")
    parser.add_argument("--native-q1-self-attention", action="store_true")
    parser.add_argument("--native-q1-rope-cache-self-attention", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    args = _load_args(cli.config_name, cli.overrides)
    try:
        report = run_weighted_h8_gate(
            args,
            config_name=cli.config_name,
            source_report_path=cli.source_report,
            source_profile_path=cli.source_profile,
            source_audio_path=cli.source_audio,
            ceiling_report_path=cli.weighted_ceiling_report,
            capture_warmup_repeats=cli.capture_warmup_repeats,
            comparison_top_k=cli.comparison_top_k,
            q1_bmm_cross_attention=cli.q1_bmm_cross_attention,
            native_q1_self_attention=cli.native_q1_self_attention,
            native_q1_rope_cache_self_attention=(
                cli.native_q1_rope_cache_self_attention
            ),
        )
    except Exception as error:
        failure_report = {
            "schema_version": WEIGHTED_H8_SCHEMA_VERSION,
            "gate": WEIGHTED_H8_GATE,
            "claim_scope": (
                "verifier_only_real_prefix_bucket576_phase_a_not_runtime"
            ),
            "pass": False,
            "exactness_pass": False,
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "runtime_metadata": {
                "git_commit": _git_commit(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "manual"),
            },
        }
        cli.report_path.parent.mkdir(parents=True, exist_ok=True)
        cli.report_path.write_text(
            json.dumps(failure_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(failure_report, indent=2, sort_keys=True), file=sys.stderr)
        raise
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    # The Slurm post-validator owns the intentional keep-bar failure after the
    # complete report has been persisted and independently revalidated.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
