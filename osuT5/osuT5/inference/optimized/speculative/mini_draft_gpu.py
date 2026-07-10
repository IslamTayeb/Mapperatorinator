"""One-window GPU feasibility scout for a greedy v32-mini speculative draft.

This module is verifier-only.  It samples the target transcript once, then
replays deterministic mini proposals against that frozen transcript.  It does
not evaluate target spans, mutate target caches, implement rollback, or expose a
production inference hook.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
from transformers import LogitsProcessor
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper
from transformers.modeling_outputs import BaseModelOutput

from osuT5.osuT5.inference.cache_utils import get_cache
from osuT5.osuT5.inference.optimized.single.session import (
    OneTokenDecodeState,
    decode_one_token_raw_logits,
    last_token_logits,
)
from osuT5.osuT5.inference.server import get_eos_token_id
from osuT5.osuT5.inference.optimized.single.logits import (
    build_single_logits_processor_list as build_logits_processor_list,
)
from osuT5.osuT5.runtime_profiling import generation_profile_context
from osuT5.osuT5.tokenizer import ContextType

from .mini_feasibility import (
    CAMPAIGN_MINIMUM_IMPROVEMENT_FRACTION,
    DEFAULT_SPECULATION_K,
    FiniteWindowProjection,
    cached_q1_forwards_for_proposal,
    clears_strict_cuda_and_wall_gate,
    replay_closed_loop_target_transcript,
    token_ids_sha256,
)
from .target_span_gpu import (
    _hash_rng_state,
    _rng_state,
    _rng_states_equal,
)


GATE_NAME = "v32_mini_closed_loop_feasibility"
EXPECTED_MAX_NEW_TOKENS = 256
APPROVED_STATEFUL_MONOTONIC_LOGITS = True
ENCODER_CONDITION_KEYS = ("beatmap_idx", "difficulty", "mapper_idx", "song_position")


@dataclass(frozen=True)
class MiniDraftGpuScoutConfig:
    speculation_k: int = DEFAULT_SPECULATION_K
    max_new_tokens: int = EXPECTED_MAX_NEW_TOKENS
    encoder_warmup: int = 2
    encoder_timing_iters: int = 8
    steady_cache_warmup: int = 2
    steady_cache_timing_iters: int = 8

    def __post_init__(self) -> None:
        if self.speculation_k != DEFAULT_SPECULATION_K:
            raise ValueError("The approved v32-mini scout is K=4 only.")
        if self.max_new_tokens != EXPECTED_MAX_NEW_TOKENS:
            raise ValueError("The approved v32-mini scout captures exactly 256 target tokens at most.")
        if self.encoder_warmup < 0 or self.encoder_timing_iters <= 0:
            raise ValueError("Encoder warmup must be non-negative and timing iters positive.")
        if self.steady_cache_warmup < 0 or self.steady_cache_timing_iters <= 0:
            raise ValueError("Steady-cache warmup must be non-negative and timing iters positive.")


@dataclass(frozen=True)
class DraftProposalMeasurement:
    iteration: int
    prefix_length: int
    requested_tokens: int
    proposal_token_ids: tuple[int, ...]
    decoder_rebuild_cuda_ms: float
    continuation_q1_cuda_ms: tuple[float, ...]
    rebuild_inclusive_cuda_ms: float
    rebuild_inclusive_wall_ms: float

    def to_manifest(self) -> dict[str, Any]:
        result = asdict(self)
        result["proposal_token_ids"] = list(self.proposal_token_ids)
        result["continuation_q1_cuda_ms"] = list(self.continuation_q1_cuda_ms)
        result["continuation_q1_cuda_ms_total"] = sum(self.continuation_q1_cuda_ms)
        return result


class _RngAuditProcessor(LogitsProcessor):
    """Record RNG immediately before each target sampling draw without logits copies."""

    def __init__(self) -> None:
        self.prefix_lengths: list[int] = []
        self.rng_hashes_before_draw: list[str] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.prefix_lengths.append(int(input_ids.shape[-1]))
        self.rng_hashes_before_draw.append(_hash_rng_state(_rng_state()))
        return scores


def _condition_kwargs(model_inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in model_inputs.items()
        if key not in {
            "frames",
            "decoder_input_ids",
            "decoder_attention_mask",
            "context_type",
            "sequence_index",
            "frame_time_ms",
            "lookback_time",
            "lookahead_time",
        }
    }


def _encoder_condition_kwargs(condition_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: condition_kwargs[key]
        for key in ENCODER_CONDITION_KEYS
        if key in condition_kwargs
    }


def _greedy_processors(args, tokenizer, device, *, lookback_time: float):
    return build_logits_processor_list(
        tokenizer,
        cfg_scale=1.0,
        timeshift_bias=args.timeshift_bias,
        types_first=args.train.data.types_first,
        temperature=args.temperature,
        timing_temperature=args.timing_temperature,
        mania_column_temperature=args.mania_column_temperature,
        taiko_hit_temperature=args.taiko_hit_temperature,
        lookback_time=lookback_time,
        device=device,
        stateful_monotonic=APPROVED_STATEFUL_MONOTONIC_LOGITS,
    )


def _target_sampling_processors(args, tokenizer, device, *, lookback_time: float):
    """Mirror server.model_generate after it pops temperature but retains top-k/p."""

    processors = build_logits_processor_list(
        tokenizer,
        cfg_scale=args.cfg_scale,
        timeshift_bias=args.timeshift_bias,
        types_first=args.train.data.types_first,
        temperature=args.temperature,
        timing_temperature=args.timing_temperature,
        mania_column_temperature=args.mania_column_temperature,
        taiko_hit_temperature=args.taiko_hit_temperature,
        lookback_time=lookback_time,
        device=device,
        stateful_monotonic=APPROVED_STATEFUL_MONOTONIC_LOGITS,
    )
    if args.top_k:
        processors.append(TopKLogitsWarper(args.top_k))
    if args.top_p < 1.0:
        processors.append(TopPLogitsWarper(args.top_p))
    return processors


@torch.no_grad()
def _capture_target_transcript(
        args,
        model,
        tokenizer,
        *,
        prompt: torch.LongTensor,
        prompt_mask: torch.Tensor,
        frames: torch.Tensor,
        condition_kwargs: Mapping[str, Any],
        lookback_time: float,
        eos_token_ids: Sequence[int],
        max_new_tokens: int,
) -> tuple[tuple[int, ...], str, Any, dict[str, Any]]:
    audit = _RngAuditProcessor()
    processors = _target_sampling_processors(
        args,
        tokenizer,
        model.device,
        lookback_time=lookback_time,
    )
    processors.append(audit)
    start_rng = _rng_state()
    cache = get_cache(model, batch_size=1, num_beams=1, cfg_scale=1.0)
    with generation_profile_context(sdpa_backend=args.profile_sdpa_backend):
        output = model.generate(
            inputs=frames,
            decoder_input_ids=prompt,
            decoder_attention_mask=prompt_mask,
            **condition_kwargs,
            do_sample=True,
            num_beams=1,
            # Production-order processors already include these transforms.
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            past_key_values=cache,
            logits_processor=processors,
            eos_token_id=list(eos_token_ids),
            return_dict_in_generate=False,
        )
    generated = tuple(int(token) for token in output[0, prompt.shape[-1]:].tolist())
    if not generated:
        raise RuntimeError("Target capture generated no tokens.")
    if len(audit.rng_hashes_before_draw) != len(generated):
        raise RuntimeError(
            "Target RNG audit step count does not match generated tokens "
            f"({len(audit.rng_hashes_before_draw)} != {len(generated)})."
        )
    eos_set = {int(token) for token in eos_token_ids}
    stop_reason = "eos" if generated[-1] in eos_set else "max_new_tokens"
    if stop_reason == "max_new_tokens" and len(generated) != max_new_tokens:
        raise RuntimeError(
            "Target capture stopped short without an EOS token "
            f"({len(generated)} != {max_new_tokens})."
        )
    end_rng = _rng_state()
    return generated, stop_reason, end_rng, {
        "start_rng_hash": _hash_rng_state(start_rng),
        "final_rng_hash": _hash_rng_state(end_rng),
        "sample_draw_count": len(audit.rng_hashes_before_draw),
        "rng_hashes_before_draw": audit.rng_hashes_before_draw,
        "processor_prefix_lengths": audit.prefix_lengths,
    }


@torch.no_grad()
def _run_mini_encoder(
        encoder,
        frames: torch.Tensor,
        encoder_condition_kwargs: Mapping[str, Any],
) -> BaseModelOutput:
    output = encoder(
        frames=frames,
        **encoder_condition_kwargs,
        return_dict=True,
    )
    if not isinstance(output, BaseModelOutput):
        output = BaseModelOutput(last_hidden_state=output[0])
    return output


@torch.no_grad()
def _time_mini_encoder(
        model,
        frames: torch.Tensor,
        encoder_condition_kwargs: Mapping[str, Any],
        *,
        warmup: int,
        iters: int,
) -> tuple[BaseModelOutput, dict[str, Any]]:
    encoder = model.get_encoder()
    torch.cuda.synchronize(model.device)
    cold_start = torch.cuda.Event(enable_timing=True)
    cold_end = torch.cuda.Event(enable_timing=True)
    cold_wall_start = time.perf_counter()
    cold_start.record()
    output = _run_mini_encoder(encoder, frames, encoder_condition_kwargs)
    cold_end.record()
    torch.cuda.synchronize(model.device)
    cold_wall_ms = (time.perf_counter() - cold_wall_start) * 1000.0
    cold_cuda_ms = cold_start.elapsed_time(cold_end)

    for _ in range(warmup):
        output = _run_mini_encoder(encoder, frames, encoder_condition_kwargs)
    torch.cuda.synchronize(model.device)
    pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    warm_wall_start = time.perf_counter()
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = _run_mini_encoder(encoder, frames, encoder_condition_kwargs)
        end.record()
        pairs.append((start, end))
    torch.cuda.synchronize(model.device)
    warm_wall_ms = (time.perf_counter() - warm_wall_start) * 1000.0 / iters
    warm_cuda_values = [start.elapsed_time(end) for start, end in pairs]
    return output, {
        "cold_cuda_ms": cold_cuda_ms,
        "cold_wall_ms": cold_wall_ms,
        "warmup": warmup,
        "iters": iters,
        "warm_cuda_ms_values": warm_cuda_values,
        "warm_cuda_ms_mean": sum(warm_cuda_values) / len(warm_cuda_values),
        "warm_wall_ms_mean": warm_wall_ms,
    }


@torch.no_grad()
def _prefill_mini_decoder(
        model,
        *,
        prefix: torch.LongTensor,
        prefix_mask: torch.Tensor,
        encoder_outputs: BaseModelOutput,
        condition_kwargs: Mapping[str, Any],
) -> OneTokenDecodeState:
    cache = get_cache(model, batch_size=1, num_beams=1, cfg_scale=1.0)
    cache_position = torch.arange(prefix.shape[-1], device=prefix.device)
    prepared = model.prepare_inputs_for_generation(
        prefix,
        past_key_values=cache,
        use_cache=True,
        encoder_outputs=encoder_outputs,
        decoder_attention_mask=prefix_mask,
        cache_position=cache_position,
        **condition_kwargs,
    )
    outputs = model(**prepared)
    return OneTokenDecodeState(
        cache=cache,
        encoder_outputs=encoder_outputs,
        prompt_length=int(prefix.shape[-1]),
        prefill_cache_position=cache_position,
        prefill_prepared_inputs=prepared,
        prefill_logits=last_token_logits(outputs.logits),
    )


def _greedy_token(processors, prefix: torch.LongTensor, logits: torch.Tensor) -> torch.LongTensor:
    scores = processors(prefix, logits.detach().clone())
    return torch.argmax(scores, dim=-1, keepdim=True).to(torch.long)


@torch.no_grad()
def _propose_with_decoder_rebuild(
        args,
        model,
        tokenizer,
        *,
        base_prompt: torch.LongTensor,
        base_prompt_mask: torch.Tensor,
        committed_target_tokens: Sequence[int],
        requested_tokens: int,
        encoder_outputs: BaseModelOutput,
        condition_kwargs: Mapping[str, Any],
        lookback_time: float,
        iteration: int,
) -> DraftProposalMeasurement:
    if requested_tokens <= 0:
        raise ValueError("requested_tokens must be positive.")
    wall_start = time.perf_counter()
    overall_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    overall_end = torch.cuda.Event(enable_timing=True)
    overall_start.record()
    committed = torch.tensor(
        [list(committed_target_tokens)],
        dtype=torch.long,
        device=base_prompt.device,
    )
    prefix = torch.cat([base_prompt, committed], dim=-1)
    prefix_mask = torch.cat(
        [base_prompt_mask, torch.ones_like(committed, dtype=base_prompt_mask.dtype)],
        dim=-1,
    )
    state = _prefill_mini_decoder(
        model,
        prefix=prefix,
        prefix_mask=prefix_mask,
        encoder_outputs=encoder_outputs,
        condition_kwargs=condition_kwargs,
    )
    prefill_end.record()
    processors = _greedy_processors(
        args,
        tokenizer,
        model.device,
        lookback_time=lookback_time,
    )
    proposal_tensors = [_greedy_token(processors, prefix, state.prefill_logits)]
    q1_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for index in range(1, requested_tokens):
        proposed = torch.cat(proposal_tensors, dim=-1)
        full_prefix = torch.cat([prefix, proposed], dim=-1)
        full_mask = torch.cat(
            [prefix_mask, torch.ones_like(proposed, dtype=prefix_mask.dtype)],
            dim=-1,
        )
        q1_start = torch.cuda.Event(enable_timing=True)
        q1_end = torch.cuda.Event(enable_timing=True)
        q1_start.record()
        result = decode_one_token_raw_logits(
            model,
            state,
            full_prefix=full_prefix,
            full_attention_mask=full_mask,
            condition_kwargs=dict(condition_kwargs),
            cache_position=torch.tensor(
                [state.prompt_length + index - 1],
                device=prefix.device,
            ),
        )
        q1_end.record()
        proposal_tensors.append(_greedy_token(processors, full_prefix, result.logits))
        q1_pairs.append((q1_start, q1_end))
    overall_end.record()
    torch.cuda.synchronize(model.device)
    proposal = tuple(int(token) for token in torch.cat(proposal_tensors, dim=-1)[0].tolist())
    if len(proposal) != requested_tokens:
        raise AssertionError("Greedy proposal length changed unexpectedly.")
    return DraftProposalMeasurement(
        iteration=iteration,
        prefix_length=len(committed_target_tokens),
        requested_tokens=requested_tokens,
        proposal_token_ids=proposal,
        decoder_rebuild_cuda_ms=overall_start.elapsed_time(prefill_end),
        continuation_q1_cuda_ms=tuple(start.elapsed_time(end) for start, end in q1_pairs),
        rebuild_inclusive_cuda_ms=overall_start.elapsed_time(overall_end),
        rebuild_inclusive_wall_ms=(time.perf_counter() - wall_start) * 1000.0,
    )


@torch.no_grad()
def _measure_optimistic_steady_cache_k_q1(
        args,
        model,
        tokenizer,
        *,
        base_prompt: torch.LongTensor,
        base_prompt_mask: torch.Tensor,
        committed_target_tokens: Sequence[int],
        encoder_outputs: BaseModelOutput,
        condition_kwargs: Mapping[str, Any],
        lookback_time: float,
        speculation_k: int,
        warmup: int,
        iters: int,
) -> dict[str, Any]:
    values: list[float] = []
    total_iters = warmup + iters
    for timing_index in range(total_iters):
        committed = torch.tensor(
            [list(committed_target_tokens)],
            dtype=torch.long,
            device=base_prompt.device,
        )
        prefix = torch.cat([base_prompt, committed], dim=-1)
        prefix_mask = torch.cat(
            [base_prompt_mask, torch.ones_like(committed, dtype=base_prompt_mask.dtype)],
            dim=-1,
        )
        state = _prefill_mini_decoder(
            model,
            prefix=prefix,
            prefix_mask=prefix_mask,
            encoder_outputs=encoder_outputs,
            condition_kwargs=condition_kwargs,
        )
        processors = _greedy_processors(
            args,
            tokenizer,
            model.device,
            lookback_time=lookback_time,
        )
        current = _greedy_token(processors, prefix, state.prefill_logits)
        generated: list[torch.LongTensor] = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        q1_forwards = cached_q1_forwards_for_proposal(speculation_k)
        for index in range(q1_forwards):
            generated.append(current)
            active = torch.cat(generated, dim=-1)
            full_prefix = torch.cat([prefix, active], dim=-1)
            full_mask = torch.cat(
                [prefix_mask, torch.ones_like(active, dtype=prefix_mask.dtype)],
                dim=-1,
            )
            result = decode_one_token_raw_logits(
                model,
                state,
                full_prefix=full_prefix,
                full_attention_mask=full_mask,
                condition_kwargs=dict(condition_kwargs),
                cache_position=torch.tensor(
                    [state.prompt_length + index],
                    device=prefix.device,
                ),
            )
            current = _greedy_token(processors, full_prefix, result.logits)
        end.record()
        torch.cuda.synchronize(model.device)
        if timing_index >= warmup:
            values.append(start.elapsed_time(end))
    return {
        "scope": (
            "K-1 sequential mini q1 calls after ready-prefix prefill logits supply draft[0]; "
            "excludes encoder and rebuild"
        ),
        "prefix_generated_tokens": len(committed_target_tokens),
        "speculation_k": speculation_k,
        "q1_forwards_per_proposal": q1_forwards,
        "warmup": warmup,
        "iters": iters,
        "cuda_ms_values": values,
        "cuda_ms_mean": sum(values) / len(values),
    }


@torch.no_grad()
def run_mini_draft_gpu_scout(
        args,
        target_model,
        target_tokenizer,
        mini_model,
        mini_tokenizer,
        model_inputs: Mapping[str, Any],
        config: MiniDraftGpuScoutConfig,
        *,
        artifact_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the single approved transcript-replay and draft-cost feasibility gate."""

    if not torch.cuda.is_available():
        raise RuntimeError("The v32-mini scout requires a Slurm-allocated CUDA GPU.")
    if target_model.device.type != "cuda" or mini_model.device.type != "cuda":
        raise RuntimeError("Both target and mini must reside on the allocated CUDA GPU.")
    if args.precision != "fp32" or target_model.dtype != torch.float32 or mini_model.dtype != torch.float32:
        raise ValueError("The v32-mini scout is FP32-only for both models.")
    if not args.do_sample or args.num_beams != 1 or args.cfg_scale != 1.0:
        raise ValueError("Target capture requires do_sample=true, num_beams=1, cfg_scale=1.0.")

    prompt = model_inputs["decoder_input_ids"]
    prompt_mask = model_inputs["decoder_attention_mask"]
    frames = model_inputs["frames"]
    condition_kwargs = _condition_kwargs(model_inputs)
    lookback_time = float(model_inputs["lookback_time"])
    lookahead_time = float(model_inputs["lookahead_time"])
    context_type = ContextType(model_inputs["context_type"])
    eos_token_ids = get_eos_token_id(
        target_tokenizer,
        lookback_time=lookback_time,
        lookahead_time=lookahead_time,
        context_type=context_type,
    )

    torch.cuda.synchronize(target_model.device)
    models_allocated_bytes = int(torch.cuda.memory_allocated(target_model.device))
    torch.cuda.reset_peak_memory_stats(target_model.device)
    target_tokens, stop_reason, target_end_rng, target_rng_audit = _capture_target_transcript(
        args,
        target_model,
        target_tokenizer,
        prompt=prompt,
        prompt_mask=prompt_mask,
        frames=frames,
        condition_kwargs=condition_kwargs,
        lookback_time=lookback_time,
        eos_token_ids=eos_token_ids,
        max_new_tokens=config.max_new_tokens,
    )

    encoder_outputs, encoder_timing = _time_mini_encoder(
        mini_model,
        frames,
        _encoder_condition_kwargs(condition_kwargs),
        warmup=config.encoder_warmup,
        iters=config.encoder_timing_iters,
    )
    measurements: list[DraftProposalMeasurement] = []

    def propose(committed: tuple[int, ...], requested: int) -> tuple[int, ...]:
        measurement = _propose_with_decoder_rebuild(
            args,
            mini_model,
            mini_tokenizer,
            base_prompt=prompt,
            base_prompt_mask=prompt_mask,
            committed_target_tokens=committed,
            requested_tokens=requested,
            encoder_outputs=encoder_outputs,
            condition_kwargs=condition_kwargs,
            lookback_time=lookback_time,
            iteration=len(measurements),
        )
        measurements.append(measurement)
        return measurement.proposal_token_ids

    replay = replay_closed_loop_target_transcript(
        target_tokens,
        speculation_k=config.speculation_k,
        propose=propose,
    )
    full_events = replay.full_span_events
    short_events = replay.short_boundary_events
    if not full_events:
        raise RuntimeError("The 256-token scout produced no full K4 proposal events.")
    timing_event = min(
        full_events,
        key=lambda event: abs(event.prefix_length_before - len(target_tokens) // 2),
    )
    steady_cache_timing = _measure_optimistic_steady_cache_k_q1(
        args,
        mini_model,
        mini_tokenizer,
        base_prompt=prompt,
        base_prompt_mask=prompt_mask,
        committed_target_tokens=target_tokens[:timing_event.prefix_length_before],
        encoder_outputs=encoder_outputs,
        condition_kwargs=condition_kwargs,
        lookback_time=lookback_time,
        speculation_k=config.speculation_k,
        warmup=config.steady_cache_warmup,
        iters=config.steady_cache_timing_iters,
    )

    rng_after_all_mini_work = _rng_state()
    target_rng_unchanged = _rng_states_equal(target_end_rng, rng_after_all_mini_work)
    full_ids = {event.iteration for event in full_events}
    short_ids = {event.iteration for event in short_events}
    full_measurements = [item for item in measurements if item.iteration in full_ids]
    short_measurements = [item for item in measurements if item.iteration in short_ids]
    full_histogram = replay.full_span_histogram()
    short_output_tokens = sum(len(event.committed_token_ids) for event in short_events)
    runnable_cuda_projection = FiniteWindowProjection(
        target_output_tokens=len(target_tokens),
        full_span_histogram=full_histogram,
        full_span_draft_cost_ms=sum(item.rebuild_inclusive_cuda_ms for item in full_measurements),
        short_boundary_output_tokens=short_output_tokens,
        short_boundary_draft_cost_ms=sum(
            item.rebuild_inclusive_cuda_ms for item in short_measurements
        ),
        fixed_draft_overhead_ms=encoder_timing["warm_cuda_ms_mean"],
    )
    runnable_wall_projection = FiniteWindowProjection(
        target_output_tokens=len(target_tokens),
        full_span_histogram=full_histogram,
        full_span_draft_cost_ms=sum(item.rebuild_inclusive_wall_ms for item in full_measurements),
        short_boundary_output_tokens=short_output_tokens,
        short_boundary_draft_cost_ms=sum(
            item.rebuild_inclusive_wall_ms for item in short_measurements
        ),
        fixed_draft_overhead_ms=encoder_timing["warm_wall_ms_mean"],
    )
    optimistic_k_ms = steady_cache_timing["cuda_ms_mean"]
    optimistic_projection = FiniteWindowProjection(
        target_output_tokens=len(target_tokens),
        full_span_histogram=full_histogram,
        full_span_draft_cost_ms=len(full_measurements) * optimistic_k_ms,
        short_boundary_output_tokens=short_output_tokens,
        short_boundary_draft_cost_ms=sum(
            optimistic_k_ms
            * cached_q1_forwards_for_proposal(item.requested_tokens)
            / cached_q1_forwards_for_proposal(config.speculation_k)
            for item in short_measurements
        ),
        fixed_draft_overhead_ms=encoder_timing["warm_cuda_ms_mean"],
    )
    peak_allocated_bytes = int(torch.cuda.max_memory_allocated(target_model.device))
    exactness_pass = replay.exact and target_rng_unchanged
    replay_stop_reason = (
        "eos"
        if replay.replayed_token_ids[-1] in {int(token) for token in eos_token_ids}
        else "max_new_tokens"
    )
    stop_reason_match = replay_stop_reason == stop_reason
    exactness_pass = exactness_pass and stop_reason_match
    strict_cost_pass = clears_strict_cuda_and_wall_gate(
        runnable_cuda_projection,
        runnable_wall_projection,
    )
    passed = exactness_pass and strict_cost_pass
    result_class = (
        "bounded_v32_mini_feasibility_pass"
        if passed
        else "bounded_v32_mini_feasibility_rejected"
    )
    return {
        "gate": GATE_NAME,
        "pass": passed,
        "result_class": result_class,
        "claim_scope": "one_real_window_transcript_replay_and_cost_scout_not_runtime_promotion",
        "decision": (
            "authorize_only_a_multi_window_acceptance_cost_repeat"
            if passed
            else "reject_v32_mini_speculative_path_below_strict_five_percent"
        ),
        "speculation_k": config.speculation_k,
        "max_new_tokens": config.max_new_tokens,
        "stateful_monotonic_logits_processor": (
            APPROVED_STATEFUL_MONOTONIC_LOGITS
        ),
        "proposal_policy": (
            "greedy_argmax_after_production_structural_logits_processors; "
            "no mini sampling and no target RNG access"
        ),
        "target_capture": {
            "generated_token_ids": list(target_tokens),
            "generated_token_ids_sha256": token_ids_sha256(target_tokens),
            "generated_tokens": len(target_tokens),
            "stop_reason": stop_reason,
            "eos_token_ids": [int(token) for token in eos_token_ids],
            **target_rng_audit,
        },
        "closed_loop_replay": replay.to_manifest(),
        "target_rng_non_consumption": {
            "pass": target_rng_unchanged,
            "target_final_rng_hash": _hash_rng_state(target_end_rng),
            "after_all_mini_work_rng_hash": _hash_rng_state(rng_after_all_mini_work),
        },
        "draft_measurements": [item.to_manifest() for item in measurements],
        "timing": {
            "mini_encoder": encoder_timing,
            "optimistic_steady_cache": steady_cache_timing,
            "runnable_rebuild_inclusive_cuda_model": {
                **runnable_cuda_projection.to_manifest(),
                "draft_cost_basis": "synchronized CUDA events",
                "baseline_cost_basis": "accepted target q1 model-time allowance",
            },
            "runnable_rebuild_inclusive_wall_conservative": {
                **runnable_wall_projection.to_manifest(),
                "draft_cost_basis": "synchronized host wall around each mini proposal",
                "baseline_cost_basis": "accepted target q1 model-time allowance",
                "warning": (
                    "This deliberately charges host-side draft overhead against a model-time denominator; "
                    "it is a conservative second gate, not a scheduler-wall throughput claim."
                ),
            },
            "optimistic_steady_cache_projection": optimistic_projection.to_manifest(),
            "strict_gate_uses": (
                "both runnable_rebuild_inclusive_cuda_model and "
                "runnable_rebuild_inclusive_wall_conservative"
            ),
        },
        "peak_vram": {
            "models_allocated_bytes_before_work": models_allocated_bytes,
            "combined_peak_allocated_bytes": peak_allocated_bytes,
            "working_peak_delta_bytes": peak_allocated_bytes - models_allocated_bytes,
        },
        "artifacts": dict(artifact_manifest),
        "exactness": {
            "pass": exactness_pass,
            "replayed_target_transcript_exact": replay.exact,
            "target_stop_reason": stop_reason,
            "replayed_stop_reason": replay_stop_reason,
            "target_stop_reason_preserved": stop_reason_match,
            "target_rng_unchanged_after_mini": target_rng_unchanged,
            "target_cache_rollback_implemented": False,
            "production_hook_implemented": False,
        },
        "keep_policy": {
            "minimum_improvement_fraction": CAMPAIGN_MINIMUM_IMPROVEMENT_FRACTION,
            "strictly_greater_required": True,
            "runnable_cuda_improvement_fraction": runnable_cuda_projection.improvement_fraction,
            "runnable_wall_improvement_fraction": runnable_wall_projection.improvement_fraction,
            "cuda_clears": runnable_cuda_projection.clears_minimum_improvement,
            "wall_clears": runnable_wall_projection.clears_minimum_improvement,
            "clears": strict_cost_pass,
        },
    }
