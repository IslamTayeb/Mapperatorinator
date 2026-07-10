"""GPU-only q_len=K target-span verifier; never imported by the V32 runtime."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import nn
from transformers import LogitsProcessor, LogitsProcessorList
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from osuT5.osuT5.inference.cache_utils import MapperatorinatorCache, get_cache
from osuT5.osuT5.inference.optimized.single.session import (
    OneTokenDecodeState,
    decode_one_token_raw_logits,
    prefill_static_cache,
)
from osuT5.osuT5.inference.server import get_eos_token_id
from osuT5.osuT5.inference.optimized.single.logits import (
    build_single_logits_processor_list as build_logits_processor_list,
)
from osuT5.osuT5.runtime_profiling import generation_profile_context
from osuT5.osuT5.tokenizer import ContextType


GATE_NAME = "target_span_q_len_numerics"
SUPPORTED_K = (2, 4, 8)


@dataclass(frozen=True)
class TargetSpanGateConfig:
    speculation_k: int = 2
    atol: float = 1e-4
    rtol: float = 1e-4
    top_k: int = 20
    timing_warmup: int = 5
    timing_iters: int = 25
    fixed_shape_cuda_graph_ceiling: bool = False

    def __post_init__(self) -> None:
        if self.speculation_k not in SUPPORTED_K:
            raise ValueError(f"speculation_k must be one of {SUPPORTED_K}.")
        if self.atol < 0 or self.rtol < 0:
            raise ValueError("atol and rtol must be non-negative.")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.timing_warmup < 0 or self.timing_iters <= 0:
            raise ValueError("timing_warmup must be non-negative and timing_iters must be positive.")


def load_and_validate_prior_gate(
        speculation_k: int,
        prior_report_path: Path | None,
) -> dict[str, Any] | None:
    """Keep K=4/8 unreachable until the immediately preceding gate passed."""

    if speculation_k not in SUPPORTED_K:
        raise ValueError(f"speculation_k must be one of {SUPPORTED_K}.")
    required_prior_k = {2: None, 4: 2, 8: 4}[speculation_k]
    if required_prior_k is None:
        return None
    if prior_report_path is None:
        raise ValueError(
            f"K={speculation_k} is gated on a passing K={required_prior_k} report; "
            "provide --prior-gate-report."
        )
    report = json.loads(prior_report_path.read_text(encoding="utf-8"))
    if report.get("gate") != GATE_NAME:
        raise ValueError(f"Prior report is not a {GATE_NAME!r} gate.")
    if report.get("speculation_k") != required_prior_k:
        raise ValueError(
            f"K={speculation_k} requires a K={required_prior_k} report, "
            f"got K={report.get('speculation_k')!r}."
        )
    if report.get("pass") is not True:
        raise ValueError(f"K={speculation_k} requires the K={required_prior_k} report to pass.")
    return report


RngState = tuple[torch.Tensor, list[torch.Tensor] | None]


def _rng_state() -> RngState:
    return (
        torch.random.get_rng_state().clone(),
        [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
    )


def _set_rng_state(state: RngState) -> None:
    cpu_state, cuda_states = state
    torch.random.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


def _hash_rng_state(state: RngState) -> str:
    digest = hashlib.sha256()
    digest.update(state[0].contiguous().numpy().tobytes())
    for cuda_state in state[1] or []:
        digest.update(cuda_state.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _rng_states_equal(left: RngState, right: RngState) -> bool:
    if not torch.equal(left[0], right[0]):
        return False
    left_cuda = left[1] or []
    right_cuda = right[1] or []
    return len(left_cuda) == len(right_cuda) and all(
        torch.equal(left_item, right_item)
        for left_item, right_item in zip(left_cuda, right_cuda)
    )


def _update_tensor_hash(digest: Any, name: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().contiguous().cpu()
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())


def _cache_tensors(
        cache: MapperatorinatorCache,
        *,
        self_prefix_length: int | None,
) -> Iterable[tuple[str, torch.Tensor]]:
    for cache_name, cache_object in (
            ("self", cache.self_attention_cache),
            ("cross", cache.cross_attention_cache),
    ):
        for layer_index, layer in enumerate(cache_object.layers):
            if not getattr(layer, "is_initialized", False):
                continue
            for tensor_name in ("keys", "values"):
                value = getattr(layer, tensor_name)
                if cache_name == "self" and self_prefix_length is not None:
                    value = value[:, :, :self_prefix_length, :]
                yield f"{cache_name}.{layer_index}.{tensor_name}", value


def hash_cache(
        cache: MapperatorinatorCache,
        *,
        self_prefix_length: int | None = None,
) -> str:
    """Hash exact cache bytes; full hashes include the zeroed static suffix."""

    digest = hashlib.sha256()
    for name, tensor in _cache_tensors(cache, self_prefix_length=self_prefix_length):
        _update_tensor_hash(digest, name, tensor)
    digest.update(repr(sorted(cache.is_updated.items())).encode("ascii"))
    return digest.hexdigest()


def compare_cache_prefixes(
        reference: MapperatorinatorCache,
        candidate: MapperatorinatorCache,
        *,
        self_prefix_length: int,
        atol: float,
        rtol: float,
) -> dict[str, Any]:
    reference_tensors = dict(_cache_tensors(reference, self_prefix_length=self_prefix_length))
    candidate_tensors = dict(_cache_tensors(candidate, self_prefix_length=self_prefix_length))
    if reference_tensors.keys() != candidate_tensors.keys():
        return {"allclose": False, "max_abs": None, "tensor_names_match": False}
    max_abs = 0.0
    allclose = True
    for name in reference_tensors:
        reference_tensor = reference_tensors[name].to(torch.float32)
        candidate_tensor = candidate_tensors[name].to(torch.float32)
        allclose = allclose and bool(torch.allclose(reference_tensor, candidate_tensor, atol=atol, rtol=rtol))
        if reference_tensor.numel():
            max_abs = max(max_abs, float((reference_tensor - candidate_tensor).abs().max().item()))
    return {"allclose": allclose, "max_abs": max_abs, "tensor_names_match": True}


def zero_static_self_cache_suffix(
        cache: MapperatorinatorCache,
        *,
        start_position: int,
        end_position: int,
) -> None:
    """Verifier-only physical rollback oracle; not a supported runtime cache API."""

    if start_position < 0 or end_position < start_position:
        raise ValueError("Invalid static-cache rollback interval.")
    for layer in cache.self_attention_cache.layers:
        if not getattr(layer, "is_initialized", False):
            raise RuntimeError("Cannot roll back an uninitialized self-attention cache layer.")
        if end_position > layer.keys.shape[2]:
            raise ValueError("Rollback interval exceeds the static self-cache capacity.")
        layer.keys[:, :, start_position:end_position, :].zero_()
        layer.values[:, :, start_position:end_position, :].zero_()


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


def _sampling_processors(args, tokenizer, device, *, lookback_time: float) -> LogitsProcessorList:
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
        stateful_monotonic=bool(getattr(args, "inference_stateful_monotonic_logits_processor", False)),
    )
    # model.generate normally appends these after the caller-provided processors.
    if args.temperature != 1.0:
        processors.append(TemperatureLogitsWarper(args.temperature))
    if args.top_k:
        processors.append(TopKLogitsWarper(args.top_k))
    if args.top_p < 1.0:
        processors.append(TopPLogitsWarper(args.top_p))
    return processors


class _CaptureProcessedScoresAndRng(LogitsProcessor):
    def __init__(self) -> None:
        self.input_ids: list[torch.Tensor] = []
        self.scores: list[torch.Tensor] = []
        self.rng_states: list[RngState] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.input_ids.append(input_ids.detach().clone())
        self.scores.append(scores.detach().to(torch.float32).clone())
        self.rng_states.append(_rng_state())
        return scores


def _capture_production_draft(
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
        speculation_k: int,
) -> tuple[torch.LongTensor, _CaptureProcessedScoresAndRng, RngState]:
    capture = _CaptureProcessedScoresAndRng()
    processors = _sampling_processors(args, tokenizer, model.device, lookback_time=lookback_time)
    processors.append(capture)
    cache = get_cache(model, batch_size=1, num_beams=1, cfg_scale=1.0)
    output = model.generate(
        inputs=frames,
        decoder_input_ids=prompt,
        decoder_attention_mask=prompt_mask,
        **condition_kwargs,
        do_sample=True,
        num_beams=1,
        # The exact production warpers are already in `processors`, in production order.
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        max_new_tokens=speculation_k,
        use_cache=True,
        past_key_values=cache,
        logits_processor=processors,
        eos_token_id=list(eos_token_ids),
    )
    generated = output[:, prompt.shape[-1]:]
    if generated.shape[-1] != speculation_k or len(capture.scores) != speculation_k:
        raise RuntimeError(
            "The production capture stopped before the requested speculative block; "
            f"tokens={generated.shape[-1]}, score_steps={len(capture.scores)}, K={speculation_k}. "
            "Choose another production prefix/window rather than weakening EOS semantics."
        )
    end_rng_state = _rng_state()
    return generated.detach().clone(), capture, end_rng_state


@torch.no_grad()
def _target_span_forward(
        model,
        state: OneTokenDecodeState,
        *,
        prompt: torch.LongTensor,
        prompt_mask: torch.Tensor,
        fixed_input_tokens: torch.LongTensor,
        condition_kwargs: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    speculation_k = int(fixed_input_tokens.shape[-1])
    full_prefix = torch.cat([prompt, fixed_input_tokens], dim=-1)
    full_mask = torch.cat(
        [prompt_mask, torch.ones_like(fixed_input_tokens, dtype=prompt_mask.dtype)],
        dim=-1,
    )
    cache_position = torch.arange(
        state.prompt_length,
        state.prompt_length + speculation_k,
        device=prompt.device,
    )
    prepared = model.prepare_inputs_for_generation(
        full_prefix,
        past_key_values=state.cache,
        use_cache=True,
        encoder_outputs=state.encoder_outputs,
        decoder_attention_mask=full_mask,
        cache_position=cache_position,
        **condition_kwargs,
    )
    prepared_tokens = prepared.get("decoder_input_ids")
    if not isinstance(prepared_tokens, torch.Tensor) or prepared_tokens.shape[-1] != speculation_k:
        raise RuntimeError(
            "prepare_inputs_for_generation did not preserve q_len=K; "
            f"prepared_shape={getattr(prepared_tokens, 'shape', None)}, K={speculation_k}."
        )
    outputs = model(**prepared, return_dict=True)
    logits = outputs.logits.detach().to(torch.float32, copy=True)
    if logits.shape[1] != speculation_k:
        raise RuntimeError(f"Target span returned q_len={logits.shape[1]}, expected K={speculation_k}.")
    return logits, prepared


@torch.no_grad()
def _sequential_target_forwards(
        model,
        state: OneTokenDecodeState,
        *,
        prompt: torch.LongTensor,
        prompt_mask: torch.Tensor,
        fixed_input_tokens: torch.LongTensor,
        condition_kwargs: Mapping[str, Any],
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    logits: list[torch.Tensor] = []
    prepared_inputs: list[dict[str, Any]] = []
    for index in range(fixed_input_tokens.shape[-1]):
        active_tokens = fixed_input_tokens[:, :index + 1]
        full_prefix = torch.cat([prompt, active_tokens], dim=-1)
        full_mask = torch.cat(
            [prompt_mask, torch.ones_like(active_tokens, dtype=prompt_mask.dtype)],
            dim=-1,
        )
        result = decode_one_token_raw_logits(
            model,
            state,
            full_prefix=full_prefix,
            full_attention_mask=full_mask,
            condition_kwargs=dict(condition_kwargs),
            cache_position=torch.tensor([state.prompt_length + index], device=prompt.device),
        )
        logits.append(result.logits)
        prepared_inputs.append(result.prepared_inputs)
    return logits, prepared_inputs


def _sample_verification_logits(
        args,
        tokenizer,
        *,
        prompt: torch.LongTensor,
        draft_tokens: torch.LongTensor,
        verification_logits: Sequence[torch.Tensor],
        start_rng_state: RngState,
        lookback_time: float,
) -> tuple[list[int], RngState, list[torch.Tensor]]:
    processors = _sampling_processors(args, tokenizer, prompt.device, lookback_time=lookback_time)
    saved_state = _rng_state()
    _set_rng_state(start_rng_state)
    sampled_tokens: list[int] = []
    processed_scores: list[torch.Tensor] = []
    for index, raw_logits in enumerate(verification_logits):
        processor_prefix = torch.cat([prompt, draft_tokens[:, :index]], dim=-1)
        scores = processors(processor_prefix, raw_logits.detach().clone())
        processed_scores.append(scores.detach().to(torch.float32).clone())
        token = torch.multinomial(nn.functional.softmax(scores, dim=-1), num_samples=1).squeeze(1)
        sampled_tokens.append(int(token.item()))
    final_state = _rng_state()
    _set_rng_state(saved_state)
    return sampled_tokens, final_state, processed_scores


def _forced_eos_no_extra_sampling_audit(
        *,
        sequential_logits: torch.Tensor,
        span_logits: torch.Tensor,
        start_rng_state: RngState,
        eos_token_id: int,
) -> dict[str, Any]:
    saved_state = _rng_state()

    def one_draw(logits: torch.Tensor) -> tuple[int, RngState, list[int]]:
        forced_scores = torch.full_like(logits, -torch.inf)
        forced_scores[:, eos_token_id] = 0.0
        _set_rng_state(start_rng_state)
        token = torch.multinomial(nn.functional.softmax(forced_scores, dim=-1), num_samples=1).squeeze(1)
        final_state = _rng_state()
        next_samples = [
            int(torch.multinomial(torch.ones_like(logits), num_samples=1).item())
            for _ in range(8)
        ]
        return int(token.item()), final_state, next_samples

    sequential_token, sequential_final, sequential_next = one_draw(sequential_logits)
    span_token, span_final, span_next = one_draw(span_logits)
    _set_rng_state(saved_state)
    passed = (
        sequential_token == eos_token_id
        and span_token == eos_token_id
        and _rng_states_equal(sequential_final, span_final)
        and sequential_next == span_next
    )
    return {
        "pass": passed,
        "forced_eos_token_id": eos_token_id,
        "sample_draw_count": 1,
        "stopped_before_position": 1,
        "sequential_token": sequential_token,
        "span_token": span_token,
        "final_rng_match": _rng_states_equal(sequential_final, span_final),
        "next_samples_match": sequential_next == span_next,
        "sequential_final_rng_hash": _hash_rng_state(sequential_final),
        "span_final_rng_hash": _hash_rng_state(span_final),
    }


def _timed_cuda_calls(
        call: Callable[[], Any],
        reset_cache: Callable[[], None],
        *,
        warmup: int,
        iters: int,
) -> float:
    for _ in range(warmup):
        reset_cache()
        call()
    torch.cuda.synchronize()
    pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for _ in range(iters):
        reset_cache()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        pairs.append((start, end))
    torch.cuda.synchronize()
    return sum(start.elapsed_time(end) for start, end in pairs) / iters


def _peak_cuda_memory(
        call: Callable[[], Any],
        reset_cache: Callable[[], None],
) -> dict[str, int]:
    reset_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    call()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return {
        "baseline_allocated_bytes": int(baseline),
        "peak_allocated_bytes": int(peak),
        "peak_delta_bytes": int(peak - baseline),
    }


def project_k4_current_stack(replay_ms: float) -> dict[str, Any]:
    """Project the K=4 span onto the current five-full structural denominator."""

    baseline_seconds = 163.088
    serial_tokens = 42_634
    empty_q1_calls = 20_078
    q4_span_calls = 14_257
    accepted_q1_ms = baseline_seconds * 1000.0 / serial_tokens
    projected_seconds = (
        empty_q1_calls * accepted_q1_ms + q4_span_calls * replay_ms
    ) / 1000.0
    improvement_fraction = (baseline_seconds - projected_seconds) / baseline_seconds
    calculated_threshold_ms = (
        0.95 * baseline_seconds * 1000.0 - empty_q1_calls * accepted_q1_ms
    ) / q4_span_calls
    strict_campaign_threshold_ms = 5.480
    return {
        "baseline_seconds": baseline_seconds,
        "serial_main_tokens": serial_tokens,
        "accepted_q1_allowance_ms": accepted_q1_ms,
        "empty_q1_calls": empty_q1_calls,
        "q4_span_calls": q4_span_calls,
        "q4_replay_ms": replay_ms,
        "projected_seconds": projected_seconds,
        "projected_main_tokens_per_second": serial_tokens / projected_seconds,
        "projected_improvement_fraction": improvement_fraction,
        "calculated_five_percent_q4_threshold_ms": calculated_threshold_ms,
        "strict_campaign_q4_threshold_ms": strict_campaign_threshold_ms,
        "clears_five_percent": replay_ms < strict_campaign_threshold_ms,
    }


def _pointer_manifest(
        prepared_inputs: Mapping[str, Any],
        cache: MapperatorinatorCache,
) -> dict[str, Any]:
    prepared_tensors: dict[str, Any] = {}
    for key, value in sorted(prepared_inputs.items()):
        tensor = value if isinstance(value, torch.Tensor) else getattr(value, "last_hidden_state", None)
        if not isinstance(tensor, torch.Tensor):
            continue
        tensor_name = key if isinstance(value, torch.Tensor) else f"{key}.last_hidden_state"
        prepared_tensors[tensor_name] = {
            "data_ptr": int(tensor.data_ptr()),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    return {
        "prepared_tensors": prepared_tensors,
        "cache_tensors": {
            name: {
                "data_ptr": int(value.data_ptr()),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in _cache_tensors(cache, self_prefix_length=None)
        },
    }


def _timed_cuda_graph_replays(
        graph: torch.cuda.CUDAGraph,
        reset_cache: Callable[[], None],
        *,
        warmup: int,
        iters: int,
) -> float:
    for _ in range(warmup):
        reset_cache()
        graph.replay()
    torch.cuda.synchronize()
    event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for _ in range(iters):
        reset_cache()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        event_pairs.append((start, end))
    torch.cuda.synchronize()
    return sum(start.elapsed_time(end) for start, end in event_pairs) / iters


@torch.no_grad()
def _fixed_shape_k4_cuda_graph_ceiling(
        args,
        model,
        tokenizer,
        *,
        prompt: torch.LongTensor,
        draft_tokens: torch.LongTensor,
        span_prepared: Mapping[str, Any],
        span_state: OneTokenDecodeState,
        sequential_state: OneTokenDecodeState,
        eager_span_logits: torch.Tensor,
        sampling_start_rng: RngState,
        production_end_rng: RngState,
        lookback_time: float,
        final_prefix_length: int,
        config: TargetSpanGateConfig,
        compare_logits: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """One fixed-shape CUDA-graph ceiling, isolated from runtime integration."""

    if config.speculation_k != 4:
        raise ValueError("The fixed-shape CUDA-graph ceiling is authorized only for K=4.")
    if span_prepared["decoder_input_ids"].shape != (1, 4):
        raise RuntimeError(
            "K4 graph capture requires a stable [1,4] decoder_input_ids tensor, "
            f"got {tuple(span_prepared['decoder_input_ids'].shape)}."
        )
    prompt_length = int(prompt.shape[-1])

    def reset_span_cache() -> None:
        zero_static_self_cache_suffix(
            span_state.cache,
            start_position=prompt_length,
            end_position=final_prefix_length,
        )

    pointers_before = _pointer_manifest(span_prepared, span_state.cache)
    warmup_started = torch.cuda.Event(enable_timing=True)
    warmup_finished = torch.cuda.Event(enable_timing=True)
    warmup_wall_started = time.perf_counter()
    warmup_started.record()
    with generation_profile_context(sdpa_backend=args.profile_sdpa_backend):
        for _ in range(config.timing_warmup):
            reset_span_cache()
            model(**span_prepared, return_dict=True)
    warmup_finished.record()
    torch.cuda.synchronize()
    warmup_wall_seconds = time.perf_counter() - warmup_wall_started
    warmup_cuda_ms = warmup_started.elapsed_time(warmup_finished)

    reset_span_cache()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    capture_wall_started = time.perf_counter()
    with torch.cuda.graph(graph):
        with generation_profile_context(sdpa_backend=args.profile_sdpa_backend):
            graph_outputs = model(**span_prepared, return_dict=True)
    torch.cuda.synchronize()
    capture_wall_seconds = time.perf_counter() - capture_wall_started
    if not isinstance(graph_outputs.logits, torch.Tensor):
        raise RuntimeError("Captured K4 graph did not expose a stable logits tensor.")
    output_data_ptr_after_capture = int(graph_outputs.logits.data_ptr())

    pointers_after_capture = _pointer_manifest(span_prepared, span_state.cache)
    pointer_stability_after_capture = pointers_before == pointers_after_capture
    reset_span_cache()
    rng_before_replay = _rng_state()
    graph.replay()
    torch.cuda.synchronize()
    rng_after_replay = _rng_state()
    replay_logits = graph_outputs.logits.detach().to(torch.float32, copy=True)
    output_data_ptr_after_replay = int(graph_outputs.logits.data_ptr())
    pointers_after_replay = _pointer_manifest(span_prepared, span_state.cache)
    pointer_stability_after_replay = pointers_before == pointers_after_replay

    logits_comparisons = [
        compare_logits(
            eager_span_logits[:, index, :],
            replay_logits[:, index, :],
            atol=config.atol,
            rtol=config.rtol,
            top_k=config.top_k,
        )
        for index in range(config.speculation_k)
    ]
    replay_verification_logits = [
        span_state.prefill_logits,
        *[replay_logits[:, index, :] for index in range(config.speculation_k - 1)],
    ]
    replay_samples, replay_final_rng, _ = _sample_verification_logits(
        args,
        tokenizer,
        prompt=prompt,
        draft_tokens=draft_tokens,
        verification_logits=replay_verification_logits,
        start_rng_state=sampling_start_rng,
        lookback_time=lookback_time,
    )
    draft_token_ids = [int(token) for token in draft_tokens[0].tolist()]
    cache_comparison = compare_cache_prefixes(
        sequential_state.cache,
        span_state.cache,
        self_prefix_length=final_prefix_length,
        atol=config.atol,
        rtol=config.rtol,
    )
    cache_bitwise_match = (
        hash_cache(sequential_state.cache, self_prefix_length=final_prefix_length)
        == hash_cache(span_state.cache, self_prefix_length=final_prefix_length)
    )
    replay_ms = _timed_cuda_graph_replays(
        graph,
        reset_span_cache,
        warmup=config.timing_warmup,
        iters=config.timing_iters,
    )
    projection = project_k4_current_stack(replay_ms)
    numeric_pass = all(
        item["allclose"] and item["topk_match"]
        for item in logits_comparisons
    )
    sampled_exact = replay_samples == draft_token_ids and _rng_states_equal(
        replay_final_rng,
        production_end_rng,
    )
    graph_forward_rng_unchanged = _rng_states_equal(rng_before_replay, rng_after_replay)
    stable_output_buffer = output_data_ptr_after_capture == output_data_ptr_after_replay
    graph_safety_pass = (
        pointer_stability_after_capture
        and pointer_stability_after_replay
        and stable_output_buffer
        and graph_forward_rng_unchanged
    )
    passed = (
        numeric_pass
        and sampled_exact
        and cache_comparison["allclose"]
        and graph_safety_pass
        and projection["clears_five_percent"]
    )
    return {
        "enabled": True,
        "pass": passed,
        "scope": "verifier_only_fixed_shape_k4_cuda_graph_ceiling",
        "graph_count": 1,
        "capture": {
            "pre_capture_warmup_calls": config.timing_warmup,
            "pre_capture_warmup_cuda_ms": warmup_cuda_ms,
            "pre_capture_warmup_wall_seconds": warmup_wall_seconds,
            "capture_wall_seconds": capture_wall_seconds,
        },
        "replay": {
            "warmup_replays": config.timing_warmup,
            "measured_replays": config.timing_iters,
            "cuda_ms_per_replay": replay_ms,
            "cache_reset_outside_timed_region": True,
        },
        "graph_safety": {
            "pass": graph_safety_pass,
            "fixed_decoder_input_shape": list(span_prepared["decoder_input_ids"].shape),
            "prepared_and_cache_pointers_stable_after_capture": pointer_stability_after_capture,
            "prepared_and_cache_pointers_stable_after_replay": pointer_stability_after_replay,
            "output_buffer_data_ptr_after_capture": output_data_ptr_after_capture,
            "output_buffer_data_ptr_after_replay": output_data_ptr_after_replay,
            "output_buffer_stable": stable_output_buffer,
            "model_forward_rng_unchanged": graph_forward_rng_unchanged,
        },
        "numeric": {
            "pass": numeric_pass,
            "logit_topk_comparisons": logits_comparisons,
            "sampled_token_ids": replay_samples,
            "captured_production_token_ids": draft_token_ids,
            "sampled_tokens_and_final_rng_exact": sampled_exact,
            "replay_final_rng_hash": _hash_rng_state(replay_final_rng),
            "production_final_rng_hash": _hash_rng_state(production_end_rng),
            "active_prefix_cache_allclose": cache_comparison,
            "active_prefix_cache_bitwise_hash_match": cache_bitwise_match,
        },
        "current_stack_projection": projection,
        "decision": (
            "ceiling_clears_5pct_continue_verifier_only"
            if passed
            else "reject_ngram_speculative_runtime_below_current_stack_keep_bar"
        ),
        "revisit_conditions": (
            "Revisit only if the accepted q1 denominator materially slows, the proposal structure changes, "
            "or a new fixed-shape target kernel/graph ceiling demonstrates q4 below the recorded threshold."
        ),
    }


@torch.no_grad()
def run_target_span_gpu_gate(
        args,
        model,
        tokenizer,
        model_inputs: Mapping[str, Any],
        config: TargetSpanGateConfig,
        *,
        compare_logits: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Compare one q_len=K target call with K q_len=1 calls on fixed production tokens."""

    if not torch.cuda.is_available() or model.device.type != "cuda":
        raise RuntimeError("The target-span numerical gate requires a Slurm-allocated CUDA GPU.")
    if args.precision != "fp32" or model.dtype != torch.float32:
        raise ValueError("The target-span numerical gate is FP32-only.")
    if not args.do_sample or args.num_beams != 1 or args.cfg_scale != 1.0:
        raise ValueError("The target-span gate requires do_sample=true, num_beams=1, and cfg_scale=1.0.")

    prompt = model_inputs["decoder_input_ids"]
    prompt_mask = model_inputs["decoder_attention_mask"]
    frames = model_inputs["frames"]
    prompt_length = int(prompt.shape[-1])
    condition_kwargs = _condition_kwargs(model_inputs)
    lookback_time = float(model_inputs["lookback_time"])
    lookahead_time = float(model_inputs["lookahead_time"])
    context_type = ContextType(model_inputs["context_type"])
    eos_token_ids = get_eos_token_id(
        tokenizer,
        lookback_time=lookback_time,
        lookahead_time=lookahead_time,
        context_type=context_type,
    )

    capture_start_rng = _rng_state()
    with generation_profile_context(sdpa_backend=args.profile_sdpa_backend):
        draft_tokens, production_capture, production_end_rng = _capture_production_draft(
            args,
            model,
            tokenizer,
            prompt=prompt,
            prompt_mask=prompt_mask,
            frames=frames,
            condition_kwargs=condition_kwargs,
            lookback_time=lookback_time,
            eos_token_ids=eos_token_ids,
            speculation_k=config.speculation_k,
        )
    _set_rng_state(capture_start_rng)

    sequential_state = prefill_static_cache(
        model,
        prompt=prompt,
        prompt_attention_mask=prompt_mask,
        frames=frames,
        condition_kwargs=condition_kwargs,
    )
    span_state = prefill_static_cache(
        model,
        prompt=prompt,
        prompt_attention_mask=prompt_mask,
        frames=frames,
        condition_kwargs=condition_kwargs,
    )
    sequential_prefill_hash = hash_cache(sequential_state.cache)
    span_prefill_hash = hash_cache(span_state.cache)

    with generation_profile_context(sdpa_backend=args.profile_sdpa_backend):
        sequential_logits, sequential_prepared = _sequential_target_forwards(
            model,
            sequential_state,
            prompt=prompt,
            prompt_mask=prompt_mask,
            fixed_input_tokens=draft_tokens,
            condition_kwargs=condition_kwargs,
        )
        span_logits, span_prepared = _target_span_forward(
            model,
            span_state,
            prompt=prompt,
            prompt_mask=prompt_mask,
            fixed_input_tokens=draft_tokens,
            condition_kwargs=condition_kwargs,
        )

    q_len_comparisons = [
        compare_logits(
            sequential_logits[index],
            span_logits[:, index, :],
            atol=config.atol,
            rtol=config.rtol,
            top_k=config.top_k,
        )
        for index in range(config.speculation_k)
    ]

    # Correct speculative mapping: prefill verifies draft[0], span[i-1] verifies draft[i].
    sequential_verification_logits = [sequential_state.prefill_logits, *sequential_logits[:-1]]
    span_verification_logits = [span_state.prefill_logits, *[
        span_logits[:, index, :]
        for index in range(config.speculation_k - 1)
    ]]
    verification_comparisons = [
        compare_logits(
            sequential_verification_logits[index],
            span_verification_logits[index],
            atol=config.atol,
            rtol=config.rtol,
            top_k=config.top_k,
        )
        for index in range(config.speculation_k)
    ]

    sampling_start_rng = production_capture.rng_states[0]
    sequential_samples, sequential_final_rng, sequential_processed = _sample_verification_logits(
        args,
        tokenizer,
        prompt=prompt,
        draft_tokens=draft_tokens,
        verification_logits=sequential_verification_logits,
        start_rng_state=sampling_start_rng,
        lookback_time=lookback_time,
    )
    span_samples, span_final_rng, span_processed = _sample_verification_logits(
        args,
        tokenizer,
        prompt=prompt,
        draft_tokens=draft_tokens,
        verification_logits=span_verification_logits,
        start_rng_state=sampling_start_rng,
        lookback_time=lookback_time,
    )
    processed_vs_production = {
        route_name: [
            compare_logits(
                route_scores[index],
                production_capture.scores[index],
                atol=config.atol,
                rtol=config.rtol,
                top_k=config.top_k,
            )
            for index in range(config.speculation_k)
        ]
        for route_name, route_scores in (
            ("sequential", sequential_processed),
            ("span", span_processed),
        )
    }

    final_prefix_length = prompt_length + config.speculation_k
    sequential_final_hash = hash_cache(sequential_state.cache, self_prefix_length=final_prefix_length)
    span_final_hash = hash_cache(span_state.cache, self_prefix_length=final_prefix_length)
    cache_allclose = compare_cache_prefixes(
        sequential_state.cache,
        span_state.cache,
        self_prefix_length=final_prefix_length,
        atol=config.atol,
        rtol=config.rtol,
    )
    span_full_hash_before_rollback = hash_cache(span_state.cache)
    zero_static_self_cache_suffix(
        span_state.cache,
        start_position=prompt_length,
        end_position=final_prefix_length,
    )
    span_full_hash_after_rollback = hash_cache(span_state.cache)
    rollback_oracle_match = span_full_hash_after_rollback == span_prefill_hash

    forced_eos_audit = _forced_eos_no_extra_sampling_audit(
        sequential_logits=sequential_verification_logits[0],
        span_logits=span_verification_logits[0],
        start_rng_state=sampling_start_rng,
        eos_token_id=int(eos_token_ids[0]),
    )

    # Reuse prepared calls for model-only CUDA timing; cache zeroing is outside each event interval.
    def sequential_call():
        outputs = None
        with generation_profile_context(sdpa_backend=args.profile_sdpa_backend):
            for prepared in sequential_prepared:
                outputs = model(**prepared, return_dict=True)
        return outputs

    def span_call():
        with generation_profile_context(sdpa_backend=args.profile_sdpa_backend):
            return model(**span_prepared, return_dict=True)
    reset_sequential = lambda: zero_static_self_cache_suffix(
        sequential_state.cache,
        start_position=prompt_length,
        end_position=final_prefix_length,
    )
    reset_span = lambda: zero_static_self_cache_suffix(
        span_state.cache,
        start_position=prompt_length,
        end_position=final_prefix_length,
    )
    sequential_ms = _timed_cuda_calls(
        sequential_call,
        reset_sequential,
        warmup=config.timing_warmup,
        iters=config.timing_iters,
    )
    span_ms = _timed_cuda_calls(
        span_call,
        reset_span,
        warmup=config.timing_warmup,
        iters=config.timing_iters,
    )
    memory = {
        "sequential": _peak_cuda_memory(sequential_call, reset_sequential),
        "span": _peak_cuda_memory(span_call, reset_span),
    }
    fixed_shape_cuda_graph = (
        _fixed_shape_k4_cuda_graph_ceiling(
            args,
            model,
            tokenizer,
            prompt=prompt,
            draft_tokens=draft_tokens,
            span_prepared=span_prepared,
            span_state=span_state,
            sequential_state=sequential_state,
            eager_span_logits=span_logits,
            sampling_start_rng=sampling_start_rng,
            production_end_rng=production_end_rng,
            lookback_time=lookback_time,
            final_prefix_length=final_prefix_length,
            config=config,
            compare_logits=compare_logits,
        )
        if config.fixed_shape_cuda_graph_ceiling
        else {"enabled": False, "pass": None}
    )

    draft_token_ids = [int(token) for token in draft_tokens[0].tolist()]
    q_len_pass = all(item["allclose"] and item["topk_match"] for item in q_len_comparisons)
    verification_pass = all(item["allclose"] and item["topk_match"] for item in verification_comparisons)
    production_score_pass = all(
        item["allclose"] and item["topk_match"]
        for comparisons in processed_vs_production.values()
        for item in comparisons
    )
    sampled_token_match = sequential_samples == span_samples == draft_token_ids
    rng_match = (
        _rng_states_equal(sequential_final_rng, span_final_rng)
        and _rng_states_equal(sequential_final_rng, production_end_rng)
    )
    cache_hash_match = sequential_final_hash == span_final_hash
    passed = all((
        q_len_pass,
        verification_pass,
        production_score_pass,
        sampled_token_match,
        rng_match,
        sequential_prefill_hash == span_prefill_hash,
        cache_allclose["allclose"],
        rollback_oracle_match,
        forced_eos_audit["pass"],
        fixed_shape_cuda_graph["pass"] if fixed_shape_cuda_graph["enabled"] else True,
    ))
    return {
        "gate": GATE_NAME,
        "pass": passed,
        "result_class": "exact-output",
        "claim_scope": "bounded_fp32_target_span_numerical_gate_not_full_output_promotion",
        "speculation_k": config.speculation_k,
        "semantic_offset": {
            "fixed_input_tokens": "draft[0:K] are fed to both forward shapes",
            "q_len_outputs": "q_len output[i] scores the token after fixed input draft[i]",
            "draft_verification": "draft[0] uses prefill logits; draft[i] uses q_len output[i-1] for i>0",
            "lookahead_only_output_index": config.speculation_k - 1,
            "lookahead_warning": "The final q_len output is not verification evidence for draft[K-1].",
        },
        "prompt_tokens": prompt_length,
        "draft_token_ids": draft_token_ids,
        "q_len_output_comparisons": q_len_comparisons,
        "q_len_allclose": q_len_pass,
        "verification_logit_comparisons": verification_comparisons,
        "verification_mapping_pass": verification_pass,
        "processed_scores_vs_production": processed_vs_production,
        "processed_scores_match_production": production_score_pass,
        "sampling": {
            "sequential_token_ids": sequential_samples,
            "span_token_ids": span_samples,
            "captured_production_token_ids": draft_token_ids,
            "token_match": sampled_token_match,
            "final_rng_match": rng_match,
            "sequential_final_rng_hash": _hash_rng_state(sequential_final_rng),
            "span_final_rng_hash": _hash_rng_state(span_final_rng),
            "production_final_rng_hash": _hash_rng_state(production_end_rng),
        },
        "cache": {
            "prefill_hash_match": sequential_prefill_hash == span_prefill_hash,
            "sequential_prefill_full_hash": sequential_prefill_hash,
            "span_prefill_full_hash": span_prefill_hash,
            "final_prefix_length": final_prefix_length,
            "sequential_final_prefix_hash": sequential_final_hash,
            "span_final_prefix_hash": span_final_hash,
            "final_prefix_bitwise_hash_match": cache_hash_match,
            "final_prefix_allclose": cache_allclose,
            "span_full_hash_before_rollback": span_full_hash_before_rollback,
            "span_full_hash_after_rollback": span_full_hash_after_rollback,
            "verifier_physical_rollback_oracle_match": rollback_oracle_match,
            "physical_rollback_status": "blocked_no_supported_static_cache_crop",
            "logical_exact_output_rollback_status": "not_tested_before_k2_numeric_pass",
            "warning": (
                "Suffix zeroing is verifier-only proof. Static EncoderDecoderCache.crop rejects StaticCache; "
                "this is not a supported production rollback implementation."
            ),
        },
        "forced_eos_no_extra_sampling": forced_eos_audit,
        "timing": {
            "scope": "model_forward_cuda_events_excludes_cache_reset_and_input_preparation",
            "warmup": config.timing_warmup,
            "iters": config.timing_iters,
            "sequential_k_calls_ms": sequential_ms,
            "span_one_call_ms": span_ms,
            "speedup": sequential_ms / span_ms if span_ms > 0 else None,
            "sequential_ms_per_scored_position": sequential_ms / config.speculation_k,
            "span_ms_per_scored_position": span_ms / config.speculation_k,
        },
        "peak_vram": memory,
        "fixed_shape_cuda_graph_ceiling": fixed_shape_cuda_graph,
        "prepared_shapes": {
            "sequential_decoder_input_ids": [
                list(prepared["decoder_input_ids"].shape)
                for prepared in sequential_prepared
            ],
            "span_decoder_input_ids": list(span_prepared["decoder_input_ids"].shape),
        },
    }
