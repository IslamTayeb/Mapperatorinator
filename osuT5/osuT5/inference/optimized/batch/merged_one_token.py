"""Real-tensor merged-batch one-token verifier and measurement gate.

The code here is diagnostic only. It builds independent B1 references and one
merged DecodeSession for an identical prompt, then proves each merged row's
prefill/decode logits, top-k, sampled tokens, and private generator state. It
does not implement a scheduler, lane pool, server path, or production runtime.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from ..benchmark import (
    BatchPhysicsExecutionFamily,
    BatchPhysicsObservation,
    MERGED_STATE_OWNERSHIP_CONTRACT,
)
from ..exactness import ExactnessResultClass
from ...direct_decode import DecodeSession


LogitsProcessorFactory = Callable[[], Any]


@dataclass(frozen=True)
class MergedOneTokenConfig:
    """Bounded correctness and timing settings for one merged shape."""

    batch_size: int
    seeds: tuple[int, ...]
    do_sample: bool
    atol: float = 1e-4
    rtol: float = 1e-4
    top_k: int = 20
    warmup_repeats: int = 5
    timing_repeats: int = 50
    active_prefix_prefill: bool = False
    active_prefix_decode: bool = False
    active_prefix_decode_length: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size not in {1, 2, 5, 8}:
            raise ValueError("real merged one-token gates support B=1, B=2, B=5, or B=8.")
        if len(self.seeds) != self.batch_size:
            raise ValueError("seeds must contain exactly one private seed per batch row.")
        if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in self.seeds):
            raise TypeError("seeds must contain integers.")
        if self.atol < 0 or self.rtol < 0:
            raise ValueError("atol and rtol must be non-negative.")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.warmup_repeats < 0:
            raise ValueError("warmup_repeats must be non-negative.")
        if self.timing_repeats <= 0:
            raise ValueError("timing_repeats must be positive.")
        if self.active_prefix_decode_length is not None:
            if not self.active_prefix_decode:
                raise ValueError("active_prefix_decode_length requires active-prefix decode.")
            if self.active_prefix_decode_length <= 0:
                raise ValueError("active_prefix_decode_length must be positive.")


def repeat_batch_tensor(value: torch.Tensor, batch_size: int, *, name: str) -> torch.Tensor:
    """Repeat a B1 tensor into one contiguous merged-batch input."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if value.ndim == 0:
        return value
    if value.shape[0] != 1:
        raise ValueError(
            f"{name} must have batch dimension 1 for the identical-prompt gate; "
            f"got {list(value.shape)}."
        )
    return value.repeat(batch_size, *([1] * (value.ndim - 1))).contiguous()


def repeat_batch_kwargs(values: Mapping[str, Any], batch_size: int) -> dict[str, Any]:
    return {
        key: (
            repeat_batch_tensor(value, batch_size, name=key)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in values.items()
    }


def slice_batch_kwargs(values: Mapping[str, Any], row: int) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] > 1:
            sliced[key] = value[row:row + 1].contiguous()
        else:
            sliced[key] = value
    return sliced


def compare_logits(
        reference: torch.Tensor,
        candidate: torch.Tensor,
        *,
        atol: float,
        rtol: float,
        top_k: int,
) -> dict[str, Any]:
    """Compare raw FP32 logits including non-finite layout and top-k identity."""

    reference = reference.detach().to(device="cpu", dtype=torch.float32)
    candidate = candidate.detach().to(device="cpu", dtype=torch.float32)
    if reference.shape != candidate.shape:
        return {
            "allclose": False,
            "topk_match": False,
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    finite_mask = torch.isfinite(reference) & torch.isfinite(candidate)
    reference_nonfinite = ~torch.isfinite(reference)
    candidate_nonfinite = ~torch.isfinite(candidate)
    nonfinite_match = bool(torch.equal(reference_nonfinite, candidate_nonfinite))
    positive_inf_match = bool(torch.equal(torch.isposinf(reference), torch.isposinf(candidate)))
    negative_inf_match = bool(torch.equal(torch.isneginf(reference), torch.isneginf(candidate)))
    finite_reference = reference[finite_mask]
    finite_candidate = candidate[finite_mask]
    finite_allclose = bool(torch.allclose(finite_reference, finite_candidate, atol=atol, rtol=rtol))
    abs_diff = torch.abs(finite_reference - finite_candidate)
    rel_diff = abs_diff / torch.clamp(torch.abs(finite_reference), min=1e-12)
    k = min(top_k, reference.shape[-1])
    reference_topk = torch.topk(reference, k=k, dim=-1).indices
    candidate_topk = torch.topk(candidate, k=k, dim=-1).indices
    topk_match = bool(torch.equal(reference_topk, candidate_topk))
    return {
        "allclose": finite_allclose and nonfinite_match and positive_inf_match and negative_inf_match,
        "finite_allclose": finite_allclose,
        "nonfinite_match": nonfinite_match,
        "positive_inf_match": positive_inf_match,
        "negative_inf_match": negative_inf_match,
        "topk_match": topk_match,
        "shape_match": True,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "finite_count": int(finite_mask.sum().item()),
        "nonfinite_mismatch_count": int((reference_nonfinite != candidate_nonfinite).sum().item()),
        "max_abs": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "mean_abs": float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
        "max_rel": float(rel_diff.max().item()) if rel_diff.numel() else 0.0,
        "reference_topk": reference_topk[0].tolist(),
        "candidate_topk": candidate_topk[0].tolist(),
    }


def resolve_row_seeds(seed_values: Sequence[int], *, batch_size: int, default_seed: int) -> tuple[int, ...]:
    """Resolve zero, one, or B explicit CLI seeds into private row seeds."""

    if not seed_values:
        return (int(default_seed),) * batch_size
    if len(seed_values) == 1:
        return (int(seed_values[0]),) * batch_size
    if len(seed_values) != batch_size:
        raise ValueError("provide zero, one, or exactly batch_size --row-seed values.")
    return tuple(int(seed) for seed in seed_values)


def validate_previous_gate(batch_size: int, report: Mapping[str, Any] | None) -> None:
    """Enforce B1 -> B2 -> B5 graduation using a passed prior report."""

    if batch_size == 1:
        if report is not None:
            raise ValueError("B=1 is the first gate and must not receive a previous report.")
        return
    expected_previous = {2: 1, 5: 2, 8: 5}.get(batch_size)
    if expected_previous is None:
        raise ValueError("real merged one-token gates currently stop at B=8.")
    if report is None:
        raise ValueError(f"B={batch_size} requires a passed B={expected_previous} report.")
    if not bool(report.get("pass")):
        raise ValueError("previous merged one-token gate did not pass.")
    if int(report.get("batch_size", -1)) != expected_previous:
        raise ValueError(
            f"B={batch_size} requires a B={expected_previous} previous report; "
            f"got B={report.get('batch_size')}."
        )
    observation = report.get("observation")
    if not isinstance(observation, Mapping):
        raise ValueError("previous report is missing its normalized observation.")
    if observation.get("execution_family") != BatchPhysicsExecutionFamily.MERGED_BATCH.value:
        raise ValueError("previous report is not a merged-batch observation.")
    if int(observation.get("parallelism", -1)) != expected_previous:
        raise ValueError("previous observation parallelism does not match its batch gate.")


def summarize_previous_gate_scaling(
        *,
        previous_batch_size: int,
        previous_tokens_per_second: float,
        candidate_batch_size: int,
        candidate_tokens_per_second: float,
) -> dict[str, Any]:
    """Apply the five-percent gate and report loss from ideal scaled capacity."""

    if previous_batch_size <= 0 or candidate_batch_size <= previous_batch_size:
        raise ValueError("candidate_batch_size must be greater than a positive previous batch size.")
    if previous_tokens_per_second <= 0 or candidate_tokens_per_second <= 0:
        raise ValueError("throughput values must be positive.")
    relative_gain = candidate_tokens_per_second / previous_tokens_per_second - 1.0
    ideal_scaled_tps = (
        previous_tokens_per_second * candidate_batch_size / previous_batch_size
    )
    return {
        "previous_batch_size": previous_batch_size,
        "candidate_batch_size": candidate_batch_size,
        "previous_complete_wall_tokens_per_second": previous_tokens_per_second,
        "candidate_complete_wall_tokens_per_second": candidate_tokens_per_second,
        "relative_complete_throughput_gain": relative_gain,
        "clears_five_percent_gain_gate": relative_gain >= 0.05,
        "ideal_scaled_previous_capacity_tokens_per_second": ideal_scaled_tps,
        "ideal_capacity_loss": candidate_tokens_per_second / ideal_scaled_tps - 1.0,
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to("cpu").contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _new_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _sample_next_token(
        *,
        input_ids: torch.LongTensor,
        raw_logits: torch.Tensor,
        logits_processor: Any,
        generator: torch.Generator,
        do_sample: bool,
) -> torch.LongTensor:
    scores = logits_processor(input_ids, raw_logits.clone(memory_format=torch.contiguous_format))
    if do_sample:
        probabilities = torch.nn.functional.softmax(scores, dim=-1)
        return torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(1)
    return torch.argmax(scores, dim=-1)


def _shape_metadata(values: Mapping[str, Any]) -> dict[str, list[int]]:
    return {
        key: list(value.shape)
        for key, value in sorted(values.items())
        if isinstance(value, torch.Tensor)
    }


def _cache_part_metadata(cache: Any) -> dict[str, Any]:
    max_shape = None
    if hasattr(cache, "get_max_cache_shape"):
        try:
            max_shape = cache.get_max_cache_shape()
        except (RuntimeError, TypeError, ValueError):
            max_shape = "unavailable"
    sequence_length = None
    if hasattr(cache, "get_seq_length"):
        try:
            sequence_length = cache.get_seq_length()
            if isinstance(sequence_length, torch.Tensor):
                sequence_length = int(sequence_length.detach().cpu().item())
            elif sequence_length is not None:
                sequence_length = int(sequence_length)
        except (RuntimeError, TypeError, ValueError):
            sequence_length = "unavailable"
    layers = getattr(cache, "layers", None)
    return {
        "type": type(cache).__name__,
        "max_cache_shape": max_shape,
        "sequence_length": sequence_length,
        "layer_count": len(layers) if layers is not None else None,
    }


def _session_cache_metadata(session: DecodeSession) -> dict[str, Any]:
    cache = session.cache_state.cache
    if cache is None:
        return {"type": None}
    metadata = {
        "type": type(cache).__name__,
        "self_attention": _cache_part_metadata(cache.self_attention_cache),
        "cross_attention": _cache_part_metadata(cache.cross_attention_cache),
        "session": session.metadata(),
    }
    metadata["metadata_sha256"] = _json_sha256(metadata)
    return metadata


@torch.no_grad()
def run_merged_one_token_gate(
        model: Any,
        *,
        prompt: torch.LongTensor,
        prompt_attention_mask: torch.Tensor,
        frames: torch.Tensor,
        condition_kwargs: Mapping[str, Any],
        logits_processor_factory: LogitsProcessorFactory,
        eos_token_ids: Sequence[int],
        config: MergedOneTokenConfig,
        runtime_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Run exact row gates and a warmed fixed-shape merged-step measurement."""

    device = torch.device(model.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the real merged one-token gate requires a CUDA GPU allocation.")
    if model.dtype != torch.float32:
        raise ValueError(f"merged one-token gate is FP32-only; model dtype is {model.dtype}.")
    if prompt.ndim != 2 or prompt.shape[0] != 1:
        raise ValueError("prompt must be a B1 rank-2 tensor.")

    batch_size = config.batch_size
    prompt_batched = repeat_batch_tensor(prompt, batch_size, name="decoder_input_ids")
    prompt_mask_batched = repeat_batch_tensor(
        prompt_attention_mask,
        batch_size,
        name="decoder_attention_mask",
    )
    frames_batched = repeat_batch_tensor(frames, batch_size, name="frames")
    condition_batched = repeat_batch_kwargs(condition_kwargs, batch_size)

    independent_rows: list[dict[str, Any]] = []
    independent_prefill_logits: list[torch.Tensor] = []
    independent_decode_logits: list[torch.Tensor] = []
    independent_anchor_tokens: list[torch.LongTensor] = []
    independent_next_tokens: list[torch.LongTensor] = []
    independent_rng_hashes: list[str] = []
    for row, seed in enumerate(config.seeds):
        row_processor = logits_processor_factory()
        row_generator = _new_generator(device, seed)
        session = DecodeSession.prefill(
            model,
            prompt=prompt,
            prompt_attention_mask=prompt_attention_mask,
            frames=frames,
            condition_kwargs=slice_batch_kwargs(condition_batched, row),
            active_prefix_self_attention=config.active_prefix_prefill,
        )
        prefill_logits = session.one_token_state().prefill_logits
        if prefill_logits is None:
            raise RuntimeError("independent DecodeSession prefill did not retain logits.")
        anchor_token = _sample_next_token(
            input_ids=prompt,
            raw_logits=prefill_logits,
            logits_processor=row_processor,
            generator=row_generator,
            do_sample=config.do_sample,
        )
        full_prefix = torch.cat([prompt, anchor_token[:, None]], dim=-1)
        full_mask = torch.cat(
            [prompt_attention_mask, torch.ones_like(anchor_token[:, None], dtype=prompt_attention_mask.dtype)],
            dim=-1,
        )
        decode_result = session.decode_one_token_raw_logits(
            full_prefix=full_prefix,
            full_attention_mask=full_mask,
            active_prefix_self_attention=config.active_prefix_decode,
            active_prefix_self_attention_length=config.active_prefix_decode_length,
        )
        next_token = _sample_next_token(
            input_ids=full_prefix,
            raw_logits=decode_result.logits,
            logits_processor=row_processor,
            generator=row_generator,
            do_sample=config.do_sample,
        )
        independent_prefill_logits.append(prefill_logits.detach().to("cpu", dtype=torch.float32))
        independent_decode_logits.append(decode_result.logits.detach().to("cpu", dtype=torch.float32))
        independent_anchor_tokens.append(anchor_token.detach().to("cpu"))
        independent_next_tokens.append(next_token.detach().to("cpu"))
        independent_rng_hashes.append(_tensor_sha256(row_generator.get_state()))
        independent_rows.append({
            "row": row,
            "seed": seed,
            "anchor_token_id": int(anchor_token.detach().cpu().item()),
            "sampled_token_id": int(next_token.detach().cpu().item()),
            "final_rng_state_hash": independent_rng_hashes[-1],
            "cache_state": _session_cache_metadata(session),
        })
        del (
            anchor_token,
            decode_result,
            full_mask,
            full_prefix,
            next_token,
            prefill_logits,
            row_generator,
            row_processor,
            session,
        )

    torch.cuda.empty_cache()
    batched_session = DecodeSession.prefill(
        model,
        prompt=prompt_batched,
        prompt_attention_mask=prompt_mask_batched,
        frames=frames_batched,
        condition_kwargs=condition_batched,
        active_prefix_self_attention=config.active_prefix_prefill,
    )
    batched_prefill_logits = batched_session.one_token_state().prefill_logits
    if batched_prefill_logits is None:
        raise RuntimeError("merged DecodeSession prefill did not retain logits.")
    batched_processors = [logits_processor_factory() for _ in range(batch_size)]
    batched_generators = [
        _new_generator(device, seed)
        for seed in config.seeds
    ]
    batched_anchor_tokens = [
        _sample_next_token(
            input_ids=prompt_batched[row:row + 1],
            raw_logits=batched_prefill_logits[row:row + 1],
            logits_processor=batched_processors[row],
            generator=batched_generators[row],
            do_sample=config.do_sample,
        )
        for row in range(batch_size)
    ]
    anchor_tokens_batched = torch.cat(batched_anchor_tokens, dim=0)
    full_prefix_batched = torch.cat([prompt_batched, anchor_tokens_batched[:, None]], dim=-1)
    full_mask_batched = torch.cat(
        [
            prompt_mask_batched,
            torch.ones_like(anchor_tokens_batched[:, None], dtype=prompt_mask_batched.dtype),
        ],
        dim=-1,
    )
    batched_decode_result = batched_session.decode_one_token_raw_logits(
        full_prefix=full_prefix_batched,
        full_attention_mask=full_mask_batched,
        active_prefix_self_attention=config.active_prefix_decode,
        active_prefix_self_attention_length=config.active_prefix_decode_length,
    )
    batched_next_tokens = [
        _sample_next_token(
            input_ids=full_prefix_batched[row:row + 1],
            raw_logits=batched_decode_result.logits[row:row + 1],
            logits_processor=batched_processors[row],
            generator=batched_generators[row],
            do_sample=config.do_sample,
        )
        for row in range(batch_size)
    ]
    batched_rng_hashes = [
        _tensor_sha256(generator.get_state())
        for generator in batched_generators
    ]

    row_reports: list[dict[str, Any]] = []
    token_hashes: dict[str, str] = {}
    rng_hashes: dict[str, str] = {}
    stop_reasons: dict[str, str] = {}
    eos_ids = {int(token_id) for token_id in eos_token_ids}
    for row in range(batch_size):
        request_id = f"row-{row}"
        prefill_comparison = compare_logits(
            independent_prefill_logits[row],
            batched_prefill_logits[row:row + 1],
            atol=config.atol,
            rtol=config.rtol,
            top_k=config.top_k,
        )
        decode_comparison = compare_logits(
            independent_decode_logits[row],
            batched_decode_result.logits[row:row + 1],
            atol=config.atol,
            rtol=config.rtol,
            top_k=config.top_k,
        )
        anchor_token_id = int(batched_anchor_tokens[row].detach().cpu().item())
        sampled_token_id = int(batched_next_tokens[row].detach().cpu().item())
        anchor_token_match = bool(torch.equal(independent_anchor_tokens[row], batched_anchor_tokens[row].cpu()))
        sampled_token_match = bool(torch.equal(independent_next_tokens[row], batched_next_tokens[row].cpu()))
        rng_state_match = independent_rng_hashes[row] == batched_rng_hashes[row]
        row_pass = bool(
            prefill_comparison["allclose"]
            and prefill_comparison["topk_match"]
            and decode_comparison["allclose"]
            and decode_comparison["topk_match"]
            and anchor_token_match
            and sampled_token_match
            and rng_state_match
        )
        token_hashes[request_id] = _json_sha256([anchor_token_id, sampled_token_id])
        rng_hashes[request_id] = batched_rng_hashes[row]
        stop_reasons[request_id] = "eos" if sampled_token_id in eos_ids else "one_token_gate"
        row_reports.append({
            "row": row,
            "request_id": request_id,
            "seed": config.seeds[row],
            "anchor_token_id": anchor_token_id,
            "sampled_token_id": sampled_token_id,
            "anchor_token_match": anchor_token_match,
            "sampled_token_match": sampled_token_match,
            "rng_state_match": rng_state_match,
            "reference_final_rng_state_hash": independent_rng_hashes[row],
            "batched_final_rng_state_hash": batched_rng_hashes[row],
            "reference_prefill_logits_sha256": _tensor_sha256(independent_prefill_logits[row]),
            "batched_prefill_logits_sha256": _tensor_sha256(
                batched_prefill_logits[row:row + 1].to("cpu", dtype=torch.float32)
            ),
            "reference_decode_logits_sha256": _tensor_sha256(independent_decode_logits[row]),
            "batched_decode_logits_sha256": _tensor_sha256(
                batched_decode_result.logits[row:row + 1].to("cpu", dtype=torch.float32)
            ),
            "prefill_logits": prefill_comparison,
            "decode_logits": decode_comparison,
            "pass": row_pass,
        })

    warmup_processors = [logits_processor_factory() for _ in range(batch_size)]
    warmup_generators = [_new_generator(device, seed) for seed in config.seeds]
    for _ in range(config.warmup_repeats):
        warmup_result = batched_session.decode_one_token_raw_logits(
            full_prefix=full_prefix_batched,
            full_attention_mask=full_mask_batched,
            active_prefix_self_attention=config.active_prefix_decode,
            active_prefix_self_attention_length=config.active_prefix_decode_length,
        )
        for row in range(batch_size):
            _sample_next_token(
                input_ids=full_prefix_batched[row:row + 1],
                raw_logits=warmup_result.logits[row:row + 1],
                logits_processor=warmup_processors[row],
                generator=warmup_generators[row],
                do_sample=config.do_sample,
            )
    torch.cuda.synchronize(device)

    model_start = torch.cuda.Event(enable_timing=True)
    model_end = torch.cuda.Event(enable_timing=True)
    model_start.record()
    for _ in range(config.timing_repeats):
        batched_session.decode_one_token_raw_logits(
            full_prefix=full_prefix_batched,
            full_attention_mask=full_mask_batched,
            active_prefix_self_attention=config.active_prefix_decode,
            active_prefix_self_attention_length=config.active_prefix_decode_length,
        )
    model_end.record()
    model_end.synchronize()
    model_cuda_seconds = model_start.elapsed_time(model_end) / 1000.0

    timing_processors = [logits_processor_factory() for _ in range(batch_size)]
    timing_generators = [_new_generator(device, seed) for seed in config.seeds]
    torch.cuda.reset_peak_memory_stats(device)
    memory_before_allocated = torch.cuda.memory_allocated(device)
    memory_before_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.synchronize(device)
    complete_start = torch.cuda.Event(enable_timing=True)
    complete_end = torch.cuda.Event(enable_timing=True)
    complete_wall_started = time.perf_counter()
    complete_start.record()
    for _ in range(config.timing_repeats):
        complete_result = batched_session.decode_one_token_raw_logits(
            full_prefix=full_prefix_batched,
            full_attention_mask=full_mask_batched,
            active_prefix_self_attention=config.active_prefix_decode,
            active_prefix_self_attention_length=config.active_prefix_decode_length,
        )
        for row in range(batch_size):
            _sample_next_token(
                input_ids=full_prefix_batched[row:row + 1],
                raw_logits=complete_result.logits[row:row + 1],
                logits_processor=timing_processors[row],
                generator=timing_generators[row],
                do_sample=config.do_sample,
            )
    complete_end.record()
    complete_end.synchronize()
    complete_wall_seconds = time.perf_counter() - complete_wall_started
    complete_cuda_seconds = complete_start.elapsed_time(complete_end) / 1000.0
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    model_seconds_per_step = model_cuda_seconds / config.timing_repeats
    complete_wall_seconds_per_step = complete_wall_seconds / config.timing_repeats
    rowwise_sampling_host_overhead_seconds_per_step = max(
        0.0,
        complete_wall_seconds_per_step - model_seconds_per_step,
    )
    model_only_tokens_per_second = batch_size / model_seconds_per_step
    complete_wall_tokens_per_second = (
        batch_size * config.timing_repeats / complete_wall_seconds
    )

    workload_contract = {
        "batch_size": batch_size,
        "seeds": list(config.seeds),
        "do_sample": config.do_sample,
        "prompt_sha256": _tensor_sha256(prompt),
        "prompt_attention_mask_sha256": _tensor_sha256(prompt_attention_mask),
        "frames_sha256": _tensor_sha256(frames),
        "condition_tensor_hashes": {
            key: _tensor_sha256(value)
            for key, value in sorted(condition_kwargs.items())
            if isinstance(value, torch.Tensor)
        },
        "active_prefix_prefill": config.active_prefix_prefill,
        "active_prefix_decode": config.active_prefix_decode,
        "active_prefix_decode_length": config.active_prefix_decode_length,
        "runtime_contract": {
            key: runtime_metadata.get(key)
            for key in (
                "config_name",
                "model_path",
                "precision",
                "attn_implementation",
                "global_seed",
                "row_seeds",
                "do_sample",
                "top_p",
                "top_k_sampling",
                "temperature",
                "stateful_monotonic_logits_processor",
                "eos_token_ids",
                "probe",
            )
        },
    }
    observation = BatchPhysicsObservation(
        execution_family=BatchPhysicsExecutionFamily.MERGED_BATCH,
        parallelism=batch_size,
        state_ownership_contract=MERGED_STATE_OWNERSHIP_CONTRACT,
        workload_contract_hash=_json_sha256(workload_contract),
        result_class=ExactnessResultClass.EXACT_OUTPUT,
        seeds={f"row-{row}": seed for row, seed in enumerate(config.seeds)},
        generated_tokens=batch_size * config.timing_repeats,
        scheduler_wall_seconds=complete_wall_seconds,
        model_seconds=model_cuda_seconds,
        cuda_seconds=complete_cuda_seconds,
        peak_memory_bytes=peak_allocated,
        graph_capture_count=0,
        graph_replay_count=0,
        active_batch_size_histogram={batch_size: config.timing_repeats},
        token_hashes=token_hashes,
        final_rng_state_hashes=rng_hashes,
        stop_reasons=stop_reasons,
    )
    merged_cache_state = _session_cache_metadata(batched_session)
    return {
        "pass": all(row["pass"] for row in row_reports),
        "batch_size": batch_size,
        "gate": "merged_identical_prompt_one_token",
        "claim_scope": "verifier_and_fixed_shape_scout_not_runtime_throughput",
        "rows": row_reports,
        "independent_rows": independent_rows,
        "input_shapes": {
            "prompt": list(prompt.shape),
            "prompt_batched": list(prompt_batched.shape),
            "prompt_attention_mask": list(prompt_attention_mask.shape),
            "frames": list(frames.shape),
            "frames_batched": list(frames_batched.shape),
            "condition_kwargs": _shape_metadata(condition_kwargs),
            "condition_kwargs_batched": _shape_metadata(condition_batched),
            "full_prefix_batched": list(full_prefix_batched.shape),
            "batched_prefill_logits": list(batched_prefill_logits.shape),
            "batched_decode_logits": list(batched_decode_result.logits.shape),
        },
        "cache_state": merged_cache_state,
        "graph_activity": {
            "decode_session_graph_count": batched_session.metadata()["graph_count"],
            "capture_count": 0,
            "replay_count": 0,
            "note": "direct merged DecodeSession path is eager; graph execution is not implemented here",
        },
        "timing": {
            "measurement_shape": "replayed_fixed_one_token_shape",
            "warmup_repeats": config.warmup_repeats,
            "timing_repeats": config.timing_repeats,
            "generated_tokens": batch_size * config.timing_repeats,
            "model_only_cuda_seconds": model_cuda_seconds,
            "model_only_cuda_seconds_per_step": model_seconds_per_step,
            "model_only_tokens_per_second": model_only_tokens_per_second,
            "complete_sampled_step_wall_seconds": complete_wall_seconds,
            "complete_sampled_step_wall_seconds_per_step": complete_wall_seconds_per_step,
            "complete_sampled_step_cuda_seconds": complete_cuda_seconds,
            "complete_wall_tokens_per_second": complete_wall_tokens_per_second,
            "complete_cuda_tokens_per_second": (
                batch_size * config.timing_repeats / complete_cuda_seconds
            ),
            "rowwise_sampling_host_overhead_seconds_per_step": (
                rowwise_sampling_host_overhead_seconds_per_step
            ),
            "rowwise_sampling_host_overhead_fraction_of_complete_wall": (
                rowwise_sampling_host_overhead_seconds_per_step
                / complete_wall_seconds_per_step
            ),
            "complete_capacity_loss_vs_model_only": (
                complete_wall_tokens_per_second / model_only_tokens_per_second - 1.0
            ),
        },
        "memory": {
            "before_allocated_bytes": memory_before_allocated,
            "before_reserved_bytes": memory_before_reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "workload_contract": workload_contract,
        "observation": observation.as_dict(),
        "runtime_metadata": dict(runtime_metadata),
    }
