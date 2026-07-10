"""Verifier-only short merged-B8 decode loop.

This module advances the identical-prompt processor control by exactly one
gate: sixteen changing-prefix sampled steps.  It is not a scheduler or runtime
implementation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from ...direct_decode import DecodeSession
from .loop_state import ActiveRowLedger, ActiveRowState
from .merged_one_token import (
    _new_generator,
    _sample_after_base_processing,
    _tensor_sha256,
    compare_logits,
    repeat_batch_kwargs,
    repeat_batch_tensor,
    slice_batch_kwargs,
)


LogitsProcessorFactory = Callable[[], Any]


@dataclass(frozen=True)
class MergedLoopConfig:
    """Bounded settings for the first changing-prefix B8 gate."""

    seeds: tuple[int, ...]
    max_new_tokens: int = 16
    do_sample: bool = True
    pad_token_id: int = 0
    atol: float = 1e-4
    rtol: float = 1e-4
    top_k: int = 20
    active_prefix_prefill: bool = False
    active_prefix_decode: bool = True
    active_prefix_decode_length: int | None = 64

    def __post_init__(self) -> None:
        if len(self.seeds) != 8:
            raise ValueError("the first merged loop gate is fixed at B=8.")
        if self.max_new_tokens != 16:
            raise ValueError("the first merged loop gate is fixed at exactly 16 steps.")
        if self.atol < 0 or self.rtol < 0:
            raise ValueError("atol and rtol must be non-negative.")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.active_prefix_decode_length is not None and not self.active_prefix_decode:
            raise ValueError("active_prefix_decode_length requires active-prefix decode.")


def _cache_parts(session: DecodeSession) -> dict[str, Any]:
    cache = session.cache_state.cache
    if cache is None:
        raise RuntimeError("DecodeSession has no cache.")
    return {
        "self_attention": cache.self_attention_cache,
        "cross_attention": cache.cross_attention_cache,
    }


def _cache_sequence_length(cache: Any) -> int:
    value = cache.get_seq_length()
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().item())
    return int(value)


def _compare_cache_tensors(
        reference: torch.Tensor,
        candidate: torch.Tensor,
        *,
        atol: float,
        rtol: float,
) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "allclose": False,
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    finite_layout_match = bool(torch.equal(torch.isfinite(reference), torch.isfinite(candidate)))
    finite = torch.isfinite(reference) & torch.isfinite(candidate)
    reference_finite = reference[finite]
    candidate_finite = candidate[finite]
    finite_allclose = bool(torch.allclose(reference_finite, candidate_finite, atol=atol, rtol=rtol))
    difference = torch.abs(reference_finite - candidate_finite)
    return {
        "allclose": bool(finite_layout_match and finite_allclose),
        "shape_match": True,
        "finite_layout_match": finite_layout_match,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "max_abs": float(difference.max().item()) if difference.numel() else 0.0,
        "mean_abs": float(difference.mean().item()) if difference.numel() else 0.0,
    }


def compare_active_session_caches(
        reference_sessions: Sequence[DecodeSession],
        merged_session: DecodeSession,
        *,
        active_rows: Sequence[int],
        self_sequence_length: int,
        atol: float,
        rtol: float,
) -> dict[str, Any]:
    """Compare active B1 cache prefixes against corresponding merged rows."""

    if not active_rows:
        return {"pass": True, "active_rows": [], "parts": {}}
    merged_parts = _cache_parts(merged_session)
    reference_parts = [_cache_parts(session) for session in reference_sessions]
    row_index = torch.tensor(active_rows, dtype=torch.long, device=merged_session.model.device)
    part_reports: dict[str, Any] = {}
    all_pass = True
    for part_name, merged_part in merged_parts.items():
        merged_layers = getattr(merged_part, "layers", None)
        if merged_layers is None:
            raise TypeError(f"{part_name} cache does not expose layers.")
        reference_lengths = [
            _cache_sequence_length(reference_parts[row][part_name])
            for row in active_rows
        ]
        merged_length = _cache_sequence_length(merged_part)
        if part_name == "self_attention":
            used_length = int(self_sequence_length)
            sequence_length_match = all(length == used_length for length in reference_lengths)
            sequence_length_match = sequence_length_match and merged_length == used_length
        else:
            used_length = merged_length
            sequence_length_match = all(length == used_length for length in reference_lengths)
        layer_reports: list[dict[str, Any]] = []
        for layer_index, merged_layer in enumerate(merged_layers):
            if not getattr(merged_layer, "is_initialized", False):
                raise RuntimeError(f"merged {part_name} cache layer {layer_index} is uninitialized.")
            key_rows = []
            value_rows = []
            for row in active_rows:
                reference_layer = reference_parts[row][part_name].layers[layer_index]
                if not getattr(reference_layer, "is_initialized", False):
                    raise RuntimeError(
                        f"reference row {row} {part_name} cache layer {layer_index} is uninitialized."
                    )
                key_rows.append(reference_layer.keys[:, :, :used_length, :])
                value_rows.append(reference_layer.values[:, :, :used_length, :])
            reference_keys = torch.cat(key_rows, dim=0)
            reference_values = torch.cat(value_rows, dim=0)
            candidate_keys = merged_layer.keys.index_select(0, row_index)[:, :, :used_length, :]
            candidate_values = merged_layer.values.index_select(0, row_index)[:, :, :used_length, :]
            key_comparison = _compare_cache_tensors(
                reference_keys,
                candidate_keys,
                atol=atol,
                rtol=rtol,
            )
            value_comparison = _compare_cache_tensors(
                reference_values,
                candidate_values,
                atol=atol,
                rtol=rtol,
            )
            layer_pass = bool(key_comparison["allclose"] and value_comparison["allclose"])
            all_pass = all_pass and layer_pass
            layer_reports.append({
                "layer": layer_index,
                "keys": key_comparison,
                "values": value_comparison,
                "pass": layer_pass,
            })
        all_pass = all_pass and sequence_length_match
        part_reports[part_name] = {
            "used_sequence_length": used_length,
            "reference_sequence_lengths": reference_lengths,
            "merged_sequence_length": merged_length,
            "sequence_length_match": sequence_length_match,
            "layers": layer_reports,
            "pass": bool(sequence_length_match and all(layer["pass"] for layer in layer_reports)),
        }
    return {
        "active_rows": list(active_rows),
        "parts": part_reports,
        "pass": all_pass,
    }


def _new_ledger(eos_token_ids: Sequence[int], *, max_new_tokens: int) -> ActiveRowLedger:
    eos_ids = tuple(int(token_id) for token_id in eos_token_ids)
    return ActiveRowLedger([
        ActiveRowState(
            request_id=f"row-{row}",
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_ids,
        )
        for row in range(8)
    ])


def _append_step(
        prefix: torch.LongTensor,
        attention_mask: torch.Tensor,
        tokens: Sequence[int],
        active_mask: Sequence[bool],
) -> tuple[torch.LongTensor, torch.Tensor]:
    token_tensor = torch.tensor(tokens, dtype=prefix.dtype, device=prefix.device).unsqueeze(1)
    mask_tensor = torch.tensor(active_mask, dtype=attention_mask.dtype, device=attention_mask.device).unsqueeze(1)
    return torch.cat([prefix, token_tensor], dim=-1), torch.cat([attention_mask, mask_tensor], dim=-1)


def _decode_logits(
        session: DecodeSession,
        *,
        prefix: torch.LongTensor,
        attention_mask: torch.Tensor,
        step: int,
        config: MergedLoopConfig,
) -> torch.Tensor:
    if step == 0:
        logits = session.one_token_state().prefill_logits
        if logits is None:
            raise RuntimeError("DecodeSession prefill did not retain logits.")
        return logits
    cache_position = torch.tensor(
        [session.cache_state.prompt_length + step - 1],
        dtype=torch.long,
        device=prefix.device,
    )
    return session.decode_one_token_raw_logits(
        full_prefix=prefix,
        full_attention_mask=attention_mask,
        cache_position=cache_position,
        active_prefix_self_attention=config.active_prefix_decode,
        active_prefix_self_attention_length=config.active_prefix_decode_length,
    ).logits


@torch.no_grad()
def _run_timed_merged_loop(
        model: Any,
        *,
        prompt: torch.LongTensor,
        prompt_attention_mask: torch.Tensor,
        frames: torch.Tensor,
        condition_kwargs: Mapping[str, Any],
        eos_token_ids: Sequence[int],
        base_logits_processor_factory: LogitsProcessorFactory,
        logits_warper_factory: LogitsProcessorFactory,
        config: MergedLoopConfig,
        shared_base_processor: bool,
) -> dict[str, Any]:
    prompt_batched = repeat_batch_tensor(prompt, 8, name="decoder_input_ids")
    mask_batched = repeat_batch_tensor(prompt_attention_mask, 8, name="decoder_attention_mask")
    session = DecodeSession.prefill(
        model,
        prompt=prompt_batched,
        prompt_attention_mask=mask_batched,
        frames=repeat_batch_tensor(frames, 8, name="frames"),
        condition_kwargs=repeat_batch_kwargs(condition_kwargs, 8),
        active_prefix_self_attention=config.active_prefix_prefill,
    )
    prefix = prompt_batched
    attention_mask = mask_batched
    ledger = _new_ledger(eos_token_ids, max_new_tokens=config.max_new_tokens)
    processors = (
        [base_logits_processor_factory()]
        if shared_base_processor
        else [base_logits_processor_factory() for _ in range(8)]
    )
    warpers = [logits_warper_factory() for _ in range(8)]
    generators = [_new_generator(torch.device(model.device), seed) for seed in config.seeds]
    torch.cuda.synchronize(model.device)
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    started.record()
    for step in range(config.max_new_tokens):
        active_rows = ledger.active_indices()
        if not active_rows:
            break
        raw_logits = _decode_logits(
            session,
            prefix=prefix,
            attention_mask=attention_mask,
            step=step,
            config=config,
        )
        if shared_base_processor:
            processed_scores = processors[0](
                prefix,
                raw_logits.clone(memory_format=torch.contiguous_format),
            )
            row_scores = {row: processed_scores[row:row + 1] for row in active_rows}
        else:
            row_scores = {
                row: processors[row](
                    prefix[row:row + 1],
                    raw_logits[row:row + 1].clone(memory_format=torch.contiguous_format),
                )
                for row in active_rows
            }
        sampled_tensors = [
            _sample_after_base_processing(
                input_ids=prefix[row:row + 1],
                processed_scores=row_scores[row],
                logits_warper=warpers[row],
                generator=generators[row],
                do_sample=config.do_sample,
            )
            for row in active_rows
        ]
        sampled_values = torch.cat(sampled_tensors, dim=0).detach().cpu().tolist()
        sampled = dict(zip(active_rows, (int(token_id) for token_id in sampled_values), strict=True))
        full_tokens, active_mask = ledger.materialize_step(
            sampled,
            pad_token_id=config.pad_token_id,
        )
        prefix, attention_mask = _append_step(prefix, attention_mask, full_tokens, active_mask)
    finished.record()
    finished.synchronize()
    wall_seconds = time.perf_counter() - wall_started
    cuda_seconds = started.elapsed_time(finished) / 1000.0
    row_report = ledger.report()
    token_count = sum(int(row["draw_count"]) for row in row_report if row is not None)
    return {
        "base_processor_shape": "shared_B8" if shared_base_processor else "eight_private_B1",
        "rows": row_report,
        "final_rng_state_hashes": [_tensor_sha256(generator.get_state()) for generator in generators],
        "aggregate_main_tokens": token_count,
        "wall_seconds": wall_seconds,
        "cuda_seconds": cuda_seconds,
        "wall_tokens_per_second": token_count / wall_seconds,
        "cuda_tokens_per_second": token_count / cuda_seconds,
        "complete_steps": max((len(row["generated_token_ids"]) for row in row_report if row), default=0),
    }


@torch.no_grad()
def run_merged_loop_gate(
        model: Any,
        *,
        prompt: torch.LongTensor,
        prompt_attention_mask: torch.Tensor,
        frames: torch.Tensor,
        condition_kwargs: Mapping[str, Any],
        base_logits_processor_factory: LogitsProcessorFactory,
        logits_warper_factory: LogitsProcessorFactory,
        eos_token_ids: Sequence[int],
        config: MergedLoopConfig,
        runtime_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare independent B1 loops with one merged B8 shared-processor loop."""

    device = torch.device(model.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the merged loop gate requires a CUDA GPU allocation.")
    if model.dtype != torch.float32:
        raise ValueError(f"merged loop gate is FP32-only; model dtype is {model.dtype}.")
    if prompt.ndim != 2 or prompt.shape[0] != 1:
        raise ValueError("prompt must be a B1 rank-2 tensor.")

    condition_batched = repeat_batch_kwargs(condition_kwargs, 8)
    reference_sessions = [
        DecodeSession.prefill(
            model,
            prompt=prompt,
            prompt_attention_mask=prompt_attention_mask,
            frames=frames,
            condition_kwargs=slice_batch_kwargs(condition_batched, row),
            active_prefix_self_attention=config.active_prefix_prefill,
        )
        for row in range(8)
    ]
    prompt_batched = repeat_batch_tensor(prompt, 8, name="decoder_input_ids")
    mask_batched = repeat_batch_tensor(prompt_attention_mask, 8, name="decoder_attention_mask")
    merged_session = DecodeSession.prefill(
        model,
        prompt=prompt_batched,
        prompt_attention_mask=mask_batched,
        frames=repeat_batch_tensor(frames, 8, name="frames"),
        condition_kwargs=condition_batched,
        active_prefix_self_attention=config.active_prefix_prefill,
    )

    reference_prefixes = [prompt.clone() for _ in range(8)]
    reference_masks = [prompt_attention_mask.clone() for _ in range(8)]
    merged_prefix = prompt_batched
    merged_mask = mask_batched
    reference_ledger = _new_ledger(eos_token_ids, max_new_tokens=config.max_new_tokens)
    merged_ledger = _new_ledger(eos_token_ids, max_new_tokens=config.max_new_tokens)
    reference_processors = [base_logits_processor_factory() for _ in range(8)]
    merged_processor = base_logits_processor_factory()
    reference_warpers = [logits_warper_factory() for _ in range(8)]
    merged_warpers = [logits_warper_factory() for _ in range(8)]
    reference_generators = [_new_generator(device, seed) for seed in config.seeds]
    merged_generators = [_new_generator(device, seed) for seed in config.seeds]

    step_reports: list[dict[str, Any]] = []
    exactness_pass = True
    for step in range(config.max_new_tokens):
        active_rows = reference_ledger.active_indices()
        if active_rows != merged_ledger.active_indices():
            raise AssertionError("reference and merged active rows diverged before sampling.")
        if not active_rows:
            break
        reference_logits = {
            row: _decode_logits(
                reference_sessions[row],
                prefix=reference_prefixes[row],
                attention_mask=reference_masks[row],
                step=step,
                config=config,
            )
            for row in active_rows
        }
        merged_logits = _decode_logits(
            merged_session,
            prefix=merged_prefix,
            attention_mask=merged_mask,
            step=step,
            config=config,
        )
        cache_report = compare_active_session_caches(
            reference_sessions,
            merged_session,
            active_rows=active_rows,
            self_sequence_length=int(prompt.shape[-1]) + step,
            atol=config.atol,
            rtol=config.rtol,
        )
        merged_processed = merged_processor(
            merged_prefix,
            merged_logits.clone(memory_format=torch.contiguous_format),
        )
        row_reports: list[dict[str, Any]] = []
        reference_sampled: dict[int, int] = {}
        merged_sampled: dict[int, int] = {}
        for row in active_rows:
            raw_comparison = compare_logits(
                reference_logits[row],
                merged_logits[row:row + 1],
                atol=config.atol,
                rtol=config.rtol,
                top_k=config.top_k,
            )
            reference_processed = reference_processors[row](
                reference_prefixes[row],
                reference_logits[row].clone(memory_format=torch.contiguous_format),
            )
            processed_comparison = compare_logits(
                reference_processed,
                merged_processed[row:row + 1],
                atol=config.atol,
                rtol=config.rtol,
                top_k=config.top_k,
            )
            reference_token = _sample_after_base_processing(
                input_ids=reference_prefixes[row],
                processed_scores=reference_processed,
                logits_warper=reference_warpers[row],
                generator=reference_generators[row],
                do_sample=config.do_sample,
            )
            merged_token = _sample_after_base_processing(
                input_ids=merged_prefix[row:row + 1],
                processed_scores=merged_processed[row:row + 1],
                logits_warper=merged_warpers[row],
                generator=merged_generators[row],
                do_sample=config.do_sample,
            )
            reference_token_id = int(reference_token.detach().cpu().item())
            merged_token_id = int(merged_token.detach().cpu().item())
            reference_sampled[row] = reference_token_id
            merged_sampled[row] = merged_token_id
            token_match = reference_token_id == merged_token_id
            rng_match = (
                _tensor_sha256(reference_generators[row].get_state())
                == _tensor_sha256(merged_generators[row].get_state())
            )
            row_pass = bool(
                raw_comparison["allclose"]
                and raw_comparison["topk_match"]
                and processed_comparison["allclose"]
                and processed_comparison["topk_match"]
                and token_match
                and rng_match
            )
            exactness_pass = exactness_pass and row_pass
            row_reports.append({
                "row": row,
                "raw_logits": raw_comparison,
                "processed_scores": processed_comparison,
                "reference_token_id": reference_token_id,
                "merged_token_id": merged_token_id,
                "token_match": token_match,
                "rng_state_match_after_draw": rng_match,
                "pass": row_pass,
            })
        reference_tokens, reference_active_mask = reference_ledger.materialize_step(
            reference_sampled,
            pad_token_id=config.pad_token_id,
        )
        merged_tokens, merged_active_mask = merged_ledger.materialize_step(
            merged_sampled,
            pad_token_id=config.pad_token_id,
        )
        state_match = bool(
            reference_tokens == merged_tokens
            and reference_active_mask == merged_active_mask
        )
        exactness_pass = exactness_pass and cache_report["pass"] and state_match
        for row in active_rows:
            reference_prefixes[row], reference_masks[row] = _append_step(
                reference_prefixes[row],
                reference_masks[row],
                (reference_tokens[row],),
                (reference_active_mask[row],),
            )
        merged_prefix, merged_mask = _append_step(
            merged_prefix,
            merged_mask,
            merged_tokens,
            merged_active_mask,
        )
        step_reports.append({
            "step": step,
            "active_rows_before_draw": list(active_rows),
            "rows": row_reports,
            "active_cache": cache_report,
            "materialized_state_match": state_match,
            "pass": bool(cache_report["pass"] and state_match and all(row["pass"] for row in row_reports)),
        })

    reference_rows = reference_ledger.report()
    merged_rows = merged_ledger.report()
    reference_rng_hashes = [_tensor_sha256(generator.get_state()) for generator in reference_generators]
    merged_rng_hashes = [_tensor_sha256(generator.get_state()) for generator in merged_generators]
    transcript_match = reference_rows == merged_rows
    final_rng_match = reference_rng_hashes == merged_rng_hashes
    no_extra_draws = all(
        row is not None and int(row["draw_count"]) == len(row["generated_token_ids"])
        for row in merged_rows
    )
    exactness_pass = bool(exactness_pass and transcript_match and final_rng_match and no_extra_draws)

    del reference_sessions, merged_session
    torch.cuda.empty_cache()
    baseline_timing = _run_timed_merged_loop(
        model,
        prompt=prompt,
        prompt_attention_mask=prompt_attention_mask,
        frames=frames,
        condition_kwargs=condition_kwargs,
        eos_token_ids=eos_token_ids,
        base_logits_processor_factory=base_logits_processor_factory,
        logits_warper_factory=logits_warper_factory,
        config=config,
        shared_base_processor=False,
    )
    candidate_timing = _run_timed_merged_loop(
        model,
        prompt=prompt,
        prompt_attention_mask=prompt_attention_mask,
        frames=frames,
        condition_kwargs=condition_kwargs,
        eos_token_ids=eos_token_ids,
        base_logits_processor_factory=base_logits_processor_factory,
        logits_warper_factory=logits_warper_factory,
        config=config,
        shared_base_processor=True,
    )
    timed_transcript_match = baseline_timing["rows"] == candidate_timing["rows"]
    timed_rng_match = (
        baseline_timing["final_rng_state_hashes"]
        == candidate_timing["final_rng_state_hashes"]
    )
    candidate_tps = float(candidate_timing["wall_tokens_per_second"])
    baseline_tps = float(baseline_timing["wall_tokens_per_second"])
    relative_gain = candidate_tps / baseline_tps - 1.0
    performance = {
        "baseline_wall_tokens_per_second": baseline_tps,
        "candidate_wall_tokens_per_second": candidate_tps,
        "relative_gain": relative_gain,
        "clears_500_tokens_per_second": candidate_tps >= 500.0,
        "clears_five_percent_gain_gate": relative_gain >= 0.05,
        "timed_transcript_match": timed_transcript_match,
        "timed_final_rng_state_match": timed_rng_match,
        "promotion_gate_pass": bool(
            timed_transcript_match
            and timed_rng_match
            and candidate_tps >= 500.0
            and relative_gain >= 0.05
        ),
    }
    return {
        "schema_version": 1,
        "scope": "identical_prompt_B8_16_step_verifier_only",
        "request_state_status": (
            "not_a_valid_request_state_design: one shared processor object remains unproven for "
            "mixed prompts, staggered arrivals, slot reuse, or state reset"
        ),
        "batch_size": 8,
        "max_new_tokens": config.max_new_tokens,
        "result_class": "exact-output",
        "runtime": dict(runtime_metadata),
        "steps": step_reports,
        "reference_rows": reference_rows,
        "merged_rows": merged_rows,
        "reference_final_rng_state_hashes": reference_rng_hashes,
        "merged_final_rng_state_hashes": merged_rng_hashes,
        "token_transcripts_and_stops_match": transcript_match,
        "final_rng_states_match": final_rng_match,
        "no_stopped_or_dummy_draws": no_extra_draws,
        "exactness_pass": exactness_pass,
        "timing": {
            "scope": "prefill_excluded_complete_sampled_16_step_changing_prefix_loop",
            "baseline_private_processors": baseline_timing,
            "candidate_shared_processor": candidate_timing,
            "performance": performance,
        },
        "performance_gate_pass": performance["promotion_gate_pass"],
        "pass": bool(exactness_pass and performance["promotion_gate_pass"]),
    }
