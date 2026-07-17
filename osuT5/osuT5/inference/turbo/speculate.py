"""Draft-K + batched teacher verify speculative decode for turbo.

Owns the Leviathan loop only. Does not touch optimized bit-exact paths.
Distribution-equivalent under rejection sampling; TIER1 required before ship.
"""
from __future__ import annotations

import time
from typing import Any, Sequence

import torch
from transformers import DynamicCache, LogitsProcessorList

from ...event import ContextType, EventType
from ..cache_utils import MapperatorinatorCache
from ..generation_utils import build_generation_stats, eos_token_ids, sync_cuda_for_model
from ..logit_processors import (
    LookbackBiasLogitsWarper,
    MonotonicTimeShiftLogitsProcessor,
    TimeshiftBias,
)
from .rejection import apply_temp_top_p, reject_sample_prefix


def get_turbo_cache(*, cfg_scale: float = 1.0) -> MapperatorinatorCache:
    """Dynamic KV cache so EncoderDecoderCache.crop works after reject."""
    if cfg_scale != 1.0:
        raise ValueError("turbo speculative decode requires cfg_scale=1.0")
    return MapperatorinatorCache(DynamicCache(), DynamicCache(), cfg_scale)


def crop_self_cache(cache: MapperatorinatorCache, length: int) -> None:
    cache.self_attention_cache.crop(int(length))


def _align_decoder_mask(
    model_kwargs: dict[str, Any],
    *,
    length: int,
    device: torch.device,
) -> None:
    """Keep decoder_attention_mask / cache_position consistent after KV crop."""
    model_kwargs["decoder_attention_mask"] = torch.ones(
        (1, int(length)),
        device=device,
        dtype=torch.long,
    )
    model_kwargs.pop("cache_position", None)


def build_structural_processors(
    tokenizer,
    *,
    timeshift_bias: float,
    types_first: bool,
    lookback_time: float,
    device,
) -> LogitsProcessorList:
    """Logits processors without temperature/top-p (owned by apply_temp_top_p)."""
    if types_first:
        raise ValueError(
            "turbo speculative decode does not support types_first=True yet "
            "(conditional temperature warper)."
        )
    processors = LogitsProcessorList()
    processors.append(MonotonicTimeShiftLogitsProcessor(tokenizer))
    if timeshift_bias != 0:
        processors.append(
            TimeshiftBias(
                timeshift_bias,
                tokenizer.event_start[EventType.TIME_SHIFT],
                tokenizer.event_end[EventType.TIME_SHIFT],
            )
        )
    if lookback_time > 0:
        processors.append(
            LookbackBiasLogitsWarper(
                lookback_time,
                tokenizer,
                types_first,
                device,
            )
        )
    return processors


def _move_model_kwargs(model, model_kwargs: dict[str, Any]) -> dict[str, Any]:
    out = {
        key: value.to(model.device) if isinstance(value, torch.Tensor) else value
        for key, value in model_kwargs.items()
    }
    out = {
        key: (
            value.to(model.dtype)
            if key != "inputs"
            and isinstance(value, torch.Tensor)
            and value.dtype == torch.float32
            else value
        )
        for key, value in out.items()
    }
    if "inputs" in out:
        # Mapperatorinator.forward takes `frames`; keep a single audio tensor.
        frames = out.pop("inputs")
        out["frames"] = out.get("frames", frames)
    return out


def _apply_processors(
    processors: LogitsProcessorList,
    input_ids: torch.LongTensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    scores = logits.float()
    if len(processors) == 0:
        return scores
    return processors(input_ids, scores)


def _sample_from_logits(
    logits_1d: torch.Tensor,
    *,
    greedy: bool,
    temperature: float,
    top_p: float,
    rng: torch.Generator | None,
) -> tuple[int, torch.Tensor]:
    """Return (token_id, probs[V]) under turbo sampling distribution."""
    if greedy or temperature <= 1e-5:
        probs = torch.zeros_like(logits_1d, dtype=torch.float32)
        tok = int(torch.argmax(logits_1d).item())
        probs[tok] = 1.0
        return tok, probs
    probs = apply_temp_top_p(logits_1d.unsqueeze(0), temperature, top_p).squeeze(0)
    tok = int(torch.multinomial(probs, 1, generator=rng).item())
    return tok, probs


def _forward_decoder(
    model,
    *,
    decoder_input_ids: torch.LongTensor,
    model_kwargs: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Cached forward; returns (outputs, updated model_kwargs)."""
    model_inputs = model.prepare_inputs_for_generation(
        decoder_input_ids,
        **model_kwargs,
    )
    outputs = model(**model_inputs, return_dict=True)
    model_kwargs = model._update_model_kwargs_for_generation(
        outputs,
        model_kwargs,
        is_encoder_decoder=True,
    )
    return outputs, model_kwargs


def _prefill(
    model,
    *,
    prompt_ids: torch.LongTensor,
    model_kwargs: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    outputs, model_kwargs = _forward_decoder(
        model,
        decoder_input_ids=prompt_ids,
        model_kwargs=model_kwargs,
    )
    last_logits = outputs.logits[:, -1, :].float().squeeze(0)
    return last_logits, model_kwargs


def _draft_k_tokens(
    draft,
    *,
    prompt_ids: torch.LongTensor,
    last_logits: torch.Tensor,
    model_kwargs: dict[str, Any],
    gamma: int,
    greedy: bool,
    temperature: float,
    top_p: float,
    processors: LogitsProcessorList,
    rng: torch.Generator | None,
    max_new_tokens: int,
    eos_ids: set[int],
) -> tuple[list[int], torch.Tensor, dict[str, Any], bool]:
    """Autoregressive draft up to γ tokens.

    Returns draft_ids, q_probs[γ,V], updated kwargs, stopped_on_eos.
    """
    ids = prompt_ids
    probs_q: list[torch.Tensor] = []
    draft_ids: list[int] = []
    cur_logits = last_logits
    stopped = False
    mk = model_kwargs

    for _ in range(min(gamma, max_new_tokens)):
        processed = _apply_processors(processors, ids, cur_logits.unsqueeze(0)).squeeze(0)
        tok, probs = _sample_from_logits(
            processed,
            greedy=greedy,
            temperature=temperature,
            top_p=top_p,
            rng=rng,
        )
        probs_q.append(probs)
        draft_ids.append(tok)
        tok_t = torch.tensor([[tok]], device=ids.device, dtype=ids.dtype)
        ids = torch.cat([ids, tok_t], dim=-1)
        if tok in eos_ids:
            stopped = True
            break
        outputs, mk = _forward_decoder(draft, decoder_input_ids=ids, model_kwargs=mk)
        cur_logits = outputs.logits[:, -1, :].float().squeeze(0)

    if not draft_ids:
        raise RuntimeError("draft produced zero tokens")
    return draft_ids, torch.stack(probs_q, dim=0), mk, stopped


def _verify_draft_tokens(
    teacher,
    *,
    prompt_ids: torch.LongTensor,
    draft_ids: Sequence[int],
    last_logits: torch.Tensor,
    model_kwargs: dict[str, Any],
    processors: LogitsProcessorList,
    greedy: bool,
    temperature: float,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """One multi-token teacher forward over drafted ids.

    Returns p_logits[γ,V], p_probs[γ,V], bonus_logits[V], updated kwargs.
    """
    gamma = len(draft_ids)
    draft_t = torch.tensor([list(draft_ids)], device=prompt_ids.device, dtype=prompt_ids.dtype)
    ids = torch.cat([prompt_ids, draft_t], dim=-1)
    outputs, mk = _forward_decoder(
        teacher,
        decoder_input_ids=ids,
        model_kwargs=model_kwargs,
    )
    verify_logits = outputs.logits[:, -gamma:, :].float().squeeze(0)  # [γ, V]
    if gamma == 1:
        p_raw = last_logits.unsqueeze(0)
        bonus_logits = verify_logits[0]
    else:
        p_raw = torch.cat([last_logits.unsqueeze(0), verify_logits[:-1]], dim=0)
        bonus_logits = verify_logits[-1]

    p_logits_list = []
    p_probs_list = []
    for i in range(gamma):
        prefix = ids[:, : prompt_ids.shape[1] + i]
        processed = _apply_processors(processors, prefix, p_raw[i].unsqueeze(0)).squeeze(0)
        p_logits_list.append(processed)
        if greedy or temperature <= 1e-5:
            probs = torch.zeros_like(processed)
            probs[int(torch.argmax(processed).item())] = 1.0
        else:
            probs = apply_temp_top_p(processed.unsqueeze(0), temperature, top_p).squeeze(0)
        p_probs_list.append(probs)

    return (
        torch.stack(p_logits_list, dim=0),
        torch.stack(p_probs_list, dim=0),
        bonus_logits,
        mk,
    )


@torch.no_grad()
def speculative_generate_window(
    *,
    teacher,
    draft,
    tokenizer,
    model_kwargs: dict[str, Any],
    generate_kwargs: dict[str, Any],
    gamma: int,
    temperature: float,
    top_p: float,
    session: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run speculative decode for one window; return (sequences_cpu, stats)."""
    gw = dict(generate_kwargs)
    precision = gw.pop("precision", "fp32")
    cfg_scale = float(gw.pop("cfg_scale", 1.0))
    timeshift_bias = gw.pop("timeshift_bias", 0)
    types_first = bool(gw.pop("types_first", False))
    req_temperature = float(gw.pop("temperature", temperature))
    gw.pop("timing_temperature", None)
    gw.pop("mania_column_temperature", None)
    gw.pop("taiko_hit_temperature", None)
    lookback_time = float(gw.pop("lookback_time", 0.0))
    lookahead_time = float(gw.pop("lookahead_time", 0.0))
    context_type = gw.pop("context_type", None)
    sync_model_timing = bool(gw.pop("sync_model_timing", False))
    gw.pop("collect_strict_exactness", None)
    do_sample = bool(gw.pop("do_sample", True))
    max_length = int(gw.pop("max_length", teacher.config.max_target_positions))
    pad_token_id = gw.get("pad_token_id", getattr(tokenizer, "pad_id", None))
    gw.pop("num_beams", None)
    gw.pop("top_k", None)
    gw.pop("top_p", None)

    if cfg_scale != 1.0:
        raise ValueError("turbo speculative decode requires cfg_scale=1.0")
    if context_type is not None:
        context_type = ContextType(context_type)

    greedy = (not do_sample) or req_temperature <= 1e-5
    spec_temperature = 0.0 if greedy else float(temperature)
    spec_top_p = float(top_p)

    teacher_mk = _move_model_kwargs(teacher, model_kwargs)
    if "decoder_input_ids" not in teacher_mk:
        raise ValueError("turbo generate_window requires decoder_input_ids")
    prompt_ids = teacher_mk.pop("decoder_input_ids")
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("turbo speculative decode requires batch_size=1")

    eos_list = eos_token_ids(
        tokenizer,
        lookback_time=lookback_time,
        lookahead_time=lookahead_time,
        context_type=context_type,
    )
    eos_ids = set(int(x) for x in eos_list)
    processors = build_structural_processors(
        tokenizer,
        timeshift_bias=float(timeshift_bias),
        types_first=types_first,
        lookback_time=lookback_time,
        device=teacher.device,
    )

    teacher_cache = get_turbo_cache(cfg_scale=1.0)
    draft_cache = get_turbo_cache(cfg_scale=1.0)
    teacher_mk["past_key_values"] = teacher_cache
    teacher_mk["use_cache"] = True

    draft_mk = _move_model_kwargs(draft, model_kwargs)
    draft_mk.pop("decoder_input_ids", None)
    draft_mk["past_key_values"] = draft_cache
    draft_mk["use_cache"] = True

    rng = None
    if not greedy:
        rng = torch.Generator(device=teacher.device)
        seed = int(torch.randint(0, 2**31 - 1, (), device="cpu").item())
        rng.manual_seed(seed)

    if sync_model_timing:
        sync_cuda_for_model(teacher)
    start_time = time.perf_counter()

    teacher_last, teacher_mk = _prefill(
        teacher, prompt_ids=prompt_ids, model_kwargs=teacher_mk
    )
    if teacher_mk.get("encoder_outputs") is not None:
        draft_mk["encoder_outputs"] = teacher_mk["encoder_outputs"]
    draft_last, draft_mk = _prefill(
        draft, prompt_ids=prompt_ids, model_kwargs=draft_mk
    )

    sequences = prompt_ids
    prompt_len = int(prompt_ids.shape[1])
    accepted_total = 0
    verify_steps = 0
    draft_calls = 0
    stopped = False

    while sequences.shape[1] < max_length and not stopped:
        remaining = max_length - int(sequences.shape[1])
        L = int(sequences.shape[1])
        this_gamma = min(int(gamma), remaining)

        draft_ids, q_probs, draft_mk, _draft_hit_eos = _draft_k_tokens(
            draft,
            prompt_ids=sequences,
            last_logits=draft_last,
            model_kwargs=draft_mk,
            gamma=this_gamma,
            greedy=greedy,
            temperature=spec_temperature,
            top_p=spec_top_p,
            processors=processors,
            rng=rng,
            max_new_tokens=remaining,
            eos_ids=eos_ids,
        )
        draft_calls += 1
        actual_gamma = len(draft_ids)

        p_logits, p_probs, bonus_logits, teacher_mk = _verify_draft_tokens(
            teacher,
            prompt_ids=sequences,
            draft_ids=draft_ids,
            last_logits=teacher_last,
            model_kwargs=teacher_mk,
            processors=processors,
            greedy=greedy,
            temperature=spec_temperature,
            top_p=spec_top_p,
        )
        verify_steps += 1

        if greedy:
            n_accepted = 0
            residual = None
            for i, tok in enumerate(draft_ids):
                teacher_tok = int(torch.argmax(p_logits[i]).item())
                if tok == teacher_tok:
                    n_accepted += 1
                else:
                    residual = teacher_tok
                    break
        else:
            n_accepted, residual = reject_sample_prefix(
                p_probs=p_probs[:actual_gamma],
                q_probs=q_probs[:actual_gamma],
                draft_token_ids=draft_ids,
                rng=rng,
            )

        commit: list[int] = list(draft_ids[:n_accepted])
        if residual is not None:
            commit.append(int(residual))
        elif n_accepted == actual_gamma and not any(
            int(t) in eos_ids for t in commit
        ):
            # Bonus only when the full draft was accepted and did not end on EOS.
            bonus_prefix = torch.cat(
                [
                    sequences,
                    torch.tensor(
                        [draft_ids], device=sequences.device, dtype=sequences.dtype
                    ),
                ],
                dim=-1,
            )
            bonus_processed = _apply_processors(
                processors,
                bonus_prefix,
                bonus_logits.unsqueeze(0),
            ).squeeze(0)
            bonus_tok, _ = _sample_from_logits(
                bonus_processed,
                greedy=greedy,
                temperature=spec_temperature,
                top_p=spec_top_p,
                rng=rng,
            )
            commit.append(bonus_tok)

        if not commit:
            raise RuntimeError("turbo speculative step committed zero tokens")

        commit = commit[:remaining]

        # Verify/draft wrote γ slots past L; roll back and materialize only commit.
        crop_self_cache(teacher_cache, L)
        crop_self_cache(draft_cache, L)
        _align_decoder_mask(teacher_mk, length=L, device=sequences.device)
        _align_decoder_mask(draft_mk, length=L, device=sequences.device)

        commit_t = torch.tensor([commit], device=sequences.device, dtype=sequences.dtype)
        sequences = torch.cat([sequences, commit_t], dim=-1)
        _align_decoder_mask(
            teacher_mk, length=int(sequences.shape[1]), device=sequences.device
        )
        _align_decoder_mask(
            draft_mk, length=int(sequences.shape[1]), device=sequences.device
        )

        teacher_out, teacher_mk = _forward_decoder(
            teacher, decoder_input_ids=sequences, model_kwargs=teacher_mk
        )
        teacher_last = teacher_out.logits[:, -1, :].float().squeeze(0)
        draft_out, draft_mk = _forward_decoder(
            draft, decoder_input_ids=sequences, model_kwargs=draft_mk
        )
        draft_last = draft_out.logits[:, -1, :].float().squeeze(0)

        accepted_total += len(commit)
        session.accepted_tokens_total += len(commit)
        session.verify_steps += 1
        session.draft_calls += 1

        for tok in commit:
            if int(tok) in eos_ids:
                stopped = True
                break

    if sync_model_timing:
        sync_cuda_for_model(teacher)
    elapsed = time.perf_counter() - start_time

    result = sequences.cpu()
    stats = build_generation_stats(result, model_kwargs, pad_token_id, elapsed)
    stats.update(
        {
            "precision": precision,
            "context_type": context_type.value if context_type is not None else None,
            "decoder_loop_backend": "turbo_speculative_draft_verify",
            "turbo_speculative": True,
            "turbo_scaffold": False,
            "turbo_teacher_fallback": False,
            "turbo_gamma": int(gamma),
            "turbo_greedy": greedy,
            "turbo_temperature": spec_temperature,
            "turbo_top_p": spec_top_p,
            "turbo_verify_steps": verify_steps,
            "turbo_draft_calls": draft_calls,
            "turbo_accepted_tokens": accepted_total,
            "turbo_accepted_per_verify": (
                accepted_total / verify_steps if verify_steps > 0 else 0.0
            ),
            "turbo_prompt_tokens": prompt_len,
        }
    )
    return result, stats
