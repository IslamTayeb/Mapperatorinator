"""Strict Leviathan speculative window on tiger compiled decode (§58 STEP 2).

Flow: draft propose → tiger ``verify_k`` (CUDAGraphDecoder q_len=K) →
rejection sampling (§34) → keep-accepted-KV rewind on bucketed StaticCache.
No §54 fused verify. Distribution-equivalent; not bit-exact; not a 500 claim.
"""
from __future__ import annotations

import os
import statistics
import time
from typing import Any, Sequence

import torch
from transformers import LogitsProcessorList
from transformers.modeling_outputs import BaseModelOutput

from ...event import ContextType, EventType
from ..compiled_decode import _pick_bucket, _reset_cache, neutralize_dynamic_rope
from ..logit_processors import (
    LookbackBiasLogitsWarper,
    MonotonicTimeShiftLogitsProcessor,
    TimeshiftBias,
)
from ..server import _build_generation_stats, get_eos_token_id
from .draft_chain_graph import TigerDraftChainRunner, draft_chain_graph_enabled
from .kv_rollback import rewind_self_cache
from .rejection import apply_temp_top_p, reject_sample_prefix
from .tiger_verify import TigerVerifySession, ensure_tiger_verify, tiger_verify_forward_k


def structural_processors_enabled() -> bool:
    """Default off to match §43 offline probe; set env=1 to enable."""
    raw = os.environ.get("MAPPERATORINATOR_TURBO_STRUCTURAL_PROCESSORS", "0").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def build_structural_processors(
    tokenizer,
    *,
    timeshift_bias: float,
    types_first: bool,
    lookback_time: float,
    device,
) -> LogitsProcessorList:
    if types_first:
        raise ValueError("turbo-on-tiger does not support types_first=True yet")
    processors = LogitsProcessorList()
    if not structural_processors_enabled():
        return processors
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
            LookbackBiasLogitsWarper(lookback_time, tokenizer, types_first, device)
        )
    return processors


def _apply_processors(
    processors: LogitsProcessorList,
    input_ids: torch.LongTensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    scores = logits.float().clone()
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
    if greedy or temperature <= 1e-5:
        probs = torch.zeros_like(logits_1d, dtype=torch.float32)
        tok = int(torch.argmax(logits_1d).item())
        probs[tok] = 1.0
        return tok, probs
    probs = apply_temp_top_p(logits_1d.unsqueeze(0), temperature, top_p).squeeze(0)
    tok = int(torch.multinomial(probs, 1, generator=rng).item())
    return tok, probs


def _prefill_static(
    model,
    *,
    cache,
    prompt_ids: torch.LongTensor,
    enc_hidden: torch.Tensor,
) -> torch.Tensor:
    """Eager prefill into StaticCache; returns last-position logits ``(V,)``."""
    _reset_cache(cache)
    device = prompt_ids.device
    prompt_len = int(prompt_ids.shape[1])
    enc_out = BaseModelOutput(last_hidden_state=enc_hidden)
    cache_position = torch.arange(prompt_len, device=device, dtype=torch.long)
    inputs = model.prepare_inputs_for_generation(
        prompt_ids,
        past_key_values=cache,
        use_cache=True,
        encoder_outputs=enc_out,
        cache_position=cache_position,
    )
    out = model(**inputs)
    return out.logits[:, -1, :].float().squeeze(0)


@torch.no_grad()
def _eager_forward_tokens(
    model,
    *,
    cache,
    enc_hidden: torch.Tensor,
    tokens: torch.LongTensor,
    prefix_len: int,
) -> torch.Tensor:
    """Eager decode of ``tokens`` ``(1, n)`` into StaticCache; logits ``(n, V)``."""
    n = int(tokens.shape[1])
    device = tokens.device
    enc_out = BaseModelOutput(last_hidden_state=enc_hidden)
    cache_position = torch.arange(
        int(prefix_len), int(prefix_len) + n, device=device, dtype=torch.long
    )
    inputs = model.prepare_inputs_for_generation(
        tokens,
        past_key_values=cache,
        use_cache=True,
        encoder_outputs=enc_out,
        cache_position=cache_position,
    )
    out = model(**inputs)
    logits = out.logits.float()
    if logits.ndim == 3:
        return logits[0]
    return logits


def _build_p_rows(
    *,
    teacher_last: torch.Tensor,
    verify_logits: torch.Tensor,
    draft_ids: Sequence[int],
    sequences: torch.LongTensor,
    processors: LogitsProcessorList,
    greedy: bool,
    temperature: float,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return p_logits, p_probs, bonus_logits."""
    gamma = len(draft_ids)
    if gamma == 1:
        p_raw = teacher_last.unsqueeze(0)
        bonus_logits = verify_logits[0]
    else:
        p_raw = torch.cat([teacher_last.unsqueeze(0), verify_logits[:-1]], dim=0)
        bonus_logits = verify_logits[-1]
    p_logits_list = []
    p_probs_list = []
    for i in range(gamma):
        if i == 0:
            prefix = sequences
        else:
            prefix = torch.cat(
                [
                    sequences,
                    torch.tensor(
                        [list(draft_ids[:i])],
                        device=sequences.device,
                        dtype=sequences.dtype,
                    ),
                ],
                dim=-1,
            )
        processed = _apply_processors(processors, prefix, p_raw[i].unsqueeze(0)).squeeze(0)
        p_logits_list.append(processed)
        if greedy or temperature <= 1e-5:
            probs = torch.zeros_like(processed)
            probs[int(torch.argmax(processed).item())] = 1.0
        else:
            probs = apply_temp_top_p(processed.unsqueeze(0), temperature, top_p).squeeze(0)
        p_probs_list.append(probs)
    return torch.stack(p_logits_list, dim=0), torch.stack(p_probs_list, dim=0), bonus_logits


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
    """One-window speculative decode on tiger teacher + draft."""
    gw = dict(generate_kwargs)
    precision = gw.pop("precision", "fp16")
    cfg_scale = float(gw.pop("cfg_scale", 1.0))
    timeshift_bias = gw.pop("timeshift_bias", 0)
    types_first = bool(gw.pop("types_first", False))
    req_temperature = float(gw.pop("temperature", temperature))
    for _k in (
        "timing_temperature",
        "mania_column_temperature",
        "taiko_hit_temperature",
        "num_beams",
        "top_k",
        "top_p",
        "collect_strict_exactness",
        "sync_model_timing",
    ):
        gw.pop(_k, None)
    lookback_time = float(gw.pop("lookback_time", 0.0))
    lookahead_time = float(gw.pop("lookahead_time", 0.0))
    context_type = gw.pop("context_type", None)
    do_sample = bool(gw.pop("do_sample", True))
    max_length = int(gw.pop("max_length", teacher.config.max_target_positions))
    pad_token_id = gw.get("pad_token_id", getattr(tokenizer, "pad_id", None))

    if cfg_scale != 1.0:
        raise ValueError("turbo-on-tiger requires cfg_scale=1.0")
    if context_type is not None:
        context_type = ContextType(context_type)

    greedy = (not do_sample) or req_temperature <= 1e-5
    if greedy:
        raise ValueError(
            "turbo-on-tiger STEP 2 wires strict rejection-sampling only; "
            "use do_sample=true with temperature>0"
        )
    spec_temperature = float(temperature)
    spec_top_p = float(top_p)

    neutralize_dynamic_rope(teacher)
    neutralize_dynamic_rope(draft)

    mk = {
        k: (v.to(teacher.device) if isinstance(v, torch.Tensor) else v)
        for k, v in model_kwargs.items()
    }
    if "decoder_input_ids" not in mk:
        raise ValueError("turbo generate_window requires decoder_input_ids")
    prompt_ids = mk["decoder_input_ids"]
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("turbo-on-tiger requires batch_size=1")

    enc = mk.get("encoder_outputs")
    if isinstance(enc, BaseModelOutput):
        enc_hidden = enc.last_hidden_state
    elif isinstance(enc, torch.Tensor):
        enc_hidden = enc
    else:
        raise ValueError("turbo-on-tiger requires precomputed encoder_outputs")
    enc_hidden = enc_hidden.to(device=teacher.device, dtype=teacher.dtype).contiguous()

    eos_list = get_eos_token_id(
        tokenizer,
        lookback_time=lookback_time,
        lookahead_time=lookahead_time,
        context_type=context_type,
    )
    eos_ids = {int(x) for x in eos_list}
    processors = build_structural_processors(
        tokenizer,
        timeshift_bias=float(timeshift_bias),
        types_first=types_first,
        lookback_time=lookback_time,
        device=teacher.device,
    )

    model_max = int(teacher.config.max_target_positions)
    hard_cap = min(max_length, model_max)
    cache_len = _pick_bucket(int(prompt_ids.shape[1]), model_max)
    win_max = min(hard_cap, cache_len)

    if getattr(session, "tiger_verify", None) is None:
        session.tiger_verify = TigerVerifySession()
    ensure_tiger_verify(
        session.tiger_verify,
        teacher,
        enc_hidden=enc_hidden,
        cache_len=cache_len,
        q_len=int(gamma),
        cfg_scale=1.0,
    )
    teacher_cache = session.tiger_verify.cache

    use_draft_chain = bool(
        draft_chain_graph_enabled() and prompt_ids.device.type == "cuda"
    )
    chain_runner: TigerDraftChainRunner | None = None
    if use_draft_chain:
        chain_runner = getattr(session, "draft_chain_runner", None)
        need_new = (
            chain_runner is None
            or chain_runner.draft is not draft
            or int(chain_runner.gamma) != int(gamma)
            or float(chain_runner.temperature) != float(spec_temperature)
            or float(chain_runner.top_p) != float(spec_top_p)
        )
        if need_new:
            chain_runner = TigerDraftChainRunner(
                draft,
                gamma=int(gamma),
                temperature=float(spec_temperature),
                top_p=float(spec_top_p),
            )
            session.draft_chain_runner = chain_runner
        chain_runner.ensure(enc_hidden=enc_hidden, cache_len=cache_len)

    rng = torch.Generator(device=teacher.device)
    rng.manual_seed(int(torch.randint(0, 2**31 - 1, (), device="cpu").item()))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    teacher_last = _prefill_static(
        teacher, cache=teacher_cache, prompt_ids=prompt_ids, enc_hidden=enc_hidden
    )
    if use_draft_chain and chain_runner is not None:
        draft_last = _prefill_static(
            draft, cache=chain_runner.cache, prompt_ids=prompt_ids, enc_hidden=enc_hidden
        )
    else:
        # Eager draft Dynamic/Static fallback uses a dedicated cache on session.
        from ..cache_utils import get_cache

        if getattr(session, "draft_cache", None) is None:
            session.draft_cache = get_cache(
                draft, batch_size=1, num_beams=1, cfg_scale=1.0, max_cache_len=cache_len
            )
        else:
            _reset_cache(session.draft_cache)
        draft_last = _prefill_static(
            draft, cache=session.draft_cache, prompt_ids=prompt_ids, enc_hidden=enc_hidden
        )

    sequences = prompt_ids
    prompt_len = int(prompt_ids.shape[1])
    accepted_total = 0
    verify_steps = 0
    draft_calls = 0
    window_keep_kv = 0
    window_draft_chain = 0
    window_draft_eager = 0
    window_teacher_verify = 0
    window_teacher_extra_q1 = 0
    accepted_per_cycle: list[int] = []
    stopped = False

    while int(sequences.shape[1]) < win_max and not stopped:
        remaining = win_max - int(sequences.shape[1])
        L = int(sequences.shape[1])
        this_gamma = min(int(gamma), remaining)

        used_chain = bool(
            use_draft_chain
            and chain_runner is not None
            and this_gamma == int(chain_runner.gamma)
        )
        if used_chain:
            proposed = chain_runner.propose_chain(
                seed_logits=draft_last, past_length=L, rng=rng
            )
            draft_ids = [int(t) for t in proposed["tokens"]]
            q_probs = proposed["q_probs"]
            draft_logits_after = [
                row.float().view(-1) for row in proposed["logits_after"]
            ]
            trimmed: list[int] = []
            for i, tok in enumerate(draft_ids):
                trimmed.append(tok)
                if tok in eos_ids:
                    break
            draft_ids = trimmed
            q_probs = q_probs[: len(draft_ids)]
            draft_logits_after = draft_logits_after[: len(draft_ids)]
            window_draft_chain += 1
        else:
            # Eager autoregressive draft on session.draft_cache / chain cache.
            draft_cache = (
                chain_runner.cache
                if chain_runner is not None and chain_runner.cache is not None
                else session.draft_cache
            )
            draft_ids = []
            q_rows = []
            draft_logits_after = []
            cur_logits = draft_last
            cur_len = L
            for _ in range(this_gamma):
                processed = _apply_processors(
                    processors, sequences if not draft_ids else torch.cat(
                        [
                            sequences,
                            torch.tensor(
                                [draft_ids], device=sequences.device, dtype=sequences.dtype
                            ),
                        ],
                        dim=-1,
                    ),
                    cur_logits.unsqueeze(0),
                ).squeeze(0)
                tok, probs = _sample_from_logits(
                    processed,
                    greedy=False,
                    temperature=spec_temperature,
                    top_p=spec_top_p,
                    rng=rng,
                )
                q_rows.append(probs)
                draft_ids.append(tok)
                if tok in eos_ids:
                    draft_logits_after.append(cur_logits)
                    break
                step_logits = _eager_forward_tokens(
                    draft,
                    cache=draft_cache,
                    enc_hidden=enc_hidden,
                    tokens=torch.tensor([[tok]], device=sequences.device, dtype=torch.long),
                    prefix_len=cur_len,
                )
                cur_logits = step_logits[-1].float().view(-1)
                draft_logits_after.append(cur_logits)
                cur_len += 1
            q_probs = torch.stack(q_rows, dim=0)
            window_draft_eager += 1

        draft_calls += 1
        actual_gamma = len(draft_ids)
        if actual_gamma == 0:
            raise RuntimeError("draft produced zero tokens")
        # Teacher verify writes actual_gamma; draft chain always wrote full γ.
        teacher_occupied_end = L + actual_gamma
        draft_occupied_end = (
            L + int(chain_runner.gamma)
            if used_chain and chain_runner is not None
            else L + actual_gamma
        )
        draft_t = torch.tensor(
            [draft_ids], device=sequences.device, dtype=sequences.dtype
        )

        # Teacher verify: graph when length matches captured K; else eager.
        if actual_gamma == int(session.tiger_verify.q_len):
            verify_logits = tiger_verify_forward_k(
                session.tiger_verify,
                draft_tokens=draft_t,
                prefix_len=L,
                rewind_to_prefix=False,
            )
        else:
            verify_logits = _eager_forward_tokens(
                teacher,
                cache=teacher_cache,
                enc_hidden=enc_hidden,
                tokens=draft_t,
                prefix_len=L,
            )
        window_teacher_verify += 1
        verify_steps += 1

        p_logits, p_probs, bonus_logits = _build_p_rows(
            teacher_last=teacher_last,
            verify_logits=verify_logits,
            draft_ids=draft_ids,
            sequences=sequences,
            processors=processors,
            greedy=False,
            temperature=spec_temperature,
            top_p=spec_top_p,
        )
        # For rejection, draft chain q_probs already applied temp/top_p; when
        # structural processors are on, rebuild q from processed draft logits
        # is not available for the chain path — keep chain q as-is (default
        # structural off). Eager path used processors above.
        n_accepted, residual = reject_sample_prefix(
            p_probs=p_probs[:actual_gamma],
            q_probs=q_probs[:actual_gamma],
            draft_token_ids=draft_ids,
            rng=rng,
        )
        n_keep = min(int(n_accepted), remaining)
        residual_tok: int | None = None
        bonus_tok: int | None = None
        if residual is not None and n_keep < remaining:
            residual_tok = int(residual)
        elif (
            residual is None
            and n_accepted == actual_gamma
            and n_keep == actual_gamma
            and n_keep < remaining
            and not any(t in eos_ids for t in draft_ids[:n_keep])
        ):
            bonus_prefix = torch.cat([sequences, draft_t], dim=-1)
            bonus_processed = _apply_processors(
                processors, bonus_prefix, bonus_logits.unsqueeze(0)
            ).squeeze(0)
            bonus_tok, _ = _sample_from_logits(
                bonus_processed,
                greedy=False,
                temperature=spec_temperature,
                top_p=spec_top_p,
                rng=rng,
            )

        extra: list[int] = []
        if residual_tok is not None:
            extra.append(residual_tok)
        elif bonus_tok is not None:
            extra.append(int(bonus_tok))
        commit = (list(draft_ids[:n_keep]) + extra)[:remaining]
        if not commit:
            raise RuntimeError("turbo speculative step committed zero tokens")
        accepted_per_cycle.append(len(commit))

        # Keep-accepted-KV: rewind rejected band; commit accepted without reforward.
        keep_len = L + n_keep
        rewind_self_cache(
            teacher_cache, keep_len, occupied_end=teacher_occupied_end
        )
        if used_chain and chain_runner is not None:
            rewind_self_cache(
                chain_runner.cache, keep_len, occupied_end=draft_occupied_end
            )
        else:
            draft_cache = session.draft_cache
            rewind_self_cache(draft_cache, keep_len, occupied_end=draft_occupied_end)
        window_keep_kv += 1

        if n_keep > 0:
            sequences = torch.cat(
                [
                    sequences,
                    torch.tensor(
                        [draft_ids[:n_keep]],
                        device=sequences.device,
                        dtype=sequences.dtype,
                    ),
                ],
                dim=-1,
            )

        if extra:
            sequences = torch.cat(
                [
                    sequences,
                    torch.tensor(
                        [extra], device=sequences.device, dtype=sequences.dtype
                    ),
                ],
                dim=-1,
            )
            teacher_step = _eager_forward_tokens(
                teacher,
                cache=teacher_cache,
                enc_hidden=enc_hidden,
                tokens=torch.tensor(
                    [[extra[0]]], device=sequences.device, dtype=torch.long
                ),
                prefix_len=keep_len,
            )
            teacher_last = teacher_step[-1].float().view(-1)
            window_teacher_extra_q1 += 1
            draft_cache = (
                chain_runner.cache
                if used_chain and chain_runner is not None
                else session.draft_cache
            )
            draft_step = _eager_forward_tokens(
                draft,
                cache=draft_cache,
                enc_hidden=enc_hidden,
                tokens=torch.tensor(
                    [[extra[0]]], device=sequences.device, dtype=torch.long
                ),
                prefix_len=keep_len,
            )
            draft_last = draft_step[-1].float().view(-1)
        else:
            if n_keep <= 0:
                raise RuntimeError("keep-KV commit with n_keep==0 and no residual")
            teacher_last = verify_logits[n_keep - 1].float().clone()
            draft_last = draft_logits_after[n_keep - 1].float().clone()

        accepted_total += len(commit)
        session.accepted_tokens_total = (
            getattr(session, "accepted_tokens_total", 0) + len(commit)
        )
        session.verify_steps = getattr(session, "verify_steps", 0) + 1
        session.draft_calls = getattr(session, "draft_calls", 0) + 1

        for tok in commit:
            if int(tok) in eos_ids:
                stopped = True
                break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    result = sequences.cpu()
    stats = _build_generation_stats(result, model_kwargs, pad_token_id, elapsed)
    e_acc = accepted_total / verify_steps if verify_steps else 0.0
    stats.update(
        {
            "precision": precision,
            "decoder_loop_backend": "turbo_on_tiger_speculative",
            "turbo_speculative": True,
            "turbo_base": "tiger14n_feat_compiled_decode",
            "turbo_verify_path": "tiger_cuda_graph_qk",
            "turbo_kv_commit_mode": "keep_accepted_o1_rewind",
            "turbo_gamma": int(gamma),
            "turbo_temperature": spec_temperature,
            "turbo_top_p": spec_top_p,
            "turbo_verify_steps": verify_steps,
            "turbo_draft_calls": draft_calls,
            "turbo_accepted_tokens": accepted_total,
            "turbo_accepted_per_verify": e_acc,
            "turbo_E_accepted_per_verify": e_acc,
            "turbo_E_mean_per_cycle": (
                float(statistics.mean(accepted_per_cycle)) if accepted_per_cycle else 0.0
            ),
            "turbo_E_median_per_cycle": (
                float(statistics.median(accepted_per_cycle))
                if accepted_per_cycle
                else 0.0
            ),
            "turbo_accepted_per_cycle": list(accepted_per_cycle),
            "turbo_structural_processors": structural_processors_enabled(),
            "turbo_prompt_tokens": prompt_len,
            "turbo_teacher_verify_forwards_window": window_teacher_verify,
            "turbo_teacher_extra_q1_window": window_teacher_extra_q1,
            "turbo_keep_accepted_o1_rewind_cycles_window": window_keep_kv,
            "turbo_draft_chain_cycles_window": window_draft_chain,
            "turbo_draft_eager_cycles_window": window_draft_eager,
            "turbo_draft_chain_enabled": bool(use_draft_chain),
            "turbo_path_hit_counters": {
                "tiger_verify_graph_replay": int(
                    session.tiger_verify.hit_counters.get("tiger_verify_graph_replay", 0)
                ),
                "tiger_verify_graph_capture": int(
                    session.tiger_verify.hit_counters.get("tiger_verify_graph_capture", 0)
                ),
                "keep_accepted_o1_rewind_cycles": int(window_keep_kv),
                "draft_chain_cycles_window": int(window_draft_chain),
                "draft_eager_cycles_window": int(window_draft_eager),
            },
            **(chain_runner.meta() if chain_runner is not None else {}),
        }
    )
    return result, stats


__all__ = [
    "speculative_generate_window",
    "structural_processors_enabled",
    "build_structural_processors",
]
