"""Draft-K + batched teacher verify speculative decode for turbo.

Owns the Leviathan loop only. Does not touch optimized bit-exact paths.
Distribution-equivalent under rejection sampling; TIER1 required before ship.
"""
from __future__ import annotations

import os
import time
from contextlib import nullcontext
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
from .draft_chain_graph import (
    DraftChainGraphRunner,
    draft_chain_graph_enabled,
)
from .kv_rollback import (
    align_kwargs_after_rewind,
    reset_self_cache,
    rewind_self_cache,
)
from .rejection import apply_temp_top_p, reject_sample_prefix
from .verify_fastpath import (
    TeacherVerifyFastpath,
    allocate_teacher_static_cache,
    build_teacher_verify_fastpath,
    teacher_aligned_runtime_context,
    verify_fastpath_enabled,
)


def get_turbo_cache(*, cfg_scale: float = 1.0) -> MapperatorinatorCache:
    """Dynamic KV cache so EncoderDecoderCache.crop works after reject."""
    if cfg_scale != 1.0:
        raise ValueError("turbo speculative decode requires cfg_scale=1.0")
    return MapperatorinatorCache(DynamicCache(), DynamicCache(), cfg_scale)


def get_teacher_cache(teacher, *, cfg_scale: float = 1.0) -> MapperatorinatorCache:
    """Teacher cache: StaticCache when §41 verify fastpath is on, else Dynamic."""
    if verify_fastpath_enabled():
        return allocate_teacher_static_cache(teacher, cfg_scale=cfg_scale)
    return get_turbo_cache(cfg_scale=cfg_scale)


def crop_self_cache(cache: MapperatorinatorCache, length: int) -> None:
    """Crop teacher/draft self KV after reject (canary / full wipe from length).

    DynamicCache uses HF crop. StaticCache (§41 teacher) has no working
    EncoderDecoderCache.crop path — zero trailing slots so get_seq_length
    (nonzero occupancy) returns ``length``. Sampled keep-KV prefers
    :func:`rewind_self_cache` (O(γ) band zero) instead.
    """
    length = int(length)
    self_cache = cache.self_attention_cache
    if isinstance(self_cache, DynamicCache):
        cache.crop(length)
        return
    from transformers.cache_utils import StaticCache

    if not isinstance(self_cache, StaticCache):
        raise TypeError(
            f"unsupported self cache for crop: {type(self_cache).__name__}"
        )
    for layer in self_cache.layers:
        if not getattr(layer, "is_initialized", False):
            continue
        if length < layer.keys.shape[2]:
            layer.keys[:, :, length:, :].zero_()
            layer.values[:, :, length:, :].zero_()
        if hasattr(layer, "cumulative_length"):
            layer.cumulative_length = length


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


class _PhaseTimer:
    """Accumulate draft / verify / rebuild / host gaps (CUDA-synced when armed)."""

    __slots__ = (
        "sync",
        "device",
        "draft_s",
        "verify_s",
        "rebuild_s",
        "host_s",
        "steps",
        "_mark",
    )

    def __init__(self, *, sync: bool, device: torch.device | None) -> None:
        self.sync = bool(sync) and device is not None and device.type == "cuda"
        self.device = device
        self.draft_s = 0.0
        self.verify_s = 0.0
        self.rebuild_s = 0.0
        self.host_s = 0.0
        self.steps = 0
        self._mark = time.perf_counter()

    def _sync(self) -> None:
        if self.sync:
            torch.cuda.synchronize(self.device)

    def begin(self) -> None:
        self._sync()
        self._mark = time.perf_counter()

    def add(self, bucket: str) -> None:
        self._sync()
        now = time.perf_counter()
        dt = now - self._mark
        if bucket == "draft":
            self.draft_s += dt
        elif bucket == "verify":
            self.verify_s += dt
        elif bucket == "rebuild":
            self.rebuild_s += dt
        elif bucket == "host":
            self.host_s += dt
        self._mark = now

    def as_stats(self, *, accepted_total: int, verify_steps: int) -> dict[str, Any]:
        total = self.draft_s + self.verify_s + self.rebuild_s + self.host_s
        return {
            "turbo_profile_steps": int(self.steps),
            "turbo_draft_ms_total": 1000.0 * self.draft_s,
            "turbo_verify_ms_total": 1000.0 * self.verify_s,
            "turbo_rebuild_ms_total": 1000.0 * self.rebuild_s,
            "turbo_host_gap_ms_total": 1000.0 * self.host_s,
            "turbo_draft_ms_per_verify": (
                1000.0 * self.draft_s / verify_steps if verify_steps else 0.0
            ),
            "turbo_verify_ms_per_verify": (
                1000.0 * self.verify_s / verify_steps if verify_steps else 0.0
            ),
            "turbo_rebuild_ms_per_verify": (
                1000.0 * self.rebuild_s / verify_steps if verify_steps else 0.0
            ),
            "turbo_host_gap_ms_per_verify": (
                1000.0 * self.host_s / verify_steps if verify_steps else 0.0
            ),
            "turbo_phase_ms_per_token": (
                1000.0 * total / accepted_total if accepted_total else 0.0
            ),
        }


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
    # Always clone: MonotonicTimeShift mutates scores in-place; sharing storage
    # with cached teacher logits corrupts later verify / bonus steps.
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
    """Return (token_id, probs[V]) under turbo sampling distribution."""
    if greedy or temperature <= 1e-5:
        probs = torch.zeros_like(logits_1d, dtype=torch.float32)
        tok = int(torch.argmax(logits_1d).item())
        probs[tok] = 1.0
        return tok, probs
    probs = apply_temp_top_p(logits_1d.unsqueeze(0), temperature, top_p).squeeze(0)
    tok = int(torch.multinomial(probs, 1, generator=rng).item())
    return tok, probs


def _stash_encoder_outputs(outputs: Any, model_kwargs: dict[str, Any]) -> dict[str, Any]:
    from transformers.modeling_outputs import BaseModelOutput

    if model_kwargs.get("encoder_outputs") is not None:
        return model_kwargs
    enc = getattr(outputs, "encoder_last_hidden_state", None)
    if enc is None:
        return model_kwargs
    model_kwargs["encoder_outputs"] = BaseModelOutput(last_hidden_state=enc)
    return model_kwargs


def _forward_decoder(
    model,
    *,
    decoder_input_ids: torch.LongTensor,
    model_kwargs: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Cached forward; returns (outputs, updated model_kwargs)."""
    # Drop stale 1-token cache_position from the previous HF update helper so
    # multi-token verify/draft commits recompute arange(past, past+n).
    model_kwargs.pop("cache_position", None)
    _align_decoder_mask(
        model_kwargs,
        length=int(decoder_input_ids.shape[1]),
        device=decoder_input_ids.device,
    )
    model_inputs = model.prepare_inputs_for_generation(
        decoder_input_ids,
        **model_kwargs,
    )
    # HF _update_model_kwargs_for_generation expects cache_position on kwargs.
    if "cache_position" in model_inputs:
        model_kwargs["cache_position"] = model_inputs["cache_position"]
    outputs = model(**model_inputs, return_dict=True)
    model_kwargs = model._update_model_kwargs_for_generation(
        outputs,
        model_kwargs,
        is_encoder_decoder=True,
    )
    model_kwargs = _stash_encoder_outputs(outputs, model_kwargs)
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
) -> tuple[list[int], torch.Tensor, dict[str, Any], bool, torch.Tensor, list[torch.Tensor]]:
    """Autoregressive draft up to γ tokens.

    Returns draft_ids, q_probs[γ,V], updated kwargs, stopped_on_eos,
    logits after the last drafted token, and raw logits-after each drafted
    token (length == len(draft_ids); EOS-stopped last entry may equal the
    pre-forward logits used to sample that EOS).
    """
    ids = prompt_ids
    probs_q: list[torch.Tensor] = []
    draft_ids: list[int] = []
    logits_after: list[torch.Tensor] = []
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
            logits_after.append(cur_logits)
            break
        outputs, mk = _forward_decoder(draft, decoder_input_ids=ids, model_kwargs=mk)
        cur_logits = outputs.logits[:, -1, :].float().squeeze(0)
        logits_after.append(cur_logits)

    if not draft_ids:
        raise RuntimeError("draft produced zero tokens")
    return (
        draft_ids,
        torch.stack(probs_q, dim=0),
        mk,
        stopped,
        cur_logits,
        logits_after,
    )


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
    verify_fastpath: TeacherVerifyFastpath | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], int]:
    """Teacher verify over drafted ids.

    Returns p_logits[γ,V], p_probs[γ,V], bonus_logits[V],
    raw_after[γ,V] (logits after each draft token), updated kwargs,
    teacher_forward_count for this verify.

    §41: greedy + graph_aligned → sequential Q=1 (matches optimized cuda-graph
    decode for TIER1a). Otherwise one multi-token K-forward for c_verify speed.
    """
    gamma = len(draft_ids)
    draft_t = torch.tensor([list(draft_ids)], device=prompt_ids.device, dtype=prompt_ids.dtype)
    ids = torch.cat([prompt_ids, draft_t], dim=-1)
    use_aligned_q1 = (
        verify_fastpath is not None
        and verify_fastpath.graph_aligned
        and (greedy or temperature <= 1e-5)
    )
    teacher_fwd = 0
    if use_aligned_q1:
        # Sequential Q=1: logits after each draft token.
        # p_raw[0] is still last_logits (pre-draft); verify rows shift like HF.
        step_logits, _bonus_unused, mk = verify_fastpath.verify_sequential_q1(
            prompt_ids=prompt_ids,
            draft_ids=draft_ids,
            model_kwargs=model_kwargs,
        )
        # step_logits[i] = teacher logits AFTER consuming draft_ids[i]
        # Same layout as multi-token verify_logits[:, -gamma:].
        verify_logits = step_logits
        teacher_fwd = int(gamma)
    elif verify_fastpath is not None:
        outputs, mk = verify_fastpath.forward_k(
            decoder_input_ids=ids,
            model_kwargs=model_kwargs,
            k=gamma,
        )
        verify_logits = outputs.logits[:, -gamma:, :].float().squeeze(0)
        teacher_fwd = 1
    else:
        outputs, mk = _forward_decoder(
            teacher,
            decoder_input_ids=ids,
            model_kwargs=model_kwargs,
        )
        verify_logits = outputs.logits[:, -gamma:, :].float().squeeze(0)
        teacher_fwd = 1
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
        verify_logits,
        mk,
        teacher_fwd,
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

    # Persist StaticCache + verify_fp across windows (session); never rebuild
    # graph/bucket caches per window (§47/§52).
    if getattr(session, "teacher_cache", None) is None:
        session.teacher_cache = get_teacher_cache(teacher, cfg_scale=1.0)
    else:
        reset_self_cache(session.teacher_cache)
    if getattr(session, "draft_cache", None) is None:
        session.draft_cache = get_turbo_cache(cfg_scale=1.0)
    else:
        reset_self_cache(session.draft_cache)
    teacher_cache = session.teacher_cache
    draft_cache = session.draft_cache
    teacher_mk["past_key_values"] = teacher_cache
    teacher_mk["use_cache"] = True
    if getattr(session, "verify_fp", None) is None:
        session.verify_fp = build_teacher_verify_fastpath(teacher)
    verify_fp = session.verify_fp

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

    # One outer context — @contextmanager CMs are one-shot and cannot be reused
    # per verify/rebuild step (AttributeError: args on second enter).
    precision_key = "fp16" if precision == "fp16" else "fp32"
    # Greedy + aligned teacher = TIER1a canary mode: crop-to-L + Q=1 rebuild.
    # Sampled turbo uses keep-accepted-KV O(1) rewind (DOCUMENTED DRIFT §34/§47).
    aligned_greedy = bool(
        verify_fp is not None and verify_fp.graph_aligned and greedy
    )
    canary_kv_mode = bool(aligned_greedy)
    kv_commit_mode = (
        "crop_to_L_full_rebuild" if canary_kv_mode else "keep_accepted_o1_rewind"
    )
    # §49/§52: graphed draft chain on sampled path only (canary stays eager).
    use_draft_chain = bool(
        draft_chain_graph_enabled()
        and (not canary_kv_mode)
        and (not greedy)
        and prompt_ids.device.type == "cuda"
    )
    draft_dtype = torch.float16 if precision_key == "fp16" else torch.float32
    chain_runner: DraftChainGraphRunner | None = None
    if use_draft_chain:
        chain_runner = getattr(session, "draft_chain_runner", None)
        if chain_runner is None or chain_runner.model is not draft:
            chain_runner = DraftChainGraphRunner(
                draft,
                dtype=draft_dtype,
                gamma=int(gamma),
                temperature=float(spec_temperature),
            )
            session.draft_chain_runner = chain_runner
    teacher_runtime = (
        teacher_aligned_runtime_context(precision=precision_key)
        if verify_fp is not None
        else nullcontext()
    )

    window_teacher_forwards = 0
    window_teacher_verify_forwards = 0
    window_teacher_extra_q1 = 0
    window_teacher_accepted_reforwards = 0
    window_draft_accepted_reforwards = 0
    window_keep_kv_rewinds = 0
    window_draft_chain_cycles = 0
    window_draft_eager_cycles = 0

    with teacher_runtime:
        teacher_last, teacher_mk = _prefill(
            teacher, prompt_ids=prompt_ids, model_kwargs=teacher_mk
        )
        window_teacher_forwards += 1
        # Drop raw audio after prefill so Q=1 CUDA graphs never re-enter the encoder
        # (capture-illegal host sync / dtype branches — see microbench 50147547).
        if teacher_mk.get("encoder_outputs") is not None:
            for _audio_key in ("frames", "inputs", "input_features"):
                teacher_mk.pop(_audio_key, None)
            draft_mk["encoder_outputs"] = teacher_mk["encoder_outputs"]
        if use_draft_chain and chain_runner is not None:
            chain_bind_mk = {
                key: value
                for key, value in draft_mk.items()
                if key != "past_key_values"
            }
            chain_runner.bind(model_kwargs=chain_bind_mk, prompt_ids=prompt_ids)
            draft_last = chain_runner.prefill()
        else:
            draft_last, draft_mk = _prefill(
                draft, prompt_ids=prompt_ids, model_kwargs=draft_mk
            )
            if draft_mk.get("encoder_outputs") is not None:
                for _audio_key in ("frames", "inputs", "input_features"):
                    draft_mk.pop(_audio_key, None)

        sequences = prompt_ids
        prompt_len = int(prompt_ids.shape[1])
        accepted_total = 0
        verify_steps = 0
        draft_calls = 0
        stopped = False
        # Phase profile when outer sync timing is on (profiler) or env opt-in.
        step_profile = sync_model_timing or (
            os.environ.get("MAPPERATORINATOR_TURBO_STEP_PROFILE", "").strip().lower()
            in {"1", "true", "on", "yes"}
        )
        timer = _PhaseTimer(sync=step_profile, device=teacher.device)

        while sequences.shape[1] < max_length and not stopped:
            remaining = max_length - int(sequences.shape[1])
            L = int(sequences.shape[1])
            this_gamma = min(int(gamma), remaining)

            timer.begin()
            used_chain_this_cycle = bool(
                use_draft_chain
                and chain_runner is not None
                and this_gamma == int(chain_runner.gamma)
            )
            if used_chain_this_cycle:
                (
                    draft_ids,
                    q_probs,
                    draft_logits_after,
                    _draft_tail_logits,
                ) = chain_runner.replay_chain(seed_logits=draft_last)
                # Truncate on EOS (chain does not early-stop in-graph).
                trimmed: list[int] = []
                trimmed_logits: list[torch.Tensor] = []
                for i, tok in enumerate(draft_ids):
                    trimmed.append(int(tok))
                    trimmed_logits.append(draft_logits_after[i])
                    if int(tok) in eos_ids:
                        break
                draft_ids = trimmed
                draft_logits_after = trimmed_logits
                q_probs = q_probs[: len(draft_ids)]
                _draft_hit_eos = bool(draft_ids and int(draft_ids[-1]) in eos_ids)
                window_draft_chain_cycles += 1
            else:
                if use_draft_chain and chain_runner is not None:
                    chain_runner.stats.hit_counters["draft_chain_eager_fallback"] += 1
                    # Resync eager DynamicCache from current sequences once.
                    reset_self_cache(draft_cache)
                    draft_mk["past_key_values"] = draft_cache
                    draft_last, draft_mk = _prefill(
                        draft, prompt_ids=sequences, model_kwargs=draft_mk
                    )
                    if draft_mk.get("encoder_outputs") is not None:
                        for _audio_key in ("frames", "inputs", "input_features"):
                            draft_mk.pop(_audio_key, None)
                (
                    draft_ids,
                    q_probs,
                    draft_mk,
                    _draft_hit_eos,
                    _draft_tail_logits,
                    draft_logits_after,
                ) = _draft_k_tokens(
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
                window_draft_eager_cycles += 1
            timer.add("draft")
            draft_calls += 1
            actual_gamma = len(draft_ids)
            occupied_end = L + actual_gamma

            (
                p_logits,
                p_probs,
                bonus_logits,
                teacher_raw_after,
                teacher_mk,
                verify_fwd,
            ) = _verify_draft_tokens(
                teacher,
                prompt_ids=sequences,
                draft_ids=draft_ids,
                last_logits=teacher_last,
                model_kwargs=teacher_mk,
                processors=processors,
                greedy=greedy,
                temperature=spec_temperature,
                top_p=spec_top_p,
                verify_fastpath=verify_fp,
            )
            timer.add("verify")
            verify_steps += 1
            window_teacher_forwards += int(verify_fwd)
            window_teacher_verify_forwards += int(verify_fwd)

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

            n_keep = min(n_accepted, remaining)
            bonus_tok: int | None = None
            residual_tok: int | None = None
            if residual is not None and n_keep < remaining:
                residual_tok = int(residual)
            elif (
                residual is None
                and n_accepted == actual_gamma
                and n_keep == actual_gamma
                and n_keep < remaining
                and not any(int(t) in eos_ids for t in draft_ids[:n_keep])
            ):
                bonus_prefix = torch.cat(
                    [
                        sequences,
                        torch.tensor(
                            [draft_ids],
                            device=sequences.device,
                            dtype=sequences.dtype,
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

            extra: list[int] = []
            if residual_tok is not None:
                extra.append(residual_tok)
            elif bonus_tok is not None:
                extra.append(int(bonus_tok))

            commit = list(draft_ids[:n_keep]) + extra
            commit = commit[:remaining]
            if not commit:
                raise RuntimeError("turbo speculative step committed zero tokens")

            timer.add("host")
            if canary_kv_mode:
                # TIER1a canary: crop-to-L + full replay (aligned Q=1).
                crop_self_cache(teacher_cache, L)
                crop_self_cache(draft_cache, L)
                _align_decoder_mask(teacher_mk, length=L, device=sequences.device)
                _align_decoder_mask(draft_mk, length=L, device=sequences.device)

                commit_t = torch.tensor(
                    [commit], device=sequences.device, dtype=sequences.dtype
                )
                sequences = torch.cat([sequences, commit_t], dim=-1)
                _align_decoder_mask(
                    teacher_mk, length=int(sequences.shape[1]), device=sequences.device
                )
                _align_decoder_mask(
                    draft_mk, length=int(sequences.shape[1]), device=sequences.device
                )
                if verify_fp is not None:
                    teacher_last_local = teacher_last
                    mk_rebuild = teacher_mk
                    ids_rebuild = sequences[:, :L]
                    for tok in commit:
                        ids_rebuild = torch.cat(
                            [
                                ids_rebuild,
                                torch.tensor(
                                    [[int(tok)]],
                                    device=sequences.device,
                                    dtype=sequences.dtype,
                                ),
                            ],
                            dim=-1,
                        )
                        teacher_out, mk_rebuild = verify_fp.forward_q1(
                            decoder_input_ids=ids_rebuild,
                            model_kwargs=mk_rebuild,
                        )
                        teacher_last_local = (
                            teacher_out.logits[:, -1, :].float().squeeze(0).clone()
                        )
                        window_teacher_forwards += 1
                        window_teacher_accepted_reforwards += 1
                    teacher_mk = mk_rebuild
                    teacher_last = teacher_last_local
                else:
                    teacher_out, teacher_mk = _forward_decoder(
                        teacher, decoder_input_ids=sequences, model_kwargs=teacher_mk
                    )
                    teacher_last = (
                        teacher_out.logits[:, -1, :].float().squeeze(0).clone()
                    )
                    window_teacher_forwards += 1
                    window_teacher_accepted_reforwards += len(commit)
                draft_out, draft_mk = _forward_decoder(
                    draft, decoder_input_ids=sequences, model_kwargs=draft_mk
                )
                draft_last = draft_out.logits[:, -1, :].float().squeeze(0).clone()
                window_draft_accepted_reforwards += len(commit)
                timer.add("rebuild")
            else:
                # Sampled keep-accepted-KV: O(1) rewind; never re-forward accepted.
                keep_len = L + n_keep
                rewind_self_cache(
                    teacher_cache, keep_len, occupied_end=occupied_end
                )
                if used_chain_this_cycle and chain_runner is not None:
                    chain_runner.rewind_keep_kv(keep_len)
                else:
                    rewind_self_cache(
                        draft_cache, keep_len, occupied_end=occupied_end
                    )
                window_keep_kv_rewinds += 1
                if n_keep > 0:
                    keep_t = torch.tensor(
                        [draft_ids[:n_keep]],
                        device=sequences.device,
                        dtype=sequences.dtype,
                    )
                    sequences = torch.cat([sequences, keep_t], dim=-1)
                align_kwargs_after_rewind(
                    teacher_mk, length=keep_len, device=sequences.device
                )
                if not (used_chain_this_cycle and chain_runner is not None):
                    align_kwargs_after_rewind(
                        draft_mk, length=keep_len, device=sequences.device
                    )

                if extra:
                    extra_t = torch.tensor(
                        [extra], device=sequences.device, dtype=sequences.dtype
                    )
                    sequences = torch.cat([sequences, extra_t], dim=-1)
                    # q1 bucket from rolled-back length (+ appended extra).
                    if verify_fp is not None:
                        teacher_out, teacher_mk = verify_fp.forward_q1(
                            decoder_input_ids=sequences,
                            model_kwargs=teacher_mk,
                        )
                    else:
                        teacher_out, teacher_mk = _forward_decoder(
                            teacher,
                            decoder_input_ids=sequences,
                            model_kwargs=teacher_mk,
                        )
                    teacher_last = (
                        teacher_out.logits[:, -1, :].float().squeeze(0).clone()
                    )
                    window_teacher_forwards += 1
                    window_teacher_extra_q1 += 1
                    if used_chain_this_cycle and chain_runner is not None:
                        draft_last = chain_runner.fastpath.decode_token(int(extra[0]))
                        chain_runner._seed_logits.copy_(draft_last.view(1, -1))
                    else:
                        draft_out, draft_mk = _forward_decoder(
                            draft, decoder_input_ids=sequences, model_kwargs=draft_mk
                        )
                        draft_last = (
                            draft_out.logits[:, -1, :].float().squeeze(0).clone()
                        )
                else:
                    if n_keep <= 0:
                        raise RuntimeError(
                            "keep-KV commit with n_keep==0 and no residual"
                        )
                    teacher_last = teacher_raw_after[n_keep - 1].float().clone()
                    draft_last = draft_logits_after[n_keep - 1].float().clone()
                    if used_chain_this_cycle and chain_runner is not None:
                        chain_runner._seed_logits.copy_(draft_last.view(1, -1))
                timer.add("rebuild")
            timer.steps += 1

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

    session.teacher_forwards += window_teacher_forwards
    session.teacher_verify_forwards += window_teacher_verify_forwards
    session.teacher_extra_q1_forwards += window_teacher_extra_q1
    session.teacher_accepted_reforwards += window_teacher_accepted_reforwards
    session.draft_accepted_reforwards += window_draft_accepted_reforwards

    # §47 gate "teacher forwards/cycle == 1": Leviathan verify forwards only
    # (sampled K-path ⇒ 1). Accepted reforward must be 0. Residual/bonus Q=1 is
    # tracked separately and is not an accepted-token rebuild.
    teacher_forwards_per_cycle = (
        float(window_teacher_verify_forwards) / float(verify_steps)
        if verify_steps
        else 0.0
    )
    teacher_extra_q1_per_cycle = (
        float(window_teacher_extra_q1) / float(verify_steps) if verify_steps else 0.0
    )

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
            "turbo_verify_fastpath": verify_fp is not None,
            "turbo_teacher_aligned_greedy": aligned_greedy,
            "turbo_kv_commit_mode": kv_commit_mode,
            "turbo_canary_kv_mode": canary_kv_mode,
            "turbo_teacher_forwards_window": window_teacher_forwards,
            "turbo_teacher_verify_forwards_window": window_teacher_verify_forwards,
            "turbo_teacher_extra_q1_window": window_teacher_extra_q1,
            "turbo_teacher_accepted_reforwards_window": (
                window_teacher_accepted_reforwards
            ),
            "turbo_draft_accepted_reforwards_window": (
                window_draft_accepted_reforwards
            ),
            "turbo_teacher_forwards_per_cycle": teacher_forwards_per_cycle,
            "turbo_teacher_extra_q1_per_cycle": teacher_extra_q1_per_cycle,
            "turbo_persistent_verify_fp": True,
            "turbo_draft_chain_enabled": bool(use_draft_chain),
            "turbo_draft_chain_cycles_window": window_draft_chain_cycles,
            "turbo_draft_eager_cycles_window": window_draft_eager_cycles,
            "turbo_keep_accepted_o1_rewind_cycles_window": window_keep_kv_rewinds,
            "turbo_verify_graph_entries": (
                len(verify_fp.graph_cache) if verify_fp is not None else 0
            ),
            **timer.as_stats(
                accepted_total=accepted_total, verify_steps=verify_steps
            ),
            **(verify_fp.profile_metadata() if verify_fp is not None else {}),
            **(
                chain_runner.profile_metadata()
                if chain_runner is not None
                else {"turbo_draft_chain_graph": False}
            ),
        }
    )
    # Flat path-hit counters for §52 scout harvest (also nested above).
    verify_meta = verify_fp.profile_metadata() if verify_fp is not None else {}
    chain_hits = (
        dict(chain_runner.stats.hit_counters) if chain_runner is not None else {}
    )
    stats["turbo_path_hit_counters"] = {
        "draft_chain_graph_replay": int(chain_hits.get("draft_chain_graph_replay", 0)),
        "draft_chain_graph_capture": int(chain_hits.get("draft_chain_graph_capture", 0)),
        "draft_chain_eager_fallback": int(
            chain_hits.get("draft_chain_eager_fallback", 0)
        ),
        "verify_graph_native_replays": int(
            verify_meta.get("turbo_verify_graph_native_replays", 0) or 0
        ),
        "verify_graph_replays": int(
            verify_meta.get("turbo_verify_graph_replays", 0) or 0
        ),
        "verify_graph_captures": int(
            verify_meta.get("turbo_verify_graph_captures", 0) or 0
        ),
        "verify_prepare_inputs_calls": int(
            verify_meta.get("turbo_verify_prepare_inputs_calls", 0) or 0
        ),
        "keep_accepted_o1_rewind_cycles": int(window_keep_kv_rewinds),
        "crop_to_L_full_rebuild": int(1 if canary_kv_mode else 0),
        "draft_chain_cycles_window": int(window_draft_chain_cycles),
        "draft_eager_cycles_window": int(window_draft_eager_cycles),
    }
    return result, stats
