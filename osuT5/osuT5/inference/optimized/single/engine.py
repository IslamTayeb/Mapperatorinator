"""Accepted FP32 optimized single-song inference engine."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from transformers import (
    ClassifierFreeGuidanceLogitsProcessor,
    LogitsProcessorList,
    TemperatureLogitsWarper,
)

from ....event import ContextType, EventType
from ....runtime_profiling import generation_profile_context
from ...engine_binding import InferenceEngineBinding
from ...logit_processors import (
    ConditionalTemperatureLogitsWarper,
    LookbackBiasLogitsWarper,
    TimeshiftBias,
    get_beat_type_tokens,
    get_mania_type_tokens,
    get_scroll_speed_tokens,
)
from .config import ACCEPTED_OPTIMIZED_SINGLE_CONFIG, OptimizedSingleConfig
from .decode_loop import active_prefix_decode_generate
from .logits import MonotonicTimeShiftLogitsProcessor
from .state import ProductionDecodeSession


MILLISECONDS_PER_STEP = 10


def _prompt_token_counts(
    model_kwargs: dict[str, Any],
    pad_token_id: int | None,
) -> torch.Tensor | None:
    decoder_attention_mask = model_kwargs.get("decoder_attention_mask")
    if isinstance(decoder_attention_mask, torch.Tensor):
        return decoder_attention_mask.to(torch.long).sum(dim=-1).cpu()

    decoder_input_ids = model_kwargs.get("decoder_input_ids")
    if not isinstance(decoder_input_ids, torch.Tensor):
        return None
    if pad_token_id is None:
        return torch.full(
            (decoder_input_ids.shape[0],),
            decoder_input_ids.shape[1],
            dtype=torch.long,
        )
    return decoder_input_ids.ne(pad_token_id).to(torch.long).sum(dim=-1).cpu()


def _output_token_counts(
    result: torch.Tensor,
    pad_token_id: int | None,
) -> torch.Tensor:
    if pad_token_id is None:
        return torch.full(
            (result.shape[0],),
            result.shape[1],
            dtype=torch.long,
        )
    return result.ne(pad_token_id).to(torch.long).sum(dim=-1)


def _build_generation_stats(
    result: torch.Tensor,
    model_kwargs: dict[str, Any],
    pad_token_id: int | None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    prompt_token_counts = _prompt_token_counts(model_kwargs, pad_token_id)
    output_token_counts = _output_token_counts(result, pad_token_id).cpu()
    generated_token_counts = output_token_counts.clone()
    if prompt_token_counts is not None:
        generated_token_counts = torch.clamp(
            generated_token_counts - prompt_token_counts,
            min=0,
        )

    generated_tokens = int(generated_token_counts.sum().item())
    tokens_per_second = (
        generated_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
    )
    prompt_tokens = (
        int(prompt_token_counts.sum().item())
        if prompt_token_counts is not None
        else None
    )
    return {
        "batch_size": int(result.shape[0]),
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_per_sample": (
            prompt_token_counts.tolist() if prompt_token_counts is not None else None
        ),
        "output_tokens": int(output_token_counts.sum().item()),
        "output_tokens_per_sample": output_token_counts.tolist(),
        "generated_tokens": generated_tokens,
        "generated_tokens_per_sample": generated_token_counts.tolist(),
        "elapsed_seconds": float(elapsed_seconds),
        "tokens_per_second": tokens_per_second,
    }


def _eos_token_ids(
    tokenizer,
    *,
    lookback_time: float,
    lookahead_time: float,
    context_type: ContextType | None,
) -> list[int]:
    eos_token_ids = [tokenizer.eos_id]
    if context_type is not None and context_type in tokenizer.context_eos:
        eos_token_ids.append(tokenizer.context_eos[context_type])
    if lookback_time > 0:
        eos_token_ids.extend(
            range(
                tokenizer.event_start[EventType.TIME_SHIFT],
                tokenizer.event_start[EventType.TIME_SHIFT]
                + int(lookback_time / MILLISECONDS_PER_STEP),
            )
        )
    if lookahead_time > 0:
        eos_token_ids.extend(
            range(
                tokenizer.event_end[EventType.TIME_SHIFT]
                - int(lookahead_time / MILLISECONDS_PER_STEP),
                tokenizer.event_end[EventType.TIME_SHIFT],
            )
        )
    return eos_token_ids


def _build_logits_processor_list(
    tokenizer,
    *,
    cfg_scale: float,
    timeshift_bias: float,
    types_first: bool,
    temperature: float,
    timing_temperature: float,
    mania_column_temperature: float,
    taiko_hit_temperature: float,
    lookback_time: float,
    device,
) -> LogitsProcessorList:
    processors = LogitsProcessorList()
    if cfg_scale > 1.0:
        processors.append(ClassifierFreeGuidanceLogitsProcessor(cfg_scale))

    processors.append(
        MonotonicTimeShiftLogitsProcessor(
            tokenizer,
            stateful_batch1=True,
        )
    )
    if timeshift_bias != 0:
        processors.append(
            TimeshiftBias(
                timeshift_bias,
                tokenizer.event_start[EventType.TIME_SHIFT],
                tokenizer.event_end[EventType.TIME_SHIFT],
            )
        )
    if types_first:
        processors.append(
            ConditionalTemperatureLogitsWarper(
                temperature,
                timing_temperature,
                mania_column_temperature,
                taiko_hit_temperature,
                types_first,
                get_beat_type_tokens(tokenizer),
                get_mania_type_tokens(tokenizer),
                get_scroll_speed_tokens(tokenizer),
            )
        )
    else:
        processors.append(TemperatureLogitsWarper(temperature))
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


def _sync_cuda_for_model(model) -> None:
    if (
        torch.cuda.is_available()
        and getattr(getattr(model, "device", None), "type", None) == "cuda"
    ):
        torch.cuda.synchronize(model.device)


@torch.no_grad()
def _generate_window(
    model,
    tokenizer,
    model_kwargs: dict[str, Any],
    generate_kwargs: dict[str, Any],
    *,
    context_state: ProductionDecodeSession,
    config: OptimizedSingleConfig,
):
    model_kwargs = {
        key: value.to(model.device) if isinstance(value, torch.Tensor) else value
        for key, value in model_kwargs.items()
    }
    model_kwargs = {
        key: (
            value.to(model.dtype)
            if key != "inputs"
            and isinstance(value, torch.Tensor)
            and value.dtype == torch.float32
            else value
        )
        for key, value in model_kwargs.items()
    }
    batch_size = model_kwargs["inputs"].shape[0]

    precision = generate_kwargs.pop("precision", config.precision)
    cfg_scale = generate_kwargs.pop("cfg_scale", config.cfg_scale)
    timeshift_bias = generate_kwargs.pop("timeshift_bias", 0)
    types_first = generate_kwargs.pop("types_first", False)
    temperature = generate_kwargs.pop("temperature", 1.0)
    timing_temperature = generate_kwargs.pop("timing_temperature", temperature)
    mania_column_temperature = generate_kwargs.pop(
        "mania_column_temperature",
        temperature,
    )
    taiko_hit_temperature = generate_kwargs.pop(
        "taiko_hit_temperature",
        temperature,
    )
    lookback_time = generate_kwargs.pop("lookback_time", 0.0)
    lookahead_time = generate_kwargs.pop("lookahead_time", 0.0)
    context_type = generate_kwargs.pop("context_type", None)
    sync_model_timing = bool(generate_kwargs.pop("sync_model_timing", False))
    profile_model_generate_cuda_ledger = bool(
        generate_kwargs.pop("profile_model_generate_cuda_ledger", False)
    )
    profile_generation_detail_ranges = bool(
        generate_kwargs.pop("profile_generation_detail_ranges", False)
    )
    profile_active_prefix_decode_diagnostics = bool(
        generate_kwargs.pop("profile_active_prefix_decode_diagnostics", False)
    )
    profile_sdpa_backend = generate_kwargs.pop("profile_sdpa_backend", None)
    active_prefix_decode_loop = bool(
        generate_kwargs.pop(
            "active_prefix_decode_loop",
            config.active_prefix_decode_loop,
        )
    )
    active_prefix_decode_bucket_size = int(
        generate_kwargs.pop(
            "active_prefix_decode_bucket_size",
            config.active_prefix_decode_bucket_size,
        )
    )
    active_prefix_decode_cuda_graph = bool(
        generate_kwargs.pop(
            "active_prefix_decode_cuda_graph",
            config.active_prefix_decode_cuda_graph,
        )
    )
    active_prefix_decode_cuda_graph_warmup = int(
        generate_kwargs.pop(
            "active_prefix_decode_cuda_graph_warmup",
            config.active_prefix_decode_cuda_graph_warmup,
        )
    )
    active_prefix_decode_cuda_graph_min_decode_steps = int(
        generate_kwargs.pop(
            "active_prefix_decode_cuda_graph_min_decode_steps",
            config.active_prefix_decode_cuda_graph_min_decode_steps,
        )
    )
    stateful_monotonic_logits_processor = bool(
        generate_kwargs.pop(
            "stateful_monotonic_logits_processor",
            config.stateful_monotonic_logits_processor,
        )
    )
    q1_bmm_cross_attention = bool(
        generate_kwargs.pop(
            "q1_bmm_cross_attention",
            config.q1_bmm_cross_attention,
        )
    )
    native_q1_self_attention_requested = bool(
        generate_kwargs.pop(
            "native_q1_self_attention",
            config.native_q1_self_attention,
        )
    )
    native_q1_rope_cache_self_attention_requested = bool(
        generate_kwargs.pop(
            "native_q1_rope_cache_self_attention",
            config.native_q1_rope_cache_self_attention,
        )
    )
    decode_session_cuda_graph = bool(
        generate_kwargs.pop(
            "decode_session_cuda_graph",
            config.decode_session_cuda_graph,
        )
    )
    generate_kwargs.pop("decode_session_state", None)

    if context_type is not None:
        context_type = ContextType(context_type)
    if precision != config.precision or cfg_scale != config.cfg_scale:
        raise ValueError("optimized single runtime configuration changed after validation.")
    if batch_size != config.batch_size:
        raise ValueError("optimized single runtime requires batch_size=1.")
    if int(generate_kwargs.get("num_beams", 1)) != config.num_beams:
        raise ValueError("optimized single runtime requires num_beams=1.")
    if not stateful_monotonic_logits_processor:
        raise ValueError("optimized single runtime requires stateful monotonic processing.")
    if not active_prefix_decode_loop:
        raise ValueError("optimized single runtime requires active-prefix decode.")
    if not active_prefix_decode_cuda_graph or not decode_session_cuda_graph:
        raise ValueError("optimized single runtime requires persistent CUDA graph replay.")

    native_q1_self_attention = (
        native_q1_self_attention_requested
        and context_type != ContextType.TIMING
    )
    native_q1_rope_cache_self_attention = (
        native_q1_rope_cache_self_attention_requested
        and native_q1_self_attention
    )
    processors = _build_logits_processor_list(
        tokenizer,
        cfg_scale=cfg_scale,
        timeshift_bias=timeshift_bias,
        types_first=types_first,
        temperature=temperature,
        timing_temperature=timing_temperature,
        mania_column_temperature=mania_column_temperature,
        taiko_hit_temperature=taiko_hit_temperature,
        lookback_time=lookback_time,
        device=model.device,
    )
    cache = context_state.cache_for_window(
        model,
        batch_size=batch_size,
        num_beams=generate_kwargs.get("num_beams", 1),
        cfg_scale=cfg_scale,
    )
    pad_token_id = generate_kwargs.get(
        "pad_token_id",
        getattr(tokenizer, "pad_id", None),
    )
    active_prefix_decode_diagnostics = (
        {
            "enabled": True,
            "decode_steps": 0,
            "bucket_lengths_seen": [],
            "bucket_transition_count": 0,
        }
        if profile_active_prefix_decode_diagnostics
        else None
    )
    custom_generate = partial(
        active_prefix_decode_generate,
        active_prefix_bucket_size=active_prefix_decode_bucket_size,
        cuda_graph_forward=active_prefix_decode_cuda_graph,
        cuda_graph_warmup=active_prefix_decode_cuda_graph_warmup,
        cuda_graph_min_decode_steps=(
            active_prefix_decode_cuda_graph_min_decode_steps
        ),
        active_prefix_decode_diagnostics=active_prefix_decode_diagnostics,
        **context_state.active_prefix_decode_kwargs(),
    )

    generate_start_event = generate_end_event = None
    generate_cuda_event_seconds = None
    with torch.autocast(
        device_type=model.device.type,
        dtype=torch.bfloat16,
        enabled=precision == "amp",
    ), generation_profile_context(
        detail_ranges=profile_generation_detail_ranges,
        sdpa_backend=profile_sdpa_backend,
        q1_bmm_cross_attention=q1_bmm_cross_attention,
        native_q1_self_attention=native_q1_self_attention,
        native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
    ):
        if sync_model_timing:
            _sync_cuda_for_model(model)
            if (
                profile_model_generate_cuda_ledger
                and torch.cuda.is_available()
                and getattr(getattr(model, "device", None), "type", None) == "cuda"
            ):
                generate_start_event = torch.cuda.Event(enable_timing=True)
                generate_end_event = torch.cuda.Event(enable_timing=True)
        start_time = time.perf_counter()
        if generate_start_event is not None:
            generate_start_event.record()
        result = model.generate(
            **model_kwargs,
            **generate_kwargs,
            use_cache=True,
            past_key_values=cache,
            logits_processor=processors,
            eos_token_id=_eos_token_ids(
                tokenizer,
                lookback_time=lookback_time,
                lookahead_time=lookahead_time,
                context_type=context_type,
            ),
            custom_generate=custom_generate,
        )
        if generate_end_event is not None:
            generate_end_event.record()
        if sync_model_timing:
            _sync_cuda_for_model(model)
        elapsed_seconds = time.perf_counter() - start_time
        if generate_start_event is not None and generate_end_event is not None:
            generate_cuda_event_seconds = float(
                generate_start_event.elapsed_time(generate_end_event)
            ) / 1000.0

    result = result.cpu()
    stats = _build_generation_stats(
        result,
        model_kwargs,
        pad_token_id,
        elapsed_seconds,
    )
    stats.update(
        {
            "precision": precision,
            "context_type": (
                context_type.value if context_type is not None else None
            ),
            "num_beams": int(generate_kwargs.get("num_beams", 1)),
            "cfg_scale": float(cfg_scale),
            "do_sample": bool(generate_kwargs.get("do_sample", False)),
            "sync_model_timing": sync_model_timing,
            "profile_model_generate_cuda_ledger": (
                profile_model_generate_cuda_ledger
            ),
            "model_generate_cpu_elapsed_seconds": (
                elapsed_seconds if profile_model_generate_cuda_ledger else None
            ),
            "model_generate_cuda_event_seconds": generate_cuda_event_seconds,
            "model_generate_host_gap_seconds": (
                elapsed_seconds - generate_cuda_event_seconds
                if generate_cuda_event_seconds is not None
                else None
            ),
            "generation_compile_enabled": not bool(
                getattr(
                    getattr(model, "generation_config", None),
                    "disable_compile",
                    True,
                )
            ),
            "profile_generation_detail_ranges": profile_generation_detail_ranges,
            "profile_active_prefix_decode_diagnostics": (
                profile_active_prefix_decode_diagnostics
            ),
            "profile_sdpa_backend": profile_sdpa_backend,
            "stateful_monotonic_logits_processor": (
                stateful_monotonic_logits_processor
            ),
            "q1_bmm_cross_attention_enabled": q1_bmm_cross_attention,
            "native_q1_self_attention_requested": (
                native_q1_self_attention_requested
            ),
            "native_q1_self_attention_enabled": native_q1_self_attention,
            "native_q1_self_attention_disabled_reason": (
                "timing_context"
                if native_q1_self_attention_requested
                and not native_q1_self_attention
                else None
            ),
            "native_q1_rope_cache_self_attention_requested": (
                native_q1_rope_cache_self_attention_requested
            ),
            "native_q1_rope_cache_self_attention_enabled": (
                native_q1_rope_cache_self_attention
            ),
            "native_q1_rope_cache_self_attention_disabled_reason": (
                "timing_context"
                if native_q1_rope_cache_self_attention_requested
                and not native_q1_rope_cache_self_attention
                else None
            ),
            "decode_session_runtime_enabled": True,
            "decode_session_cuda_graph_enabled": decode_session_cuda_graph,
            "decode_session_graph_count": context_state.graph_count,
            "active_prefix_decode_loop_enabled": active_prefix_decode_loop,
            "active_prefix_decode_bucket_size": active_prefix_decode_bucket_size,
            "active_prefix_decode_cuda_graph_enabled": (
                active_prefix_decode_cuda_graph
            ),
            "active_prefix_decode_cuda_graph_warmup": (
                active_prefix_decode_cuda_graph_warmup
            ),
            "active_prefix_decode_cuda_graph_min_decode_steps": (
                active_prefix_decode_cuda_graph_min_decode_steps
            ),
            "optimized_effective_config_version": config.version,
            "optimized_runtime_owner": (
                "osuT5.osuT5.inference.optimized.single.engine"
            ),
            "optimized_result_class": config.result_class,
        }
    )
    if active_prefix_decode_diagnostics is not None:
        stats["active_prefix_decode_diagnostics"] = (
            active_prefix_decode_diagnostics
        )
    return result, stats


@dataclass(frozen=True, slots=True)
class OptimizedSingleRuntime:
    config: OptimizedSingleConfig = ACCEPTED_OPTIMIZED_SINGLE_CONFIG

    def new_context_state(self) -> ProductionDecodeSession:
        return ProductionDecodeSession()

    def generate_window(
        self,
        *,
        model,
        tokenizer,
        model_kwargs: dict[str, Any],
        generate_kwargs: dict[str, Any],
        context_state: ProductionDecodeSession,
    ):
        if not isinstance(context_state, ProductionDecodeSession):
            raise TypeError(
                "optimized single runtime requires ProductionDecodeSession."
            )
        effective_generate_kwargs = dict(generate_kwargs)
        effective_generate_kwargs.update(self.config.generation_overrides())
        return _generate_window(
            model,
            tokenizer,
            model_kwargs,
            effective_generate_kwargs,
            context_state=context_state,
            config=self.config,
        )

    def profile_metadata(self) -> dict[str, Any]:
        return {
            "optimized_effective_config_version": self.config.version,
            "optimized_effective_config": self.config.metadata(),
            "optimized_runtime_owner": (
                "osuT5.osuT5.inference.optimized.single.engine"
            ),
            "optimized_result_class": self.config.result_class,
        }


def load_optimized_single_engine(
    *,
    model_loader,
    loader_kwargs: dict[str, Any],
):
    config = ACCEPTED_OPTIMIZED_SINGLE_CONFIG
    if loader_kwargs.get("precision") != config.precision:
        raise ValueError("optimized single loader requires precision=fp32.")
    if loader_kwargs.get("attn_implementation") != config.attn_implementation:
        raise ValueError("optimized single loader requires attn_implementation=sdpa.")
    if loader_kwargs.get("use_server"):
        raise ValueError("optimized single loader requires use_server=false.")

    effective_loader_kwargs = dict(loader_kwargs)
    effective_loader_kwargs.update(
        {
            "precision": config.precision,
            "attn_implementation": config.attn_implementation,
            "generation_compile": config.generation_compile,
            "use_server": False,
        }
    )
    raw_model, tokenizer = model_loader(**effective_loader_kwargs)
    if bool(
        getattr(
            getattr(raw_model, "generation_config", None),
            "disable_compile",
            True,
        )
    ):
        raise RuntimeError(
            "optimized single loader requested generation compile, but the raw "
            "model reports compilation disabled."
        )
    return (
        InferenceEngineBinding(
            raw_model=raw_model,
            runtime=OptimizedSingleRuntime(config=config),
        ),
        tokenizer,
    )
