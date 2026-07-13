"""Accepted FP32 optimized single-song inference engine."""

from __future__ import annotations

import time
from functools import partial
from typing import Any

import torch

from ....event import ContextType
from ....runtime_profiling import generation_profile_context
from ...engine_binding import InferenceEngineBinding
from ...generation_utils import (
    build_generation_stats,
    eos_token_ids,
    sync_cuda_for_model,
)
from .decode_loop import active_prefix_decode_generate
from .logits import build_single_logits_processor_list
from .state import ProductionDecodeSession


OPTIMIZED_CONFIG_VERSION = "accepted-fp32-270.475-v1"
OPTIMIZED_RESULT_CLASS = "exact-output"
OPTIMIZED_PRECISION = "fp32"
OPTIMIZED_ATTN_IMPLEMENTATION = "sdpa"
ACTIVE_PREFIX_BUCKET_SIZE = 64


def _optimized_config_metadata() -> dict[str, Any]:
    """Describe the immutable accepted engine preset for profiles."""

    return {
        "version": OPTIMIZED_CONFIG_VERSION,
        "result_class": OPTIMIZED_RESULT_CLASS,
        "precision": OPTIMIZED_PRECISION,
        "attn_implementation": OPTIMIZED_ATTN_IMPLEMENTATION,
        "decoder_loop_backend": "active_prefix_cuda_graph",
        "torch_compile_enabled": False,
        "batch_size": 1,
        "cfg_scale": 1.0,
        "num_beams": 1,
        "active_prefix_decode_loop": True,
        "active_prefix_decode_bucket_size": ACTIVE_PREFIX_BUCKET_SIZE,
        "active_prefix_decode_cuda_graph": True,
        "active_prefix_decode_cuda_graph_warmup": 0,
        "active_prefix_decode_cuda_graph_min_decode_steps": 1,
        "stateful_monotonic_logits_processor": True,
        "q1_bmm_cross_attention": True,
        "decode_session_runtime": True,
        "decode_session_cuda_graph": True,
        "native_decode_kernels": True,
        "native_q1_self_attention": True,
        "native_q1_rope_cache_self_attention": True,
    }


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
):
    return build_single_logits_processor_list(
        tokenizer,
        cfg_scale=cfg_scale,
        timeshift_bias=timeshift_bias,
        types_first=types_first,
        temperature=temperature,
        timing_temperature=timing_temperature,
        mania_column_temperature=mania_column_temperature,
        taiko_hit_temperature=taiko_hit_temperature,
        lookback_time=lookback_time,
        device=device,
        stateful_monotonic=True,
    )


@torch.no_grad()
def _generate_window(
    model,
    tokenizer,
    model_kwargs: dict[str, Any],
    generate_kwargs: dict[str, Any],
    *,
    context_state: ProductionDecodeSession,
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

    precision = generate_kwargs.pop("precision", OPTIMIZED_PRECISION)
    cfg_scale = generate_kwargs.pop("cfg_scale", 1.0)
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
    if context_type is not None:
        context_type = ContextType(context_type)
    if precision != OPTIMIZED_PRECISION or cfg_scale != 1.0:
        raise ValueError("optimized single runtime configuration changed after validation.")
    if batch_size != 1:
        raise ValueError("optimized single runtime requires batch_size=1.")
    if int(generate_kwargs.get("num_beams", 1)) != 1:
        raise ValueError("optimized single runtime requires num_beams=1.")

    native_q1_self_attention = context_type != ContextType.TIMING
    native_q1_rope_cache_self_attention = native_q1_self_attention
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
    custom_generate = partial(
        active_prefix_decode_generate,
        active_prefix_bucket_size=ACTIVE_PREFIX_BUCKET_SIZE,
        cuda_graph_forward=True,
        cuda_graph_warmup=0,
        cuda_graph_min_decode_steps=1,
        **context_state.active_prefix_decode_kwargs(),
    )

    with torch.autocast(
        device_type=model.device.type,
        dtype=torch.bfloat16,
        enabled=precision == "amp",
    ), generation_profile_context(
        q1_bmm_cross_attention=True,
        native_q1_self_attention=native_q1_self_attention,
        native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
    ):
        if sync_model_timing:
            sync_cuda_for_model(model)
        start_time = time.perf_counter()
        result = model.generate(
            **model_kwargs,
            **generate_kwargs,
            use_cache=True,
            past_key_values=cache,
            logits_processor=processors,
            eos_token_id=eos_token_ids(
                tokenizer,
                lookback_time=lookback_time,
                lookahead_time=lookahead_time,
                context_type=context_type,
            ),
            custom_generate=custom_generate,
        )
        if sync_model_timing:
            sync_cuda_for_model(model)
        elapsed_seconds = time.perf_counter() - start_time

    result = result.cpu()
    stats = build_generation_stats(
        result,
        model_kwargs,
        pad_token_id,
        elapsed_seconds,
    )
    stats.update({
        "precision": precision,
        "context_type": context_type.value if context_type is not None else None,
        "decoder_loop_backend": "active_prefix_cuda_graph",
        "torch_compile_enabled": False,
        "optimized_effective_config_version": OPTIMIZED_CONFIG_VERSION,
    })
    return result, stats


class OptimizedSingleRuntime:
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
        return _generate_window(
            model,
            tokenizer,
            model_kwargs,
            dict(generate_kwargs),
            context_state=context_state,
        )

    def profile_metadata(self) -> dict[str, Any]:
        return {
            "optimized_effective_config_version": OPTIMIZED_CONFIG_VERSION,
            "optimized_effective_config": _optimized_config_metadata(),
            "optimized_runtime_owner": (
                "osuT5.osuT5.inference.optimized.single.engine"
            ),
            "optimized_result_class": OPTIMIZED_RESULT_CLASS,
        }


def load_optimized_single_engine(
    *,
    model_loader,
    loader_kwargs: dict[str, Any],
):
    if loader_kwargs.get("precision") != OPTIMIZED_PRECISION:
        raise ValueError("optimized single loader requires precision=fp32.")
    if loader_kwargs.get("attn_implementation") != OPTIMIZED_ATTN_IMPLEMENTATION:
        raise ValueError("optimized single loader requires attn_implementation=sdpa.")
    if loader_kwargs.get("use_server"):
        raise ValueError("optimized single loader requires use_server=false.")

    effective_loader_kwargs = dict(loader_kwargs)
    effective_loader_kwargs.update(
        {
            "precision": OPTIMIZED_PRECISION,
            "attn_implementation": OPTIMIZED_ATTN_IMPLEMENTATION,
            "generation_compile": True,
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
            "optimized single loader requested the custom generation path, but "
            "the raw model reports it disabled."
        )
    return (
        InferenceEngineBinding(
            raw_model=raw_model,
            runtime=OptimizedSingleRuntime(),
        ),
        tokenizer,
    )
