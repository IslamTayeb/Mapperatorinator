"""Accepted immutable FP32 and FP16 optimized single-song presets."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType
from typing import Any

import torch

from ....event import ContextType
from ....runtime_profiling import generation_profile_context, profile_range
from ...engine_binding import InferenceEngineBinding
from ...generation_utils import (
    build_generation_stats,
    eos_token_ids,
    sync_cuda_for_model,
)
from .decode_loop import active_prefix_decode_generate
from .logits import build_single_logits_processor_list
from .state import ProductionDecodeSession


OPTIMIZED_ATTN_IMPLEMENTATION = "sdpa"
ACTIVE_PREFIX_BUCKET_SIZE = 64
SPLIT_KV_Q1_PREFIX_BUCKETS = tuple(range(192, 833, 64))
SPLIT_KV_Q1_SPLIT_COUNT = 8


@dataclass(frozen=True, slots=True)
class OptimizedPreset:
    version: str
    result_class: str
    precision: str
    torch_dtype: torch.dtype


OPTIMIZED_PRESETS = MappingProxyType(
    {
        "fp32": OptimizedPreset(
            version="accepted-fp32-native-cross-mlp-289-v3",
            result_class="documented-drift",
            precision="fp32",
            torch_dtype=torch.float32,
        ),
        "fp16": OptimizedPreset(
            version="accepted-fp16-all-fused-v2",
            result_class="documented-drift",
            precision="fp16",
            torch_dtype=torch.float16,
        ),
    }
)

VALIDATED_SUPER_TIMING_ACTUAL_BATCH_SIZES = MappingProxyType(
    {
        "fp32": (1, 2, 3, 4, 8, 10, 11),
        "fp16": (10, 11),
    }
)
SUPER_TIMING_MIN_CONFIGURED_BATCH_SIZE = MappingProxyType(
    {
        "fp32": 1,
        "fp16": 16,
    }
)


def _exact_batch_partition(
    total: int,
    *,
    allowed_shapes: tuple[int, ...],
) -> tuple[int, ...] | None:
    """Return the fewest deterministic chunks, preferring larger shapes."""

    if isinstance(total, bool) or not isinstance(total, int):
        raise TypeError("super-timing sample count must be an integer")
    if total <= 0:
        raise ValueError("super-timing sample count must be positive")
    if (
        not allowed_shapes
        or any(
            isinstance(shape, bool) or not isinstance(shape, int) or shape <= 0
            for shape in allowed_shapes
        )
        or tuple(sorted(set(allowed_shapes))) != allowed_shapes
    ):
        raise ValueError(
            "super-timing allowed shapes must be unique positive ascending integers"
        )

    plans: list[tuple[int, ...] | None] = [()] + [None] * total
    for count in range(1, total + 1):
        candidates = [
            (shape,) + tail
            for shape in reversed(allowed_shapes)
            if shape <= count and (tail := plans[count - shape]) is not None
        ]
        if candidates:
            plans[count] = min(
                candidates,
                key=lambda plan: (len(plan), tuple(-shape for shape in plan)),
            )
    return plans[total]


def _optimized_config_metadata(preset: OptimizedPreset) -> dict[str, Any]:
    """Describe one immutable accepted engine preset for profiles."""

    return {
        "version": preset.version,
        "result_class": preset.result_class,
        "precision": preset.precision,
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
        "native_q1_rope_cache_split_kv": preset.torch_dtype == torch.float32,
        "native_q1_rope_cache_split_kv_split_count": (
            SPLIT_KV_Q1_SPLIT_COUNT
            if preset.torch_dtype == torch.float32
            else None
        ),
        "native_q1_rope_cache_split_kv_prefix_buckets": (
            SPLIT_KV_Q1_PREFIX_BUCKETS
            if preset.torch_dtype == torch.float32
            else ()
        ),
        "native_cross_mlp_tail": True,
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
    stateful_monotonic: bool = True,
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
        stateful_monotonic=stateful_monotonic,
    )


def _native_cross_mlp_tail_enabled(
    *,
    context_type: ContextType | None,
) -> bool:
    return context_type != ContextType.TIMING


@torch.no_grad()
def _generate_window(
    model,
    tokenizer,
    model_kwargs: dict[str, Any],
    generate_kwargs: dict[str, Any],
    *,
    context_state: ProductionDecodeSession,
    preset: OptimizedPreset,
    allow_batched_decode: bool = False,
    specialized_dispatch_batch_size: int | None = None,
):
    expected_dtype = preset.torch_dtype
    if model.dtype != expected_dtype:
        raise TypeError(
            f"optimized {preset.precision} runtime loaded model dtype "
            f"{model.dtype}, expected {expected_dtype}"
        )
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

    precision = generate_kwargs.pop("precision", preset.precision)
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
    if precision != preset.precision or cfg_scale != 1.0:
        raise ValueError("optimized single runtime configuration changed after validation.")
    if batch_size != 1 and not allow_batched_decode:
        raise ValueError("optimized single runtime requires batch_size=1.")
    if int(generate_kwargs.get("num_beams", 1)) != 1:
        raise ValueError("optimized single runtime requires num_beams=1.")

    if specialized_dispatch_batch_size is not None and (
        specialized_dispatch_batch_size <= 0
    ):
        raise ValueError("specialized dispatch batch size must be positive")
    specialized_batch = (
        batch_size == 1
        and specialized_dispatch_batch_size in {None, 1}
    )
    batched_policy_disabled_reason = (
        "batch_gt_1"
        if batch_size > 1
        else "nominal_batched_policy"
        if not specialized_batch
        else None
    )
    q1_bmm_cross_attention = specialized_batch
    native_q1_self_attention = (
        specialized_batch and context_type != ContextType.TIMING
    )
    native_q1_rope_cache_self_attention = native_q1_self_attention
    native_cross_mlp_tail = (
        specialized_batch
        and _native_cross_mlp_tail_enabled(context_type=context_type)
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
        stateful_monotonic=specialized_batch,
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
    graph_count_before = int(getattr(context_state, "graph_count", 0))
    graph_capture_seconds_before = float(
        getattr(context_state, "graph_capture_seconds", 0.0)
    )
    graph_decode_replays_before = int(
        getattr(context_state, "graph_decode_replays", 0)
    )
    dispatch_counts = {
        "native_q1_rope_cache_self_attention": 0,
        "native_q1_self_attention": 0,
        "q1_bmm_cross_attention": 0,
        "native_cross_mlp_tail": 0,
    }

    with torch.autocast(
        device_type=model.device.type,
        dtype=torch.bfloat16,
        enabled=precision == "amp",
    ), generation_profile_context(
        q1_bmm_cross_attention=q1_bmm_cross_attention,
        native_q1_self_attention=native_q1_self_attention,
        native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
        native_cross_mlp_tail=native_cross_mlp_tail,
        optimized_expected_dtype=expected_dtype,
        optimized_dispatch_counts=dispatch_counts,
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

    with profile_range("generation.final_device_to_host"):
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
        "optimized_effective_config_version": preset.version,
        "optimized_batched_super_timing": allow_batched_decode,
        "optimized_dispatch_mode": (
            "accepted_batch1" if specialized_batch else "framework_batch"
        ),
        "optimized_dispatch_policy": {
            "q1_bmm_cross_attention": {
                "requested": specialized_batch,
                "enabled": q1_bmm_cross_attention,
                "disabled_reason": (
                    None
                    if q1_bmm_cross_attention
                    else batched_policy_disabled_reason
                ),
            },
            "native_q1_self_attention": {
                "requested": specialized_batch,
                "enabled": native_q1_self_attention,
                "disabled_reason": (
                    None
                    if native_q1_self_attention
                    else batched_policy_disabled_reason
                    if not specialized_batch
                    else "timing_context"
                ),
            },
            "native_q1_rope_cache_self_attention": {
                "requested": specialized_batch,
                "enabled": native_q1_rope_cache_self_attention,
                "disabled_reason": (
                    None
                    if native_q1_rope_cache_self_attention
                    else batched_policy_disabled_reason
                    if not specialized_batch
                    else "timing_context"
                ),
            },
            "native_cross_mlp_tail": {
                "requested": specialized_batch,
                "enabled": native_cross_mlp_tail,
                "disabled_reason": (
                    None
                    if native_cross_mlp_tail
                    else batched_policy_disabled_reason
                    if not specialized_batch
                    else "timing_context"
                ),
            },
        },
        "native_cross_mlp_tail_requested": specialized_batch,
        "native_cross_mlp_tail_enabled": native_cross_mlp_tail,
        "native_cross_mlp_tail_disabled_reason": (
            batched_policy_disabled_reason
            if not specialized_batch
            else "timing_context"
            if not native_cross_mlp_tail
            else None
        ),
        "optimized_dispatch_capture_hits": dict(dispatch_counts),
        "decode_graph_count_before": graph_count_before,
        "decode_graph_count_after": int(getattr(context_state, "graph_count", 0)),
        "decode_graph_count_delta": (
            int(getattr(context_state, "graph_count", 0))
            - graph_count_before
        ),
        "decode_graph_capture_seconds_delta": (
            float(getattr(context_state, "graph_capture_seconds", 0.0))
            - graph_capture_seconds_before
        ),
        "decode_graph_replays_delta": (
            int(getattr(context_state, "graph_decode_replays", 0))
            - graph_decode_replays_before
        ),
        "optimized_cuda_graphs": context_state.graph_profile_summary(),
    })
    return result, stats


@dataclass(frozen=True, slots=True)
class OptimizedSingleRuntime:
    preset: OptimizedPreset

    def __post_init__(self):
        if not isinstance(self.preset, OptimizedPreset):
            raise TypeError(
                "optimized runtime requires an immutable OptimizedPreset."
            )
        if not any(
            self.preset is candidate
            for candidate in OPTIMIZED_PRESETS.values()
        ):
            raise ValueError("optimized runtime requires a registered precision preset.")

    def new_context_state(self) -> ProductionDecodeSession:
        return ProductionDecodeSession()

    def for_super_timing(
        self,
        *,
        max_batch_size: int,
    ) -> "OptimizedSuperTimingRuntime":
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int):
            raise TypeError("optimized super timing max_batch_size must be an integer")
        if max_batch_size <= 0:
            raise ValueError("optimized super timing max_batch_size must be positive")
        minimum = SUPER_TIMING_MIN_CONFIGURED_BATCH_SIZE[self.preset.precision]
        if max_batch_size < minimum:
            raise ValueError(
                f"optimized {self.preset.precision} super timing requires "
                f"max_batch_size>={minimum}"
            )
        return OptimizedSuperTimingRuntime(
            self.preset,
            configured_max_batch_size=max_batch_size,
            public_wiring=True,
        )

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
            preset=self.preset,
        )

    def profile_metadata(self) -> dict[str, Any]:
        return {
            "optimized_effective_config_version": self.preset.version,
            "optimized_effective_config": _optimized_config_metadata(self.preset),
            "optimized_runtime_owner": (
                "osuT5.osuT5.inference.optimized.single.engine"
            ),
            "optimized_result_class": self.preset.result_class,
        }


@dataclass(frozen=True, slots=True)
class OptimizedSuperTimingRuntime:
    """Request-local fixed-batch runtime for optimized super timing."""

    preset: OptimizedPreset
    configured_max_batch_size: int
    public_wiring: bool
    _session: ProductionDecodeSession = field(
        default_factory=ProductionDecodeSession,
        init=False,
        repr=False,
    )

    def __post_init__(self):
        if not any(
            self.preset is candidate
            for candidate in OPTIMIZED_PRESETS.values()
        ):
            raise ValueError(
                "optimized super-timing runtime requires a registered preset"
            )
        if (
            isinstance(self.configured_max_batch_size, bool)
            or not isinstance(self.configured_max_batch_size, int)
        ):
            raise TypeError(
                "optimized super-timing configured max batch size must be an integer"
            )
        if self.configured_max_batch_size <= 0:
            raise ValueError(
                "optimized super-timing configured max batch size must be positive"
            )
        if not isinstance(self.public_wiring, bool):
            raise TypeError("optimized super-timing public_wiring must be boolean")
        minimum = SUPER_TIMING_MIN_CONFIGURED_BATCH_SIZE[self.preset.precision]
        if self.public_wiring and self.configured_max_batch_size < minimum:
            raise ValueError(
                f"optimized {self.preset.precision} super timing requires "
                f"configured max batch size >= {minimum}"
            )

    @property
    def is_super_timing_runtime(self) -> bool:
        return True

    def new_context_state(self) -> ProductionDecodeSession:
        return self._session

    def plan_batches(
        self,
        *,
        num_samples: int,
        max_batch_size: int,
    ) -> tuple[int, ...]:
        if max_batch_size != self.configured_max_batch_size:
            raise RuntimeError(
                "optimized super-timing Processor batch limit changed after setup"
            )
        if self.public_wiring:
            allowed = tuple(
                shape
                for shape in VALIDATED_SUPER_TIMING_ACTUAL_BATCH_SIZES[
                    self.preset.precision
                ]
                if shape <= max_batch_size
            )
            plan = _exact_batch_partition(
                num_samples,
                allowed_shapes=allowed,
            )
            if plan is None:
                raise ValueError(
                    f"optimized {self.preset.precision} super timing cannot "
                    f"partition {num_samples} windows into validated actual "
                    f"graph shapes {list(allowed)}"
                )
            return plan

        if isinstance(num_samples, bool) or not isinstance(num_samples, int):
            raise TypeError("profiling super-timing sample count must be an integer")
        if num_samples <= 0:
            raise ValueError("profiling super-timing sample count must be positive")
        return tuple(
            min(max_batch_size, num_samples - start)
            for start in range(0, num_samples, max_batch_size)
        )

    def generate_window(
        self,
        *,
        model,
        tokenizer,
        model_kwargs: dict[str, Any],
        generate_kwargs: dict[str, Any],
        context_state: ProductionDecodeSession,
    ):
        if context_state is not self._session:
            raise RuntimeError(
                "optimized super-timing runtime requires its persistent session"
            )
        return _generate_window(
            model,
            tokenizer,
            model_kwargs,
            dict(generate_kwargs),
            context_state=context_state,
            preset=self.preset,
            allow_batched_decode=True,
            specialized_dispatch_batch_size=self.configured_max_batch_size,
        )

    def super_timing_profile_metadata(self) -> dict[str, Any]:
        actual_shapes = (
            list(
                shape
                for shape in VALIDATED_SUPER_TIMING_ACTUAL_BATCH_SIZES[
                    self.preset.precision
                ]
                if shape <= self.configured_max_batch_size
            )
            if self.public_wiring
            else None
        )
        return {
            "optimized_super_timing_enabled": True,
            "optimized_super_timing_public_wiring": self.public_wiring,
            "optimized_super_timing_config": {
                "precision": self.preset.precision,
                "configured_max_batch_size": self.configured_max_batch_size,
                "validated_actual_batch_sizes": actual_shapes,
                "batch_planner": (
                    "validated_exact_partition"
                    if self.public_wiring
                    else "profiling_fixed_chunks_with_arbitrary_tail"
                ),
                "framework_dispatch_batch_gt_1": True,
                "framework_dispatch_batch1_under_batched_policy": (
                    self.configured_max_batch_size > 1
                ),
                "accepted_specialized_dispatch_batch1": (
                    self.configured_max_batch_size == 1
                ),
                "native_q1_self_attention_batch_gt_1": False,
                "native_q1_rope_cache_self_attention_batch_gt_1": False,
                "q1_bmm_cross_attention_batch_gt_1": False,
                "native_cross_mlp_tail_batch_gt_1": False,
            },
            "optimized_super_timing_runtime_owner": (
                "osuT5.osuT5.inference.optimized.single.engine"
            ),
            "optimized_super_timing_result_class": (
                self.preset.result_class
                if self.public_wiring
                else "component-scout"
            ),
        }


def build_profiling_batched_super_timing_runtime(
    precision: str,
    *,
    nominal_batch_size: int,
) -> OptimizedSuperTimingRuntime:
    try:
        preset = OPTIMIZED_PRESETS[precision]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "batched super-timing runtime requires precision in: fp16, fp32"
        ) from exc
    return OptimizedSuperTimingRuntime(
        preset,
        configured_max_batch_size=nominal_batch_size,
        public_wiring=False,
    )


def load_optimized_single_engine(
    *,
    model_loader,
    loader_kwargs: dict[str, Any],
):
    precision = loader_kwargs.get("precision")
    try:
        preset = OPTIMIZED_PRESETS[precision]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "optimized single loader requires precision in: fp16, fp32."
        ) from exc
    if loader_kwargs.get("attn_implementation") != OPTIMIZED_ATTN_IMPLEMENTATION:
        raise ValueError("optimized single loader requires attn_implementation=sdpa.")
    if loader_kwargs.get("use_server"):
        raise ValueError("optimized single loader requires use_server=false.")

    effective_loader_kwargs = dict(loader_kwargs)
    effective_loader_kwargs.update(
        {
            "precision": preset.precision,
            "attn_implementation": OPTIMIZED_ATTN_IMPLEMENTATION,
            "generation_compile": False,
            "use_server": False,
        }
    )
    raw_model, tokenizer = model_loader(**effective_loader_kwargs)
    disable_compile = getattr(
        getattr(raw_model, "generation_config", None),
        "disable_compile",
        None,
    )
    if disable_compile is not True:
        raise RuntimeError(
            "optimized single loader requires generation_config.disable_compile "
            "to be boolean True after loading; the model loader ignored or did "
            "not preserve generation_compile=False."
        )
    return (
        InferenceEngineBinding(
            raw_model=raw_model,
            runtime=OptimizedSingleRuntime(preset),
        ),
        tokenizer,
    )
