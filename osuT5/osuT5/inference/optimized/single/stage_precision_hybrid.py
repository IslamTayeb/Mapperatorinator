"""Opt-in stage-selective precision runtime for reciprocal profiling only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ...engine_binding import InferenceEngineBinding
from ....event import ContextType
from .engine import (
    OPTIMIZED_PRESETS,
    OptimizedSingleRuntime,
    _generate_window,
)
from .state import ProductionDecodeSession
from .k8_runtime import k8_block_size_context


HYBRID_VERSION = "timing-fp16-main-mixed-fp32-k4-v1"


def _context_type(generate_kwargs: dict[str, Any]) -> ContextType | None:
    raw = generate_kwargs.get("context_type")
    return None if raw is None else ContextType(raw)


def _require_raw_model(
    binding: InferenceEngineBinding,
    *,
    precision: str,
) -> tuple[torch.nn.Module, OptimizedSingleRuntime]:
    if not isinstance(binding, InferenceEngineBinding):
        raise TypeError("stage-precision hybrid requires an optimized engine binding")
    runtime = binding.runtime
    if type(runtime) is not OptimizedSingleRuntime:
        raise TypeError(
            "stage-precision hybrid requires the ordinary optimized single runtime"
        )
    if runtime.preset is not OPTIMIZED_PRESETS[precision]:
        raise RuntimeError(
            f"stage-precision hybrid expected the registered {precision} preset"
        )
    model = binding.raw_model
    if not isinstance(model, torch.nn.Module):
        raise TypeError("stage-precision hybrid requires a torch module model")
    if model.dtype != runtime.preset.torch_dtype:
        raise TypeError(
            f"stage-precision hybrid {precision} model has dtype {model.dtype}"
        )
    return model, runtime


@dataclass(slots=True)
class StagePrecisionHybridCoordinator:
    """Own the exact main-then-timing load order and distinct model bindings."""

    main_model: torch.nn.Module | None = None
    main_runtime: OptimizedSingleRuntime | None = None
    timing_model: torch.nn.Module | None = None
    timing_runtime: OptimizedSingleRuntime | None = None

    def bind_main(self, binding: InferenceEngineBinding) -> InferenceEngineBinding:
        if self.main_model is not None or self.main_runtime is not None:
            raise RuntimeError("stage-precision hybrid main model loaded more than once")
        if self.timing_model is not None or self.timing_runtime is not None:
            raise RuntimeError("stage-precision hybrid timing model loaded before main")
        model, runtime = _require_raw_model(binding, precision="fp32")
        state = runtime._approximate_weight_only_state
        if state is None:
            raise RuntimeError(
                "stage-precision hybrid main model must initialize mixed weights first"
            )
        state.validate_owner(model)
        self.main_model = model
        self.main_runtime = runtime
        return InferenceEngineBinding(
            raw_model=model,
            runtime=_HybridMainRuntime(self),
        )

    def bind_timing(self, binding: InferenceEngineBinding) -> InferenceEngineBinding:
        if self.main_model is None or self.main_runtime is None:
            raise RuntimeError("stage-precision hybrid timing model loaded before main")
        if self.timing_model is not None or self.timing_runtime is not None:
            raise RuntimeError("stage-precision hybrid timing model loaded more than once")
        model, runtime = _require_raw_model(binding, precision="fp16")
        if model is self.main_model:
            raise RuntimeError("timing and main stages must own distinct model objects")
        if runtime is self.main_runtime:
            raise RuntimeError("timing and main stages must own distinct runtimes")
        if runtime._approximate_weight_only_state is not None:
            raise RuntimeError("timing FP16 model must not initialize mixed FP32 weights")
        self.timing_model = model
        self.timing_runtime = runtime
        return InferenceEngineBinding(
            raw_model=model,
            runtime=_HybridTimingRuntime(self),
        )

    def assert_complete(self) -> None:
        if any(
            value is None
            for value in (
                self.main_model,
                self.main_runtime,
                self.timing_model,
                self.timing_runtime,
            )
        ):
            raise RuntimeError(
                "stage-precision hybrid requires exactly one main and one timing model"
            )
        if self.main_model is self.timing_model:
            raise RuntimeError("stage-precision hybrid model identity collapsed")
        self.main_runtime._approximate_weight_only_state.validate_owner(
            self.main_model
        )
        if self.timing_runtime._approximate_weight_only_state is not None:
            raise RuntimeError("timing model unexpectedly owns mixed weights")

    def metadata(self) -> dict[str, Any]:
        self.assert_complete()
        weight_metadata = self.main_runtime._approximate_weight_only_state.metadata()
        return {
            "version": HYBRID_VERSION,
            "result_class": "documented-drift",
            "exactness_claim": False,
            "load_order": ["main_gamemode_fp32", "base_timing_fp16"],
            "models_distinct": True,
            "decode_block_sizes": {
                "timing_context": 1,
                "main_generation": 4,
            },
            "timing": {
                "model_role": "separate_base_timing_model",
                "precision": "fp16",
                "torch_dtype": "torch.float16",
                "preset_version": self.timing_runtime.preset.version,
                "mixed_weight_state": False,
                "q1_bmm_cross_attention": True,
                "native_q1_rope_cache_self_attention": True,
                "native_cross_mlp_tail": True,
                "specialized_timing_dispatch_override": True,
            },
            "main": {
                "model_role": "gamemode_model",
                "precision": "fp32",
                "torch_dtype": "torch.float32",
                "preset_version": self.main_runtime.preset.version,
                "fp32_activations_caches_reductions_logits": True,
                "mixed_weight_state": True,
                "mixed_weight_version": weight_metadata["version"],
                "split_kv_q1": True,
            },
        }


class _HybridMainRuntime:
    def __init__(self, coordinator: StagePrecisionHybridCoordinator):
        self._coordinator = coordinator

    @property
    def preset(self):
        return self._coordinator.main_runtime.preset

    def new_context_state(self) -> ProductionDecodeSession:
        self._coordinator.assert_complete()
        return self._coordinator.main_runtime.new_context_state()

    def generate_window(self, **kwargs):
        self._coordinator.assert_complete()
        model = kwargs.get("model")
        if model is not self._coordinator.main_model:
            raise RuntimeError("hybrid main runtime received the wrong model identity")
        generate_kwargs = kwargs.get("generate_kwargs")
        if not isinstance(generate_kwargs, dict):
            raise TypeError("hybrid main runtime requires generate_kwargs")
        if _context_type(generate_kwargs) == ContextType.TIMING:
            raise RuntimeError("hybrid main runtime cannot execute timing context")
        result, stats = self._coordinator.main_runtime.generate_window(**kwargs)
        stats["optimized_stage_precision_hybrid_role"] = "main_mixed_fp32"
        return result, stats

    def profile_metadata(self) -> dict[str, Any]:
        self._coordinator.assert_complete()
        metadata = self._coordinator.main_runtime.profile_metadata()
        metadata["optimized_stage_precision_hybrid"] = (
            self._coordinator.metadata()
        )
        metadata["optimized_result_class"] = "documented-drift"
        return metadata


class _HybridTimingRuntime:
    def __init__(self, coordinator: StagePrecisionHybridCoordinator):
        self._coordinator = coordinator

    @property
    def preset(self):
        return self._coordinator.timing_runtime.preset

    def new_context_state(self) -> ProductionDecodeSession:
        self._coordinator.assert_complete()
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
        self._coordinator.assert_complete()
        if model is not self._coordinator.timing_model:
            raise RuntimeError("hybrid timing runtime received the wrong model identity")
        if not isinstance(context_state, ProductionDecodeSession):
            raise TypeError("hybrid timing runtime requires ProductionDecodeSession")
        if _context_type(generate_kwargs) != ContextType.TIMING:
            raise RuntimeError("hybrid timing runtime may execute only timing context")
        requested_precision = generate_kwargs.get("precision")
        if requested_precision != "fp32":
            raise RuntimeError(
                "hybrid timing bridge expected the public FP32 request precision"
            )
        timing_kwargs = dict(generate_kwargs)
        timing_kwargs["precision"] = "fp16"
        with k8_block_size_context(1):
            result, stats = _generate_window(
                model,
                tokenizer,
                model_kwargs,
                timing_kwargs,
                context_state=context_state,
                preset=self._coordinator.timing_runtime.preset,
                allow_specialized_timing_dispatch=True,
            )
        stats["optimized_stage_precision_hybrid_role"] = "timing_fp16"
        return result, stats

    def profile_metadata(self) -> dict[str, Any]:
        self._coordinator.assert_complete()
        metadata = self._coordinator.timing_runtime.profile_metadata()
        metadata["optimized_stage_precision_hybrid"] = (
            self._coordinator.metadata()
        )
        return metadata


__all__ = ["HYBRID_VERSION", "StagePrecisionHybridCoordinator"]
