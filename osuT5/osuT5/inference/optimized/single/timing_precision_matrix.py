"""Opt-in timing-stage precision variants for one reciprocal experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ....event import ContextType
from ...engine_binding import InferenceEngineBinding
from ..kernels.weight_only_runtime import CROSS_FP16_PACKED
from .engine import OPTIMIZED_PRESETS, OptimizedSingleRuntime, _generate_window
from .k8_runtime import k8_block_size_context
from .state import ProductionDecodeSession


FULL_FP16 = "full_fp16"
FP16_WEIGHTS_FP32_STATE = "fp16_weights_fp32_state"
TIMING_PRECISION_MODES = frozenset({FULL_FP16, FP16_WEIGHTS_FP32_STATE})
MATRIX_VERSION = "selected-timing-precision-matrix-v1"


def _context_type(generate_kwargs: dict[str, Any]) -> ContextType | None:
    raw = generate_kwargs.get("context_type")
    return None if raw is None else ContextType(raw)


def _require_binding(
    binding: InferenceEngineBinding,
    *,
    precision: str,
) -> tuple[torch.nn.Module, OptimizedSingleRuntime]:
    if not isinstance(binding, InferenceEngineBinding):
        raise TypeError("timing precision matrix requires an optimized binding")
    runtime = binding.runtime
    if type(runtime) is not OptimizedSingleRuntime:
        raise TypeError("timing precision matrix requires the ordinary single runtime")
    if runtime.preset is not OPTIMIZED_PRESETS[precision]:
        raise RuntimeError(
            f"timing precision matrix expected the registered {precision} preset"
        )
    model = binding.raw_model
    if not isinstance(model, torch.nn.Module):
        raise TypeError("timing precision matrix requires a torch module")
    if model.dtype != runtime.preset.torch_dtype:
        raise TypeError(
            f"timing precision matrix {precision} model has dtype {model.dtype}"
        )
    return model, runtime


@dataclass(slots=True)
class TimingPrecisionMatrixRuntime:
    """Own one separate timing model without changing the selected main runtime."""

    mode: str
    model: torch.nn.Module
    runtime: OptimizedSingleRuntime

    @classmethod
    def from_binding(
        cls,
        binding: InferenceEngineBinding,
        *,
        mode: str,
    ) -> tuple[InferenceEngineBinding, dict[str, Any]]:
        if mode not in TIMING_PRECISION_MODES:
            raise ValueError(
                f"timing precision mode must be one of {sorted(TIMING_PRECISION_MODES)}"
            )
        precision = "fp16" if mode == FULL_FP16 else "fp32"
        model, runtime = _require_binding(binding, precision=precision)
        if runtime._timing_native_self_owner is not None:
            raise RuntimeError("timing precision model unexpectedly owns exact self overlay")
        if runtime._approximate_weight_only_state is not None:
            raise RuntimeError("timing precision model was already initialized")

        initialization: dict[str, Any] = {}
        if mode == FP16_WEIGHTS_FP32_STATE:
            initialization = runtime.initialize_approximate_weight_only_cross(
                model,
                mode=CROSS_FP16_PACKED,
            )
        owner = cls(mode=mode, model=model, runtime=runtime)
        metadata = owner.metadata(initialization=initialization)
        return (
            InferenceEngineBinding(raw_model=model, runtime=owner),
            metadata,
        )

    @property
    def preset(self):
        return self.runtime.preset

    def new_context_state(self) -> ProductionDecodeSession:
        return ProductionDecodeSession()

    def _assert_owner(self, model: torch.nn.Module) -> None:
        if model is not self.model:
            raise RuntimeError("timing precision runtime received the wrong model")
        if self.mode == FP16_WEIGHTS_FP32_STATE:
            state = self.runtime._approximate_weight_only_state
            if state is None:
                raise RuntimeError("timing weight-only state disappeared")
            state.validate_owner(model)
        elif self.runtime._approximate_weight_only_state is not None:
            raise RuntimeError("full-FP16 timing model unexpectedly owns packed weights")

    def generate_window(
        self,
        *,
        model,
        tokenizer,
        model_kwargs: dict[str, Any],
        generate_kwargs: dict[str, Any],
        context_state: ProductionDecodeSession,
    ):
        self._assert_owner(model)
        if not isinstance(context_state, ProductionDecodeSession):
            raise TypeError("timing precision runtime requires ProductionDecodeSession")
        if _context_type(generate_kwargs) != ContextType.TIMING:
            raise RuntimeError("timing precision runtime may execute only timing context")
        if generate_kwargs.get("precision") != "fp32":
            raise RuntimeError("timing precision bridge expects a public FP32 request")
        timing_kwargs = dict(generate_kwargs)
        timing_kwargs["precision"] = self.runtime.preset.precision
        state = self.runtime._approximate_weight_only_state
        with k8_block_size_context(1):
            result, stats = _generate_window(
                model,
                tokenizer,
                model_kwargs,
                timing_kwargs,
                context_state=context_state,
                preset=self.runtime.preset,
                approximate_weight_only_state=state,
                allow_specialized_timing_dispatch=self.mode == FULL_FP16,
                allow_timing_approximate_weight_only=(
                    self.mode == FP16_WEIGHTS_FP32_STATE
                ),
            )
        stats["optimized_timing_precision_matrix"] = self.metadata(
            initialization=None
        )
        return result, stats

    def metadata(self, *, initialization: dict[str, Any] | None) -> dict[str, Any]:
        self._assert_owner(self.model)
        full_fp16 = self.mode == FULL_FP16
        payload: dict[str, Any] = {
            "version": MATRIX_VERSION,
            "mode": self.mode,
            "scope": "separate-timing-model-only",
            "result_class": "documented-drift",
            "exactness_claim": False,
            "precision": "fp16" if full_fp16 else "fp32",
            "torch_dtype": "torch.float16" if full_fp16 else "torch.float32",
            "timing_block_size": 1,
            "q1_bmm_cross_attention": True,
            "original_decoder_forward_retained": True,
            "production_selector_unchanged": True,
            "main_runtime_unchanged": True,
            "full_fp16_model_storage": full_fp16,
            "fp16_packed_weights": not full_fp16,
            "fp32_activations_caches_norm_attention_reductions_logits": not full_fp16,
            "native_q1_rope_cache_self_attention": True,
            "native_cross_mlp_tail": full_fp16,
        }
        if initialization is not None:
            payload["weight_initialization"] = initialization
        return payload

    def profile_metadata(self) -> dict[str, Any]:
        self._assert_owner(self.model)
        metadata = self.runtime.profile_metadata()
        metadata["optimized_timing_precision_matrix"] = self.metadata(
            initialization=None
        )
        metadata["optimized_result_class"] = "documented-drift"
        return metadata


__all__ = [
    "FP16_WEIGHTS_FP32_STATE",
    "FULL_FP16",
    "MATRIX_VERSION",
    "TIMING_PRECISION_MODES",
    "TimingPrecisionMatrixRuntime",
]
