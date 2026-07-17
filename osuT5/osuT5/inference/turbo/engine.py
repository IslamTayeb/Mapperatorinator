"""Turbo runtime — §38 TIER2 relaxed fused decoder step.

Immutable preset: tip ``optimized`` teacher stack + opt-in fp32-accumulate
fused decoder numerics (7-stage launch-collapse target). Not bit-exact.
Not speculative (§45 STOP_DEAD_END). TIER2 evidence pack before ship.
Campaign tip remains ``55949274`` / FP16 366.11.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..engine_binding import InferenceEngineBinding
from .fused_step import install_tier2_fused_numerics

TURBO_PRESET_VERSION = "turbo-tier2-fused-step-s38-v2"


@dataclass(frozen=True, slots=True)
class TurboPreset:
    version: str
    precision: str
    result_class: str = "relaxed-numerics-pending-tier2"


TURBO_PRESETS = MappingProxyType(
    {
        "fp16": TurboPreset(
            version=TURBO_PRESET_VERSION,
            precision="fp16",
        ),
        "fp32": TurboPreset(
            version=TURBO_PRESET_VERSION,
            precision="fp32",
        ),
    }
)


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no"}


@dataclass
class TurboRuntime:
    """Opt-in turbo runtime: TIER2 fused numerics over tip optimized."""

    preset: TurboPreset
    fused_enabled: bool = True
    _install_cm: Any = None
    _install_meta: dict[str, Any] | None = None

    def profile_metadata(self) -> dict[str, Any]:
        return {
            "turbo_effective_config_version": self.preset.version,
            "turbo_runtime_owner": "osuT5.osuT5.inference.turbo.engine",
            "turbo_result_class": self.preset.result_class,
            "turbo_precision": self.preset.precision,
            "turbo_tier2_fused_step": self.fused_enabled,
            "turbo_tier2_required": True,
            "turbo_speculative": False,
            "note": (
                "§38 TIER2 fused decoder step (7-stage; storage-dtype GEMM; "
                "fp32 reductions). Not bit-exact. Not a 500 claim. "
                "Campaign tip 55949274/366.11."
            ),
            **(self._install_meta or {}),
        }

    def attach_fused(self, raw_model) -> None:
        """Install fused numerics on the bound raw model (process lifetime)."""
        if not self.fused_enabled:
            return
        if self._install_cm is not None:
            return
        self._install_cm = install_tier2_fused_numerics(raw_model)
        self._install_meta = dict(self._install_cm.__enter__())


class _TurboBoundRuntime:
    """Thin facade: optimized decode + turbo profile metadata."""

    def __init__(self, opt_rt, turbo_rt: TurboRuntime):
        self._opt = opt_rt
        self._turbo = turbo_rt

    def __getattr__(self, name: str):
        return getattr(self._opt, name)

    def profile_metadata(self) -> dict[str, Any]:
        meta = {}
        if hasattr(self._opt, "profile_metadata"):
            meta.update(self._opt.profile_metadata())
        meta.update(self._turbo.profile_metadata())
        return meta


def load_turbo_engine(
    *,
    model_loader,
    loader_kwargs: dict[str, Any],
):
    """Load tip optimized engine, then attach TIER2 fused numerics (turbo only)."""
    precision = str(loader_kwargs.get("precision", "fp32"))
    if precision not in TURBO_PRESETS:
        raise ValueError(f"turbo preset unsupported for precision={precision}")

    from ..optimized.adapter import load_optimized_engine

    binding, tokenizer = load_optimized_engine(
        model_loader=model_loader,
        loader_kwargs=loader_kwargs,
    )
    if not isinstance(binding, InferenceEngineBinding):
        raise TypeError("turbo requires InferenceEngineBinding from optimized loader")

    fused_enabled = _env_flag("MAPPERATORINATOR_TURBO_TIER2_FUSED", default=True)
    runtime = TurboRuntime(
        preset=TURBO_PRESETS[precision],
        fused_enabled=fused_enabled,
    )
    if fused_enabled:
        runtime.attach_fused(binding.raw_model)

    return (
        InferenceEngineBinding(
            raw_model=binding.raw_model,
            runtime=_TurboBoundRuntime(binding.runtime, runtime),
        ),
        tokenizer,
    )
