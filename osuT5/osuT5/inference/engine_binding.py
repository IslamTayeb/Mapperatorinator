"""Neutral binding between a raw model and an opt-in inference runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class InferenceEngineBinding:
    raw_model: Any
    runtime: Any


def unwrap_engine_binding(model: Any) -> tuple[Any, Any | None]:
    if isinstance(model, InferenceEngineBinding):
        return model.raw_model, model.runtime
    return model, None
