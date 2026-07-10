"""Narrow request, state, and result contracts for the optimized runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .exactness import ExactnessEvidence
from .metrics import ThroughputMeasurement


def _require_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")


@dataclass(frozen=True)
class OptimizedRequest:
    """One independently seeded song/window/context generation request."""

    request_id: str
    song_id: str
    window_id: str
    context_id: str
    model_inputs: Mapping[str, Any]
    generation_settings: Mapping[str, Any]
    seed: int

    def __post_init__(self) -> None:
        for name in ("request_id", "song_id", "window_id", "context_id"):
            _require_identifier(name, getattr(self, name))
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer.")


@dataclass
class RequestState:
    """All mutable generation state owned by exactly one request."""

    request: OptimizedRequest
    generator: Any
    logits_processors: Sequence[Any]
    stopping_state: Any
    encoder_output: Any
    self_attention_cache: Any
    cross_attention_cache: Any
    token_buffer: list[int] = field(default_factory=list)
    cache_position: int = 0
    slot_id: int | None = None
    slot_generation: int = 0

    def __post_init__(self) -> None:
        if self.cache_position < 0:
            raise ValueError("cache_position must be non-negative.")
        if self.slot_id is not None and self.slot_id < 0:
            raise ValueError("slot_id must be non-negative when present.")
        if self.slot_generation < 0:
            raise ValueError("slot_generation must be non-negative.")


@dataclass(frozen=True)
class OptimizedResult:
    """Generated output and the evidence needed to classify and profile it."""

    request: OptimizedRequest
    generated_output: Any
    exactness: ExactnessEvidence
    throughput: ThroughputMeasurement | None = None
    timing_metadata: Mapping[str, Any] = field(default_factory=dict)
    cache_metadata: Mapping[str, Any] = field(default_factory=dict)
    graph_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.request.request_id == "":
            raise ValueError("result request must have a non-empty request_id.")
