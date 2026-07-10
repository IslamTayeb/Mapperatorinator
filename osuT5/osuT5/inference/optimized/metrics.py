"""Small, model-independent metric primitives for optimized inference."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ThroughputMeasurement:
    """Token throughput over an explicit scheduler-wall interval."""

    generated_tokens: int
    started_at_seconds: float
    finished_at_seconds: float

    def __post_init__(self) -> None:
        if self.generated_tokens < 0:
            raise ValueError("generated_tokens must be non-negative.")
        if not math.isfinite(self.started_at_seconds) or not math.isfinite(self.finished_at_seconds):
            raise ValueError("throughput timestamps must be finite.")
        if self.finished_at_seconds <= self.started_at_seconds:
            raise ValueError("finished_at_seconds must be greater than started_at_seconds.")

    @property
    def scheduler_wall_seconds(self) -> float:
        return self.finished_at_seconds - self.started_at_seconds

    @property
    def scheduler_wall_tokens_per_second(self) -> float:
        return self.generated_tokens / self.scheduler_wall_seconds

    def as_dict(self) -> dict[str, float | int]:
        return {
            "generated_tokens": self.generated_tokens,
            "scheduler_wall_seconds": self.scheduler_wall_seconds,
            "scheduler_wall_tokens_per_second": self.scheduler_wall_tokens_per_second,
        }


@dataclass(frozen=True)
class RequestTiming:
    """Per-request scheduler timing kept separate from aggregate throughput."""

    queue_wait_seconds: float
    latency_seconds: float

    def __post_init__(self) -> None:
        if self.queue_wait_seconds < 0 or not math.isfinite(self.queue_wait_seconds):
            raise ValueError("queue_wait_seconds must be finite and non-negative.")
        if self.latency_seconds < 0 or not math.isfinite(self.latency_seconds):
            raise ValueError("latency_seconds must be finite and non-negative.")
        if self.queue_wait_seconds > self.latency_seconds:
            raise ValueError("queue_wait_seconds cannot exceed total latency_seconds.")


def active_batch_size_histogram(samples: Iterable[int]) -> dict[int, int]:
    """Count scheduler iterations by active batch size, rejecting idle/invalid samples."""

    histogram = Counter(samples)
    if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in histogram):
        raise ValueError("active batch sizes must be positive integers.")
    return dict(sorted(histogram.items()))
