"""Opt-in untraced decode budget instrumentation.

This module is imported only by the optimized runtime.  A recorder is created
only for ``profile_pass_kind=untraced_budget``; production and ordinary profile
runs do not allocate CUDA events or retain timing state.
"""

from __future__ import annotations

import time
import math
from contextlib import contextmanager
from typing import Any, Iterator

import torch


BUDGET_SCHEMA_VERSION = 1
BUDGET_RECONCILIATION_LIMIT_FRACTION = 0.02
BUDGET_REGION_NAMES = (
    "prefill",
    "graph_input_copies",
    "graph_capture",
    "decode_replay",
    "cache_update",
    "logits_processors",
    "sampling",
    "append_stopping",
    "sync",
)
COPY_DIRECTIONS = ("h2d", "d2h", "d2d", "h2h", "other")
EXTERNAL_TRANSFER_NAMES = (
    "model_input_to_device",
    "model_input_dtype_conversion",
    "final_device_to_host",
)
CUDA_EVENT_REGION_NAMES = frozenset(
    {
        "prefill",
        "graph_input_copies",
        "decode_replay",
        "cache_update",
        "logits_processors",
        "sampling",
        "append_stopping",
    }
)


class UntracedBudgetRecorder:
    """Accumulate non-overlapping host timing and deferred CUDA-event timing."""

    def __init__(self) -> None:
        self._active_region: str | None = None
        self._finished = False
        self._cuda_available = torch.cuda.is_available()
        self._regions: dict[str, dict[str, Any]] = {
            name: {
                "calls": 0,
                "host_wall_seconds": 0.0,
                "cuda_event_samples": 0,
                "cuda_event_seconds": 0.0,
                "copy_count": 0,
                "copy_bytes": 0,
                "copy_count_by_direction": {
                    direction: 0 for direction in COPY_DIRECTIONS
                },
                "copy_bytes_by_direction": {
                    direction: 0 for direction in COPY_DIRECTIONS
                },
            }
            for name in BUDGET_REGION_NAMES
        }
        self._cuda_events: dict[str, list[tuple[Any, Any]]] = {
            name: [] for name in CUDA_EVENT_REGION_NAMES
        }
        self._external_transfers: dict[str, dict[str, Any]] = {
            name: {
                "calls": 0,
                "host_wall_seconds": 0.0,
                "copy_count": 0,
                "copy_bytes": 0,
                "copy_count_by_direction": {
                    direction: 0 for direction in COPY_DIRECTIONS
                },
                "copy_bytes_by_direction": {
                    direction: 0 for direction in COPY_DIRECTIONS
                },
            }
            for name in EXTERNAL_TRANSFER_NAMES
        }

    @contextmanager
    def region(self, name: str) -> Iterator[None]:
        if self._finished:
            raise RuntimeError("untraced budget recorder is already finished")
        if name not in self._regions:
            raise ValueError(f"unknown untraced budget region: {name}")
        if self._active_region is not None:
            raise RuntimeError(
                "untraced budget regions must not overlap: "
                f"{self._active_region} -> {name}"
            )

        use_cuda_events = self._cuda_available and name in CUDA_EVENT_REGION_NAMES
        start_event = end_event = None
        if use_cuda_events:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        self._active_region = name
        started = time.perf_counter()
        try:
            yield
        finally:
            finished = time.perf_counter()
            if use_cuda_events:
                end_event.record()
                self._cuda_events[name].append((start_event, end_event))
            region = self._regions[name]
            region["calls"] += 1
            region["host_wall_seconds"] += finished - started
            self._active_region = None

    def record_copy(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
    ) -> None:
        """Record an explicit tensor copy owned by the active budget region."""

        if self._active_region is None:
            raise RuntimeError("explicit copy must be recorded inside a budget region")
        if not isinstance(source, torch.Tensor) or not isinstance(
            destination,
            torch.Tensor,
        ):
            raise TypeError("budget copy endpoints must be tensors")
        if source.numel() != destination.numel():
            raise ValueError("budget copy endpoints must contain the same elements")
        direction, byte_count = self._copy_shape(source, destination)
        region = self._regions[self._active_region]
        region["copy_count"] += 1
        region["copy_bytes"] += byte_count
        region["copy_count_by_direction"][direction] += 1
        region["copy_bytes_by_direction"][direction] += byte_count

    def record_external_copy(
        self,
        name: str,
        source: torch.Tensor,
        destination: torch.Tensor,
        *,
        host_wall_seconds: float,
    ) -> None:
        """Record an explicit transfer outside synchronized model time."""

        if self._finished:
            raise RuntimeError("untraced budget recorder is already finished")
        if name not in self._external_transfers:
            raise ValueError(f"unknown external transfer: {name}")
        if not math.isfinite(float(host_wall_seconds)) or host_wall_seconds < 0:
            raise ValueError("external transfer host wall must be finite and non-negative")
        direction, byte_count = self._copy_shape(source, destination)
        transfer = self._external_transfers[name]
        transfer["calls"] += 1
        transfer["host_wall_seconds"] += float(host_wall_seconds)
        transfer["copy_count"] += 1
        transfer["copy_bytes"] += byte_count
        transfer["copy_count_by_direction"][direction] += 1
        transfer["copy_bytes_by_direction"][direction] += byte_count

    @staticmethod
    def _copy_shape(
        source: torch.Tensor,
        destination: torch.Tensor,
    ) -> tuple[str, int]:
        if not isinstance(source, torch.Tensor) or not isinstance(
            destination,
            torch.Tensor,
        ):
            raise TypeError("budget copy endpoints must be tensors")
        if source.numel() != destination.numel():
            raise ValueError("budget copy endpoints must contain the same elements")
        source_type = source.device.type
        destination_type = destination.device.type
        if source_type == "cpu" and destination_type == "cuda":
            direction = "h2d"
        elif source_type == "cuda" and destination_type == "cpu":
            direction = "d2h"
        elif source_type == destination_type == "cuda":
            direction = "d2d"
        elif source_type == destination_type == "cpu":
            direction = "h2h"
        else:
            direction = "other"
        # Use destination bytes as the transferred payload.  This also handles
        # explicit dtype-converting ``to(copy=True)`` operations, which are
        # kernels rather than raw cudaMemcpy calls but still move one output
        # payload across the recorded transition.
        return direction, destination.numel() * destination.element_size()

    def finish(
        self,
        *,
        model_elapsed_seconds: float,
        generated_tokens: int,
    ) -> dict[str, Any]:
        if self._finished:
            raise RuntimeError("untraced budget recorder can only finish once")
        if self._active_region is not None:
            raise RuntimeError(
                f"cannot finish untraced budget inside region {self._active_region}"
            )
        if not isinstance(model_elapsed_seconds, (int, float)) or isinstance(
            model_elapsed_seconds,
            bool,
        ):
            raise TypeError("model_elapsed_seconds must be numeric")
        model_elapsed_seconds = float(model_elapsed_seconds)
        if model_elapsed_seconds <= 0:
            raise ValueError("model_elapsed_seconds must be positive")
        if (
            isinstance(generated_tokens, bool)
            or not isinstance(generated_tokens, int)
            or generated_tokens <= 0
        ):
            raise ValueError("generated_tokens must be a positive integer")

        if self._cuda_available:
            for name, pairs in self._cuda_events.items():
                cuda_seconds = 0.0
                for start_event, end_event in pairs:
                    cuda_seconds += float(start_event.elapsed_time(end_event)) / 1000.0
                self._regions[name]["cuda_event_samples"] = len(pairs)
                self._regions[name]["cuda_event_seconds"] = cuda_seconds

        measured_host_seconds = sum(
            float(region["host_wall_seconds"])
            for region in self._regions.values()
        )
        host_unattributed_seconds = max(
            0.0,
            model_elapsed_seconds - measured_host_seconds,
        )
        accounted_host_seconds = measured_host_seconds + host_unattributed_seconds
        reconciliation_error_seconds = abs(
            model_elapsed_seconds - accounted_host_seconds
        )
        reconciliation_error_fraction = (
            reconciliation_error_seconds / model_elapsed_seconds
        )
        reconciliation_pass = (
            reconciliation_error_fraction
            <= BUDGET_RECONCILIATION_LIMIT_FRACTION
        )
        cuda_event_measured_seconds = sum(
            float(region["cuda_event_seconds"])
            for region in self._regions.values()
        )
        explicit_sync_wall_seconds = float(
            self._regions["sync"]["host_wall_seconds"]
        )
        cpu_region_wall_seconds = (
            measured_host_seconds - explicit_sync_wall_seconds
        )
        transition_count_by_direction = {
            direction: sum(
                int(region["copy_count_by_direction"][direction])
                for region in self._regions.values()
            )
            for direction in COPY_DIRECTIONS
        }
        transition_bytes_by_direction = {
            direction: sum(
                int(region["copy_bytes_by_direction"][direction])
                for region in self._regions.values()
            )
            for direction in COPY_DIRECTIONS
        }
        per_token = {
            "model_elapsed_ms": model_elapsed_seconds * 1000.0 / generated_tokens,
            "measured_host_ms": measured_host_seconds * 1000.0 / generated_tokens,
            "host_unattributed_ms": (
                host_unattributed_seconds * 1000.0 / generated_tokens
            ),
            "cuda_event_measured_ms": (
                cuda_event_measured_seconds * 1000.0 / generated_tokens
            ),
            "cpu_region_wall_ms": (
                cpu_region_wall_seconds * 1000.0 / generated_tokens
            ),
            "explicit_sync_wall_ms": (
                explicit_sync_wall_seconds * 1000.0 / generated_tokens
            ),
            "explicit_sync_count": (
                int(self._regions["sync"]["calls"]) / generated_tokens
            ),
            "regions": {
                name: {
                    "host_wall_ms": (
                        float(region["host_wall_seconds"])
                        * 1000.0
                        / generated_tokens
                    ),
                    "cuda_event_ms": (
                        float(region["cuda_event_seconds"])
                        * 1000.0
                        / generated_tokens
                    ),
                    "calls_per_token": (
                        int(region["calls"]) / generated_tokens
                    ),
                    "copy_bytes_per_token": (
                        int(region["copy_bytes"]) / generated_tokens
                    ),
                }
                for name, region in self._regions.items()
            },
        }
        self._finished = True

        return {
            "schema_version": BUDGET_SCHEMA_VERSION,
            "cuda_available": self._cuda_available,
            "generated_tokens": generated_tokens,
            "model_elapsed_seconds": model_elapsed_seconds,
            "regions": self._regions,
            "measured_host_seconds": measured_host_seconds,
            "host_unattributed_seconds": host_unattributed_seconds,
            "cuda_event_measured_seconds": cuda_event_measured_seconds,
            "cpu_region_wall_seconds": cpu_region_wall_seconds,
            "explicit_sync_wall_seconds": explicit_sync_wall_seconds,
            "explicit_sync_count": int(self._regions["sync"]["calls"]),
            "transition_count_by_direction": transition_count_by_direction,
            "transition_bytes_by_direction": transition_bytes_by_direction,
            "external_transfers": self._external_transfers,
            "accounted_host_seconds": accounted_host_seconds,
            "reconciliation_error_seconds": reconciliation_error_seconds,
            "reconciliation_error_fraction": reconciliation_error_fraction,
            "reconciliation_limit_fraction": (
                BUDGET_RECONCILIATION_LIMIT_FRACTION
            ),
            "reconciliation_pass": reconciliation_pass,
            "per_token": per_token,
        }


def budget_region(
    recorder: UntracedBudgetRecorder | None,
    name: str,
):
    """Return a cold no-op when the budget pass is not active."""

    if recorder is None:
        return _null_budget_region()
    return recorder.region(name)


@contextmanager
def _null_budget_region() -> Iterator[None]:
    yield
