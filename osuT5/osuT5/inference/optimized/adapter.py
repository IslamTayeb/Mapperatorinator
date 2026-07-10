"""Lazy entrypoint for the isolated optimized inference engine."""

from __future__ import annotations

from typing import Any


SUPPORTED_OPTIMIZED_MODES = frozenset({"single", "offline_batch", "server"})


class OptimizedInferenceNotImplementedError(NotImplementedError):
    """Raised until a validated optimized runtime is available for a mode."""


def load_optimized_engine(*, mode: str, **_: Any):
    """Fail before loading models, CUDA code, or native extensions."""

    if mode not in SUPPORTED_OPTIMIZED_MODES:
        raise ValueError(
            "optimized_inference_mode must be one of: "
            + ", ".join(sorted(SUPPORTED_OPTIMIZED_MODES))
            + "."
        )
    raise OptimizedInferenceNotImplementedError(
        f"optimized inference mode {mode!r} is control-plane scaffolding only; "
        "no optimized model runtime has been implemented or validated yet."
    )
