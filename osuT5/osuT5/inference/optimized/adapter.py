"""Lazy entrypoint for the isolated optimized inference engine."""

from __future__ import annotations

from importlib import import_module
from typing import Any


SUPPORTED_OPTIMIZED_MODES = frozenset({"single", "offline_batch", "server"})


class OptimizedInferenceNotImplementedError(NotImplementedError):
    """Raised until a validated optimized runtime is available for a mode."""


def load_optimized_engine(
    *,
    mode: str,
    model_loader=None,
    loader_kwargs: dict[str, Any] | None = None,
):
    """Validate mode before loading models, CUDA code, or native extensions."""

    if mode not in SUPPORTED_OPTIMIZED_MODES:
        raise ValueError(
            "optimized_inference_mode must be one of: "
            + ", ".join(sorted(SUPPORTED_OPTIMIZED_MODES))
            + "."
        )
    if mode != "single":
        raise OptimizedInferenceNotImplementedError(
            f"optimized inference mode {mode!r} is control-plane scaffolding only; "
            "no optimized model runtime has been implemented or validated yet."
        )
    if not callable(model_loader):
        raise TypeError("optimized single inference requires an injected model_loader.")
    if not isinstance(loader_kwargs, dict):
        raise TypeError("optimized single inference requires loader_kwargs.")

    engine = import_module("osuT5.osuT5.inference.optimized.single.engine")
    return engine.load_optimized_single_engine(
        model_loader=model_loader,
        loader_kwargs=loader_kwargs,
    )
