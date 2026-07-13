"""Lazy entrypoint for the isolated optimized inference engine."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def load_optimized_engine(
    *,
    model_loader=None,
    loader_kwargs: dict[str, Any] | None = None,
):
    """Load the single accepted optimized engine preset."""

    if not callable(model_loader):
        raise TypeError("optimized inference requires an injected model_loader.")
    if not isinstance(loader_kwargs, dict):
        raise TypeError("optimized inference requires loader_kwargs.")

    engine = import_module("osuT5.osuT5.inference.optimized.single.engine")
    return engine.load_optimized_single_engine(
        model_loader=model_loader,
        loader_kwargs=loader_kwargs,
    )
