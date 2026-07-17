"""Lazy entrypoint for the opt-in turbo (TIER2 fused) inference engine."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def load_turbo_engine(
    *,
    model_loader=None,
    loader_kwargs: dict[str, Any] | None = None,
):
    """Load the immutable turbo TIER2 fused-step preset."""

    if not callable(model_loader):
        raise TypeError("turbo inference requires an injected model_loader.")
    if not isinstance(loader_kwargs, dict):
        raise TypeError("turbo inference requires loader_kwargs.")

    engine = import_module("osuT5.osuT5.inference.turbo.engine")
    return engine.load_turbo_engine(
        model_loader=model_loader,
        loader_kwargs=loader_kwargs,
    )
