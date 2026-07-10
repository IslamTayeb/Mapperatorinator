"""Lazy compatibility API for the optimized q1 attention kernels."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def _implementation():
    return import_module(
        "osuT5.osuT5.inference.optimized.kernels.q1_attention"
    )


def preload_native_q1_attention():
    return _implementation().preload_native_q1_attention()


def native_q1_attention(*args: Any, **kwargs: Any):
    return _implementation().native_q1_attention(*args, **kwargs)


def native_q1_rope_cache_attention(*args: Any, **kwargs: Any):
    return _implementation().native_q1_rope_cache_attention(*args, **kwargs)
