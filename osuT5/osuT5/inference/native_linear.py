"""Lazy compatibility API for verifier-only optimized linear kernels."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def _implementation():
    return import_module("osuT5.osuT5.inference.optimized.kernels.linear")


def preload_native_linear():
    return _implementation().preload_native_linear()


def native_one_token_linear(*args: Any, **kwargs: Any):
    return _implementation().native_one_token_linear(*args, **kwargs)


def native_one_token_linear_warp_group(*args: Any, **kwargs: Any):
    return _implementation().native_one_token_linear_warp_group(*args, **kwargs)
