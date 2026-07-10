"""Lazy compatibility API for verifier-only optimized decoder-layer kernels."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def _implementation():
    return import_module(
        "osuT5.osuT5.inference.optimized.kernels.decoder_layer"
    )


def preload_native_decoder_layer():
    return _implementation().preload_native_decoder_layer()


def native_one_token_mlp_residual(*args: Any, **kwargs: Any):
    return _implementation().native_one_token_mlp_residual(*args, **kwargs)


def native_one_token_rmsnorm_linear(*args: Any, **kwargs: Any):
    return _implementation().native_one_token_rmsnorm_linear(*args, **kwargs)


def native_one_token_linear_residual(*args: Any, **kwargs: Any):
    return _implementation().native_one_token_linear_residual(*args, **kwargs)
