"""Verifier-only optimized inference scouts.

Nothing in this package is imported by the accepted runtime.  Scouts may drift
numerically and must pass their own promotion gates before runtime wiring.

W3 batched encoder prefill is intentionally lazy: import
``osuT5.osuT5.inference.optimized.scout.batched_encoder_prefill`` (or attribute
access below) only from reciprocal/scout runners — never from selectors.
"""

from __future__ import annotations

from typing import Any

from .native_prefix import (
    accepted_fp32_prefix,
    fp32_rms_norm,
    framework_prefix,
    native_prefix,
    native_prefix_attention_context,
    specialized_prefix_attention_context,
)

__all__ = [
    "accepted_fp32_prefix",
    "fp32_rms_norm",
    "framework_prefix",
    "native_prefix",
    "native_prefix_attention_context",
    "specialized_prefix_attention_context",
    "install_batched_encoder_prefill_candidate",
    "BatchedEncoderPrefillStore",
    "DEFAULT_ENCODER_BATCH_SIZE",
    "PREFILL_VERSION",
]


def __getattr__(name: str) -> Any:
    if name in {
        "install_batched_encoder_prefill_candidate",
        "BatchedEncoderPrefillStore",
        "DEFAULT_ENCODER_BATCH_SIZE",
        "PREFILL_VERSION",
    }:
        from . import batched_encoder_prefill as _batched

        # importlib may skip rebinding when the submodule is already cached;
        # pin it so package attribute access stays observable and stable.
        globals()["batched_encoder_prefill"] = _batched
        return getattr(_batched, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
