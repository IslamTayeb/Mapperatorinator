"""Verifier-only optimized inference scouts.

Nothing in this package is imported by the accepted runtime.  Scouts may drift
numerically and must pass their own promotion gates before runtime wiring.
"""

from .native_prefix import (
    accepted_fp32_prefix,
    fp32_rms_norm,
    framework_prefix,
    native_prefix,
    native_prefix_attention_context,
)

__all__ = [
    "accepted_fp32_prefix",
    "fp32_rms_norm",
    "framework_prefix",
    "native_prefix",
    "native_prefix_attention_context",
]
