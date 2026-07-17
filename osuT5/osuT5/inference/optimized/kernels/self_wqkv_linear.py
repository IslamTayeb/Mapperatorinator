"""Opt-in owned rectangular self-Wqkv one-token linear.

Tip still runs the decode-path self-attention QKV projection through eager
``module.Wqkv(hidden_states)`` (an ``nn.Linear`` GEMV) *after* tip-eager
``self_attn_layer_norm`` RMSNorm. Candidate replaces only that projection
with an owned warp-group rectangular ``one_token_linear_rect`` kernel
(``output_dim = 3 * hidden_dim``), computed on the same already-normalized
``hidden_states``. RMSNorm itself is not fused (§7 failed exactness) and
Wo is not touched (§11 failed exactness).

Math is unchanged; only the projection kernel launch geometry differs.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

TIP_DEFAULT_OUTPUTS_PER_BLOCK = 8
_ALLOWED = frozenset({2, 4, 8})

_REQUESTED = False
_OVERRIDE: ContextVar[int | None] = ContextVar(
    "self_wqkv_linear_outputs_per_block_override", default=None
)


def self_wqkv_linear_requested() -> bool:
    return _REQUESTED


def self_wqkv_linear_override() -> int | None:
    return _OVERRIDE.get()


def effective_self_wqkv_linear_outputs_per_block(
    default: int = TIP_DEFAULT_OUTPUTS_PER_BLOCK,
) -> int:
    override = _OVERRIDE.get()
    if override is None:
        value = int(default)
    else:
        value = int(override)
    if value not in _ALLOWED:
        raise ValueError(
            "self-Wqkv linear outputs_per_block must be one of "
            f"{sorted(_ALLOWED)}; got {value}"
        )
    return value


@contextmanager
def self_wqkv_linear_candidate_context(
    outputs_per_block: int = TIP_DEFAULT_OUTPUTS_PER_BLOCK,
) -> Iterator[None]:
    global _REQUESTED
    if (
        isinstance(outputs_per_block, bool)
        or not isinstance(outputs_per_block, int)
        or outputs_per_block not in _ALLOWED
    ):
        raise ValueError(
            "self-Wqkv linear candidate outputs_per_block must be one of "
            f"{sorted(_ALLOWED)}"
        )
    previous = _REQUESTED
    token = _OVERRIDE.set(int(outputs_per_block))
    _REQUESTED = True
    try:
        yield
    finally:
        _OVERRIDE.reset(token)
        _REQUESTED = previous


__all__ = [
    "TIP_DEFAULT_OUTPUTS_PER_BLOCK",
    "effective_self_wqkv_linear_outputs_per_block",
    "self_wqkv_linear_candidate_context",
    "self_wqkv_linear_override",
    "self_wqkv_linear_requested",
]
