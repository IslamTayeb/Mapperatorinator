"""Opt-in native cross+MLP `outputs_per_block` occupancy schedule.

Tip hardcodes `outputs_per_block=8` for native cross Q / Wo / MLP residual
kernels (~15% fused-MLP nsight family on tip). Candidate forces **4** (allowed
2/4/8) to launch more CTAs with fewer outputs per block — SM75 occupancy /
latency-hiding reformulation, not q1 headgroup CTA (§12), not bucket/pad,
not proj_out-compile, not allocator-env.

Math is unchanged; only CUDA launch geometry differs.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

TIP_DEFAULT_OUTPUTS_PER_BLOCK = 8
CANDIDATE_OUTPUTS_PER_BLOCK = 4
_ALLOWED = frozenset({2, 4, 8})

_REQUESTED = False
_OVERRIDE: ContextVar[int | None] = ContextVar(
    "native_mlp_outputs_per_block_override", default=None
)


def native_mlp_outputs_per_block_requested() -> bool:
    return _REQUESTED


def native_mlp_outputs_per_block_override() -> int | None:
    return _OVERRIDE.get()


def effective_native_mlp_outputs_per_block(default: int = TIP_DEFAULT_OUTPUTS_PER_BLOCK) -> int:
    override = _OVERRIDE.get()
    if override is None:
        value = int(default)
    else:
        value = int(override)
    if value not in _ALLOWED:
        raise ValueError(
            "native MLP outputs_per_block must be one of "
            f"{sorted(_ALLOWED)}; got {value}"
        )
    return value


@contextmanager
def native_mlp_outputs_per_block_candidate_context(
    outputs_per_block: int = CANDIDATE_OUTPUTS_PER_BLOCK,
) -> Iterator[None]:
    global _REQUESTED
    if (
        isinstance(outputs_per_block, bool)
        or not isinstance(outputs_per_block, int)
        or outputs_per_block not in _ALLOWED
    ):
        raise ValueError(
            "native MLP outputs_per_block candidate must be one of "
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
    "CANDIDATE_OUTPUTS_PER_BLOCK",
    "TIP_DEFAULT_OUTPUTS_PER_BLOCK",
    "effective_native_mlp_outputs_per_block",
    "native_mlp_outputs_per_block_candidate_context",
    "native_mlp_outputs_per_block_override",
    "native_mlp_outputs_per_block_requested",
]
