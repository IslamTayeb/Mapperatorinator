"""Opt-in exact q1 RoPE/cache attention head-group CTA candidate.

Importing this module does not change production dispatch. The candidate context
patches the optimized engine's profiling context so the native q1 RoPE/cache
attention path schedules two independent heads per CTA.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Iterator


def scout_metadata() -> dict[str, Any]:
    return {
        "version": "q1-rope-cache-headgroup-v1",
        "scope": "native-q1-rope-cache-attention-only",
        "evidence_mode": "full_song_reciprocal",
        "int8": False,
        "candidate": "headgroup-cta-packing",
        "heads_per_cta": 2,
        "exactness_hypothesis": (
            "Packing independent heads into one CTA preserves each head's "
            "RoPE, cache, softmax, and reduction order."
        ),
    }


@dataclass(slots=True)
class Q1RopeCacheHeadgroupActivation:
    calls: int = 0


@contextmanager
def q1_rope_cache_headgroup_candidate_context(
) -> Iterator[Q1RopeCacheHeadgroupActivation]:
    """Enable head-group scheduling on top of the accepted tip engine."""

    from ..single import engine

    original = engine.generation_profile_context
    activation = Q1RopeCacheHeadgroupActivation()

    @contextmanager
    @wraps(original)
    def candidate(*args, **kwargs):
        if "native_q1_rope_cache_headgroup" in kwargs:
            raise RuntimeError(
                "q1 RoPE/cache headgroup candidate owns "
                "native_q1_rope_cache_headgroup"
            )
        activation.calls += 1
        with original(
            *args,
            native_q1_rope_cache_headgroup=True,
            **kwargs,
        ) as ctx:
            yield ctx

    engine.generation_profile_context = candidate
    try:
        yield activation
    finally:
        engine.generation_profile_context = original


__all__ = [
    "Q1RopeCacheHeadgroupActivation",
    "q1_rope_cache_headgroup_candidate_context",
    "scout_metadata",
]
