"""Opt-in K1 CUDA-graph static-input arena candidate."""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from typing import Iterator


_ACTIVE = False


@contextmanager
def install_shared_static_input_arena_candidate() -> Iterator[None]:
    """Temporarily enable address-stable inputs without replacing the decoder."""

    global _ACTIVE
    if _ACTIVE:
        raise RuntimeError("shared static-input arena candidate cannot be nested")

    from . import engine

    original = engine.active_prefix_decode_generate

    @wraps(original)
    def candidate(*args, **kwargs):
        requested = kwargs.pop("shared_static_input_arena", True)
        if requested is not True:
            raise ValueError(
                "shared static-input arena candidate requires its opt-in flag"
            )
        return original(
            *args,
            shared_static_input_arena=True,
            **kwargs,
        )

    _ACTIVE = True
    engine.active_prefix_decode_generate = candidate
    try:
        yield
    finally:
        replaced = engine.active_prefix_decode_generate is not candidate
        engine.active_prefix_decode_generate = original
        _ACTIVE = False
        if replaced:
            raise RuntimeError(
                "shared static-input arena candidate dispatch changed while active"
            )


__all__ = ["install_shared_static_input_arena_candidate"]
