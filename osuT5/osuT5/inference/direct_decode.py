"""Lazy compatibility exports for optimized single-session verifier helpers."""

from __future__ import annotations

from importlib import import_module


__all__ = [
    "OneTokenDecodeState",
    "OneTokenDecodeResult",
    "EncoderState",
    "DecoderCacheState",
    "StaticInputBuffers",
    "SamplerState",
    "GraphState",
    "GeneratedTokenState",
    "DecodeLoopState",
    "DecodeSession",
    "last_token_logits",
    "prefill_static_cache",
    "decode_one_token_raw_logits",
    "prepare_one_token_decode_inputs_fast",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    implementation = import_module(
        "osuT5.osuT5.inference.optimized.single.session"
    )
    value = getattr(implementation, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
