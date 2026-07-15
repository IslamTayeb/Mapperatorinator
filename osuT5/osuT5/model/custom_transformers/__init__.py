"""Lazy exports for custom backbone implementations."""

from __future__ import annotations

from importlib import import_module
from typing import Final


_EXPORT_MODULE: Final = {
    "NWhisperConfig": ".configuration_nwhisper",
    "NWhisperModel": ".modeling_nwhisper",
    "NWhisperForConditionalGeneration": ".modeling_nwhisper",
    "RoPEWhisperConfig": ".configuration_ropewhisper",
    "RoPEWhisperModel": ".modeling_ropewhisper",
    "RoPEWhisperForConditionalGeneration": ".modeling_ropewhisper",
    "VarWhisperConfig": ".configuration_varwhisper",
    "VarWhisperModel": ".modeling_varwhisper",
    "VarWhisperForConditionalGeneration": ".modeling_varwhisper",
    "T5": ".t5",
}
__all__ = tuple(_EXPORT_MODULE)


def __getattr__(name: str):
    module_name = _EXPORT_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
