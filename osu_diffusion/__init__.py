"""Lazy public diffusion exports.

Inference configuration needs ``osu_diffusion.config`` even when position
generation is disabled.  Keep that configuration import from constructing the
entire diffusion runtime.
"""

from __future__ import annotations

from importlib import import_module
from typing import Final


_EXPORT_MODULE: Final = {
    "repeat_type": ".utils.data_loading",
    "Tokenizer": ".utils.tokenizer",
    "timestep_embedding": ".utils.positional_embedding",
    "create_diffusion": ".utils.diffusion",
    "DiT_models": ".utils.models",
    "DiT": ".utils.models",
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
