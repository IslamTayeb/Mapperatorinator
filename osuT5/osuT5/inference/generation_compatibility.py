from __future__ import annotations

from typing import Any

import torch

try:
    from ..event import ContextType
except ImportError:
    class ContextType:  # type: ignore[no-redef]
        pass


def freeze_for_generation_compatibility_key(value: Any, path: str) -> Any:
    if isinstance(value, ContextType):
        return ("ContextType", value.value)
    if isinstance(value, torch.dtype):
        return ("torch.dtype", str(value))
    if isinstance(value, torch.device):
        return ("torch.device", str(value))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return ("tuple", tuple(
            freeze_for_generation_compatibility_key(item, f"{path}[{idx}]")
            for idx, item in enumerate(value)
        ))
    if isinstance(value, list):
        return ("list", tuple(
            freeze_for_generation_compatibility_key(item, f"{path}[{idx}]")
            for idx, item in enumerate(value)
        ))
    if isinstance(value, (set, frozenset)):
        frozen_items = [
            freeze_for_generation_compatibility_key(item, f"{path}[{idx}]")
            for idx, item in enumerate(value)
        ]
        return ("set", tuple(sorted(frozen_items, key=repr)))
    if isinstance(value, dict):
        frozen_items = [
            (
                freeze_for_generation_compatibility_key(key, f"{path}.key"),
                freeze_for_generation_compatibility_key(item, f"{path}[{key!r}]"),
            )
            for key, item in value.items()
        ]
        return ("dict", tuple(sorted(frozen_items, key=repr)))

    raise TypeError(
        f"Cannot use {path}={type(value).__name__} in the server batching compatibility key. "
        "Move mutable runtime state out of generate_kwargs or add an explicit stable key."
    )


def generation_compatibility_key(generate_kwargs: dict[str, Any]) -> tuple[Any, ...]:
    frozen_items = [
        (
            freeze_for_generation_compatibility_key(key, "generate_kwargs.key"),
            freeze_for_generation_compatibility_key(value, f"generate_kwargs[{key!r}]"),
        )
        for key, value in generate_kwargs.items()
    ]
    return tuple(sorted(frozen_items, key=repr))
