"""Cache ownership and repeatability checks for experimental decode candidates.

The verifier is intentionally runtime-agnostic: callers provide a candidate
that operates on an already prepared cache.  It snapshots and restores the
cache around every repeat so candidate order cannot affect the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


_SELF_KEYS = "self_keys"
_SELF_VALUES = "self_values"
_CROSS_KEYS = "cross_keys"
_CROSS_VALUES = "cross_values"


@dataclass(frozen=True)
class TensorOwnership:
    data_ptr: int
    storage_data_ptr: int
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass
class CacheTensorSnapshot:
    value: torch.Tensor
    ownership: TensorOwnership


@dataclass
class CacheStateSnapshot:
    layer_idx: int
    cache_position: int
    tensors: dict[str, CacheTensorSnapshot]


def _scalar_position(cache_position: int | torch.Tensor) -> int:
    if isinstance(cache_position, int):
        return cache_position
    if not isinstance(cache_position, torch.Tensor) or cache_position.numel() != 1:
        raise ValueError("cache_position must be an int or scalar tensor")
    return int(cache_position.detach().item())


def _ownership(tensor: torch.Tensor) -> TensorOwnership:
    return TensorOwnership(
        data_ptr=int(tensor.data_ptr()),
        storage_data_ptr=int(tensor.untyped_storage().data_ptr()),
        storage_offset=int(tensor.storage_offset()),
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        dtype=tensor.dtype,
        device=tensor.device,
    )


def _cache_layer(cache: Any, cache_name: str, layer_idx: int) -> Any:
    attention_cache = getattr(cache, cache_name, None)
    layers = getattr(attention_cache, "layers", None)
    if not isinstance(layers, (list, tuple)) or not 0 <= layer_idx < len(layers):
        raise ValueError(f"missing {cache_name} layer {layer_idx}")
    layer = layers[layer_idx]
    if getattr(layer, "is_initialized", True) is not True:
        raise ValueError(f"{cache_name} layer {layer_idx} is not initialized")
    return layer


def _cache_tensors(cache: Any, layer_idx: int) -> dict[str, torch.Tensor]:
    self_layer = _cache_layer(cache, "self_attention_cache", layer_idx)
    cross_layer = _cache_layer(cache, "cross_attention_cache", layer_idx)
    tensors = {
        _SELF_KEYS: getattr(self_layer, "keys", None),
        _SELF_VALUES: getattr(self_layer, "values", None),
        _CROSS_KEYS: getattr(cross_layer, "keys", None),
        _CROSS_VALUES: getattr(cross_layer, "values", None),
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"cache tensor {name} is missing")
        if tensor.ndim != 4:
            raise ValueError(
                f"cache tensor {name} must have shape [batch, heads, length, dim], "
                f"got {tuple(tensor.shape)}"
            )
    if tensors[_SELF_KEYS].shape != tensors[_SELF_VALUES].shape:
        raise ValueError("self-attention key/value cache shapes differ")
    if tensors[_CROSS_KEYS].shape != tensors[_CROSS_VALUES].shape:
        raise ValueError("cross-attention key/value cache shapes differ")
    return tensors


def snapshot_cache_state(
    cache: Any,
    *,
    layer_idx: int,
    cache_position: int | torch.Tensor,
) -> CacheStateSnapshot:
    """Clone one layer's self/cross caches and record their storage identity."""

    position = _scalar_position(cache_position)
    tensors = _cache_tensors(cache, layer_idx)
    self_length = int(tensors[_SELF_KEYS].shape[-2])
    if not 0 <= position < self_length:
        raise ValueError(
            f"cache_position {position} is outside self-cache length {self_length}"
        )
    return CacheStateSnapshot(
        layer_idx=layer_idx,
        cache_position=position,
        tensors={
            name: CacheTensorSnapshot(
                value=tensor.detach().clone(),
                ownership=_ownership(tensor),
            )
            for name, tensor in tensors.items()
        },
    )


def cache_ownership_matches(cache: Any, snapshot: CacheStateSnapshot) -> dict[str, bool]:
    tensors = _cache_tensors(cache, snapshot.layer_idx)
    return {
        name: _ownership(tensor) == snapshot.tensors[name].ownership
        for name, tensor in tensors.items()
    }


def restore_cache_state(cache: Any, snapshot: CacheStateSnapshot) -> None:
    """Restore cache bytes without allowing tensor/storage replacement."""

    tensors = _cache_tensors(cache, snapshot.layer_idx)
    ownership = cache_ownership_matches(cache, snapshot)
    changed = [name for name, matches in ownership.items() if not matches]
    if changed:
        raise RuntimeError(
            "candidate replaced or resized cache storage: " + ", ".join(changed)
        )
    with torch.no_grad():
        for name, tensor in tensors.items():
            tensor.copy_(snapshot.tensors[name].value)


def _is_finite(tensor: torch.Tensor) -> bool:
    if tensor.is_floating_point() or tensor.is_complex():
        return bool(torch.isfinite(tensor).all().item())
    return True


def _exact(reference: torch.Tensor, candidate: torch.Tensor) -> bool:
    return (
        reference.shape == candidate.shape
        and reference.dtype == candidate.dtype
        and bool(torch.equal(reference, candidate))
    )


def tensor_difference(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    """Return dtype-generic exactness, finiteness, and numeric drift metadata."""

    shape_matches = reference.shape == candidate.shape
    dtype_matches = reference.dtype == candidate.dtype
    result: dict[str, Any] = {
        "shape_matches": bool(shape_matches),
        "dtype_matches": bool(dtype_matches),
        "reference_finite": _is_finite(reference),
        "candidate_finite": _is_finite(candidate),
        "exact": bool(_exact(reference, candidate)),
        "max_abs": None,
    }
    if shape_matches and reference.numel() > 0:
        result["max_abs"] = float(
            (
                reference.detach().to(torch.float64)
                - candidate.detach().to(torch.float64)
            )
            .abs()
            .max()
            .item()
        )
    return result


def _slot(tensor: torch.Tensor, position: int) -> torch.Tensor:
    return tensor[..., position : position + 1, :]


def _outside_slot_checks(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    position: int,
) -> dict[str, bool]:
    before = bool(torch.equal(reference[..., :position, :], candidate[..., :position, :]))
    after = bool(
        torch.equal(
            reference[..., position + 1 :, :],
            candidate[..., position + 1 :, :],
        )
    )
    return {
        "before_slot_unchanged": before,
        "future_unchanged": after,
        "outside_slot_unchanged": bool(before and after),
    }


def verify_candidate_cache_behavior(
    cache: Any,
    *,
    layer_idx: int,
    cache_position: int | torch.Tensor,
    candidate: Callable[[], torch.Tensor],
    repeats: int = 2,
) -> dict[str, Any]:
    """Verify one cache-writing candidate from identical state on every repeat.

    A candidate must fully overwrite the current self-attention K/V slot, leave
    every other self-cache element and the complete cross cache unchanged,
    preserve tensor/storage ownership, return finite output, and reproduce its
    output and written slot exactly across repeats.  The original cache state is
    restored before each repeat and before returning.
    """

    if repeats < 2:
        raise ValueError("candidate self-repeat verification requires repeats >= 2")
    snapshot = snapshot_cache_state(
        cache,
        layer_idx=layer_idx,
        cache_position=cache_position,
    )
    for name in (_SELF_KEYS, _SELF_VALUES):
        if not snapshot.tensors[name].value.is_floating_point():
            raise ValueError(f"cache tensor {name} must be floating point")

    repeat_reports: list[dict[str, Any]] = []
    outputs: list[torch.Tensor] = []
    written_keys: list[torch.Tensor] = []
    written_values: list[torch.Tensor] = []
    try:
        for repeat_index in range(repeats):
            restore_cache_state(cache, snapshot)
            tensors = _cache_tensors(cache, layer_idx)
            with torch.no_grad():
                _slot(tensors[_SELF_KEYS], snapshot.cache_position).fill_(float("nan"))
                _slot(tensors[_SELF_VALUES], snapshot.cache_position).fill_(float("nan"))

            output = candidate()
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"candidate must return a tensor, got {type(output).__name__}"
                )
            tensors = _cache_tensors(cache, layer_idx)
            ownership = cache_ownership_matches(cache, snapshot)
            self_key_slot = _slot(tensors[_SELF_KEYS], snapshot.cache_position)
            self_value_slot = _slot(tensors[_SELF_VALUES], snapshot.cache_position)
            self_key_outside = _outside_slot_checks(
                snapshot.tensors[_SELF_KEYS].value,
                tensors[_SELF_KEYS],
                snapshot.cache_position,
            )
            self_value_outside = _outside_slot_checks(
                snapshot.tensors[_SELF_VALUES].value,
                tensors[_SELF_VALUES],
                snapshot.cache_position,
            )
            cross_keys_unchanged = _exact(
                snapshot.tensors[_CROSS_KEYS].value,
                tensors[_CROSS_KEYS],
            )
            cross_values_unchanged = _exact(
                snapshot.tensors[_CROSS_VALUES].value,
                tensors[_CROSS_VALUES],
            )
            finite = {
                "output": _is_finite(output),
                **{name: _is_finite(tensor) for name, tensor in tensors.items()},
            }
            current_slot_written = bool(
                _is_finite(self_key_slot) and _is_finite(self_value_slot)
            )
            repeat_pass = bool(
                all(ownership.values())
                and current_slot_written
                and self_key_outside["outside_slot_unchanged"]
                and self_value_outside["outside_slot_unchanged"]
                and cross_keys_unchanged
                and cross_values_unchanged
                and all(finite.values())
            )
            repeat_reports.append(
                {
                    "repeat": repeat_index,
                    "pass": repeat_pass,
                    "ownership_stable": ownership,
                    "current_slot_written": current_slot_written,
                    "self_keys": self_key_outside,
                    "self_values": self_value_outside,
                    "cross_keys_unchanged": cross_keys_unchanged,
                    "cross_values_unchanged": cross_values_unchanged,
                    "finite": finite,
                }
            )
            outputs.append(output.detach().clone())
            written_keys.append(self_key_slot.detach().clone())
            written_values.append(self_value_slot.detach().clone())
    finally:
        restore_cache_state(cache, snapshot)

    first_output = outputs[0]
    first_keys = written_keys[0]
    first_values = written_values[0]
    output_repeat_exact = all(_exact(first_output, output) for output in outputs[1:])
    key_repeat_exact = all(_exact(first_keys, keys) for keys in written_keys[1:])
    value_repeat_exact = all(
        _exact(first_values, values) for values in written_values[1:]
    )
    restored = _cache_tensors(cache, layer_idx)
    restored_exact = {
        name: _exact(snapshot.tensors[name].value, tensor)
        for name, tensor in restored.items()
    }
    restored_ownership = cache_ownership_matches(cache, snapshot)
    return {
        "pass": bool(
            all(report["pass"] for report in repeat_reports)
            and output_repeat_exact
            and key_repeat_exact
            and value_repeat_exact
            and all(restored_exact.values())
            and all(restored_ownership.values())
        ),
        "layer_idx": layer_idx,
        "cache_position": snapshot.cache_position,
        "repeat_count": repeats,
        "candidate_repeat_output_exact": output_repeat_exact,
        "candidate_repeat_self_key_slot_exact": key_repeat_exact,
        "candidate_repeat_self_value_slot_exact": value_repeat_exact,
        "restored_exact": restored_exact,
        "restored_ownership": restored_ownership,
        "repeats": repeat_reports,
    }
