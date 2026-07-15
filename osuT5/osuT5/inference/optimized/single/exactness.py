"""Opt-in exactness evidence for non-authoritative verifier runs."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch


def _tensor_sha256(tensor: torch.Tensor) -> str:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("exactness signature requires a tensor")
    value = tensor.detach().contiguous()
    metadata = json.dumps(
        {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload = value.view(torch.uint8).cpu().numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(metadata)
    digest.update(b"\0")
    digest.update(payload)
    return digest.hexdigest()


def rng_progression_signature(device: torch.device | str) -> dict[str, Any]:
    """Hash global CPU and active-device RNG state without advancing either."""

    device = torch.device(device)
    signature: dict[str, Any] = {
        "cpu_sha256": _tensor_sha256(torch.random.get_rng_state()),
        "cuda_device": None,
        "cuda_sha256": None,
    }
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA exactness evidence requested without CUDA")
        signature["cuda_device"] = str(device)
        signature["cuda_sha256"] = _tensor_sha256(
            torch.cuda.get_rng_state(device)
        )
    return signature


def _cache_layers(cache: Any, name: str) -> list[Any] | tuple[Any, ...]:
    attention_cache = getattr(cache, name, None)
    layers = getattr(attention_cache, "layers", None)
    if not isinstance(layers, (list, tuple)) or not layers:
        raise ValueError(f"strict exactness cache is missing {name} layers")
    return layers


def _layer_pair(
    layer: Any,
    *,
    name: str,
    expected_dtype: torch.dtype,
    expected_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if getattr(layer, "is_initialized", True) is not True:
        raise ValueError(f"strict exactness cache layer {name} is not initialized")
    keys = getattr(layer, "keys", None)
    values = getattr(layer, "values", None)
    if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
        raise ValueError(f"strict exactness cache layer {name} has no key/value tensors")
    if keys.ndim != 4 or values.ndim != 4 or keys.shape != values.shape:
        raise ValueError(
            f"strict exactness cache layer {name} must expose matching rank-4 "
            "key/value tensors"
        )
    for kind, tensor in (("keys", keys), ("values", values)):
        if tensor.dtype != expected_dtype:
            raise TypeError(
                f"strict exactness cache {name}.{kind} has dtype {tensor.dtype}, "
                f"expected {expected_dtype}"
            )
        if tensor.device != expected_device:
            raise ValueError(
                f"strict exactness cache {name}.{kind} is on {tensor.device}, "
                f"expected {expected_device}"
            )
    return keys, values


def _layer_sequence_length(layer: Any, *, name: str) -> int:
    get_seq_length = getattr(layer, "get_seq_length", None)
    if not callable(get_seq_length):
        raise ValueError(
            f"strict exactness cache layer {name} has no sequence-length API"
        )
    length = get_seq_length()
    if isinstance(length, torch.Tensor):
        if length.numel() != 1:
            raise ValueError(
                f"strict exactness cache layer {name} returned non-scalar length"
            )
        length = int(length.detach().item())
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError(
            f"strict exactness cache layer {name} returned invalid length {length!r}"
        )
    return length


def _require_zero_tail(
    tensor: torch.Tensor,
    *,
    sequence_length: int,
    name: str,
) -> None:
    tail = tensor[..., sequence_length:, :]
    if tail.numel() and bool(torch.count_nonzero(tail).detach().cpu().item()):
        raise RuntimeError(
            f"strict exactness cache {name} has nonzero state beyond written length "
            f"{sequence_length}"
        )


def cache_write_signature(
    cache: Any,
    *,
    self_sequence_length: int,
    expected_dtype: torch.dtype,
    expected_device: torch.device | str,
) -> dict[str, Any]:
    """Hash every written self slot and the initialized cross cache.

    The unwritten static-cache tails must remain zero and are checked before
    hashing the logically initialized prefixes. This helper copies cache data
    to the host and is therefore only valid in a separately classified,
    non-authoritative exactness run.
    """

    if isinstance(self_sequence_length, bool) or not isinstance(
        self_sequence_length, int
    ):
        raise TypeError("self_sequence_length must be an integer")
    if self_sequence_length <= 0:
        raise ValueError("self_sequence_length must be positive")
    if not isinstance(expected_dtype, torch.dtype):
        raise TypeError("expected_dtype must be a torch dtype")
    expected_device = torch.device(expected_device)

    self_layers = _cache_layers(cache, "self_attention_cache")
    cross_layers = _cache_layers(cache, "cross_attention_cache")
    if len(self_layers) != len(cross_layers):
        raise ValueError("strict exactness self/cross cache layer counts differ")

    cross_sequence_length: int | None = None
    layer_signatures = []
    aggregate = hashlib.sha256()
    for layer_idx, (self_layer, cross_layer) in enumerate(
        zip(self_layers, cross_layers, strict=True)
    ):
        self_keys, self_values = _layer_pair(
            self_layer,
            name=f"self[{layer_idx}]",
            expected_dtype=expected_dtype,
            expected_device=expected_device,
        )
        cross_keys, cross_values = _layer_pair(
            cross_layer,
            name=f"cross[{layer_idx}]",
            expected_dtype=expected_dtype,
            expected_device=expected_device,
        )
        if self_sequence_length > self_keys.shape[-2]:
            raise ValueError(
                f"self_sequence_length {self_sequence_length} exceeds cache "
                f"capacity {self_keys.shape[-2]} at layer {layer_idx}"
            )
        reported_self_length = _layer_sequence_length(
            self_layer,
            name=f"self[{layer_idx}]",
        )
        if reported_self_length != self_sequence_length:
            raise RuntimeError(
                f"strict exactness self[{layer_idx}] reports written length "
                f"{reported_self_length}, expected result_length-1="
                f"{self_sequence_length}"
            )
        reported_cross_length = _layer_sequence_length(
            cross_layer,
            name=f"cross[{layer_idx}]",
        )
        if reported_cross_length > cross_keys.shape[-2]:
            raise RuntimeError(
                f"strict exactness cross[{layer_idx}] reports length "
                f"{reported_cross_length} beyond capacity {cross_keys.shape[-2]}"
            )
        if cross_sequence_length is None:
            cross_sequence_length = reported_cross_length
        elif reported_cross_length != cross_sequence_length:
            raise RuntimeError(
                "strict exactness cross-cache written lengths differ across layers"
            )
        for kind, tensor in (("keys", self_keys), ("values", self_values)):
            _require_zero_tail(
                tensor,
                sequence_length=self_sequence_length,
                name=f"self[{layer_idx}].{kind}",
            )
        for kind, tensor in (("keys", cross_keys), ("values", cross_values)):
            _require_zero_tail(
                tensor,
                sequence_length=reported_cross_length,
                name=f"cross[{layer_idx}].{kind}",
            )
        tensors = {
            "self_keys": self_keys[..., :self_sequence_length, :],
            "self_values": self_values[..., :self_sequence_length, :],
            "cross_keys": cross_keys[..., :reported_cross_length, :],
            "cross_values": cross_values[..., :reported_cross_length, :],
        }
        hashes = {name: _tensor_sha256(value) for name, value in tensors.items()}
        for name in sorted(hashes):
            aggregate.update(f"{layer_idx}:{name}:{hashes[name]}\n".encode("ascii"))
        layer_signatures.append({"layer_idx": layer_idx, **hashes})

    return {
        "schema_version": 1,
        "self_sequence_length": self_sequence_length,
        "cross_sequence_length": cross_sequence_length,
        "layer_count": len(layer_signatures),
        "aggregate_sha256": aggregate.hexdigest(),
        "layers": layer_signatures,
    }


__all__ = ["cache_write_signature", "rng_progression_signature"]
