"""Request-local sharing for exact-equivalent decoder RoPE modules."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Iterator, Mapping

import torch


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    try:
        payload = value.numpy().tobytes()
    except TypeError:
        payload = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("RoPE configuration contains a non-finite float")
        return value
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "sha256": _tensor_digest(value),
        }
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _canonical(child)
            for key, child in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(child) for child in value]
    if hasattr(value, "value") and isinstance(value.value, (bool, int, float, str)):
        return _canonical(value.value)
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def _digest(value: Any) -> str:
    payload = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _callable_name(value: Any) -> str:
    module = getattr(value, "__module__", "")
    qualname = getattr(value, "__qualname__", repr(value))
    return f"{module}.{qualname}"


def _rope_signature(module: torch.nn.Module) -> tuple[Any, ...]:
    inv_freq = getattr(module, "inv_freq", None)
    original_inv_freq = getattr(module, "original_inv_freq", None)
    config = getattr(module, "config", None)
    rope_type = getattr(module, "rope_type", None)
    if not isinstance(inv_freq, torch.Tensor) or not inv_freq.is_floating_point():
        raise TypeError("shared decoder RoPE requires floating inv_freq state")
    if not isinstance(original_inv_freq, torch.Tensor):
        raise TypeError("shared decoder RoPE requires original_inv_freq state")
    if config is None or not hasattr(config, "to_dict"):
        raise TypeError("shared decoder RoPE requires a serializable config")
    if rope_type != "default":
        raise ValueError(
            "shared decoder RoPE supports only static default RoPE"
        )
    config_payload = config.to_dict()
    if not isinstance(config_payload, dict):
        raise TypeError("RoPE config.to_dict() must return a dictionary")
    return (
        f"{type(module).__module__}.{type(module).__qualname__}",
        rope_type,
        int(getattr(module, "max_seq_len_cached")),
        int(getattr(module, "original_max_seq_len")),
        _callable_name(getattr(module, "rope_init_fn", None)),
        _digest(inv_freq),
        _digest(original_inv_freq),
        _digest(getattr(module, "attention_scaling", None)),
        _digest(config_payload),
    )


@dataclass(frozen=True, slots=True)
class SharedDecoderRopeMember:
    name: str
    layer_idx: int
    module: torch.nn.Module
    group_id: int


@dataclass(frozen=True, slots=True)
class SharedDecoderRopePlan:
    model_id: int
    members: tuple[SharedDecoderRopeMember, ...]
    members_by_module_id: Mapping[int, SharedDecoderRopeMember]
    group_count: int

    @property
    def module_count(self) -> int:
        return len(self.members)

    def summary(self) -> dict[str, Any]:
        return {
            "module_count": self.module_count,
            "group_count": self.group_count,
            "eliminated_per_forward": self.module_count - self.group_count,
            "member_names": [member.name for member in self.members],
        }


def build_shared_decoder_rope_plan(
    model: torch.nn.Module,
) -> SharedDecoderRopePlan:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("shared decoder RoPE owner must be a torch.nn.Module")
    from ....model.custom_transformers.modeling_varwhisper import (
        VarWhisperDecoderLayer,
    )

    decoder_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, VarWhisperDecoderLayer)
    ]
    if not decoder_layers:
        raise RuntimeError("shared decoder RoPE found no decoder layers")

    group_by_signature: dict[tuple[Any, ...], int] = {}
    members: list[SharedDecoderRopeMember] = []
    seen_layer_indices: set[int] = set()
    for layer_name, layer in decoder_layers:
        if (
            "forward" in layer.__dict__
            or getattr(layer.forward, "__func__", None)
            is not VarWhisperDecoderLayer.forward
        ):
            raise RuntimeError(
                f"shared decoder RoPE requires the original forward at {layer_name}"
            )
        self_attention = getattr(layer, "self_attn", None)
        rotary = getattr(self_attention, "rotary_emb", None)
        layer_idx = getattr(self_attention, "layer_idx", None)
        if not isinstance(rotary, torch.nn.Module):
            raise TypeError(f"decoder layer {layer_name} has no RoPE module")
        if (
            isinstance(layer_idx, bool)
            or not isinstance(layer_idx, int)
            or layer_idx < 0
        ):
            raise ValueError(
                f"decoder layer {layer_name} has invalid layer_idx={layer_idx!r}"
            )
        if layer_idx in seen_layer_indices:
            raise RuntimeError(f"decoder layer index {layer_idx} is repeated")
        seen_layer_indices.add(layer_idx)
        signature = _rope_signature(rotary)
        group_id = group_by_signature.setdefault(signature, len(group_by_signature))
        members.append(
            SharedDecoderRopeMember(
                name=f"{layer_name}.self_attn.rotary_emb",
                layer_idx=layer_idx,
                module=rotary,
                group_id=group_id,
            )
        )
    members.sort(key=lambda member: member.layer_idx)
    by_module_id = {id(member.module): member for member in members}
    if len(by_module_id) != len(members):
        raise RuntimeError("decoder layers unexpectedly share one RoPE module")
    return SharedDecoderRopePlan(
        model_id=id(model),
        members=tuple(members),
        members_by_module_id=MappingProxyType(by_module_id),
        group_count=len(group_by_signature),
    )


@dataclass(frozen=True, slots=True)
class _CachedRope:
    input_signature: tuple[Any, ...]
    output: tuple[torch.Tensor, torch.Tensor]


class _SharedDecoderRopeForward:
    def __init__(self, plan: SharedDecoderRopePlan):
        self.plan = plan
        self.cache: dict[int, _CachedRope] = {}
        self.visited_module_ids: set[int] = set()

    @staticmethod
    def _input_signature(
        module: torch.nn.Module,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[Any, ...]:
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError("shared decoder RoPE input must be floating point")
        if (
            not isinstance(position_ids, torch.Tensor)
            or position_ids.dtype != torch.long
        ):
            raise TypeError("shared decoder RoPE position_ids must use torch.long")
        inv_freq = getattr(module, "inv_freq", None)
        if not isinstance(inv_freq, torch.Tensor):
            raise TypeError("shared decoder RoPE module lost inv_freq")
        if x.device != inv_freq.device or position_ids.device != x.device:
            raise ValueError("shared decoder RoPE tensors must use one device")
        return (
            tuple(x.shape),
            tuple(x.stride()),
            str(x.dtype),
            str(x.device),
            tuple(position_ids.shape),
            tuple(position_ids.stride()),
            str(position_ids.device),
            int(position_ids.data_ptr()),
        )

    def run(
        self,
        module: torch.nn.Module,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        dispatch_counts: dict[str, int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        member = self.plan.members_by_module_id.get(id(module))
        if member is None or member.module is not module:
            raise RuntimeError("shared decoder RoPE received an unplanned module")
        module_id = id(module)
        if module_id in self.visited_module_ids:
            raise RuntimeError("shared decoder RoPE module ran twice in one forward")
        self.visited_module_ids.add(module_id)

        signature = self._input_signature(module, x, position_ids)
        cached = self.cache.get(member.group_id)
        if cached is None:
            output = module(x, position_ids=position_ids)
            if (
                not isinstance(output, tuple)
                or len(output) != 2
                or not all(isinstance(value, torch.Tensor) for value in output)
            ):
                raise TypeError("RoPE forward must return two tensors")
            cos, sin = output
            if (
                cos.shape != sin.shape
                or cos.dtype != x.dtype
                or sin.dtype != x.dtype
                or cos.device != x.device
                or sin.device != x.device
            ):
                raise RuntimeError("shared decoder RoPE output changed shape or storage")
            cached = _CachedRope(signature, (cos, sin))
            self.cache[member.group_id] = cached
            if dispatch_counts is not None:
                dispatch_counts["shared_decoder_rope_compute"] += 1
        else:
            if cached.input_signature != signature:
                raise RuntimeError(
                    f"shared decoder RoPE group {member.group_id} inputs differ"
                )
            if dispatch_counts is not None:
                dispatch_counts["shared_decoder_rope_reuse"] += 1
        return cached.output

    def finish(self) -> None:
        expected = set(self.plan.members_by_module_id)
        if self.visited_module_ids != expected:
            missing = sorted(
                member.name
                for member in self.plan.members
                if id(member.module) not in self.visited_module_ids
            )
            raise RuntimeError(
                "shared decoder RoPE did not visit every decoder layer: "
                + ", ".join(missing)
            )
        if len(self.cache) != self.plan.group_count:
            raise RuntimeError("shared decoder RoPE did not compute every group")


_ACTIVE_FORWARD: ContextVar[_SharedDecoderRopeForward | None] = ContextVar(
    "optimized_shared_decoder_rope_forward",
    default=None,
)


@contextmanager
def shared_decoder_rope_forward_context(
    plan: SharedDecoderRopePlan,
) -> Iterator[None]:
    if not isinstance(plan, SharedDecoderRopePlan):
        raise TypeError("shared decoder RoPE context requires a validated plan")
    if _ACTIVE_FORWARD.get() is not None:
        raise RuntimeError("shared decoder RoPE forward contexts cannot nest")
    state = _SharedDecoderRopeForward(plan)
    token = _ACTIVE_FORWARD.set(state)
    try:
        yield
    except BaseException:
        raise
    else:
        state.finish()
    finally:
        _ACTIVE_FORWARD.reset(token)


def shared_decoder_rope(
    module: torch.nn.Module,
    x: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    dispatch_counts: dict[str, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = _ACTIVE_FORWARD.get()
    if state is None:
        raise RuntimeError("shared decoder RoPE executed outside a forward context")
    return state.run(
        module,
        x,
        position_ids,
        dispatch_counts=dispatch_counts,
    )


__all__ = [
    "SharedDecoderRopePlan",
    "build_shared_decoder_rope_plan",
    "shared_decoder_rope",
    "shared_decoder_rope_forward_context",
]
