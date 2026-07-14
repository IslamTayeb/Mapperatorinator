"""Opt-in sharing of exact-equivalent decoder RoPE calculations."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MethodType
from typing import Any, Iterator

import torch


_ACTIVE_MODELS: set[int] = set()


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
            "type": "tensor",
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


def _json_digest(value: Any) -> str:
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
        raise TypeError("shared RoPE requires a floating inv_freq tensor")
    if not isinstance(original_inv_freq, torch.Tensor):
        raise TypeError("shared RoPE requires tensor original_inv_freq state")
    if config is None or not hasattr(config, "to_dict"):
        raise TypeError("shared RoPE requires a serializable RoPE config")
    if rope_type != "default":
        raise ValueError(
            "shared RoPE supports only static default RoPE; dynamic scaling mutates "
            "per-layer state"
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
        _json_digest(inv_freq),
        str(inv_freq.dtype),
        str(inv_freq.device),
        tuple(inv_freq.shape),
        tuple(inv_freq.stride()),
        _json_digest(original_inv_freq),
        str(original_inv_freq.dtype),
        str(original_inv_freq.device),
        tuple(original_inv_freq.shape),
        tuple(original_inv_freq.stride()),
        _json_digest(getattr(module, "attention_scaling", None)),
        _json_digest(config_payload),
    )


@dataclass(frozen=True)
class RopeMember:
    name: str
    layer_idx: int
    module: torch.nn.Module
    group_id: str


@dataclass(frozen=True)
class SharedRopePlan:
    members: tuple[RopeMember, ...]
    group_signatures: dict[str, tuple[Any, ...]]

    @property
    def module_count(self) -> int:
        return len(self.members)

    @property
    def group_count(self) -> int:
        return len(self.group_signatures)


@dataclass
class SharedRopeStats:
    module_count: int = 0
    group_count: int = 0
    forwards: int = 0
    computes: int = 0
    reuses: int = 0
    group_computes: dict[str, int] = field(default_factory=dict)
    member_names: tuple[str, ...] = ()
    group_members: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def reset(self, plan: SharedRopePlan) -> None:
        self.module_count = plan.module_count
        self.group_count = plan.group_count
        self.forwards = 0
        self.computes = 0
        self.reuses = 0
        self.group_computes = {
            group_id: 0 for group_id in plan.group_signatures
        }
        self.member_names = tuple(member.name for member in plan.members)
        self.group_members = {
            group_id: tuple(
                member.name
                for member in plan.members
                if member.group_id == group_id
            )
            for group_id in plan.group_signatures
        }

    def as_dict(self) -> dict[str, Any]:
        eliminated_per_forward = self.module_count - self.group_count
        return {
            "module_count": self.module_count,
            "group_count": self.group_count,
            "forwards": self.forwards,
            "computes": self.computes,
            "reuses": self.reuses,
            "eliminated_per_forward": eliminated_per_forward,
            "expected_computes": self.forwards * self.group_count,
            "expected_reuses": self.forwards * eliminated_per_forward,
            "group_computes": dict(self.group_computes),
            "member_names": list(self.member_names),
            "group_members": {
                group_id: list(names)
                for group_id, names in self.group_members.items()
            },
        }


def build_shared_rope_plan(model: torch.nn.Module) -> SharedRopePlan:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("shared RoPE owner must be a torch.nn.Module")
    from osuT5.osuT5.model.custom_transformers.modeling_varwhisper import (
        VarWhisperDecoderLayer,
    )

    named_modules = dict(model.named_modules())
    decoder_layers = [
        (name, module)
        for name, module in named_modules.items()
        if isinstance(module, VarWhisperDecoderLayer)
    ]
    if not decoder_layers:
        raise RuntimeError("shared RoPE found no VarWhisper decoder layers")
    signatures: dict[tuple[Any, ...], str] = {}
    group_signatures: dict[str, tuple[Any, ...]] = {}
    members: list[RopeMember] = []
    seen_layer_indices: set[int] = set()
    for layer_name, layer in decoder_layers:
        if (
            "forward" in layer.__dict__
            or getattr(layer.forward, "__func__", None)
            is not VarWhisperDecoderLayer.forward
        ):
            raise RuntimeError(
                f"shared RoPE requires the original decoder forward at {layer_name}"
            )
        self_attn = getattr(layer, "self_attn", None)
        rotary = getattr(self_attn, "rotary_emb", None)
        layer_idx = getattr(self_attn, "layer_idx", None)
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
        group_id = signatures.get(signature)
        if group_id is None:
            digest = _json_digest(signature)
            group_id = f"rope-{len(signatures)}-{digest[:12]}"
            if group_id in group_signatures:
                raise RuntimeError("shared RoPE group digest collision")
            signatures[signature] = group_id
            group_signatures[group_id] = signature
        members.append(
            RopeMember(
                name=f"{layer_name}.self_attn.rotary_emb",
                layer_idx=layer_idx,
                module=rotary,
                group_id=group_id,
            )
        )
    members.sort(key=lambda member: member.layer_idx)
    return SharedRopePlan(tuple(members), group_signatures)


@dataclass
class _CachedRope:
    input_signature: tuple[Any, ...]
    output: tuple[torch.Tensor, torch.Tensor]


class _SharedRopeState:
    def __init__(self, plan: SharedRopePlan, stats: SharedRopeStats):
        self.plan = plan
        self.stats = stats
        self.in_forward = False
        self.cache: dict[str, _CachedRope] = {}

    def begin_forward(self) -> None:
        if self.in_forward:
            raise RuntimeError("shared RoPE does not support recursive model.forward")
        self.in_forward = True
        self.cache.clear()
        self.stats.forwards += 1

    def end_forward(self) -> None:
        if not self.in_forward:
            raise RuntimeError("shared RoPE forward lifecycle is unbalanced")
        if set(self.cache) != set(self.plan.group_signatures):
            missing = sorted(set(self.plan.group_signatures) - set(self.cache))
            raise RuntimeError(
                f"shared RoPE model forward did not visit every group; missing={missing}"
            )
        self.cache.clear()
        self.in_forward = False

    def abort_forward(self) -> None:
        self.cache.clear()
        self.in_forward = False

    @staticmethod
    def _input_signature(
        module: torch.nn.Module,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[Any, ...]:
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError("shared RoPE input must be a floating tensor")
        if (
            not isinstance(position_ids, torch.Tensor)
            or position_ids.dtype != torch.long
        ):
            raise TypeError("shared RoPE position_ids must be an int64 tensor")
        inv_freq = getattr(module, "inv_freq", None)
        if not isinstance(inv_freq, torch.Tensor):
            raise TypeError("shared RoPE module lost inv_freq")
        if x.device != inv_freq.device or position_ids.device != x.device:
            raise ValueError("shared RoPE tensors and inv_freq must use one device")
        return (
            tuple(x.shape),
            tuple(x.stride()),
            str(x.dtype),
            str(x.device),
            tuple(position_ids.shape),
            tuple(position_ids.stride()),
            str(position_ids.dtype),
            str(position_ids.device),
            int(position_ids.data_ptr()),
        )

    def run(
        self,
        member: RopeMember,
        original_forward,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.in_forward:
            raise RuntimeError("shared RoPE module executed outside the owner forward")
        input_signature = self._input_signature(member.module, x, position_ids)
        cached = self.cache.get(member.group_id)
        if cached is None:
            output = original_forward(x, position_ids=position_ids)
            if (
                not isinstance(output, tuple)
                or len(output) != 2
                or not all(isinstance(value, torch.Tensor) for value in output)
            ):
                raise TypeError("RoPE forward must return a pair of tensors")
            cos, sin = output
            if (
                cos.shape != sin.shape
                or cos.dtype != x.dtype
                or sin.dtype != x.dtype
                or cos.device != x.device
                or sin.device != x.device
            ):
                raise RuntimeError("RoPE output shape, dtype, or device changed")
            cached = _CachedRope(input_signature=input_signature, output=(cos, sin))
            self.cache[member.group_id] = cached
            self.stats.computes += 1
            self.stats.group_computes[member.group_id] += 1
        else:
            if cached.input_signature != input_signature:
                raise RuntimeError(
                    f"shared RoPE group {member.group_id} received mismatched inputs"
                )
            self.stats.reuses += 1
        return cached.output


@dataclass
class _Patch:
    module: torch.nn.Module
    had_instance_forward: bool
    instance_forward: Any

    def restore(self) -> None:
        if self.had_instance_forward:
            self.module.forward = self.instance_forward
        else:
            delattr(self.module, "forward")


def _patch_forward(module: torch.nn.Module, replacement) -> _Patch:
    patch = _Patch(
        module=module,
        had_instance_forward="forward" in module.__dict__,
        instance_forward=module.__dict__.get("forward"),
    )
    module.forward = MethodType(replacement, module)
    return patch


@contextmanager
def shared_decoder_rope_context(
    model: torch.nn.Module,
    *,
    stats: SharedRopeStats | None = None,
) -> Iterator[SharedRopeStats]:
    """Share RoPE output within each exact-equivalent decoder group."""

    owner_id = id(model)
    if owner_id in _ACTIVE_MODELS:
        raise RuntimeError("shared RoPE is already active for this model")
    plan = build_shared_rope_plan(model)
    active_stats = stats if stats is not None else SharedRopeStats()
    active_stats.reset(plan)
    state = _SharedRopeState(plan, active_stats)
    patches: list[_Patch] = []
    _ACTIVE_MODELS.add(owner_id)
    try:
        for member in plan.members:
            original_forward = member.module.forward

            def rope_forward(
                self,
                x,
                position_ids,
                *,
                _member=member,
                _original=original_forward,
            ):
                if self is not _member.module:
                    raise RuntimeError("shared RoPE forward owner changed")
                return state.run(_member, _original, x, position_ids)

            patches.append(_patch_forward(member.module, rope_forward))

        original_model_forward = model.forward

        def model_forward(self, *args, **kwargs):
            if self is not model:
                raise RuntimeError("shared RoPE model owner changed")
            state.begin_forward()
            try:
                output = original_model_forward(*args, **kwargs)
            except BaseException:
                state.abort_forward()
                raise
            else:
                state.end_forward()
                return output

        patches.append(_patch_forward(model, model_forward))
        yield active_stats
    finally:
        for patch in reversed(patches):
            patch.restore()
        _ACTIVE_MODELS.discard(owner_id)


__all__ = [
    "SharedRopePlan",
    "SharedRopeStats",
    "build_shared_rope_plan",
    "shared_decoder_rope_context",
]
