"""Shared FP32/FP16 native-prefix scout components.

The four verifier variants are formed by calling :func:`framework_prefix` or
:func:`native_prefix` with FP32 or FP16 captures.  Norms, attention scores,
softmax, and value reductions accumulate in FP32.  Linear layers retain the
capture storage dtype so FP16 runs can use framework Tensor Core kernels.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Iterator

import torch
import torch.nn.functional as F

from ...runtime_dispatch import (
    AttentionRuntimeHooks,
    attention_runtime_hooks,
    attention_runtime_hooks_context,
)
from .q1_attention import (
    native_q1_attention,
    native_q1_rope_cache_attention,
    preload_native_q1_attention,
)


SUPPORTED_DTYPES = (torch.float32, torch.float16)


def _require_storage_dtype(dtype: torch.dtype) -> None:
    if dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"native-prefix scout requires float32 or float16, got {dtype}")


def fp32_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float | None,
) -> torch.Tensor:
    """RMSNorm with explicit FP32 reduction and storage-dtype output."""

    if not isinstance(hidden_states, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError("hidden_states and weight must be tensors")
    _require_storage_dtype(hidden_states.dtype)
    if weight.dtype != hidden_states.dtype:
        raise TypeError("RMSNorm input and weight dtypes must match")
    if weight.dim() != 1 or weight.numel() != hidden_states.shape[-1]:
        raise ValueError("RMSNorm weight must match the hidden dimension")
    effective_eps = torch.finfo(hidden_states.dtype).eps if eps is None else float(eps)
    values = hidden_states.float()
    inverse_rms = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + effective_eps)
    return (values * inverse_rms * weight.float()).to(dtype=hidden_states.dtype)


def framework_q1_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Framework q1 attention with explicit FP32 scores and reductions."""

    dtype, heads, key_length, head_dim = _validate_q1_tensors(query, key, value)
    q = query.reshape(heads, 1, head_dim)
    k = key.reshape(heads, key_length, head_dim)
    v = value.reshape(heads, key_length, head_dim)
    if dtype == torch.float16 and query.device.type == "cuda":
        scores = torch.bmm(q, k.transpose(1, 2), out_dtype=torch.float32)
    else:
        scores = torch.bmm(q.float(), k.float().transpose(1, 2))
    scores.mul_(head_dim ** -0.5)
    if attention_mask is not None:
        mask = _attention_mask(attention_mask, query.device, key_length)
        scores.add_(mask.view(1, 1, key_length))
    probabilities = torch.softmax(scores, dim=-1)
    # Keeping probabilities and values FP32 makes the value reduction policy
    # explicit.  This is the conservative scout; native kernels avoid the
    # materialized FP32 value copy while retaining the same accumulation type.
    output = torch.bmm(probabilities, v.float()).to(dtype=dtype)
    return output.view(1, heads, 1, head_dim)


def _validate_q1_tensors(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.dtype, int, int, int]:
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        _require_storage_dtype(tensor.dtype)
        if tensor.dim() != 4:
            raise ValueError(f"{name} must be 4D")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise TypeError("query/key/value dtypes must match")
    if query.shape[0] != 1 or query.shape[2] != 1:
        raise ValueError("query must have shape [1, heads, 1, head_dim]")
    if key.shape != value.shape or key.shape[0] != 1:
        raise ValueError("key/value must have matching batch-1 shapes")
    if key.shape[1] != query.shape[1] or key.shape[3] != query.shape[3]:
        raise ValueError("query/key/value head shapes must match")
    return query.dtype, int(query.shape[1]), int(key.shape[2]), int(query.shape[3])


def _attention_mask(
    attention_mask: torch.Tensor,
    device: torch.device,
    key_length: int,
) -> torch.Tensor:
    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError("attention_mask must be a tensor")
    if attention_mask.device != device or not attention_mask.is_floating_point():
        raise ValueError("attention_mask must be floating point on the attention device")
    if attention_mask.numel() != key_length:
        raise ValueError("attention_mask must contain one value per key position")
    return attention_mask.reshape(-1).float()


def _cache_tensors(past_key_value: Any, kind: str, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    cache = getattr(past_key_value, f"{kind}_attention_cache", None)
    layers = getattr(cache, "layers", None)
    if not isinstance(layers, (list, tuple)) or layer_idx >= len(layers):
        raise ValueError(f"missing {kind}-attention cache layer {layer_idx}")
    layer = layers[layer_idx]
    if not bool(getattr(layer, "is_initialized", False)):
        raise ValueError(f"{kind}-attention cache layer {layer_idx} is not initialized")
    keys = getattr(layer, "keys", None)
    values = getattr(layer, "values", None)
    if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
        raise TypeError(f"{kind}-attention cache layer must own tensor keys/values")
    return keys, values


def _norm(module: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError("native-prefix scout requires affine RMSNorm weights")
    return fp32_rms_norm(hidden_states, weight, getattr(module, "eps", None))


def _validate_prefix_inputs(
    module: torch.nn.Module,
    *,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    past_key_value: Any,
    cache_position: torch.Tensor,
    position_ids: torch.Tensor,
    active_prefix_length: int,
) -> tuple[torch.dtype, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if module.training or module.self_attn.training or module.cross_attn.training:
        raise ValueError("native-prefix scout requires eval mode")
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.shape[:2] != (1, 1):
        raise ValueError("hidden_states must have shape [1, 1, hidden_dim]")
    dtype = hidden_states.dtype
    _require_storage_dtype(dtype)
    if not isinstance(encoder_hidden_states, torch.Tensor) or encoder_hidden_states.shape[0] != 1:
        raise ValueError("encoder_hidden_states must be a batch-1 tensor")
    if encoder_hidden_states.dtype != dtype or encoder_hidden_states.device != hidden_states.device:
        raise ValueError("encoder and decoder states must share dtype/device")
    if not isinstance(cache_position, torch.Tensor) or cache_position.dtype != torch.long or cache_position.numel() != 1:
        raise ValueError("cache_position must be a scalar int64 tensor")
    if not isinstance(position_ids, torch.Tensor) or position_ids.dtype != torch.long or position_ids.shape != (1, 1):
        raise ValueError("position_ids must have shape [1, 1] and dtype int64")
    if cache_position.device != hidden_states.device or position_ids.device != hidden_states.device:
        raise ValueError("position tensors must be on the hidden-state device")
    if int(active_prefix_length) <= 0:
        raise ValueError("active_prefix_length must be positive")
    layer_idx = int(module.self_attn.layer_idx)
    if int(module.cross_attn.layer_idx) != layer_idx:
        raise ValueError("self/cross attention layer indices must match")
    for owner_name in ("self_attn_layer_norm", "cross_attn_layer_norm", "final_layer_norm"):
        weight = getattr(getattr(module, owner_name), "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.dtype != dtype:
            raise TypeError(f"{owner_name} weight must have dtype {dtype}")
    for owner, linear_name in (
        (module.self_attn, "Wqkv"),
        (module.self_attn, "Wo"),
        (module.cross_attn, "Wq"),
        (module.cross_attn, "Wo"),
        (module, "fc1"),
        (module, "fc2"),
    ):
        linear = getattr(owner, linear_name)
        if linear.weight.dtype != dtype:
            raise TypeError(f"{linear_name} weight must have dtype {dtype}")
    self_keys, self_values = _cache_tensors(past_key_value, "self", layer_idx)
    cross_keys, cross_values = _cache_tensors(past_key_value, "cross", layer_idx)
    for name, tensor in (
        ("self cache keys", self_keys),
        ("self cache values", self_values),
        ("cross cache keys", cross_keys),
        ("cross cache values", cross_values),
    ):
        if tensor.dtype != dtype or tensor.device != hidden_states.device:
            raise ValueError(f"{name} must share hidden-state dtype/device")
    if int(active_prefix_length) > int(self_keys.shape[2]):
        raise ValueError("active_prefix_length exceeds the self cache")
    updated = getattr(past_key_value, "is_updated", None)
    if not isinstance(updated, dict) or updated.get(layer_idx) is not True:
        raise ValueError("native-prefix scout requires a populated reusable cross cache")
    return dtype, layer_idx, self_keys, self_values, cross_keys, cross_values


def _trim_mask(attention_mask: torch.Tensor | None, active_prefix_length: int) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.shape[-1] < active_prefix_length:
        raise ValueError("attention mask is shorter than active_prefix_length")
    return attention_mask[..., :active_prefix_length]


def _mlp_tail(module: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    residual = hidden_states
    hidden_states = _norm(module.final_layer_norm, hidden_states)
    hidden_states = F.linear(hidden_states, module.fc1.weight, module.fc1.bias)
    hidden_states = module.activation_fn(hidden_states)
    hidden_states = F.dropout(
        hidden_states,
        p=float(module.activation_dropout),
        training=module.training,
    )
    hidden_states = F.linear(hidden_states, module.fc2.weight, module.fc2.bias)
    hidden_states = F.dropout(
        hidden_states,
        p=float(module.dropout),
        training=module.training,
    )
    return residual + hidden_states


def _prefix(
    module: torch.nn.Module,
    *,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    encoder_hidden_states: torch.Tensor,
    past_key_value: Any,
    cache_position: torch.Tensor,
    position_ids: torch.Tensor,
    active_prefix_length: int,
    self_attention_fn: Callable[..., torch.Tensor] | None,
    cross_attention_fn: Callable[..., torch.Tensor] | None,
) -> torch.Tensor:
    dtype, _, self_keys, self_values, cross_keys, cross_values = _validate_prefix_inputs(
        module,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        past_key_value=past_key_value,
        cache_position=cache_position,
        position_ids=position_ids,
        active_prefix_length=active_prefix_length,
    )
    self_attn = module.self_attn
    cross_attn = module.cross_attn
    active_mask = _trim_mask(attention_mask, active_prefix_length)

    residual = hidden_states
    normalized = _norm(module.self_attn_layer_norm, hidden_states)
    qkv_projection = F.linear(
        normalized,
        self_attn.Wqkv.weight,
        self_attn.Wqkv.bias,
    )
    qkv = qkv_projection.view(1, 1, 3, self_attn.num_heads, self_attn.head_dim)
    cos, sin = self_attn.rotary_emb(qkv, position_ids=position_ids)
    if self_attention_fn is not None:
        self_output = self_attention_fn(
            qkv,
            self_keys,
            self_values,
            cos,
            sin,
            cache_position,
            active_mask,
            active_prefix_length,
        )
    else:
        from ....model.custom_transformers.modeling_varwhisper import apply_rotary_pos_emb

        query, key, value = qkv.transpose(1, 3).unbind(dim=2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        self_keys.index_copy_(2, cache_position, key)
        self_values.index_copy_(2, cache_position, value)
        self_output = framework_q1_attention(
            query,
            self_keys[:, :, :active_prefix_length, :],
            self_values[:, :, :active_prefix_length, :],
            active_mask,
        )
    self_output = self_output.transpose(1, 2).contiguous().view(
        1, 1, self_attn.all_head_size
    )
    hidden_states = residual + F.linear(
        self_output,
        self_attn.Wo.weight,
        self_attn.Wo.bias,
    )

    residual = hidden_states
    normalized = _norm(module.cross_attn_layer_norm, hidden_states)
    query = F.linear(normalized, cross_attn.Wq.weight, cross_attn.Wq.bias)
    query = query.view(1, 1, cross_attn.num_heads, cross_attn.head_dim).transpose(1, 2)
    cross_output = (
        cross_attention_fn(query, cross_keys, cross_values, None)
        if cross_attention_fn is not None
        else framework_q1_attention(query, cross_keys, cross_values, None)
    )
    cross_output = cross_output.transpose(1, 2).contiguous().view(
        1, 1, cross_attn.all_head_size
    )
    hidden_states = residual + F.linear(
        cross_output,
        cross_attn.Wo.weight,
        cross_attn.Wo.bias,
    )
    output = _mlp_tail(module, hidden_states)
    if output.dtype != dtype:
        raise RuntimeError("native-prefix scout changed the storage dtype")
    return output


def framework_prefix(
    module: torch.nn.Module,
    *,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    encoder_hidden_states: torch.Tensor,
    past_key_value: Any,
    cache_position: torch.Tensor,
    position_ids: torch.Tensor,
    active_prefix_length: int,
) -> torch.Tensor:
    return _prefix(
        module,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        past_key_value=past_key_value,
        cache_position=cache_position,
        position_ids=position_ids,
        active_prefix_length=active_prefix_length,
        self_attention_fn=None,
        cross_attention_fn=None,
    )


def accepted_fp32_prefix(
    module: torch.nn.Module,
    *,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    encoder_hidden_states: torch.Tensor,
    past_key_value: Any,
    cache_position: torch.Tensor,
    position_ids: torch.Tensor,
    active_prefix_length: int,
) -> torch.Tensor:
    if hidden_states.dtype != torch.float32:
        raise TypeError("accepted component control requires FP32 storage")
    from ..kernels.q1_attention import native_q1_rope_cache_attention as accepted_self

    return _prefix(
        module,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        past_key_value=past_key_value,
        cache_position=cache_position,
        position_ids=position_ids,
        active_prefix_length=active_prefix_length,
        self_attention_fn=accepted_self,
        cross_attention_fn=None,
    )


def native_prefix(
    module: torch.nn.Module,
    *,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    encoder_hidden_states: torch.Tensor,
    past_key_value: Any,
    cache_position: torch.Tensor,
    position_ids: torch.Tensor,
    active_prefix_length: int,
) -> torch.Tensor:
    return _prefix(
        module,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        past_key_value=past_key_value,
        cache_position=cache_position,
        position_ids=position_ids,
        active_prefix_length=active_prefix_length,
        self_attention_fn=native_q1_rope_cache_attention,
        cross_attention_fn=None,
    )


def _hook_cross_attention(
    *,
    expected_dtype: torch.dtype,
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bs: int,
    dim: int,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor] | None:
    if not module.is_cross_attention or query.shape[-2] != 1:
        return None
    if module.training or bs != 1 or query.dtype != expected_dtype:
        raise ValueError("native-prefix hook requires eval-mode batch-1 q1 tensors of its configured dtype")
    output = native_q1_attention(query, key, value, attention_mask)
    output = output.transpose(1, 2).contiguous().view(bs, -1, dim)
    return (output,)


def _hook_self_rope_cache(
    *,
    expected_dtype: torch.dtype,
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    bs: int,
    is_varlen: bool,
    past_key_value: Any,
    cache_position: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    profile_ranges: bool,
    range_prefix: str,
    attention_mask: torch.Tensor | None,
    sliding_window_mask: torch.Tensor | None,
) -> tuple[torch.Tensor] | None:
    del profile_ranges, range_prefix
    from ..single.runtime_context import active_prefix_self_attention_length

    prefix_length = active_prefix_self_attention_length()
    if prefix_length is None or hidden_states.shape[1] != 1:
        return None
    if (
        module.training
        or is_varlen
        or bs != 1
        or hidden_states.dtype != expected_dtype
        or cache_position is None
        or position_ids is None
    ):
        raise ValueError("native-prefix self hook requires eval-mode batch-1 static q1 tensors of its configured dtype")
    layers = getattr(past_key_value, "layers", None)
    if not isinstance(layers, (list, tuple)) or module.layer_idx >= len(layers):
        raise ValueError("native-prefix self hook requires a static cache layer")
    cache_layer = layers[module.layer_idx]
    if not bool(getattr(cache_layer, "is_initialized", False)):
        raise ValueError("native-prefix self hook requires an initialized cache")
    qkv = F.linear(hidden_states, module.Wqkv.weight, module.Wqkv.bias).view(
        1, 1, 3, module.num_heads, module.head_dim
    )
    cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
    mask = sliding_window_mask if module.local_attention != (-1, -1) else attention_mask
    mask = _trim_mask(mask, prefix_length)
    output = native_q1_rope_cache_attention(
        qkv,
        cache_layer.keys,
        cache_layer.values,
        cos,
        sin,
        cache_position,
        mask,
        prefix_length,
    )
    output = output.transpose(1, 2).contiguous().view(bs, -1, module.all_head_size)
    return (output,)


@contextmanager
def native_prefix_attention_context(*, dtype: torch.dtype) -> Iterator[None]:
    """Install verifier-only full-model native q1 hooks for one storage dtype."""

    _require_storage_dtype(dtype)
    preload_native_q1_attention()
    previous = attention_runtime_hooks()
    hooks = AttentionRuntimeHooks(
        sdpa_attention_inputs=previous.sdpa_attention_inputs,
        sdpa_attention_forward=partial(_hook_cross_attention, expected_dtype=dtype),
        q1_rope_cache_self_attention_forward=partial(
            _hook_self_rope_cache,
            expected_dtype=dtype,
        ),
    )
    with attention_runtime_hooks_context(hooks):
        yield


@contextmanager
def specialized_prefix_attention_context(
    *,
    dtype: torch.dtype,
    native_self: bool,
) -> Iterator[None]:
    """Install the accepted attention policy without replacing decoder layers.

    FP32 reuses the production q1 kernels. FP16 uses their scout equivalents.
    Both dtypes retain the framework q1 BMM cross-attention path; the slower
    native cross-attention control is intentionally not part of this policy.
    """

    _require_storage_dtype(dtype)
    from ..kernels.dispatch import (
        q1_rope_cache_self_attention_forward,
        sdpa_q1_attention_forward,
    )

    selected_q1 = None
    if native_self:
        if dtype == torch.float32:
            from ..kernels import q1_attention as selected_q1
        else:
            from . import q1_attention as selected_q1
        selected_q1.preload_native_q1_attention()
    previous = attention_runtime_hooks()
    hooks = AttentionRuntimeHooks(
        sdpa_attention_inputs=previous.sdpa_attention_inputs,
        sdpa_attention_forward=partial(
            sdpa_q1_attention_forward,
            q1_bmm_cross_attention=True,
            native_q1_self_attention=native_self,
            native_q1_attention=(
                selected_q1.native_q1_attention
                if selected_q1 is not None
                else None
            ),
            expected_dtype=dtype,
        ),
        q1_rope_cache_self_attention_forward=partial(
            q1_rope_cache_self_attention_forward,
            native_q1_rope_cache_attention=(
                selected_q1.native_q1_rope_cache_attention
                if selected_q1 is not None
                else None
            ),
            expected_dtype=dtype,
        ) if native_self else None,
    )
    with attention_runtime_hooks_context(hooks):
        yield


__all__ = [
    "SUPPORTED_DTYPES",
    "accepted_fp32_prefix",
    "fp32_rms_norm",
    "framework_prefix",
    "framework_q1_attention",
    "native_prefix",
    "native_prefix_attention_context",
    "specialized_prefix_attention_context",
]
