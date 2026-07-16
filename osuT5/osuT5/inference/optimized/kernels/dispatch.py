"""Eligibility and orchestration for optimized q_len=1 attention paths."""

from __future__ import annotations

from typing import Callable

import torch
from transformers.cache_utils import StaticCache

from ....runtime_profiling import profile_range
from ..single.runtime_context import (
    active_prefix_self_attention_length,
)


def _q1_bmm_cross_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    expected_dtype: torch.dtype,
) -> torch.Tensor:
    """Run the accepted q1 cross-attention math for one storage dtype."""

    if expected_dtype not in (torch.float32, torch.float16):
        raise TypeError("q1 BMM cross attention supports only float32 or float16")
    if not all(isinstance(tensor, torch.Tensor) for tensor in (query, key, value)):
        raise TypeError("q1 BMM query, key, and value must be tensors")
    if (
        query.dtype != expected_dtype
        or key.dtype != expected_dtype
        or value.dtype != expected_dtype
    ):
        raise TypeError(
            "q1 BMM query, key, and value must match the expected storage dtype"
        )
    if query.device != key.device or query.device != value.device:
        raise ValueError("q1 BMM query, key, and value must use one device")
    if query.dim() != 4 or query.shape[0] != 1 or query.shape[-2] != 1:
        raise ValueError("q1 BMM query must have shape [1, heads, 1, head_dim]")
    if key.dim() != 4 or value.dim() != 4 or key.shape != value.shape:
        raise ValueError("q1 BMM key and value must have matching 4D shapes")
    if (
        key.shape[0] != 1
        or key.shape[1] != query.shape[1]
        or key.shape[3] != query.shape[3]
        or key.shape[2] <= 0
    ):
        raise ValueError("q1 BMM key/value shapes must match the query heads")

    num_heads = query.shape[1]
    head_dim = query.shape[-1]
    q = query.reshape(num_heads, 1, head_dim)
    k = key.reshape(num_heads, -1, head_dim)
    v = value.reshape(num_heads, -1, head_dim)
    if expected_dtype == torch.float32:
        # Preserve the accepted FP32 calculation order exactly.
        scores = torch.bmm(q, k.transpose(1, 2)) * (head_dim ** -0.5)
        output = torch.bmm(
            torch.softmax(scores, dim=-1),
            v,
        )
    else:
        if query.device.type == "cuda":
            scores = torch.bmm(
                q,
                k.transpose(1, 2),
                out_dtype=torch.float32,
            )
        else:
            scores = torch.bmm(q.float(), k.float().transpose(1, 2))
        scores.mul_(head_dim ** -0.5)
        output = torch.bmm(
            torch.softmax(scores, dim=-1),
            v.float(),
        ).to(dtype=torch.float16)
    return output.view(1, num_heads, 1, head_dim)


def active_prefix_attention_inputs(
    *,
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Trim self-attention inputs without changing views or operation order."""
    prefix_length = active_prefix_self_attention_length()
    if (
        not module.is_cross_attention
        and prefix_length is not None
        and prefix_length > 0
        and key.shape[-2] > prefix_length
    ):
        key = key[:, :, :prefix_length, :]
        value = value[:, :, :prefix_length, :]
        if (
            isinstance(attention_mask, torch.Tensor)
            and attention_mask.shape[-1] > prefix_length
        ):
            attention_mask = attention_mask[..., :prefix_length]
    return query, key, value, attention_mask


def sdpa_q1_attention_forward(
    *,
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bs: int,
    dim: int,
    attention_mask: torch.Tensor | None,
    q1_bmm_cross_attention: bool,
    native_q1_self_attention: bool,
    native_q1_attention: Callable[..., torch.Tensor] | None,
    expected_dtype: torch.dtype = torch.float32,
    dispatch_counts: dict[str, int] | None = None,
) -> tuple[torch.Tensor] | None:
    use_q1_bmm_cross_attention = (
        q1_bmm_cross_attention
        and module.is_cross_attention
        and not module.training
        and query.dtype == expected_dtype
        and key.dtype == expected_dtype
        and value.dtype == expected_dtype
        and attention_mask is None
        and query.shape[0] == 1
        and query.shape[-2] == 1
        and key.shape[0] == 1
        and value.shape[0] == 1
    )
    use_native_q1_self_attention = (
        native_q1_self_attention
        and native_q1_attention is not None
        and not module.is_cross_attention
        and not module.training
        and query.dtype == expected_dtype
        and key.dtype == expected_dtype
        and value.dtype == expected_dtype
        and query.shape[0] == 1
        and query.shape[-2] == 1
        and key.shape[0] == 1
        and value.shape[0] == 1
    )
    if use_native_q1_self_attention:
        attn_output = native_q1_attention(
            query,
            key,
            value,
            attention_mask,
        ).transpose(1, 2).contiguous()
        if dispatch_counts is not None:
            dispatch_counts["native_q1_self_attention"] += 1
    elif use_q1_bmm_cross_attention:
        attn_output = _q1_bmm_cross_attention(
            query,
            key,
            value,
            expected_dtype=expected_dtype,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        if dispatch_counts is not None:
            dispatch_counts["q1_bmm_cross_attention"] += 1
    else:
        return None
    return (attn_output.view(bs, -1, dim),)


def q1_rope_cache_self_attention_forward(
    *,
    module,
    hidden_states: torch.Tensor,
    bs: int,
    is_varlen: bool,
    past_key_value,
    cache_position: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    profile_ranges: bool,
    range_prefix: str,
    attention_mask: torch.Tensor | None,
    sliding_window_mask: torch.Tensor | None,
    native_q1_rope_cache_attention: Callable[..., torch.Tensor] | None,
    expected_dtype: torch.dtype = torch.float32,
    dispatch_counts: dict[str, int] | None = None,
    compiled_wqkv=None,
) -> tuple[torch.Tensor] | None:
    prefix_length = active_prefix_self_attention_length()
    use_native_q1_rope_cache_self_attention = (
        native_q1_rope_cache_attention is not None
        and not is_varlen
        and not module.training
        and hidden_states.dtype == expected_dtype
        and hidden_states.shape[0] == 1
        and hidden_states.shape[1] == 1
        and isinstance(past_key_value, StaticCache)
        and isinstance(cache_position, torch.Tensor)
        and isinstance(position_ids, torch.Tensor)
        and prefix_length is not None
        and prefix_length > 0
    )
    if not use_native_q1_rope_cache_self_attention:
        return None

    if compiled_wqkv is not None:
        if profile_ranges:
            with profile_range(f"{range_prefix}.qkv_proj"):
                qkv = compiled_wqkv(
                    hidden_states,
                    module.Wqkv.weight,
                    module.Wqkv.bias,
                ).view(
                    bs,
                    -1,
                    3,
                    module.num_heads,
                    module.head_dim,
                )
        else:
            qkv = compiled_wqkv(
                hidden_states,
                module.Wqkv.weight,
                module.Wqkv.bias,
            ).view(
                bs,
                -1,
                3,
                module.num_heads,
                module.head_dim,
            )
        if dispatch_counts is not None:
            dispatch_counts["compiled_self_wqkv"] = (
                dispatch_counts.get("compiled_self_wqkv", 0) + 1
            )
    elif profile_ranges:
        with profile_range(f"{range_prefix}.qkv_proj"):
            qkv = module.Wqkv(hidden_states).view(
                bs,
                -1,
                3,
                module.num_heads,
                module.head_dim,
            )
    else:
        qkv = module.Wqkv(hidden_states).view(
            bs,
            -1,
            3,
            module.num_heads,
            module.head_dim,
        )

    if profile_ranges:
        with profile_range(f"{range_prefix}.rope"):
            cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
    else:
        cos, sin = module.rotary_emb(qkv, position_ids=position_ids)

    cache_layer = past_key_value.layers[module.layer_idx]
    if not getattr(cache_layer, "is_initialized", False):
        raise RuntimeError(
            "native q1 RoPE/cache self-attention requires an initialized "
            "StaticCache"
        )
    attention_mask_for_native = (
        sliding_window_mask
        if module.local_attention != (-1, -1)
        else attention_mask
    )
    if (
        isinstance(attention_mask_for_native, torch.Tensor)
        and attention_mask_for_native.shape[-1] > prefix_length
    ):
        attention_mask_for_native = attention_mask_for_native[..., :prefix_length]
    if profile_ranges:
        with profile_range(f"{range_prefix}.rope_cache_native_q1"):
            output = native_q1_rope_cache_attention(
                qkv,
                cache_layer.keys,
                cache_layer.values,
                cos,
                sin,
                cache_position,
                attention_mask_for_native,
                int(prefix_length),
            )
    else:
        output = native_q1_rope_cache_attention(
            qkv,
            cache_layer.keys,
            cache_layer.values,
            cos,
            sin,
            cache_position,
            attention_mask_for_native,
            int(prefix_length),
        )
    output = output.transpose(1, 2).contiguous()
    output = output.view(bs, -1, module.all_head_size)
    if dispatch_counts is not None:
        dispatch_counts["native_q1_rope_cache_self_attention"] += 1
    return (output,)
