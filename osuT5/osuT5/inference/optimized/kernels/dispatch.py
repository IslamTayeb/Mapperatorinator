"""Eligibility and orchestration for optimized q_len=1 attention paths."""

from __future__ import annotations

from typing import Callable

import torch
from transformers.cache_utils import StaticCache

from ....runtime_profiling import (
    active_prefix_self_attention_length,
    profile_range,
)


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
) -> tuple[torch.Tensor] | None:
    use_q1_bmm_cross_attention = (
        q1_bmm_cross_attention
        and module.is_cross_attention
        and not module.training
        and query.dtype == torch.float32
        and key.dtype == torch.float32
        and value.dtype == torch.float32
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
        and query.dtype == torch.float32
        and key.dtype == torch.float32
        and value.dtype == torch.float32
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
    elif use_q1_bmm_cross_attention:
        num_heads = query.shape[1]
        head_dim = query.shape[-1]
        q = query.reshape(num_heads, 1, head_dim)
        k = key.reshape(num_heads, -1, head_dim)
        v = value.reshape(num_heads, -1, head_dim)
        scores = torch.bmm(q, k.transpose(1, 2)) * (head_dim ** -0.5)
        attn_output = torch.bmm(
            torch.softmax(scores, dim=-1),
            v,
        ).view(1, num_heads, 1, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
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
) -> tuple[torch.Tensor] | None:
    prefix_length = active_prefix_self_attention_length()
    use_native_q1_rope_cache_self_attention = (
        native_q1_rope_cache_attention is not None
        and not is_varlen
        and not module.training
        and hidden_states.dtype == torch.float32
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

    if profile_ranges:
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
    return (output,)
