"""Accepted FP32 native cross-attention and MLP tail dispatch."""

from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import EncoderDecoderCache

from ....runtime_profiling import profile_range
from .decoder_layer import (
    native_one_token_linear_residual,
    native_one_token_mlp_residual,
    native_one_token_rmsnorm_linear,
)


def _norm_eps(module: torch.nn.Module, dtype: torch.dtype) -> float:
    eps = getattr(module, "eps", None)
    return float(torch.finfo(dtype).eps if eps is None else eps)


def native_cross_mlp_tail_forward(
    *,
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None,
    past_key_value: Any,
    self_attn_outputs: tuple[Any, ...],
    output_attentions: bool,
    cu_seqlens: torch.Tensor | None,
    encoder_cu_seqlens: torch.Tensor | None,
    layer_name: str,
) -> tuple[Any, ...] | None:
    """Run the historical exact FP32 cross+MLP island when decode inputs qualify."""

    if (
        encoder_hidden_states is None
        or module.training
        or module.activation_dropout != 0
        or module.dropout != 0
        or hidden_states.device.type != "cuda"
        or hidden_states.dtype != torch.float32
        or hidden_states.shape[:2] != (1, 1)
        or not isinstance(past_key_value, EncoderDecoderCache)
        or output_attentions
        or cu_seqlens is not None
        or encoder_cu_seqlens is not None
    ):
        return None

    layer_idx = module.cross_attn.layer_idx
    cross_cache = past_key_value.cross_attention_cache
    cache_layer = cross_cache.layers[layer_idx]
    if not (
        bool(past_key_value.is_updated.get(layer_idx))
        and getattr(cache_layer, "is_initialized", False)
    ):
        return None

    outputs_per_block = 8
    with profile_range(f"{layer_name}.native_cross_mlp_tail.cross_q"):
        query_states = native_one_token_rmsnorm_linear(
            hidden_states,
            module.cross_attn_layer_norm.weight,
            module.cross_attn.Wq.weight,
            module.cross_attn.Wq.bias,
            eps=_norm_eps(module.cross_attn_layer_norm, hidden_states.dtype),
            outputs_per_block=outputs_per_block,
        ).view(
            1,
            1,
            module.cross_attn.num_heads,
            module.cross_attn.head_dim,
        ).transpose(1, 2)

    key_states, value_states = cache_layer.keys, cache_layer.values
    num_heads = query_states.shape[1]
    head_dim = query_states.shape[-1]
    q = query_states.reshape(num_heads, 1, head_dim)
    k = key_states.reshape(num_heads, -1, head_dim)
    v = value_states.reshape(num_heads, -1, head_dim)
    with profile_range(f"{layer_name}.native_cross_mlp_tail.cross_bmm"):
        scores = torch.bmm(q, k.transpose(1, 2)) * (head_dim ** -0.5)
        cross_attention_output = torch.bmm(
            torch.softmax(scores, dim=-1),
            v,
        ).view(1, num_heads, 1, head_dim)

    with profile_range(f"{layer_name}.native_cross_mlp_tail.cross_out"):
        hidden_states = native_one_token_linear_residual(
            cross_attention_output,
            hidden_states,
            module.cross_attn.Wo.weight,
            module.cross_attn.Wo.bias,
            outputs_per_block=outputs_per_block,
        )

    with profile_range(f"{layer_name}.native_cross_mlp_tail.mlp"):
        hidden_states = native_one_token_mlp_residual(
            hidden_states,
            module.final_layer_norm.weight,
            module.fc1.weight,
            module.fc1.bias,
            module.fc2.weight,
            module.fc2.bias,
            eps=_norm_eps(module.final_layer_norm, hidden_states.dtype),
            outputs_per_block=outputs_per_block,
        )

    return (hidden_states,) + self_attn_outputs[1:] + (past_key_value,)


__all__ = ["native_cross_mlp_tail_forward"]
