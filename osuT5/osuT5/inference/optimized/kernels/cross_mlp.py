"""Accepted dtype-aware native cross-attention and MLP tail dispatch."""

from __future__ import annotations

from typing import Any, Callable

import torch
from transformers.activations import GELUActivation
from transformers.cache_utils import EncoderDecoderCache

from ....runtime_profiling import profile_range
from .decoder_layer import (
    native_one_token_linear_residual,
    native_one_token_mlp_residual,
    native_one_token_rmsnorm_linear,
)
from .dispatch import _q1_bmm_cross_attention


CompiledCrossBmm = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


_SUPPORTED_DTYPES = (torch.float32, torch.float16)


def _norm_eps(module: torch.nn.Module, dtype: torch.dtype) -> float:
    eps = getattr(module, "eps", None)
    return float(torch.finfo(dtype).eps if eps is None else eps)


def _require_matching_tensor(
    name: str,
    value: Any,
    reference: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype != reference.dtype:
        raise TypeError(
            f"{name} dtype {value.dtype} does not match hidden dtype "
            f"{reference.dtype}"
        )
    if value.device != reference.device:
        raise ValueError(
            f"{name} device {value.device} does not match hidden device "
            f"{reference.device}"
        )
    return value


def _validate_cross_mlp_tensors(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    cache_layer: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden_states.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "native cross+MLP tail supports only float32 or float16 storage"
        )

    key_states = _require_matching_tensor(
        "cross cache keys",
        getattr(cache_layer, "keys", None),
        hidden_states,
    )
    value_states = _require_matching_tensor(
        "cross cache values",
        getattr(cache_layer, "values", None),
        hidden_states,
    )
    tensors = (
        ("cross RMSNorm weight", module.cross_attn_layer_norm.weight),
        ("cross query weight", module.cross_attn.Wq.weight),
        ("cross query bias", module.cross_attn.Wq.bias),
        ("cross output weight", module.cross_attn.Wo.weight),
        ("cross output bias", module.cross_attn.Wo.bias),
        ("MLP RMSNorm weight", module.final_layer_norm.weight),
        ("MLP FC1 weight", module.fc1.weight),
        ("MLP FC1 bias", module.fc1.bias),
        ("MLP FC2 weight", module.fc2.weight),
        ("MLP FC2 bias", module.fc2.bias),
    )
    for name, tensor in tensors:
        if tensor is not None:
            _require_matching_tensor(name, tensor, hidden_states)
    return key_states, value_states


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
    dispatch_counts: dict[str, int] | None = None,
    compiled_cross_bmm: CompiledCrossBmm | None = None,
) -> tuple[Any, ...] | None:
    """Run the dtype-aware native cross+MLP island when decode inputs qualify."""

    if (
        encoder_hidden_states is None
        or module.training
        or module.activation_dropout != 0
        or module.dropout != 0
        or hidden_states.device.type != "cuda"
        or hidden_states.shape[:2] != (1, 1)
        or not isinstance(getattr(module, "activation_fn", None), GELUActivation)
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
    key_states, value_states = _validate_cross_mlp_tensors(
        module,
        hidden_states,
        cache_layer,
    )

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

    with profile_range(f"{layer_name}.native_cross_mlp_tail.cross_bmm"):
        if compiled_cross_bmm is not None:
            cross_attention_output = compiled_cross_bmm(
                query_states,
                key_states,
                value_states,
            )
        else:
            cross_attention_output = _q1_bmm_cross_attention(
                query_states,
                key_states,
                value_states,
                expected_dtype=hidden_states.dtype,
            )

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

    if dispatch_counts is not None:
        dispatch_counts["native_cross_mlp_tail"] += 1
        dispatch_counts["q1_bmm_cross_attention"] += 1
        if compiled_cross_bmm is not None:
            dispatch_counts["compiled_q1_bmm_cross_attention"] = (
                dispatch_counts.get("compiled_q1_bmm_cross_attention", 0) + 1
            )
    return (hidden_states,) + self_attn_outputs[1:] + (past_key_value,)


__all__ = ["native_cross_mlp_tail_forward"]
