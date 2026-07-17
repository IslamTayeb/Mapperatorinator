"""§54 Stage A: turbo-only m≤8 verify path around SDPA.

Arms m-row warp-group cross/MLP island + rope/KV-cache-write fusion.
Self-attention core stays SDPA (Stage B owns fused split-KV attention).
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from functools import partial
from typing import Any, Iterator

import torch
from transformers.activations import GELUActivation
from transformers.cache_utils import EncoderDecoderCache, StaticCache

from ...runtime_profiling import profile_range
from ..optimized.kernels.m_row import (
    MAX_M,
    native_m_row_linear_residual,
    native_m_row_mlp_residual,
    native_m_row_rmsnorm_linear,
    native_m_row_rope_cache_write,
    preload_native_m_row,
)
from ..optimized.single.runtime_context import active_prefix_self_attention_length


def mrow_verify_enabled() -> bool:
    raw = os.environ.get("MAPPERATORINATOR_TURBO_VERIFY_MROW", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _norm_eps(module: torch.nn.Module, dtype: torch.dtype) -> float:
    eps = getattr(module, "eps", None)
    return float(torch.finfo(dtype).eps if eps is None else eps)


def _m_row_bmm_cross_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    expected_dtype: torch.dtype,
) -> torch.Tensor:
    """Batched cross-attn for query [1, H, m, D], key/value [1, H, L, D]."""
    if query.dim() != 4 or query.shape[0] != 1:
        raise ValueError("query must have shape [1, heads, m, head_dim]")
    m = int(query.shape[2])
    if m < 1 or m > MAX_M:
        raise ValueError(f"m-row cross attention supports m in [1,{MAX_M}]")
    num_heads = query.shape[1]
    head_dim = query.shape[-1]
    q = query.reshape(num_heads, m, head_dim)
    k = key.reshape(num_heads, -1, head_dim)
    v = value.reshape(num_heads, -1, head_dim)
    if expected_dtype == torch.float32:
        scores = torch.bmm(q, k.transpose(1, 2)) * (head_dim ** -0.5)
        output = torch.bmm(torch.softmax(scores, dim=-1), v)
    else:
        if query.device.type == "cuda":
            scores = torch.bmm(q, k.transpose(1, 2), out_dtype=torch.float32)
        else:
            scores = torch.bmm(q.float(), k.float().transpose(1, 2))
        scores.mul_(head_dim ** -0.5)
        output = torch.bmm(torch.softmax(scores, dim=-1), v.float()).to(
            dtype=torch.float16
        )
    return output.view(1, num_heads, m, head_dim)


def m_row_cross_mlp_tail_forward(
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
) -> tuple[Any, ...] | None:
    m = int(hidden_states.shape[1]) if hidden_states.ndim >= 2 else 0
    if (
        encoder_hidden_states is None
        or module.training
        or module.activation_dropout != 0
        or module.dropout != 0
        or hidden_states.device.type != "cuda"
        or hidden_states.shape[0] != 1
        or m < 1
        or m > MAX_M
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

    key_states = cache_layer.keys
    value_states = cache_layer.values
    outputs_per_block = 8
    with profile_range(f"{layer_name}.m_row_cross_mlp.cross_q"):
        query_states = native_m_row_rmsnorm_linear(
            hidden_states,
            module.cross_attn_layer_norm.weight,
            module.cross_attn.Wq.weight,
            module.cross_attn.Wq.bias,
            eps=_norm_eps(module.cross_attn_layer_norm, hidden_states.dtype),
            outputs_per_block=outputs_per_block,
        ).view(
            1,
            m,
            module.cross_attn.num_heads,
            module.cross_attn.head_dim,
        ).transpose(1, 2)

    with profile_range(f"{layer_name}.m_row_cross_mlp.cross_bmm"):
        cross_attention_output = _m_row_bmm_cross_attention(
            query_states,
            key_states,
            value_states,
            expected_dtype=hidden_states.dtype,
        ).transpose(1, 2).contiguous().view(1, m, -1)

    with profile_range(f"{layer_name}.m_row_cross_mlp.cross_out"):
        hidden_states = native_m_row_linear_residual(
            cross_attention_output,
            hidden_states,
            module.cross_attn.Wo.weight,
            module.cross_attn.Wo.bias,
            outputs_per_block=outputs_per_block,
        )

    with profile_range(f"{layer_name}.m_row_cross_mlp.mlp"):
        hidden_states = native_m_row_mlp_residual(
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
        dispatch_counts["m_row_cross_mlp_tail"] = (
            dispatch_counts.get("m_row_cross_mlp_tail", 0) + 1
        )
    return (hidden_states,) + self_attn_outputs[1:] + (past_key_value,)


def m_row_sdpa_self_attention_forward(
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
    expected_dtype: torch.dtype = torch.float16,
    dispatch_counts: dict[str, int] | None = None,
) -> tuple[torch.Tensor] | None:
    """Wqkv + fused rope/cache-write + SDPA core + Wo (m≤8)."""
    prefix_length = active_prefix_self_attention_length()
    m = int(hidden_states.shape[1]) if hidden_states.ndim >= 2 else 0
    # m==1 stays on aligned q1 hooks (TIER1a canary / greedy cert path).
    use = (
        not is_varlen
        and not module.training
        and hidden_states.dtype == expected_dtype
        and hidden_states.shape[0] == 1
        and 2 <= m <= MAX_M
        and isinstance(past_key_value, StaticCache)
        and isinstance(cache_position, torch.Tensor)
        and isinstance(position_ids, torch.Tensor)
        and int(cache_position.numel()) == m
        and prefix_length is not None
        and prefix_length > 0
        and module.local_attention == (-1, -1)
    )
    if not use:
        return None

    if profile_ranges:
        with profile_range(f"{range_prefix}.qkv_proj"):
            qkv = module.Wqkv(hidden_states).view(
                bs, m, 3, module.num_heads, module.head_dim
            )
    else:
        qkv = module.Wqkv(hidden_states).view(
            bs, m, 3, module.num_heads, module.head_dim
        )

    if profile_ranges:
        with profile_range(f"{range_prefix}.rope"):
            cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
    else:
        cos, sin = module.rotary_emb(qkv, position_ids=position_ids)

    cache_layer = past_key_value.layers[module.layer_idx]
    if not getattr(cache_layer, "is_initialized", False):
        raise RuntimeError("m-row rope/cache write requires initialized StaticCache")

    # Mark cache length so StaticCache.get_seq_length sees the write.
    # StaticCache.update normally sets this; we write tensors directly.
    if hasattr(cache_layer, "cumulative_length"):
        # Prefer bumping via update metadata if present.
        pass

    if profile_ranges:
        with profile_range(f"{range_prefix}.m_row_rope_cache_write"):
            query_states = native_m_row_rope_cache_write(
                qkv,
                cache_layer.keys,
                cache_layer.values,
                cos,
                sin,
                cache_position.view(-1),
            )
    else:
        query_states = native_m_row_rope_cache_write(
            qkv,
            cache_layer.keys,
            cache_layer.values,
            cos,
            sin,
            cache_position.view(-1),
        )

    # Active prefix already buckets past+m for verify; keep graph-safe trim.
    key_states = cache_layer.keys[:, :, :prefix_length, :]
    value_states = cache_layer.values[:, :, :prefix_length, :]

    attn_mask = attention_mask
    if (
        isinstance(attn_mask, torch.Tensor)
        and attn_mask.shape[-1] > key_states.shape[2]
    ):
        attn_mask = attn_mask[..., : key_states.shape[2]]

    from torch.nn.functional import scaled_dot_product_attention

    if profile_ranges:
        with profile_range(f"{range_prefix}.sdpa"):
            attn_output = scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )
    else:
        attn_output = scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
    # Match q1_rope_cache hook: return pre-Wo attn; modeling applies Wo.
    attn_output = attn_output.transpose(1, 2).contiguous().view(
        bs, m, module.all_head_size
    )

    if dispatch_counts is not None:
        dispatch_counts["m_row_sdpa_self_attention"] = (
            dispatch_counts.get("m_row_sdpa_self_attention", 0) + 1
        )
    return (attn_output,)


def _self_attn_dispatch(
    *,
    q1_fallback,
    m_row_fwd,
    **kwargs,
):
    """Prefer m-row for K>1; fall back to q1 rope-cache for sequential Q=1."""
    out = m_row_fwd(**kwargs)
    if out is not None:
        return out
    if q1_fallback is None:
        return None
    return q1_fallback(**kwargs)


def _cross_mlp_dispatch(
    *,
    q1_fallback,
    m_row_fwd,
    **kwargs,
):
    hidden = kwargs.get("hidden_states")
    m = int(hidden.shape[1]) if isinstance(hidden, torch.Tensor) and hidden.ndim >= 2 else 0
    if 2 <= m <= MAX_M:
        out = m_row_fwd(**kwargs)
        if out is not None:
            return out
    if q1_fallback is None:
        return None
    return q1_fallback(**kwargs)


@contextmanager
def mrow_verify_runtime_context(
    *,
    precision: str,
    dispatch_counts: dict[str, int] | None = None,
) -> Iterator[None]:
    """Arm Stage A m-row hooks for turbo teacher verify.

    Nested under the existing q1/cross-mlp runtime so greedy Q=1 canary keeps
    the aligned native path unchanged.
    """
    from ..optimized.single.runtime_context import (
        attention_runtime_context,
        attention_runtime_hooks,
        attention_runtime_hooks_context,
        decoder_layer_runtime_context,
    )
    from ..runtime_dispatch import (
        AttentionRuntimeHooks,
        DecoderLayerRuntimeHooks,
        decoder_layer_runtime_hooks,
        decoder_layer_runtime_hooks_context,
    )

    dtype = torch.float16 if precision == "fp16" else torch.float32
    preload_native_m_row()
    with attention_runtime_context(
        q1_bmm_cross_attention=True,
        native_q1_self_attention=True,
        native_q1_rope_cache_self_attention=True,
        expected_dtype=dtype,
        dispatch_counts=dispatch_counts,
    ), decoder_layer_runtime_context(
        native_cross_mlp_tail=True,
        dispatch_counts=dispatch_counts,
    ):
        prev_attn = attention_runtime_hooks()
        prev_layer = decoder_layer_runtime_hooks()
        self_fwd = partial(
            _self_attn_dispatch,
            q1_fallback=prev_attn.q1_rope_cache_self_attention_forward,
            m_row_fwd=partial(
                m_row_sdpa_self_attention_forward,
                expected_dtype=dtype,
                dispatch_counts=dispatch_counts,
            ),
        )
        cross_fwd = partial(
            _cross_mlp_dispatch,
            q1_fallback=prev_layer.cross_mlp_tail_forward,
            m_row_fwd=partial(
                m_row_cross_mlp_tail_forward,
                dispatch_counts=dispatch_counts,
            ),
        )
        attn_hooks = AttentionRuntimeHooks(
            sdpa_attention_inputs=prev_attn.sdpa_attention_inputs,
            sdpa_attention_forward=prev_attn.sdpa_attention_forward,
            q1_rope_cache_self_attention_forward=self_fwd,
        )
        layer_hooks = DecoderLayerRuntimeHooks(cross_mlp_tail_forward=cross_fwd)
        with attention_runtime_hooks_context(attn_hooks), decoder_layer_runtime_hooks_context(
            layer_hooks
        ):
            yield


__all__ = [
    "mrow_verify_enabled",
    "mrow_verify_runtime_context",
    "m_row_cross_mlp_tail_forward",
    "m_row_sdpa_self_attention_forward",
]
