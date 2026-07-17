"""§38 TIER2 fused decoder-step (true 7-stage layer path).

Seven stages per decoder layer (launch-collapse target):
  1. norm + Wqkv
  2. q1 self-attn
  3. Wo + residual
  4. cross block (norm + Wq/Wkv + attn + Wo + residual)
  5. fc1 (+ GELU; final norm fused into this stage)
  6. fc2 + residual
  7. glue / dtype restore

Numerics follow tip native CUDA style — storage-dtype GEMMs with fp32
accumulate on reductions (RMSNorm variance / residual add) — not a blanket
``Linear.float()`` wrap. Opt-in only via ``inference_engine=turbo``.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


def rmsnorm_fp32(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """RMSNorm with fp32 variance accumulate; restore storage dtype."""
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    y = x_f * torch.rsqrt(var + eps)
    return (y * weight.float()).to(dtype=x.dtype)


def linear_storage(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Linear in storage dtype (TensorCore / tip cuBLAS path)."""
    return F.linear(x, weight, bias)


def linear_residual_fp32_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Linear (storage dtype) + residual add with fp32 accumulate."""
    out = F.linear(x, weight, bias)
    return (out.float() + residual.float()).to(dtype=residual.dtype)


def gelu_exact_fp32(x: torch.Tensor) -> torch.Tensor:
    """Exact GELU in fp32 (matches tip native ``erff`` path)."""
    x_f = x.float()
    return (0.5 * x_f * (1.0 + torch.erf(x_f * 0.7071067811865475))).to(dtype=x.dtype)


def _norm_eps(norm: nn.Module) -> float:
    eps = getattr(norm, "eps", None)
    if eps is None:
        eps = getattr(norm, "variance_epsilon", 1e-6)
    return float(eps)


def _one_token(hidden_states: torch.Tensor) -> bool:
    return (
        hidden_states.is_cuda
        and hidden_states.dim() == 3
        and hidden_states.shape[0] == 1
        and hidden_states.shape[1] == 1
        and hidden_states.dtype in (torch.float16, torch.float32)
    )


def stage1_norm_wqkv(
    hidden_states: torch.Tensor,
    *,
    norm: nn.Module,
    wqkv: nn.Linear,
) -> torch.Tensor:
    """Stage 1: RMSNorm + Wqkv.

    One-token: tip native fused kernel (fp32 reductions). Multi-token: module
    RMSNorm + storage-dtype Linear (tip HF path) so teacher-forced stays close.
    """
    if _one_token(hidden_states):
        try:
            from ..optimized.kernels.decoder_layer import native_one_token_rmsnorm_linear

            return native_one_token_rmsnorm_linear(
                hidden_states,
                norm.weight,
                wqkv.weight,
                wqkv.bias,
                eps=_norm_eps(norm),
                outputs_per_block=8,
            )
        except Exception:
            pass
    # Prefer module RMSNorm for multi-token tip agreement; fall back to fp32-reduce.
    if callable(norm):
        normed = norm(hidden_states)
    else:
        normed = rmsnorm_fp32(hidden_states, norm.weight, _norm_eps(norm))
    return linear_storage(normed, wqkv.weight, wqkv.bias)


def stage3_wo_residual(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    *,
    wo: nn.Linear,
    out_drop: nn.Module | None = None,
) -> torch.Tensor:
    """Stage 3: Wo + residual."""
    if _one_token(attn_out) and _one_token(residual):
        try:
            from ..optimized.kernels.decoder_layer import native_one_token_linear_residual

            return native_one_token_linear_residual(
                attn_out,
                residual,
                wo.weight,
                wo.bias,
                outputs_per_block=8,
            )
        except Exception:
            pass
    proj = linear_storage(attn_out, wo.weight, wo.bias)
    if out_drop is not None:
        proj = out_drop(proj)
    # Multi-token: tip-style residual add in storage dtype.
    return residual + proj


def stage5_fc1_gelu(
    hidden_states: torch.Tensor,
    *,
    norm: nn.Module,
    fc1: nn.Linear,
    activation_fn: Any,
) -> torch.Tensor:
    """Stage 5: final RMSNorm + fc1 + GELU."""
    normed = norm(hidden_states) if callable(norm) else rmsnorm_fp32(
        hidden_states, norm.weight, _norm_eps(norm)
    )
    projected = linear_storage(normed, fc1.weight, fc1.bias)
    # Multi-token: tip activation_fn. One-token native MLP uses exact erff.
    return activation_fn(projected)


def stage6_fc2_residual(
    activated: torch.Tensor,
    residual: torch.Tensor,
    *,
    fc2: nn.Linear,
) -> torch.Tensor:
    """Stage 6: fc2 + residual."""
    if _one_token(activated) and _one_token(residual):
        try:
            from ..optimized.kernels.decoder_layer import native_one_token_linear_residual

            return native_one_token_linear_residual(
                activated,
                residual,
                fc2.weight,
                fc2.bias,
                outputs_per_block=8,
            )
        except Exception:
            pass
    return residual + linear_storage(activated, fc2.weight, fc2.bias)


def stage7_glue(hidden_states: torch.Tensor, storage_dtype: torch.dtype) -> torch.Tensor:
    """Stage 7: dtype restore / contiguous glue."""
    if hidden_states.dtype != storage_dtype:
        hidden_states = hidden_states.to(dtype=storage_dtype)
    return hidden_states.contiguous()


def _mlp_stages(layer: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Stages 5–6 (and native 1-token pack when eligible)."""
    residual_mlp = hidden_states
    if (
        layer.dropout == 0
        and layer.activation_dropout == 0
        and _one_token(hidden_states)
    ):
        try:
            from ..optimized.kernels.decoder_layer import native_one_token_mlp_residual

            return native_one_token_mlp_residual(
                hidden_states,
                layer.final_layer_norm.weight,
                layer.fc1.weight,
                layer.fc1.bias,
                layer.fc2.weight,
                layer.fc2.bias,
                eps=_norm_eps(layer.final_layer_norm),
                outputs_per_block=8,
            )
        except Exception:
            pass
    activated = stage5_fc1_gelu(
        residual_mlp,
        norm=layer.final_layer_norm,
        fc1=layer.fc1,
        activation_fn=layer.activation_fn,
    )
    if layer.activation_dropout:
        activated = F.dropout(
            activated, p=layer.activation_dropout, training=layer.training
        )
    out = stage6_fc2_residual(activated, residual_mlp, fc2=layer.fc2)
    if layer.dropout:
        out = F.dropout(out, p=layer.dropout, training=layer.training)
    return out


def _try_native_q1_from_qkv(
    self_attn: nn.Module,
    qkv: torch.Tensor,
    *,
    past_key_value: Any,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor | None,
    position_ids: torch.Tensor | None,
) -> torch.Tensor | None:
    """Stage 2 native rope-cache q1 when StaticCache + prefix are live."""
    try:
        from transformers.cache_utils import EncoderDecoderCache, StaticCache

        from ..optimized.kernels.q1_attention import native_q1_rope_cache_attention
        from ..optimized.single.runtime_context import (
            active_prefix_self_attention_length,
        )

        cache = (
            past_key_value.self_attention_cache
            if isinstance(past_key_value, EncoderDecoderCache)
            else past_key_value
        )
        prefix_length = active_prefix_self_attention_length()
        if not (
            isinstance(cache, StaticCache)
            and isinstance(cache_position, torch.Tensor)
            and isinstance(position_ids, torch.Tensor)
            and prefix_length is not None
            and prefix_length > 0
        ):
            return None
        cache_layer = cache.layers[self_attn.layer_idx]
        if not getattr(cache_layer, "is_initialized", False):
            return None
        cos, sin = self_attn.rotary_emb(qkv, position_ids=position_ids)
        attn_mask = attention_mask
        if isinstance(attn_mask, torch.Tensor) and attn_mask.shape[-1] > prefix_length:
            attn_mask = attn_mask[..., :prefix_length]
        attn_heads = native_q1_rope_cache_attention(
            qkv,
            cache_layer.keys,
            cache_layer.values,
            cos,
            sin,
            cache_position,
            attn_mask,
            int(prefix_length),
        )
        return (
            attn_heads.transpose(1, 2)
            .contiguous()
            .view(qkv.shape[0], qkv.shape[1], self_attn.all_head_size)
        )
    except Exception:
        return None


def _self_attn_core_from_qkv(
    self_attn: nn.Module,
    qkv: torch.Tensor,
    *,
    past_key_value: Any,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    output_attentions: bool,
) -> tuple[torch.Tensor, tuple]:
    """Stage 2: RoPE + SDPA from precomputed QKV (no Wo)."""
    from ...model.custom_transformers.modeling_varwhisper import (
        VARWHISPER_ATTENTION_FUNCTION,
        apply_rotary_pos_emb,
    )

    bs = qkv.shape[0]
    query_states, key_states, value_states = qkv.transpose(1, 3).unbind(dim=2)
    cos, sin = self_attn.rotary_emb(query_states, position_ids=position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_for_update = getattr(
            past_key_value, "self_attention_cache", past_key_value
        )
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = cache_for_update.update(
            key_states, value_states, self_attn.layer_idx, cache_kwargs
        )

    attn_func = VARWHISPER_ATTENTION_FUNCTION[self_attn.config._attn_implementation]
    attn_outputs = attn_func(
        module=self_attn,
        query=query_states,
        key=key_states,
        value=value_states,
        attention_mask=attention_mask,
        local_attention=self_attn.local_attention,
        sliding_window_mask=None,
        bs=bs,
        dim=self_attn.all_head_size,
        cu_seqlens=None,
        max_seqlen=None,
        output_attentions=output_attentions,
    )
    attn_hidden, *rest = attn_outputs
    # Match VarWhisperAttention.forward: append cache handle after attn extras.
    return attn_hidden, tuple(rest) + (past_key_value,)


def _cross_block(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    *,
    encoder_hidden_states: torch.Tensor,
    past_key_value: Any,
    position_ids: torch.Tensor | None,
    output_attentions: bool,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: int | None,
    encoder_cu_seqlens: torch.Tensor | None,
    encoder_max_seqlen: int | None,
) -> tuple[torch.Tensor, tuple]:
    """Stage 4: full cross block."""
    residual = hidden_states
    cross = layer.cross_attn

    if (
        _one_token(hidden_states)
        and past_key_value is not None
        and cu_seqlens is None
        and not output_attentions
        and layer.dropout == 0
    ):
        try:
            from transformers.cache_utils import EncoderDecoderCache

            from ..optimized.kernels.decoder_layer import (
                native_one_token_linear_residual,
                native_one_token_rmsnorm_linear,
            )
            from ..optimized.kernels.dispatch import _q1_bmm_cross_attention

            if isinstance(past_key_value, EncoderDecoderCache):
                layer_idx = cross.layer_idx
                cache_layer = past_key_value.cross_attention_cache.layers[layer_idx]
                if bool(past_key_value.is_updated.get(layer_idx)) and getattr(
                    cache_layer, "is_initialized", False
                ):
                    query_states = (
                        native_one_token_rmsnorm_linear(
                            hidden_states,
                            layer.cross_attn_layer_norm.weight,
                            cross.Wq.weight,
                            cross.Wq.bias,
                            eps=_norm_eps(layer.cross_attn_layer_norm),
                            outputs_per_block=8,
                        )
                        .view(1, 1, cross.num_heads, cross.head_dim)
                        .transpose(1, 2)
                    )
                    cross_out = _q1_bmm_cross_attention(
                        query_states,
                        cache_layer.keys,
                        cache_layer.values,
                        expected_dtype=hidden_states.dtype,
                    )
                    hidden = native_one_token_linear_residual(
                        cross_out,
                        residual,
                        cross.Wo.weight,
                        cross.Wo.bias,
                        outputs_per_block=8,
                    )
                    # Match tip native_cross_mlp_tail / module cross return tail.
                    return hidden, (past_key_value,)
        except Exception:
            pass

    normed = layer.cross_attn_layer_norm(hidden_states)
    cross_outputs = cross(
        hidden_states=normed,
        key_value_states=encoder_hidden_states,
        past_key_value=past_key_value,
        position_ids=position_ids,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        cu_seqlens_k=encoder_cu_seqlens,
        max_seqlen_k=encoder_max_seqlen,
        output_attentions=output_attentions,
    )
    hidden = residual + cross_outputs[0]
    return hidden, tuple(cross_outputs[1:])


def fused_decoder_layer_forward(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    encoder_hidden_states: torch.Tensor | None = None,
    past_key_value: Any = None,
    cache_position: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    encoder_cu_seqlens: torch.Tensor | None = None,
    encoder_max_seqlen: int | None = None,
    output_attentions: bool = False,
) -> tuple:
    """True 7-stage fused decoder layer forward (turbo / TIER2 only)."""
    storage_dtype = hidden_states.dtype
    self_attn = layer.self_attn
    residual_self = hidden_states

    # Flash-attn / varlen: keep module self-attn (includes Wo) then continue 4–7.
    if self_attn.config._attn_implementation == "flash_attention_2" or cu_seqlens is not None:
        normed = layer.self_attn_layer_norm(residual_self)
        self_attn_outputs = self_attn(
            hidden_states=normed,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            output_attentions=output_attentions,
        )
        hidden_states = residual_self + self_attn_outputs[0]
        attn_rest = tuple(self_attn_outputs[1:])
    else:
        # Stage 1: norm + Wqkv
        qkv_proj = stage1_norm_wqkv(
            residual_self,
            norm=layer.self_attn_layer_norm,
            wqkv=self_attn.Wqkv,
        )
        qkv = qkv_proj.view(
            residual_self.shape[0],
            residual_self.shape[1],
            3,
            self_attn.num_heads,
            self_attn.head_dim,
        )

        # Stage 2: q1
        attn_hidden = None
        attn_rest: tuple = ()
        if (
            _one_token(residual_self)
            and past_key_value is not None
            and not output_attentions
        ):
            attn_hidden = _try_native_q1_from_qkv(
                self_attn,
                qkv,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                cache_position=cache_position,
                position_ids=position_ids,
            )
            if attn_hidden is not None:
                attn_rest = (past_key_value,)
        if attn_hidden is None:
            attn_hidden, attn_rest = _self_attn_core_from_qkv(
                self_attn,
                qkv,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                cache_position=cache_position,
                position_ids=position_ids,
                output_attentions=bool(output_attentions),
            )

        # Stage 3: Wo + residual
        hidden_states = stage3_wo_residual(
            attn_hidden,
            residual_self,
            wo=self_attn.Wo,
            out_drop=getattr(self_attn, "out_drop", None),
        )

    # Stage 4: cross block
    cross_rest: tuple = ()
    if encoder_hidden_states is not None:
        hidden_states, cross_rest = _cross_block(
            layer,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            past_key_value=past_key_value,
            position_ids=position_ids,
            output_attentions=bool(output_attentions),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            encoder_cu_seqlens=encoder_cu_seqlens,
            encoder_max_seqlen=encoder_max_seqlen,
        )

    # Stages 5–6: MLP; Stage 7: glue
    hidden_states = _mlp_stages(layer, hidden_states)
    hidden_states = stage7_glue(hidden_states, storage_dtype)
    return (hidden_states,) + attn_rest + cross_rest


def _make_bound_forward(layer: nn.Module):
    def _forward(self, *args, **kwargs):
        return fused_decoder_layer_forward(self, *args, **kwargs)

    return MethodType(_forward, layer)


@contextmanager
def install_tier2_fused_numerics(raw_model) -> Iterator[dict[str, int]]:
    """Install 7-stage fused decoder layer forwards (turbo only).

    Encoder / LM head stay tip-bitexact. Optimized engine code paths are not
    modified; this only patches decoder layer ``forward`` while the context is open.
    """
    decoder = None
    for path in (
        ("transformer", "model", "decoder"),
        ("model", "decoder"),
        ("decoder",),
    ):
        obj = raw_model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            decoder = obj
            break
    if decoder is None or not hasattr(decoder, "layers"):
        raise RuntimeError("tier2 fused install: decoder.layers not found on raw_model")

    layers = list(decoder.layers)
    saved: list[tuple[nn.Module, Any]] = []
    for layer in layers:
        saved.append((layer, layer.forward))
        layer.forward = _make_bound_forward(layer)

    meta = {
        "fused_layers": len(saved),
        "decoder_layers": len(layers),
        "accumulate": "fp32_reductions_storage_gemm",
        "stages": 7,
        "path": "seven_kernel_fused_layer",
        "blanket_linear_wrap": False,
    }
    try:
        yield meta
    finally:
        for layer, orig in saved:
            layer.forward = orig


def relative_logit_delta(
    teacher: torch.Tensor,
    candidate: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-position max relative |Δ| / max(|teacher|, eps) over vocab."""
    t = teacher.float()
    c = candidate.float()
    denom = t.abs().clamp_min(eps)
    rel = (c - t).abs() / denom
    if rel.dim() == 3:
        return rel.amax(dim=-1).reshape(-1)
    if rel.dim() == 2:
        return rel.amax(dim=-1)
    raise ValueError("expected logits rank 2 or 3")


def top1_agreement(teacher: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Boolean agreement of argmax over last dim; flattened positions."""
    t = teacher.float().argmax(dim=-1).reshape(-1)
    c = candidate.float().argmax(dim=-1).reshape(-1)
    return t == c
