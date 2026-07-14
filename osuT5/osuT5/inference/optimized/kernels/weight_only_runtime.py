"""Opt-in runtime for the measured mixed-weight FP32 decode candidate."""

from __future__ import annotations

import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Iterator

import torch
from transformers.activations import GELUActivation
from transformers.cache_utils import EncoderDecoderCache

from ....runtime_profiling import profile_range
from ...runtime_dispatch import (
    DecoderLayerRuntimeHooks,
    decoder_layer_runtime_hooks_context,
)
from ..single.runtime_context import active_prefix_self_attention_length
from .decoder_layer import (
    native_one_token_linear_residual,
    native_one_token_rmsnorm_linear,
)
from .dispatch import _q1_bmm_cross_attention
from .q1_attention import native_q1_rope_cache_attention
from .weight_only import (
    DecoderWeightPack,
    PackedLinear,
    preload_weight_only_extension,
    weight_only_linear_residual,
    weight_only_mlp_residual,
    weight_only_rmsnorm_linear,
)


SELF_QKV_OUTPUTS_PER_BLOCK = 4
SELF_OUT_OUTPUTS_PER_BLOCK = 4
MLP_FC1_OUTPUTS_PER_BLOCK = 2
MLP_FC2_OUTPUTS_PER_BLOCK = 4
FINAL_LOGITS_OUTPUTS_PER_BLOCK = 2


def _norm_eps(module: torch.nn.Module) -> float:
    eps = getattr(module, "eps", None)
    return float(torch.finfo(torch.float32).eps if eps is None else eps)


def _require_fp32_cuda(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must retain FP32 storage")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return value


def _locate_candidate_modules(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Module], torch.nn.Module, torch.nn.Module]:
    from ....model.custom_transformers.modeling_varwhisper import (
        VarWhisperDecoderLayer,
    )

    layers = [module for module in model.modules() if isinstance(module, VarWhisperDecoderLayer)]
    if len(layers) != 12:
        raise RuntimeError(
            "weight-only candidate requires exactly 12 VarWhisper decoder layers; "
            f"found {len(layers)}"
        )
    named_modules = dict(model.named_modules())
    layer_ids = {id(layer) for layer in layers}
    layer_names = [name for name, module in named_modules.items() if id(module) in layer_ids]
    decoder_parents = {
        name.rsplit(".layers.", 1)[0]
        for name in layer_names
        if ".layers." in name
    }
    if len(layer_names) != len(layers) or len(decoder_parents) != 1:
        raise RuntimeError("weight-only candidate could not locate one decoder parent")
    decoder = named_modules[next(iter(decoder_parents))]
    final_norm = getattr(decoder, "layer_norm", None)
    projections = {
        id(module): module
        for name, module in named_modules.items()
        if name == "proj_out" or name.endswith(".proj_out")
    }
    if not isinstance(final_norm, torch.nn.Module) or len(projections) != 1:
        raise RuntimeError(
            "weight-only candidate requires one decoder final norm and one proj_out"
        )
    return layers, final_norm, next(iter(projections.values()))


@dataclass(frozen=True)
class ApproximateWeightOnlyState:
    """Model-owned packed weights and explicit initialization evidence."""

    owner_ref: Any
    layer_packs: dict[int, DecoderWeightPack]
    final_norm: torch.nn.Module
    final_projection: torch.nn.Module
    final_projection_pack: PackedLinear
    extension_init_seconds: float
    pack_seconds: float
    allocated_bytes_delta: int
    reserved_bytes_delta: int
    retained_fp32_source_weight_bytes: int
    packed_weight_bytes: int

    @classmethod
    def initialize(cls, model: torch.nn.Module) -> "ApproximateWeightOnlyState":
        if not isinstance(model, torch.nn.Module):
            raise TypeError("weight-only candidate owner must be a torch module")
        if model.training:
            raise RuntimeError("weight-only candidate requires eval mode")
        layers, final_norm, final_projection = _locate_candidate_modules(model)
        for index, layer in enumerate(layers):
            if (
                layer.training
                or layer.dropout != 0
                or layer.activation_dropout != 0
                or not isinstance(layer.activation_fn, GELUActivation)
            ):
                raise RuntimeError(
                    f"weight-only candidate decoder layer {index} is not eligible"
                )
            required_fp32 = {
                "self norm": layer.self_attn_layer_norm.weight,
                "cross norm": layer.cross_attn_layer_norm.weight,
                "cross query": layer.cross_attn.Wq.weight,
                "cross output": layer.cross_attn.Wo.weight,
                "MLP norm": layer.final_layer_norm.weight,
            }
            optional_fp32 = {
                "cross query bias": layer.cross_attn.Wq.bias,
                "cross output bias": layer.cross_attn.Wo.bias,
            }
            for name, tensor in required_fp32.items():
                _require_fp32_cuda(tensor, name=f"layer {index} {name}")
            for name, tensor in optional_fp32.items():
                if tensor is not None:
                    _require_fp32_cuda(tensor, name=f"layer {index} {name}")
        _require_fp32_cuda(final_norm.weight, name="decoder final norm weight")
        _require_fp32_cuda(final_projection.weight, name="final projection weight")
        if getattr(final_projection, "bias", None) is not None:
            _require_fp32_cuda(final_projection.bias, name="final projection bias")

        extension_start = time.perf_counter()
        from .decoder_layer import preload_native_decoder_layer
        from .q1_attention import preload_native_q1_attention

        preload_native_q1_attention()
        preload_native_decoder_layer()
        preload_weight_only_extension()
        torch.cuda.synchronize()
        extension_seconds = time.perf_counter() - extension_start

        device = layers[0].self_attn.Wqkv.weight.device
        if device.type != "cuda":
            raise RuntimeError("weight-only candidate source model must be on CUDA")
        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        pack_start = time.perf_counter()
        packs = {id(layer): DecoderWeightPack.from_layer(layer) for layer in layers}
        projection_pack = PackedLinear.from_module(final_projection)
        torch.cuda.synchronize(device)
        pack_seconds = time.perf_counter() - pack_start
        allocated_after = torch.cuda.memory_allocated(device)
        reserved_after = torch.cuda.memory_reserved(device)

        source_bytes = projection_pack.source_weight_bytes
        packed_bytes = projection_pack.packed_weight_bytes
        for pack in packs.values():
            report = pack.memory_report()
            source_bytes += report["retained_fp32_source_weight_bytes"]
            packed_bytes += report["packed_weight_bytes"]
        return cls(
            owner_ref=weakref.ref(model),
            layer_packs=packs,
            final_norm=final_norm,
            final_projection=final_projection,
            final_projection_pack=projection_pack,
            extension_init_seconds=extension_seconds,
            pack_seconds=pack_seconds,
            allocated_bytes_delta=allocated_after - allocated_before,
            reserved_bytes_delta=reserved_after - reserved_before,
            retained_fp32_source_weight_bytes=source_bytes,
            packed_weight_bytes=packed_bytes,
        )

    def validate_owner(self, model: torch.nn.Module) -> None:
        if self.owner_ref() is not model:
            raise RuntimeError("weight-only packed weights belong to a different model")

    def pack_for_layer(self, layer: torch.nn.Module) -> DecoderWeightPack:
        try:
            return self.layer_packs[id(layer)]
        except KeyError as exc:
            raise RuntimeError("weight-only candidate received an unowned decoder layer") from exc

    def metadata(self) -> dict[str, Any]:
        return {
            "version": "approximate-fp16-weights-fp32-state-v1",
            "result_class": "documented-drift",
            "exactness_claim": False,
            "fp32_activations_caches_reductions_logits": True,
            "fp16_weight_regions": [
                "self_qkv",
                "self_output",
                "mlp_fc1",
                "mlp_fc2",
                "final_logits",
            ],
            "fp32_weight_regions": ["cross_query", "cross_output"],
            "outputs_per_block": {
                "self_qkv": SELF_QKV_OUTPUTS_PER_BLOCK,
                "self_output": SELF_OUT_OUTPUTS_PER_BLOCK,
                "mlp_fc1": MLP_FC1_OUTPUTS_PER_BLOCK,
                "mlp_fc2": MLP_FC2_OUTPUTS_PER_BLOCK,
                "final_logits": FINAL_LOGITS_OUTPUTS_PER_BLOCK,
            },
            "extension_init_seconds": self.extension_init_seconds,
            "weight_pack_seconds": self.pack_seconds,
            "allocated_bytes_delta": self.allocated_bytes_delta,
            "reserved_bytes_delta": self.reserved_bytes_delta,
            "retained_fp32_source_weight_bytes": self.retained_fp32_source_weight_bytes,
            "packed_weight_bytes": self.packed_weight_bytes,
        }


def _self_attention_block_forward(
    *,
    state: ApproximateWeightOnlyState,
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    past_key_value: Any,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: int | None,
    output_attentions: bool,
    layer_name: str,
    dispatch_counts: dict[str, int] | None,
) -> tuple[torch.Tensor, tuple[Any, ...]]:
    del max_seqlen
    _require_fp32_cuda(hidden_states, name="hidden states")
    if (
        module.training
        or hidden_states.shape[:2] != (1, 1)
        or not isinstance(past_key_value, EncoderDecoderCache)
        or not isinstance(cache_position, torch.Tensor)
        or not isinstance(position_ids, torch.Tensor)
        or cu_seqlens is not None
        or output_attentions
    ):
        raise RuntimeError("weight-only self-attention received unsupported decode inputs")
    if cache_position.dtype != torch.int64 or cache_position.numel() != 1:
        raise ValueError("weight-only self-attention requires one int64 cache position")
    prefix_length = active_prefix_self_attention_length()
    if prefix_length is None or prefix_length <= 0:
        raise RuntimeError("weight-only self-attention requires an active prefix")
    pack = state.pack_for_layer(module)
    self_cache = past_key_value.self_attention_cache
    layer_idx = module.self_attn.layer_idx
    cache_layer = self_cache.layers[layer_idx]
    if not getattr(cache_layer, "is_initialized", False):
        raise RuntimeError("weight-only self-attention requires initialized self cache")
    keys = _require_fp32_cuda(getattr(cache_layer, "keys", None), name="self cache keys")
    values = _require_fp32_cuda(getattr(cache_layer, "values", None), name="self cache values")

    with profile_range(f"{layer_name}.weight_only.self_qkv"):
        qkv = weight_only_rmsnorm_linear(
            hidden_states,
            module.self_attn_layer_norm.weight,
            pack.self_qkv,
            eps=_norm_eps(module.self_attn_layer_norm),
            outputs_per_block=SELF_QKV_OUTPUTS_PER_BLOCK,
        ).view(1, 1, 3, module.self_attn.num_heads, module.self_attn.head_dim)
    cos, sin = module.self_attn.rotary_emb(qkv, position_ids=position_ids)
    native_mask = attention_mask
    if isinstance(native_mask, torch.Tensor):
        _require_fp32_cuda(native_mask, name="self attention mask")
        if native_mask.shape[-1] > prefix_length:
            native_mask = native_mask[..., :prefix_length]
    with profile_range(f"{layer_name}.weight_only.self_attention"):
        self_output = native_q1_rope_cache_attention(
            qkv,
            keys,
            values,
            cos,
            sin,
            cache_position,
            native_mask,
            int(prefix_length),
        ).transpose(1, 2).contiguous().view(1, 1, module.self_attn.all_head_size)
    with profile_range(f"{layer_name}.weight_only.self_out"):
        output = weight_only_linear_residual(
            self_output,
            hidden_states,
            pack.self_out,
            outputs_per_block=SELF_OUT_OUTPUTS_PER_BLOCK,
        )
    if dispatch_counts is not None:
        dispatch_counts["weight_only_self_attention_block"] += 1
        dispatch_counts["native_q1_rope_cache_self_attention"] += 1
    return output, (self_output, self_cache)


def _cross_mlp_tail_forward(
    *,
    state: ApproximateWeightOnlyState,
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None,
    past_key_value: Any,
    self_attn_outputs: tuple[Any, ...],
    output_attentions: bool,
    cu_seqlens: torch.Tensor | None,
    encoder_cu_seqlens: torch.Tensor | None,
    layer_name: str,
    dispatch_counts: dict[str, int] | None,
) -> tuple[Any, ...]:
    _require_fp32_cuda(hidden_states, name="hidden states")
    if (
        encoder_hidden_states is None
        or module.training
        or module.activation_dropout != 0
        or module.dropout != 0
        or hidden_states.shape[:2] != (1, 1)
        or not isinstance(getattr(module, "activation_fn", None), GELUActivation)
        or not isinstance(past_key_value, EncoderDecoderCache)
        or output_attentions
        or cu_seqlens is not None
        or encoder_cu_seqlens is not None
    ):
        raise RuntimeError("weight-only cross+MLP received unsupported decode inputs")
    layer_idx = module.cross_attn.layer_idx
    cache_layer = past_key_value.cross_attention_cache.layers[layer_idx]
    if not (
        bool(past_key_value.is_updated.get(layer_idx))
        and getattr(cache_layer, "is_initialized", False)
    ):
        raise RuntimeError("weight-only cross+MLP requires initialized cross cache")
    key_states = _require_fp32_cuda(getattr(cache_layer, "keys", None), name="cross cache keys")
    value_states = _require_fp32_cuda(getattr(cache_layer, "values", None), name="cross cache values")
    pack = state.pack_for_layer(module)

    with profile_range(f"{layer_name}.weight_only.cross_q_fp32"):
        query_states = native_one_token_rmsnorm_linear(
            hidden_states,
            module.cross_attn_layer_norm.weight,
            module.cross_attn.Wq.weight,
            module.cross_attn.Wq.bias,
            eps=_norm_eps(module.cross_attn_layer_norm),
            outputs_per_block=8,
        ).view(1, 1, module.cross_attn.num_heads, module.cross_attn.head_dim).transpose(1, 2)
    with profile_range(f"{layer_name}.weight_only.cross_bmm_fp32"):
        cross_output = _q1_bmm_cross_attention(
            query_states,
            key_states,
            value_states,
            expected_dtype=torch.float32,
        )
    with profile_range(f"{layer_name}.weight_only.cross_out_fp32"):
        hidden_states = native_one_token_linear_residual(
            cross_output,
            hidden_states,
            module.cross_attn.Wo.weight,
            module.cross_attn.Wo.bias,
            outputs_per_block=8,
        )
    with profile_range(f"{layer_name}.weight_only.mlp"):
        hidden_states = weight_only_mlp_residual(
            hidden_states,
            module.final_layer_norm.weight,
            pack.fc1,
            pack.fc2,
            eps=_norm_eps(module.final_layer_norm),
            fc1_outputs_per_block=MLP_FC1_OUTPUTS_PER_BLOCK,
            fc2_outputs_per_block=MLP_FC2_OUTPUTS_PER_BLOCK,
        )
    if dispatch_counts is not None:
        dispatch_counts["weight_only_mlp_tail"] += 1
        dispatch_counts["q1_bmm_cross_attention"] += 1
    return (hidden_states,) + self_attn_outputs[1:] + (past_key_value,)


def _defer_final_norm(
    *,
    state: ApproximateWeightOnlyState,
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    output_hidden_states: bool,
) -> torch.Tensor:
    if module is not state.final_norm:
        raise RuntimeError("weight-only candidate received an unowned final norm")
    if output_hidden_states:
        raise RuntimeError("weight-only fused final projection does not support hidden-state output")
    _require_fp32_cuda(hidden_states, name="decoder final hidden states")
    return hidden_states


def _final_projection_forward(
    *,
    state: ApproximateWeightOnlyState,
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    dispatch_counts: dict[str, int] | None,
) -> torch.Tensor:
    if module is not state.final_projection:
        raise RuntimeError("weight-only candidate received an unowned output projection")
    logits = weight_only_rmsnorm_linear(
        hidden_states,
        state.final_norm.weight,
        state.final_projection_pack,
        eps=_norm_eps(state.final_norm),
        outputs_per_block=FINAL_LOGITS_OUTPUTS_PER_BLOCK,
    )
    if dispatch_counts is not None:
        dispatch_counts["weight_only_final_projection"] += 1
    return logits


@contextmanager
def approximate_weight_only_runtime_context(
    state: ApproximateWeightOnlyState,
    *,
    dispatch_counts: dict[str, int] | None = None,
) -> Iterator[None]:
    if not isinstance(state, ApproximateWeightOnlyState):
        raise TypeError("weight-only runtime context requires initialized state")
    hooks = DecoderLayerRuntimeHooks(
        self_attention_block_forward=partial(
            _self_attention_block_forward,
            state=state,
            dispatch_counts=dispatch_counts,
        ),
        cross_mlp_tail_forward=partial(
            _cross_mlp_tail_forward,
            state=state,
            dispatch_counts=dispatch_counts,
        ),
        decoder_final_norm_forward=partial(_defer_final_norm, state=state),
        output_projection_forward=partial(
            _final_projection_forward,
            state=state,
            dispatch_counts=dispatch_counts,
        ),
    )
    with decoder_layer_runtime_hooks_context(hooks):
        yield


__all__ = [
    "ApproximateWeightOnlyState",
    "approximate_weight_only_runtime_context",
]
