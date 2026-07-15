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
from .q1_attention import (
    native_q1_rope_cache_attention,
    native_q1_rope_cache_attention_variant,
)
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
ACCEPTED_Q1_VARIANT = "accepted"
SPLIT_KV_Q1_VARIANT = "split_kv_8"
CROSS_ACCEPTED = "accepted"
CROSS_FP16_PACKED = "fp16_packed_projections"
CROSS_MODES = frozenset({CROSS_ACCEPTED, CROSS_FP16_PACKED})


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


def _record_weight_only_self_attention_dispatch(
    dispatch_counts: dict[str, int] | None,
    *,
    variant: str,
    prefix_length: int,
) -> None:
    """Record the effective native kernel selected inside the weight-owned hook."""

    if variant not in {ACCEPTED_Q1_VARIANT, SPLIT_KV_Q1_VARIANT}:
        raise RuntimeError(f"weight-only self-attention selected unknown q1 variant {variant!r}")
    if dispatch_counts is None:
        return
    dispatch_counts["weight_only_self_attention_block"] += 1
    dispatch_counts["native_q1_rope_cache_self_attention"] += 1
    if variant == SPLIT_KV_Q1_VARIANT:
        aggregate = "native_q1_rope_cache_self_attention_split_kv_8"
        prefix = f"{aggregate}_prefix_{prefix_length}"
        dispatch_counts[aggregate] = dispatch_counts.get(aggregate, 0) + 1
        dispatch_counts[prefix] = dispatch_counts.get(prefix, 0) + 1


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
    extension_allocated_bytes_delta: int
    extension_reserved_bytes_delta: int
    pack_seconds: float
    allocated_bytes_delta: int
    reserved_bytes_delta: int
    retained_fp32_source_weight_bytes: int
    packed_weight_bytes: int
    cross_mode: str
    cross_packs: dict[int, tuple[PackedLinear, PackedLinear]] | None
    int8_mlp_packs: dict[int, Any] | None = None
    int8_mlp_extension_init_seconds: float = 0.0
    int8_mlp_pack_seconds: float = 0.0
    int8_mlp_packed_weight_bytes: int = 0
    dp4a_self_qkv_packs: dict[int, Any] | None = None
    dp4a_extension_init_seconds: float = 0.0
    dp4a_pack_seconds: float = 0.0
    dp4a_packed_weight_bytes: int = 0

    @classmethod
    def initialize(
        cls,
        model: torch.nn.Module,
        *,
        cross_mode: str = CROSS_ACCEPTED,
        int8_mlp: bool = False,
        dp4a_self_qkv: bool = False,
    ) -> "ApproximateWeightOnlyState":
        if not isinstance(model, torch.nn.Module):
            raise TypeError("weight-only candidate owner must be a torch module")
        if cross_mode not in CROSS_MODES:
            raise ValueError(
                f"cross candidate mode must be one of {sorted(CROSS_MODES)}, "
                f"got {cross_mode!r}"
            )
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

        device = layers[0].self_attn.Wqkv.weight.device
        if device.type != "cuda":
            raise RuntimeError("weight-only candidate source model must be on CUDA")
        extension_allocated_before = torch.cuda.memory_allocated(device)
        extension_reserved_before = torch.cuda.memory_reserved(device)
        extension_start = time.perf_counter()
        from .decoder_layer import preload_native_decoder_layer
        from .q1_attention import preload_native_q1_attention

        preload_native_q1_attention()
        preload_native_decoder_layer()
        preload_weight_only_extension()
        int8_extension_seconds = 0.0
        if int8_mlp:
            from ..scout.int8_mlp import preload_int8_mlp_extension

            int8_extension_start = time.perf_counter()
            preload_int8_mlp_extension()
            torch.cuda.synchronize()
            int8_extension_seconds = time.perf_counter() - int8_extension_start
        dp4a_extension_seconds = 0.0
        if dp4a_self_qkv:
            from ..scout.int8_dp4a_projection import preload_dp4a_projection_extension

            dp4a_extension_start = time.perf_counter()
            preload_dp4a_projection_extension()
            torch.cuda.synchronize()
            dp4a_extension_seconds = time.perf_counter() - dp4a_extension_start
        torch.cuda.synchronize()
        extension_seconds = time.perf_counter() - extension_start
        extension_allocated_after = torch.cuda.memory_allocated(device)
        extension_reserved_after = torch.cuda.memory_reserved(device)

        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        pack_start = time.perf_counter()
        packs = {id(layer): DecoderWeightPack.from_layer(layer) for layer in layers}
        cross_packs = (
            {
                id(layer): (
                    PackedLinear.from_module(layer.cross_attn.Wq),
                    PackedLinear.from_module(layer.cross_attn.Wo),
                )
                for layer in layers
            }
            if cross_mode == CROSS_FP16_PACKED
            else None
        )
        projection_pack = PackedLinear.from_module(final_projection)
        int8_packs = None
        int8_pack_seconds = 0.0
        int8_packed_bytes = 0
        if int8_mlp:
            from ..scout.int8_mlp import Int8MlpPack

            int8_pack_start = time.perf_counter()
            int8_packs = {id(layer): Int8MlpPack.from_layer(layer) for layer in layers}
            torch.cuda.synchronize(device)
            int8_pack_seconds = time.perf_counter() - int8_pack_start
            int8_packed_bytes = sum(
                pack.memory_report()["packed_weight_bytes"]
                for pack in int8_packs.values()
            )
        dp4a_packs = None
        dp4a_pack_seconds = 0.0
        dp4a_packed_bytes = 0
        if dp4a_self_qkv:
            from ..scout.int8_mlp import Int8PackedLinear

            dp4a_pack_start = time.perf_counter()
            dp4a_packs = {
                id(layer): Int8PackedLinear.from_module(layer.self_attn.Wqkv)
                for layer in layers
            }
            torch.cuda.synchronize(device)
            dp4a_pack_seconds = time.perf_counter() - dp4a_pack_start
            dp4a_packed_bytes = sum(
                pack.packed_weight_bytes for pack in dp4a_packs.values()
            )
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
        if cross_packs is not None:
            for query_pack, output_pack in cross_packs.values():
                source_bytes += (
                    query_pack.source_weight_bytes + output_pack.source_weight_bytes
                )
                packed_bytes += (
                    query_pack.packed_weight_bytes + output_pack.packed_weight_bytes
                )
        return cls(
            owner_ref=weakref.ref(model),
            layer_packs=packs,
            final_norm=final_norm,
            final_projection=final_projection,
            final_projection_pack=projection_pack,
            extension_init_seconds=extension_seconds,
            extension_allocated_bytes_delta=(
                extension_allocated_after - extension_allocated_before
            ),
            extension_reserved_bytes_delta=(
                extension_reserved_after - extension_reserved_before
            ),
            pack_seconds=pack_seconds,
            allocated_bytes_delta=allocated_after - allocated_before,
            reserved_bytes_delta=reserved_after - reserved_before,
            retained_fp32_source_weight_bytes=source_bytes,
            packed_weight_bytes=(
                packed_bytes + int8_packed_bytes + dp4a_packed_bytes
            ),
            cross_mode=cross_mode,
            cross_packs=cross_packs,
            int8_mlp_packs=int8_packs,
            int8_mlp_extension_init_seconds=int8_extension_seconds,
            int8_mlp_pack_seconds=int8_pack_seconds,
            int8_mlp_packed_weight_bytes=int8_packed_bytes,
            dp4a_self_qkv_packs=dp4a_packs,
            dp4a_extension_init_seconds=dp4a_extension_seconds,
            dp4a_pack_seconds=dp4a_pack_seconds,
            dp4a_packed_weight_bytes=dp4a_packed_bytes,
        )

    def validate_owner(self, model: torch.nn.Module) -> None:
        if self.owner_ref() is not model:
            raise RuntimeError("weight-only packed weights belong to a different model")

    def pack_for_layer(self, layer: torch.nn.Module) -> DecoderWeightPack:
        try:
            return self.layer_packs[id(layer)]
        except KeyError as exc:
            raise RuntimeError("weight-only candidate received an unowned decoder layer") from exc

    def cross_pack_for_layer(
        self,
        layer: torch.nn.Module,
    ) -> tuple[PackedLinear, PackedLinear]:
        if self.cross_mode != CROSS_FP16_PACKED or self.cross_packs is None:
            raise RuntimeError("FP16 cross projections are not initialized")
        try:
            return self.cross_packs[id(layer)]
        except KeyError as exc:
            raise RuntimeError(
                "cross projection candidate received an unowned decoder layer"
            ) from exc

    @property
    def int8_mlp_enabled(self) -> bool:
        return self.int8_mlp_packs is not None

    def int8_mlp_pack_for_layer(self, layer: torch.nn.Module) -> Any:
        if self.int8_mlp_packs is None:
            raise RuntimeError("INT8 MLP overlay was not initialized")
        try:
            return self.int8_mlp_packs[id(layer)]
        except KeyError as exc:
            raise RuntimeError("INT8 MLP overlay received an unowned decoder layer") from exc

    @property
    def dp4a_self_qkv_enabled(self) -> bool:
        return self.dp4a_self_qkv_packs is not None

    def dp4a_self_qkv_pack_for_layer(self, layer: torch.nn.Module) -> Any:
        if self.dp4a_self_qkv_packs is None:
            raise RuntimeError("DP4A self-QKV overlay was not initialized")
        try:
            return self.dp4a_self_qkv_packs[id(layer)]
        except KeyError as exc:
            raise RuntimeError(
                "DP4A self-QKV overlay received an unowned decoder layer"
            ) from exc

    def metadata(self) -> dict[str, Any]:
        fp16_regions = [
            "self_qkv",
            "self_output",
            "mlp_fc1",
            "mlp_fc2",
            "final_logits",
        ]
        fp32_regions = ["cross_query", "cross_output"]
        cross_candidate = {
            "mode": self.cross_mode,
            "scope": "main-model-only",
            "attention_accumulation": "fp32",
            "production_selector_unchanged": True,
        }
        if self.cross_mode == CROSS_FP16_PACKED:
            fp16_regions.extend(("cross_query", "cross_output"))
            fp32_regions = []
            cross_candidate.update(
                {
                    "result_class": "documented-drift",
                    "exactness_claim": False,
                    "packed_projection_weights": True,
                    "projection_delta_only": True,
                    "accepted_q1_bmm": True,
                    "incremental_exactness_required": True,
                }
            )
        else:
            cross_candidate.update(
                {
                    "result_class": "control",
                    "exactness_claim": False,
                    "accepted_q1_bmm": True,
                }
            )
        metadata = {
            "version": "approximate-fp16-weights-fp32-state-v2",
            "result_class": "documented-drift",
            "exactness_claim": False,
            "fp32_activations_caches_reductions_logits": True,
            "fp16_weight_regions": fp16_regions,
            "fp32_selected_decode_matrix_regions": fp32_regions,
            "cross_candidate": cross_candidate,
            "outputs_per_block": {
                "self_qkv": SELF_QKV_OUTPUTS_PER_BLOCK,
                "self_output": SELF_OUT_OUTPUTS_PER_BLOCK,
                "mlp_fc1": MLP_FC1_OUTPUTS_PER_BLOCK,
                "mlp_fc2": MLP_FC2_OUTPUTS_PER_BLOCK,
                "final_logits": FINAL_LOGITS_OUTPUTS_PER_BLOCK,
            },
            "extension_init_seconds": self.extension_init_seconds,
            "extension_allocated_bytes_delta": self.extension_allocated_bytes_delta,
            "extension_reserved_bytes_delta": self.extension_reserved_bytes_delta,
            "weight_pack_seconds": self.pack_seconds,
            "weight_pack_allocated_bytes_delta": self.allocated_bytes_delta,
            "weight_pack_reserved_bytes_delta": self.reserved_bytes_delta,
            "allocated_bytes_delta": self.allocated_bytes_delta,
            "reserved_bytes_delta": self.reserved_bytes_delta,
            "retained_fp32_source_weight_bytes": self.retained_fp32_source_weight_bytes,
            "packed_weight_bytes": self.packed_weight_bytes,
        }
        if self.int8_mlp_enabled:
            metadata["int8_mlp_overlay"] = {
                "version": "per-row-symmetric-int8-mlp-v1",
                "result_class": "documented-drift",
                "exactness_claim": False,
                "scope": "main-model-decoder-mlp-only",
                "composition_control": "k4-split-kv-mixed-weight-shared-rope-v1",
                "replaces_effective_regions": ["mlp_fc1", "mlp_fc2"],
                "retained_unused_fp16_regions": ["mlp_fc1", "mlp_fc2"],
                "fp32_activations_norm_bias_reductions_residual_outputs": True,
                "quantization": "symmetric-per-output-row",
                "dispatch_counter": "int8_weight_mlp_tail",
                "extension_init_seconds": self.int8_mlp_extension_init_seconds,
                "weight_pack_seconds": self.int8_mlp_pack_seconds,
                "packed_weight_bytes": self.int8_mlp_packed_weight_bytes,
            }
        if self.dp4a_self_qkv_enabled:
            metadata["dp4a_self_qkv_overlay"] = {
                "version": "sm75-dynamic-activation-int8-dp4a-self-qkv-v1",
                "result_class": "documented-drift",
                "exactness_claim": False,
                "scope": "main-model-decoder-self-qkv-only",
                "replaces_effective_regions": ["self_qkv"],
                "retained_unused_fp16_regions": ["self_qkv"],
                "activation_quantization": "dynamic-per-token-symmetric-int8",
                "weight_quantization": "symmetric-per-output-row-int8",
                "accumulation": "signed-dp4a-int32",
                "output_storage": "fp32",
                "active_warps": 8,
                "persistent_ctas": 68,
                "dispatch_counter": "dp4a_self_qkv_projection",
                "extension_init_seconds": self.dp4a_extension_init_seconds,
                "weight_pack_seconds": self.dp4a_pack_seconds,
                "packed_weight_bytes": self.dp4a_packed_weight_bytes,
            }
        return metadata


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
) -> tuple[torch.Tensor, tuple[Any, ...]] | None:
    del max_seqlen
    if hidden_states.shape[:2] != (1, 1):
        return None
    _require_fp32_cuda(hidden_states, name="hidden states")
    if (
        module.training
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
        if state.dp4a_self_qkv_enabled:
            from ..scout.int8_dp4a_projection import dp4a_rmsnorm_linear

            qkv_projection = dp4a_rmsnorm_linear(
                hidden_states,
                module.self_attn_layer_norm.weight,
                state.dp4a_self_qkv_pack_for_layer(module),
                eps=_norm_eps(module.self_attn_layer_norm),
                active_warps=8,
                persistent_ctas=68,
            )
        else:
            qkv_projection = weight_only_rmsnorm_linear(
                hidden_states,
                module.self_attn_layer_norm.weight,
                pack.self_qkv,
                eps=_norm_eps(module.self_attn_layer_norm),
                outputs_per_block=SELF_QKV_OUTPUTS_PER_BLOCK,
            )
        qkv = qkv_projection.view(
            1, 1, 3, module.self_attn.num_heads, module.self_attn.head_dim
        )
    cos, sin = module.self_attn.rotary_emb(qkv, position_ids=position_ids)
    native_mask = attention_mask
    if isinstance(native_mask, torch.Tensor):
        _require_fp32_cuda(native_mask, name="self attention mask")
        if native_mask.shape[-1] > prefix_length:
            native_mask = native_mask[..., :prefix_length]
    native_q1_variant = native_q1_rope_cache_attention_variant(
        qkv,
        int(prefix_length),
    )
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
    _record_weight_only_self_attention_dispatch(
        dispatch_counts,
        variant=native_q1_variant,
        prefix_length=int(prefix_length),
    )
    if state.dp4a_self_qkv_enabled and dispatch_counts is not None:
        dispatch_counts["dp4a_self_qkv_projection"] = (
            dispatch_counts.get("dp4a_self_qkv_projection", 0) + 1
        )
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
) -> tuple[Any, ...] | None:
    if hidden_states.shape[:2] != (1, 1):
        return None
    _require_fp32_cuda(hidden_states, name="hidden states")
    if (
        encoder_hidden_states is None
        or module.training
        or module.activation_dropout != 0
        or module.dropout != 0
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
    value_states = _require_fp32_cuda(
        getattr(cache_layer, "values", None),
        name="cross cache values",
    )
    pack = state.pack_for_layer(module)

    if state.cross_mode == CROSS_FP16_PACKED:
        query_pack, output_pack = state.cross_pack_for_layer(module)
        with profile_range(f"{layer_name}.weight_only.cross_q_fp16_weight"):
            query_states = weight_only_rmsnorm_linear(
                hidden_states,
                module.cross_attn_layer_norm.weight,
                query_pack,
                eps=_norm_eps(module.cross_attn_layer_norm),
                outputs_per_block=8,
            ).view(
                1, 1, module.cross_attn.num_heads, module.cross_attn.head_dim
            ).transpose(1, 2)
    else:
        with profile_range(f"{layer_name}.weight_only.cross_q_fp32"):
            query_states = native_one_token_rmsnorm_linear(
                hidden_states,
                module.cross_attn_layer_norm.weight,
                module.cross_attn.Wq.weight,
                module.cross_attn.Wq.bias,
                eps=_norm_eps(module.cross_attn_layer_norm),
                outputs_per_block=8,
            ).view(
                1, 1, module.cross_attn.num_heads, module.cross_attn.head_dim
            ).transpose(1, 2)
    with profile_range(f"{layer_name}.weight_only.cross_bmm_fp32"):
        cross_output = _q1_bmm_cross_attention(
            query_states,
            key_states,
            value_states,
            expected_dtype=torch.float32,
        )
    cross_output = cross_output.transpose(1, 2).contiguous().view(
        1,
        1,
        module.cross_attn.all_head_size,
    )
    if state.cross_mode == CROSS_FP16_PACKED:
        with profile_range(f"{layer_name}.weight_only.cross_out_fp16_weight"):
            hidden_states = weight_only_linear_residual(
                cross_output,
                hidden_states,
                output_pack,
                outputs_per_block=8,
            )
    else:
        with profile_range(f"{layer_name}.weight_only.cross_out_fp32"):
            hidden_states = native_one_token_linear_residual(
                cross_output,
                hidden_states,
                module.cross_attn.Wo.weight,
                module.cross_attn.Wo.bias,
                outputs_per_block=8,
            )
    if state.int8_mlp_enabled:
        from ..scout.int8_mlp import int8_weight_mlp_residual

        with profile_range(f"{layer_name}.weight_only.int8_mlp"):
            hidden_states = int8_weight_mlp_residual(
                hidden_states,
                module.final_layer_norm.weight,
                state.int8_mlp_pack_for_layer(module),
                eps=_norm_eps(module.final_layer_norm),
                fc1_outputs_per_block=MLP_FC1_OUTPUTS_PER_BLOCK,
                fc2_outputs_per_block=MLP_FC2_OUTPUTS_PER_BLOCK,
            )
    else:
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
        if state.int8_mlp_enabled:
            dispatch_counts["int8_weight_mlp_tail"] += 1
        dispatch_counts["q1_bmm_cross_attention"] += 1
        if state.cross_mode == CROSS_FP16_PACKED:
            dispatch_counts["fp16_packed_cross_projection_candidate"] = (
                dispatch_counts.get("fp16_packed_cross_projection_candidate", 0) + 1
            )
    return (hidden_states,) + self_attn_outputs[1:] + (past_key_value,)


def _defer_final_norm(
    *,
    state: ApproximateWeightOnlyState,
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    output_hidden_states: bool,
) -> torch.Tensor | None:
    if module is not state.final_norm:
        raise RuntimeError("weight-only candidate received an unowned final norm")
    if hidden_states.shape[:2] != (1, 1):
        return None
    if output_hidden_states:
        raise RuntimeError(
            "weight-only fused final projection does not support hidden-state output"
        )
    _require_fp32_cuda(hidden_states, name="decoder final hidden states")
    return hidden_states


def _final_projection_forward(
    *,
    state: ApproximateWeightOnlyState,
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    dispatch_counts: dict[str, int] | None,
) -> torch.Tensor | None:
    if module is not state.final_projection:
        raise RuntimeError("weight-only candidate received an unowned output projection")
    if hidden_states.shape[:2] != (1, 1):
        return None
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
    "CROSS_ACCEPTED",
    "CROSS_FP16_PACKED",
    "CROSS_MODES",
    "approximate_weight_only_runtime_context",
]
