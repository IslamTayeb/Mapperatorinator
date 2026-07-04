from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from inference import compile_args, load_model_with_server, setup_inference_environment
from osuT5.osuT5.inference.cache_utils import get_cache
from osuT5.osuT5.inference.direct_decode import (
    decode_one_token_raw_logits,
    last_token_logits,
    prefill_static_cache,
)
from osuT5.osuT5.inference.server import get_eos_token_id
from osuT5.osuT5.model.custom_transformers import modeling_varwhisper
from osuT5.osuT5.runtime_profiling import generation_profile_context
from osuT5.osuT5.tokenizer import ContextType
from utils.profile_decode_linear_kernels import (
    _allclose,
    _bucketed_prefix_length,
    _cuda_event_time_ms,
    _cuda_graph_replay_time_ms,
    _max_abs,
)
from utils.verify_one_token_decode import (
    _build_generation_logits_processors,
    _build_probe_inputs,
    _condition_kwargs,
    _load_args,
    _move_kwargs_for_model,
)


CACHE_WRITING_VARIANTS = {
    "repo_decoder_layer",
    "self_attn_residual_segment",
    "manual_decoder_runtime_island",
    "compiled_decoder_layer",
    "native_self_cross_prefix_warp2",
    "native_self_cross_prefix_warp4",
    "native_self_cross_prefix_warp8",
    "native_decoder_layer_mlp_tail_warp2",
    "native_decoder_layer_mlp_tail_warp4",
    "native_decoder_layer_mlp_tail_warp8",
}

NATIVE_DECODER_LAYER_CANDIDATES = (
    "native_self_cross_prefix_warp2",
    "native_self_cross_prefix_warp4",
    "native_self_cross_prefix_warp8",
    "native_decoder_layer_mlp_tail_warp2",
    "native_decoder_layer_mlp_tail_warp4",
    "native_decoder_layer_mlp_tail_warp8",
)


@dataclass
class DecoderLayerCapture:
    name: str
    layer_idx: int
    module: torch.nn.Module
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor | None
    encoder_hidden_states: torch.Tensor | None
    past_key_value: Any
    cache_position: torch.Tensor
    position_ids: torch.Tensor | None
    cu_seqlens: torch.Tensor | None
    max_seqlen: int | None
    encoder_cu_seqlens: torch.Tensor | None
    encoder_max_seqlen: int | None
    output: torch.Tensor
    active_prefix_length: int
    cache_write_fingerprint: dict[str, Any] | None


def _clone_tensor(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    return value


def _decoder_layer_signature(capture: DecoderLayerCapture) -> str:
    mask_shape = "none" if capture.attention_mask is None else "x".join(str(dim) for dim in capture.attention_mask.shape)
    encoder_shape = (
        "none"
        if capture.encoder_hidden_states is None
        else "x".join(str(dim) for dim in capture.encoder_hidden_states.shape)
    )
    return (
        f"hidden{'x'.join(str(dim) for dim in capture.hidden_states.shape)}_"
        f"encoder{encoder_shape}_prefix{capture.active_prefix_length}_mask{mask_shape}"
    )


def _tensor_metadata(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, torch.Tensor):
        return None
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "device_type": value.device.type,
        "is_cuda": bool(value.is_cuda),
        "is_contiguous": bool(value.is_contiguous()),
        "requires_grad": bool(value.requires_grad),
        "numel": int(value.numel()),
        "element_size": int(value.element_size()),
        "storage_offset": int(value.storage_offset()),
    }


def _linear_metadata(module: torch.nn.Module) -> dict[str, Any]:
    return {
        "type": module.__class__.__name__,
        "in_features": int(module.in_features),
        "out_features": int(module.out_features),
        "has_bias": module.bias is not None,
        "weight": _tensor_metadata(module.weight),
        "bias": _tensor_metadata(module.bias),
    }


def _norm_metadata(module: torch.nn.Module) -> dict[str, Any]:
    return {
        "type": module.__class__.__name__,
        "normalized_shape": list(module.normalized_shape),
        "eps": module.eps,
        "elementwise_affine": bool(getattr(module, "elementwise_affine", True)),
        "weight": _tensor_metadata(module.weight),
        "bias": _tensor_metadata(getattr(module, "bias", None)),
    }


def _cache_layer_metadata(cache: Any, layer_idx: int) -> dict[str, Any] | None:
    layers = getattr(cache, "layers", None)
    if not isinstance(layers, (list, tuple)) or layer_idx >= len(layers):
        return None
    layer = layers[layer_idx]
    return {
        "type": layer.__class__.__name__,
        "is_initialized": bool(getattr(layer, "is_initialized", False)),
        "keys": _tensor_metadata(getattr(layer, "keys", None)),
        "values": _tensor_metadata(getattr(layer, "values", None)),
    }


def _tensor_fingerprint(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, torch.Tensor):
        return None
    contiguous_cpu = value.detach().contiguous().cpu()
    try:
        payload = contiguous_cpu.numpy().tobytes()
    except TypeError:
        payload = contiguous_cpu.view(torch.uint8).numpy().tobytes()
    return {
        "tensor": _tensor_metadata(value),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "num_bytes": int(len(payload)),
        "hash_layout": "contiguous_tensor_bytes",
        "hash_dtype": str(contiguous_cpu.dtype),
    }


def _cache_write_fingerprint(
        past_key_value: Any,
        *,
        layer_idx: int,
        cache_position: torch.Tensor,
        active_prefix_length: int,
) -> dict[str, Any]:
    position_values = [int(item) for item in cache_position.detach().cpu().view(-1).tolist()]
    if len(position_values) != 1:
        return {
            "schema_version": 1,
            "available": False,
            "reason": f"expected scalar cache_position, got {position_values}",
            "cache_position_values": position_values,
        }
    cache_pos = int(position_values[0])
    self_cache = getattr(past_key_value, "self_attention_cache", None)
    layers = getattr(self_cache, "layers", None)
    if not isinstance(layers, (list, tuple)) or layer_idx >= len(layers):
        return {
            "schema_version": 1,
            "available": False,
            "reason": "missing self-attention cache layer",
            "cache_position": cache_pos,
        }
    layer = layers[layer_idx]
    keys = getattr(layer, "keys", None)
    values = getattr(layer, "values", None)
    if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
        return {
            "schema_version": 1,
            "available": False,
            "reason": "missing tensor keys/values",
            "cache_position": cache_pos,
        }
    if keys.dim() != 4 or values.dim() != 4:
        return {
            "schema_version": 1,
            "available": False,
            "reason": f"expected 4D keys/values, got {list(keys.shape)} and {list(values.shape)}",
            "cache_position": cache_pos,
        }
    max_cache_len = int(keys.shape[2])
    if cache_pos < 0 or cache_pos >= max_cache_len:
        return {
            "schema_version": 1,
            "available": False,
            "reason": f"cache_position {cache_pos} outside cache length {max_cache_len}",
            "cache_position": cache_pos,
            "max_cache_len": max_cache_len,
        }
    key_slot = keys[:, :, cache_pos:cache_pos + 1, :]
    value_slot = values[:, :, cache_pos:cache_pos + 1, :]
    return {
        "schema_version": 1,
        "available": True,
        "cache_position": cache_pos,
        "active_prefix_length": int(active_prefix_length),
        "max_cache_len": max_cache_len,
        "position_within_cache": bool(cache_pos < max_cache_len),
        "position_within_active_prefix": bool(cache_pos < int(active_prefix_length)),
        "slot_shape_contract": "[batch, heads, 1, head_dim]",
        "keys": _tensor_fingerprint(key_slot),
        "values": _tensor_fingerprint(value_slot),
    }


def _cache_slot_views(capture: DecoderLayerCapture) -> tuple[torch.Tensor, torch.Tensor]:
    self_cache = getattr(capture.past_key_value, "self_attention_cache", None)
    layers = getattr(self_cache, "layers", None)
    if not isinstance(layers, (list, tuple)) or capture.layer_idx >= len(layers):
        raise RuntimeError("missing self-attention cache layer")
    layer = layers[capture.layer_idx]
    keys = getattr(layer, "keys", None)
    values = getattr(layer, "values", None)
    if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
        raise RuntimeError("missing tensor keys/values")
    positions = [int(item) for item in capture.cache_position.detach().cpu().view(-1).tolist()]
    if len(positions) != 1:
        raise RuntimeError(f"expected scalar cache_position, got {positions}")
    pos = positions[0]
    if pos < 0 or pos >= int(keys.shape[2]):
        raise RuntimeError(f"cache_position {pos} outside cache length {int(keys.shape[2])}")
    return keys[:, :, pos:pos + 1, :], values[:, :, pos:pos + 1, :]


def _restore_cache_slot(capture: DecoderLayerCapture, key_slot: torch.Tensor, value_slot: torch.Tensor) -> None:
    keys, values = _cache_slot_views(capture)
    keys.copy_(key_slot)
    values.copy_(value_slot)


def _cache_fingerprint_matches(expected: dict[str, Any] | None, actual: dict[str, Any] | None) -> dict[str, Any]:
    expected_keys = expected.get("keys") if isinstance(expected, dict) else None
    expected_values = expected.get("values") if isinstance(expected, dict) else None
    actual_keys = actual.get("keys") if isinstance(actual, dict) else None
    actual_values = actual.get("values") if isinstance(actual, dict) else None
    keys_match = (
        isinstance(expected_keys, dict)
        and isinstance(actual_keys, dict)
        and expected_keys.get("sha256") == actual_keys.get("sha256")
    )
    values_match = (
        isinstance(expected_values, dict)
        and isinstance(actual_values, dict)
        and expected_values.get("sha256") == actual_values.get("sha256")
    )
    return {
        "keys_match": bool(keys_match),
        "values_match": bool(values_match),
        "matches_expected": bool(keys_match and values_match),
        "expected_key_sha256": expected_keys.get("sha256") if isinstance(expected_keys, dict) else None,
        "actual_key_sha256": actual_keys.get("sha256") if isinstance(actual_keys, dict) else None,
        "expected_value_sha256": expected_values.get("sha256") if isinstance(expected_values, dict) else None,
        "actual_value_sha256": actual_values.get("sha256") if isinstance(actual_values, dict) else None,
    }


def _verify_cache_write_candidate(
        capture: DecoderLayerCapture,
        *,
        name: str,
        fn: Callable[[], torch.Tensor],
        expected_output: torch.Tensor,
        atol: float,
        rtol: float,
) -> dict[str, Any]:
    expected_fingerprint = capture.cache_write_fingerprint
    if not isinstance(expected_fingerprint, dict) or expected_fingerprint.get("available") is not True:
        return {
            "checked": False,
            "pass": False,
            "variant": name,
            "reason": "missing expected cache-write fingerprint",
        }
    try:
        key_slot, value_slot = _cache_slot_views(capture)
        reference_key = key_slot.detach().clone()
        reference_value = value_slot.detach().clone()
        key_slot.fill_(float("nan"))
        value_slot.fill_(float("nan"))
        output = fn()
        if isinstance(output, torch.Tensor) and output.is_cuda:
            torch.cuda.synchronize(output.device)
        actual_fingerprint = _cache_write_fingerprint(
            capture.past_key_value,
            layer_idx=capture.layer_idx,
            cache_position=capture.cache_position,
            active_prefix_length=capture.active_prefix_length,
        )
        actual_key_slot, actual_value_slot = _cache_slot_views(capture)
        match = _cache_fingerprint_matches(expected_fingerprint, actual_fingerprint)
        keys_allclose = _allclose(reference_key, actual_key_slot, atol=atol, rtol=rtol)
        values_allclose = _allclose(reference_value, actual_value_slot, atol=atol, rtol=rtol)
        output_allclose = _allclose(expected_output, output, atol=atol, rtol=rtol)
        return {
            "checked": True,
            "pass": bool(match["matches_expected"] and output_allclose),
            "variant": name,
            "slot_reset_fill": "nan",
            "output_allclose": bool(output_allclose),
            "output_max_abs": _max_abs(expected_output, output),
            "keys_allclose": bool(keys_allclose),
            "values_allclose": bool(values_allclose),
            "keys_max_abs": _max_abs(reference_key, actual_key_slot),
            "values_max_abs": _max_abs(reference_value, actual_value_slot),
            "actual": actual_fingerprint,
            **match,
        }
    except Exception as exc:
        return {
            "checked": True,
            "pass": False,
            "variant": name,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if "reference_key" in locals() and "reference_value" in locals():
            _restore_cache_slot(capture, reference_key, reference_value)


def _decoder_layer_abi_manifest(capture: DecoderLayerCapture) -> dict[str, Any]:
    module = capture.module
    self_attn = module.self_attn
    cross_attn = module.cross_attn
    past_key_value = capture.past_key_value
    self_cache = getattr(past_key_value, "self_attention_cache", None)
    cross_cache = getattr(past_key_value, "cross_attention_cache", None)
    is_updated = getattr(past_key_value, "is_updated", {})
    is_cross_updated = None
    if isinstance(is_updated, dict):
        is_cross_updated = bool(is_updated.get(capture.layer_idx, False))

    return {
        "schema_version": 1,
        "purpose": (
            "Verifier-only ABI metadata for a future whole decoder-layer native "
            "math/memory island. This is not a speed claim."
        ),
        "boundary": [
            "self_attn_layer_norm",
            "self_attn.Wqkv",
            "self_attn.RoPE/cache write/q_len1 attention",
            "self_attn.Wo",
            "self residual add",
            "cross_attn_layer_norm",
            "cross_attn.Wq",
            "cross_attn cached q_len1 attention",
            "cross_attn.Wo",
            "cross residual add",
            "final_layer_norm",
            "fc1",
            "activation",
            "fc2",
            "mlp residual add",
        ],
        "layer": {
            "name": capture.name,
            "layer_idx": int(capture.layer_idx),
            "training": bool(module.training),
            "dropout": float(module.dropout),
            "activation_dropout": float(module.activation_dropout),
            "activation_fn": getattr(module.activation_fn, "__name__", module.activation_fn.__class__.__name__),
            "embed_dim": int(module.embed_dim),
            "active_prefix_length": int(capture.active_prefix_length),
        },
        "runtime_inputs": {
            "hidden_states": _tensor_metadata(capture.hidden_states),
            "attention_mask": _tensor_metadata(capture.attention_mask),
            "encoder_hidden_states": _tensor_metadata(capture.encoder_hidden_states),
            "cache_position": _tensor_metadata(capture.cache_position),
            "position_ids": _tensor_metadata(capture.position_ids),
            "cu_seqlens": _tensor_metadata(capture.cu_seqlens),
            "encoder_cu_seqlens": _tensor_metadata(capture.encoder_cu_seqlens),
            "output": _tensor_metadata(capture.output),
            "max_seqlen": capture.max_seqlen,
            "encoder_max_seqlen": capture.encoder_max_seqlen,
        },
        "modules": {
            "self_attn_layer_norm": _norm_metadata(module.self_attn_layer_norm),
            "cross_attn_layer_norm": _norm_metadata(module.cross_attn_layer_norm),
            "final_layer_norm": _norm_metadata(module.final_layer_norm),
            "self_attn": {
                "num_heads": int(self_attn.num_heads),
                "head_dim": int(self_attn.head_dim),
                "all_head_size": int(self_attn.all_head_size),
                "local_attention": list(self_attn.local_attention),
                "max_position_embeddings": int(self_attn.max_position_embeddings),
                "Wqkv": _linear_metadata(self_attn.Wqkv),
                "Wo": _linear_metadata(self_attn.Wo),
            },
            "cross_attn": {
                "num_heads": int(cross_attn.num_heads),
                "head_dim": int(cross_attn.head_dim),
                "all_head_size": int(cross_attn.all_head_size),
                "local_attention": list(cross_attn.local_attention),
                "max_position_embeddings": int(cross_attn.max_position_embeddings),
                "Wq": _linear_metadata(cross_attn.Wq),
                "Wkv": _linear_metadata(cross_attn.Wkv),
                "Wo": _linear_metadata(cross_attn.Wo),
            },
            "fc1": _linear_metadata(module.fc1),
            "fc2": _linear_metadata(module.fc2),
        },
        "cache": {
            "type": past_key_value.__class__.__name__,
            "self_attention_cache_type": self_cache.__class__.__name__ if self_cache is not None else None,
            "cross_attention_cache_type": cross_cache.__class__.__name__ if cross_cache is not None else None,
            "cross_attention_cache_updated_for_layer": is_cross_updated,
            "self_layer": _cache_layer_metadata(self_cache, capture.layer_idx),
            "cross_layer": _cache_layer_metadata(cross_cache, capture.layer_idx),
        },
        "cache_write_fingerprint": capture.cache_write_fingerprint,
        "guardrails": {
            "batch_size_1": (
                isinstance(capture.hidden_states, torch.Tensor)
                and capture.hidden_states.shape[:2] == (1, 1)
            ),
            "fp32_hidden_states": (
                isinstance(capture.hidden_states, torch.Tensor)
                and capture.hidden_states.dtype == torch.float32
            ),
            "has_static_cache_layers": (
                _cache_layer_metadata(self_cache, capture.layer_idx) is not None
                and _cache_layer_metadata(cross_cache, capture.layer_idx) is not None
            ),
            "cross_kv_expected_reused": bool(is_cross_updated),
            "native_candidate_must_preserve_cache_write": True,
            "native_candidate_must_match_output": True,
        },
    }


def _layer_forward(capture: DecoderLayerCapture) -> torch.Tensor:
    return capture.module(
        hidden_states=capture.hidden_states,
        attention_mask=capture.attention_mask,
        encoder_hidden_states=capture.encoder_hidden_states,
        past_key_value=capture.past_key_value,
        cache_position=capture.cache_position,
        position_ids=capture.position_ids,
        cu_seqlens=capture.cu_seqlens,
        max_seqlen=capture.max_seqlen,
        encoder_cu_seqlens=capture.encoder_cu_seqlens,
        encoder_max_seqlen=capture.encoder_max_seqlen,
        output_attentions=False,
    )[0]


def _norm_only(norm_module: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    return norm_module(hidden_states)


def _self_attn_residual_segment(capture: DecoderLayerCapture) -> torch.Tensor:
    residual = capture.hidden_states
    hidden_states = capture.module.self_attn_layer_norm(capture.hidden_states)
    self_attn_output = capture.module.self_attn(
        hidden_states=hidden_states,
        past_key_value=capture.past_key_value,
        attention_mask=capture.attention_mask,
        cache_position=capture.cache_position,
        position_ids=capture.position_ids,
        cu_seqlens=capture.cu_seqlens,
        max_seqlen=capture.max_seqlen,
        output_attentions=False,
    )[0]
    return residual + self_attn_output


def _active_prefix_attention_mask(capture: DecoderLayerCapture) -> torch.Tensor | None:
    attention_mask = capture.attention_mask
    if isinstance(attention_mask, torch.Tensor) and attention_mask.shape[-1] > capture.active_prefix_length:
        attention_mask = attention_mask[..., :capture.active_prefix_length]
    return attention_mask


def _self_attention_cache_tensors(capture: DecoderLayerCapture) -> tuple[torch.Tensor, torch.Tensor]:
    cache_layer = capture.past_key_value.self_attention_cache.layers[capture.layer_idx]
    if not getattr(cache_layer, "is_initialized", False):
        raise RuntimeError(f"self-attention cache layer {capture.layer_idx} is not initialized")
    return cache_layer.keys, cache_layer.values


def _cross_attention_cache_tensors(capture: DecoderLayerCapture) -> tuple[torch.Tensor, torch.Tensor]:
    cache_layer = capture.past_key_value.cross_attention_cache.layers[capture.layer_idx]
    if not getattr(cache_layer, "is_initialized", False):
        raise RuntimeError(f"cross-attention cache layer {capture.layer_idx} is not initialized")
    return cache_layer.keys, cache_layer.values


def _rmsnorm_eps(norm_module: torch.nn.Module, dtype: torch.dtype) -> float:
    eps = getattr(norm_module, "eps", None)
    if eps is None:
        eps = torch.finfo(dtype).eps
    return float(eps)


def _cross_attn_residual_segment(capture: DecoderLayerCapture, after_self_attn: torch.Tensor) -> torch.Tensor:
    if capture.encoder_hidden_states is None:
        return after_self_attn
    residual = after_self_attn
    hidden_states = capture.module.cross_attn_layer_norm(after_self_attn)
    cross_attn_output = capture.module.cross_attn(
        hidden_states=hidden_states,
        key_value_states=capture.encoder_hidden_states,
        past_key_value=capture.past_key_value,
        position_ids=capture.position_ids,
        cu_seqlens=capture.cu_seqlens,
        max_seqlen=capture.max_seqlen,
        cu_seqlens_k=capture.encoder_cu_seqlens,
        max_seqlen_k=capture.encoder_max_seqlen,
        output_attentions=False,
    )[0]
    return residual + cross_attn_output


def _mlp_residual_segment(capture: DecoderLayerCapture, after_cross_attn: torch.Tensor) -> torch.Tensor:
    residual = after_cross_attn
    hidden_states = capture.module.final_layer_norm(after_cross_attn)
    hidden_states = capture.module.activation_fn(capture.module.fc1(hidden_states))
    hidden_states = F.dropout(hidden_states, p=capture.module.activation_dropout, training=capture.module.training)
    hidden_states = capture.module.fc2(hidden_states)
    hidden_states = F.dropout(hidden_states, p=capture.module.dropout, training=capture.module.training)
    return residual + hidden_states


def _mlp_input_norm_only(capture: DecoderLayerCapture, after_cross_attn: torch.Tensor) -> torch.Tensor:
    return capture.module.final_layer_norm(after_cross_attn)


def _mlp_fc1_only(capture: DecoderLayerCapture, mlp_input_norm: torch.Tensor) -> torch.Tensor:
    return capture.module.fc1(mlp_input_norm)


def _mlp_activation_only(capture: DecoderLayerCapture, mlp_fc1: torch.Tensor) -> torch.Tensor:
    return capture.module.activation_fn(mlp_fc1)


def _mlp_fc2_only(capture: DecoderLayerCapture, mlp_activation: torch.Tensor) -> torch.Tensor:
    return capture.module.fc2(mlp_activation)


def _mlp_residual_add_only(after_cross_attn: torch.Tensor, mlp_fc2: torch.Tensor) -> torch.Tensor:
    return after_cross_attn + mlp_fc2


def _mlp_fc2_residual_segment(
        capture: DecoderLayerCapture,
        after_cross_attn: torch.Tensor,
        mlp_activation: torch.Tensor,
) -> torch.Tensor:
    hidden_states = capture.module.fc2(mlp_activation)
    hidden_states = F.dropout(hidden_states, p=capture.module.dropout, training=capture.module.training)
    return after_cross_attn + hidden_states


def _manual_decoder_runtime_island(capture: DecoderLayerCapture) -> torch.Tensor:
    """Exact whole-layer callable used to size broad runtime-island work.

    This intentionally composes the same module calls as VarWhisperDecoderLayer
    instead of adding a new optimized path. It calibrates the profiler and gives
    future native/CUDA candidates a single verifier boundary to replace.
    """
    hidden_states = _self_attn_residual_segment(capture)
    hidden_states = _cross_attn_residual_segment(capture, hidden_states)
    return _mlp_residual_segment(capture, hidden_states)


def _q1_bmm_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    num_heads = query.shape[1]
    head_dim = query.shape[-1]
    q = query.reshape(num_heads, 1, head_dim)
    k = key.reshape(num_heads, -1, head_dim)
    v = value.reshape(num_heads, -1, head_dim)
    scores = torch.bmm(q, k.transpose(1, 2)) * (head_dim ** -0.5)
    return torch.bmm(torch.softmax(scores, dim=-1), v).view(1, num_heads, 1, head_dim)


def _native_self_cross_prefix_variants(
        capture: DecoderLayerCapture,
) -> dict[str, Callable[[], torch.Tensor]]:
    """Full-layer verifier candidates with native self+cross attention prefix.

    The candidate replaces the adjacent self-attention and cross-attention
    residual prefix, then runs the normal MLP residual segment so the benchmark
    still compares at the full decoder-layer output boundary.
    """
    if capture.module.training or capture.module.self_attn.training or capture.module.cross_attn.training:
        raise ValueError("native self+cross prefix candidate only supports eval mode")
    if capture.encoder_hidden_states is None:
        raise ValueError("native self+cross prefix candidate requires encoder hidden states")
    if capture.hidden_states.dtype != torch.float32 or capture.hidden_states.shape[:2] != (1, 1):
        raise ValueError("native self+cross prefix candidate requires fp32 [1, 1, hidden] input")
    if capture.position_ids is None:
        raise ValueError("native self+cross prefix candidate requires position_ids")
    if capture.cache_position is None:
        raise ValueError("native self+cross prefix candidate requires cache_position")
    if capture.active_prefix_length <= 0:
        raise ValueError("native self+cross prefix candidate requires an active prefix")

    from osuT5.osuT5.inference.native_decoder_layer import (
        native_one_token_linear_residual,
        native_one_token_rmsnorm_linear,
        preload_native_decoder_layer,
    )
    from osuT5.osuT5.inference.native_q1_attention import (
        native_q1_rope_cache_attention,
        preload_native_q1_attention,
    )

    preload_native_decoder_layer()
    preload_native_q1_attention()
    self_attn = capture.module.self_attn
    cross_attn = capture.module.cross_attn
    self_cache_keys, self_cache_values = _self_attention_cache_tensors(capture)
    cross_cache_keys, cross_cache_values = _cross_attention_cache_tensors(capture)
    attention_mask = _active_prefix_attention_mask(capture)
    self_norm_eps = _rmsnorm_eps(capture.module.self_attn_layer_norm, capture.hidden_states.dtype)
    cross_norm_eps = _rmsnorm_eps(capture.module.cross_attn_layer_norm, capture.hidden_states.dtype)

    def make_prefix(outputs_per_block: int) -> Callable[[], torch.Tensor]:
        def native_self_cross_prefix() -> torch.Tensor:
            qkv = native_one_token_rmsnorm_linear(
                capture.hidden_states,
                capture.module.self_attn_layer_norm.weight,
                self_attn.Wqkv.weight,
                self_attn.Wqkv.bias,
                eps=self_norm_eps,
                outputs_per_block=outputs_per_block,
            ).view(1, 1, 3, self_attn.num_heads, self_attn.head_dim)
            cos, sin = self_attn.rotary_emb(qkv, position_ids=capture.position_ids)
            self_attention_output = native_q1_rope_cache_attention(
                qkv,
                self_cache_keys,
                self_cache_values,
                cos,
                sin,
                capture.cache_position,
                attention_mask,
                capture.active_prefix_length,
            )
            after_self_attn = native_one_token_linear_residual(
                self_attention_output,
                capture.hidden_states,
                self_attn.Wo.weight,
                self_attn.Wo.bias,
                outputs_per_block=outputs_per_block,
            )
            query = native_one_token_rmsnorm_linear(
                after_self_attn,
                capture.module.cross_attn_layer_norm.weight,
                cross_attn.Wq.weight,
                cross_attn.Wq.bias,
                eps=cross_norm_eps,
                outputs_per_block=outputs_per_block,
            ).view(1, 1, cross_attn.num_heads, cross_attn.head_dim).transpose(1, 2)
            cross_attention_output = _q1_bmm_attention(query, cross_cache_keys, cross_cache_values)
            return native_one_token_linear_residual(
                cross_attention_output,
                after_self_attn,
                cross_attn.Wo.weight,
                cross_attn.Wo.bias,
                outputs_per_block=outputs_per_block,
            )

        return native_self_cross_prefix

    return {
        "native_self_cross_prefix_warp2": make_prefix(2),
        "native_self_cross_prefix_warp4": make_prefix(4),
        "native_self_cross_prefix_warp8": make_prefix(8),
    }


def _activation_name(activation: Callable[..., Any]) -> str:
    return getattr(activation, "__name__", activation.__class__.__name__)


def _native_decoder_layer_mlp_tail_variants(
        capture: DecoderLayerCapture,
) -> dict[str, Callable[[], torch.Tensor]]:
    """Whole-layer profiler candidates with a native MLP tail.

    These variants intentionally keep the candidate at the full decoder-layer
    verifier boundary: self-attn, cross-attn, then native MLP tail. They are not
    production inference paths.
    """
    if capture.module.training or capture.module.activation_dropout != 0 or capture.module.dropout != 0:
        raise ValueError("native decoder-layer MLP-tail candidate only supports eval/no-dropout layers")
    activation_name = _activation_name(capture.module.activation_fn)
    if activation_name not in {"GELUActivation", "gelu"}:
        raise ValueError(f"native decoder-layer MLP-tail candidate requires exact GELU, got {activation_name}")

    from osuT5.osuT5.inference.native_decoder_layer import (
        native_one_token_mlp_residual,
        preload_native_decoder_layer,
    )

    preload_native_decoder_layer()
    norm = capture.module.final_layer_norm
    fc1 = capture.module.fc1
    fc2 = capture.module.fc2
    norm_eps = getattr(norm, "eps", None)

    def run(outputs_per_block: int) -> torch.Tensor:
        hidden_states = _self_attn_residual_segment(capture)
        hidden_states = _cross_attn_residual_segment(capture, hidden_states)
        eps = torch.finfo(hidden_states.dtype).eps if norm_eps is None else float(norm_eps)
        return native_one_token_mlp_residual(
            hidden_states,
            norm.weight,
            fc1.weight,
            fc1.bias,
            fc2.weight,
            fc2.bias,
            eps=eps,
            outputs_per_block=outputs_per_block,
        )

    return {
        "native_decoder_layer_mlp_tail_warp2": lambda: run(2),
        "native_decoder_layer_mlp_tail_warp4": lambda: run(4),
        "native_decoder_layer_mlp_tail_warp8": lambda: run(8),
    }


def _benchmark_capture(
        capture: DecoderLayerCapture,
        *,
        native_q1_rope_cache_self_attention: bool,
        compile_decoder_layer_variant: bool,
        candidate_decoder_runtime_island: bool,
        candidate_native_decoder_layer_island: bool,
        candidate_native_self_cross_prefix: bool,
        verify_cache_write_candidates: bool,
        warmup: int,
        iters: int,
        atol: float,
        rtol: float,
        cuda_graph_replay: bool,
) -> dict[str, Any]:
    with generation_profile_context(
            active_prefix_self_attention_length=capture.active_prefix_length,
            q1_bmm_cross_attention=True,
            native_q1_self_attention=True,
            native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
    ):
        after_self_attn = _self_attn_residual_segment(capture).detach()
        after_cross_attn = _cross_attn_residual_segment(capture, after_self_attn).detach()
        mlp_input_norm = _mlp_input_norm_only(capture, after_cross_attn).detach()
        mlp_fc1 = _mlp_fc1_only(capture, mlp_input_norm).detach()
        mlp_activation = _mlp_activation_only(capture, mlp_fc1).detach()
        mlp_fc2 = _mlp_fc2_only(capture, mlp_activation).detach()
        variants: dict[str, tuple[Callable[[], torch.Tensor], torch.Tensor]] = {
            "repo_decoder_layer": (lambda: _layer_forward(capture), capture.output),
            "self_attn_residual_segment": (
                lambda: _self_attn_residual_segment(capture),
                after_self_attn,
            ),
            "self_attn_norm_only": (
                lambda: _norm_only(capture.module.self_attn_layer_norm, capture.hidden_states),
                _norm_only(capture.module.self_attn_layer_norm, capture.hidden_states).detach(),
            ),
            "mlp_residual_segment": (
                lambda: _mlp_residual_segment(capture, after_cross_attn),
                capture.output,
            ),
            "mlp_input_norm_only": (
                lambda: _mlp_input_norm_only(capture, after_cross_attn),
                mlp_input_norm,
            ),
            "mlp_fc1_only": (
                lambda: _mlp_fc1_only(capture, mlp_input_norm),
                mlp_fc1,
            ),
            "mlp_activation_only": (
                lambda: _mlp_activation_only(capture, mlp_fc1),
                mlp_activation,
            ),
            "mlp_fc2_only": (
                lambda: _mlp_fc2_only(capture, mlp_activation),
                mlp_fc2,
            ),
            "mlp_residual_add_only": (
                lambda: _mlp_residual_add_only(after_cross_attn, mlp_fc2),
                capture.output,
            ),
            "mlp_fc2_residual_segment": (
                lambda: _mlp_fc2_residual_segment(capture, after_cross_attn, mlp_activation),
                capture.output,
            ),
        }
        if candidate_decoder_runtime_island:
            variants["manual_decoder_runtime_island"] = (
                lambda: _manual_decoder_runtime_island(capture),
                capture.output,
            )
        if candidate_native_self_cross_prefix:
            try:
                for variant_name, prefix_fn in _native_self_cross_prefix_variants(capture).items():
                    variants[variant_name] = (
                        lambda prefix_fn=prefix_fn: _mlp_residual_segment(capture, prefix_fn()),
                        capture.output,
                    )
            except Exception as exc:
                variants["native_self_cross_prefix_setup"] = (
                    lambda exc=exc: (_ for _ in ()).throw(exc),
                    capture.output,
                )
        if candidate_native_decoder_layer_island:
            try:
                for variant_name, fn in _native_decoder_layer_mlp_tail_variants(capture).items():
                    variants[variant_name] = (fn, capture.output)
            except Exception as exc:
                variants["native_decoder_layer_setup"] = (
                    lambda exc=exc: (_ for _ in ()).throw(exc),
                    capture.output,
                )
        if capture.encoder_hidden_states is not None:
            self_attn_output = capture.module.self_attn(
                hidden_states=capture.module.self_attn_layer_norm(capture.hidden_states),
                past_key_value=capture.past_key_value,
                attention_mask=capture.attention_mask,
                cache_position=capture.cache_position,
                position_ids=capture.position_ids,
                cu_seqlens=capture.cu_seqlens,
                max_seqlen=capture.max_seqlen,
                output_attentions=False,
            )[0]
            after_self = capture.hidden_states + self_attn_output
            variants["cross_attn_norm_only"] = (
                lambda: _norm_only(capture.module.cross_attn_layer_norm, after_self),
                _norm_only(capture.module.cross_attn_layer_norm, after_self).detach(),
            )
            variants["cross_attn_residual_segment"] = (
                lambda: _cross_attn_residual_segment(capture, after_self_attn),
                after_cross_attn,
            )
        variants["mlp_norm_only"] = (
            lambda: _norm_only(capture.module.final_layer_norm, capture.output),
            _norm_only(capture.module.final_layer_norm, capture.output).detach(),
        )

        results: dict[str, dict[str, Any]] = {}
        for name, (fn, expected) in variants.items():
            cache_write_check = (
                _verify_cache_write_candidate(
                    capture,
                    name=name,
                    fn=fn,
                    expected_output=expected,
                    atol=atol,
                    rtol=rtol,
                )
                if verify_cache_write_candidates and name in CACHE_WRITING_VARIANTS
                else None
            )
            ms, output = _cuda_event_time_ms(fn, warmup=warmup, iters=iters)
            results[name] = {
                "ms_per_call": ms,
                "allclose": _allclose(expected, output, atol=atol, rtol=rtol),
                "max_abs": _max_abs(expected, output),
                "output_shape": list(output.shape),
            }
            if cache_write_check is not None:
                results[name]["cache_write_check"] = cache_write_check
            if cuda_graph_replay:
                try:
                    graph_ms, graph_output = _cuda_graph_replay_time_ms(fn, warmup=warmup, iters=iters)
                    results[name]["cuda_graph_replay_ms_per_call"] = graph_ms
                    results[name]["cuda_graph_replay_allclose"] = _allclose(
                        expected,
                        graph_output,
                        atol=atol,
                        rtol=rtol,
                    )
                    results[name]["cuda_graph_replay_max_abs"] = _max_abs(expected, graph_output)
                except Exception as exc:
                    results[name]["cuda_graph_replay_error"] = f"{type(exc).__name__}: {exc}"
        if compile_decoder_layer_variant:
            compiled_module = torch.compile(capture.module, mode="reduce-overhead")

            def compiled_layer_forward() -> torch.Tensor:
                return compiled_module(
                    hidden_states=capture.hidden_states,
                    attention_mask=capture.attention_mask,
                    encoder_hidden_states=capture.encoder_hidden_states,
                    past_key_value=capture.past_key_value,
                    cache_position=capture.cache_position,
                    position_ids=capture.position_ids,
                    cu_seqlens=capture.cu_seqlens,
                    max_seqlen=capture.max_seqlen,
                    encoder_cu_seqlens=capture.encoder_cu_seqlens,
                    encoder_max_seqlen=capture.encoder_max_seqlen,
                    output_attentions=False,
                )[0]

            try:
                cache_write_check = (
                    _verify_cache_write_candidate(
                        capture,
                        name="compiled_decoder_layer",
                        fn=compiled_layer_forward,
                        expected_output=capture.output,
                        atol=atol,
                        rtol=rtol,
                    )
                    if verify_cache_write_candidates
                    else None
                )
                ms, output = _cuda_event_time_ms(compiled_layer_forward, warmup=warmup, iters=iters)
                results["compiled_decoder_layer"] = {
                    "ms_per_call": ms,
                    "allclose": _allclose(capture.output, output, atol=atol, rtol=rtol),
                    "max_abs": _max_abs(capture.output, output),
                    "output_shape": list(output.shape),
                }
                if cache_write_check is not None:
                    results["compiled_decoder_layer"]["cache_write_check"] = cache_write_check
                if cuda_graph_replay:
                    try:
                        graph_ms, graph_output = _cuda_graph_replay_time_ms(
                            compiled_layer_forward,
                            warmup=warmup,
                            iters=iters,
                        )
                        results["compiled_decoder_layer"]["cuda_graph_replay_ms_per_call"] = graph_ms
                        results["compiled_decoder_layer"]["cuda_graph_replay_allclose"] = _allclose(
                            capture.output,
                            graph_output,
                            atol=atol,
                            rtol=rtol,
                        )
                        results["compiled_decoder_layer"]["cuda_graph_replay_max_abs"] = _max_abs(
                            capture.output,
                            graph_output,
                        )
                    except Exception as exc:
                        results["compiled_decoder_layer"]["cuda_graph_replay_error"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
            except Exception as exc:
                results["compiled_decoder_layer"] = {"error": f"{type(exc).__name__}: {exc}"}
    return results


def _capture_decoder_layers(
        model,
        prepared_inputs: dict[str, Any],
        *,
        active_prefix_length: int,
        native_q1_rope_cache_self_attention: bool,
        sdpa_backend: str | None,
        include_cache_write_fingerprint: bool,
) -> tuple[dict[str, DecoderLayerCapture], torch.Tensor]:
    captures: dict[str, DecoderLayerCapture] = {}
    handles = []

    def should_capture(name: str, module: torch.nn.Module) -> bool:
        return (
            isinstance(module, modeling_varwhisper.VarWhisperDecoderLayer)
            and (
                name.startswith("model.decoder.layers.")
                or ".model.decoder.layers." in name
            )
        )

    def hook_for(name: str):
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
            if name in captures:
                return
            hidden_states = kwargs.get("hidden_states")
            if not isinstance(hidden_states, torch.Tensor):
                if not inputs or not isinstance(inputs[0], torch.Tensor):
                    return
                hidden_states = inputs[0]
            if hidden_states.shape[0] != 1 or hidden_states.shape[1] != 1:
                return
            if not isinstance(output, tuple) or not output or not isinstance(output[0], torch.Tensor):
                return
            match = re.search(r"(^|\.)model\.decoder\.layers\.(\d+)$", name)
            if match is None:
                return
            past_key_value = kwargs.get("past_key_value")
            cache_position = kwargs.get("cache_position")
            if past_key_value is None:
                raise RuntimeError(f"{name} did not receive past_key_value")
            if not isinstance(cache_position, torch.Tensor):
                raise RuntimeError(f"{name} did not receive tensor cache_position")
            layer_idx = int(match.group(2))
            cache_write_fingerprint = (
                _cache_write_fingerprint(
                    past_key_value,
                    layer_idx=layer_idx,
                    cache_position=cache_position,
                    active_prefix_length=active_prefix_length,
                )
                if include_cache_write_fingerprint
                else None
            )
            captures[name] = DecoderLayerCapture(
                name=name,
                layer_idx=layer_idx,
                module=module,
                hidden_states=hidden_states.detach(),
                attention_mask=_clone_tensor(kwargs.get("attention_mask")),
                encoder_hidden_states=_clone_tensor(kwargs.get("encoder_hidden_states")),
                past_key_value=past_key_value,
                cache_position=cache_position.detach(),
                position_ids=_clone_tensor(kwargs.get("position_ids")),
                cu_seqlens=_clone_tensor(kwargs.get("cu_seqlens")),
                max_seqlen=kwargs.get("max_seqlen"),
                encoder_cu_seqlens=_clone_tensor(kwargs.get("encoder_cu_seqlens")),
                encoder_max_seqlen=kwargs.get("encoder_max_seqlen"),
                output=output[0].detach(),
                active_prefix_length=active_prefix_length,
                cache_write_fingerprint=cache_write_fingerprint,
            )

        return hook

    for name, module in model.named_modules():
        if should_capture(name, module):
            handles.append(module.register_forward_hook(hook_for(name), with_kwargs=True))

    try:
        with generation_profile_context(
                sdpa_backend=sdpa_backend,
                active_prefix_self_attention_length=active_prefix_length,
                q1_bmm_cross_attention=True,
                native_q1_self_attention=True,
                native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
        ):
            outputs = model(**prepared_inputs)
            logits = last_token_logits(outputs.logits)
    finally:
        for handle in handles:
            handle.remove()

    return captures, logits


@torch.no_grad()
def profile_decode_decoder_layer_island(
        args,
        *,
        sequence_index: int,
        active_prefix_bucket_size: int,
        active_prefix_decode_length: int | None,
        native_q1_rope_cache_self_attention: bool,
        compile_decoder_layer_variant: bool,
        candidate_decoder_runtime_island: bool,
        candidate_native_decoder_layer_island: bool,
        candidate_native_self_cross_prefix: bool,
        warmup: int,
        iters: int,
        atol: float,
        rtol: float,
        cuda_graph_replay: bool,
        full_song_decode_steps: int,
        full_song_main_tokens: int,
        full_song_model_time_s: float,
        include_cache_write_fingerprint: bool,
        verify_cache_write_candidates: bool,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Decode decoder-layer island profiling requires CUDA")

    compile_args(args, verbose=False)
    setup_inference_environment(args.seed)
    model, tokenizer = load_model_with_server(
        args.model_path,
        args.train,
        args.device,
        max_batch_size=args.max_batch_size,
        use_server=False,
        precision=args.precision,
        attn_implementation=args.attn_implementation,
        lora_path=args.lora_path,
        gamemode=args.gamemode,
        auto_select_gamemode_model=args.auto_select_gamemode_model,
        generation_compile=args.inference_generation_compile,
    )
    model.eval()
    model_inputs = _move_kwargs_for_model(
        model,
        _build_probe_inputs(args, model, tokenizer, sequence_index=sequence_index),
    )
    metadata = {
        "model_path": args.model_path,
        "precision": args.precision,
        "attn_implementation": args.attn_implementation,
        "profile_sdpa_backend": args.profile_sdpa_backend,
        "torch_version": torch.__version__,
        "sequence_index": model_inputs.pop("sequence_index"),
        "frame_time_ms": model_inputs.pop("frame_time_ms"),
        "context_type": model_inputs.pop("context_type"),
        "lookback_time": model_inputs.pop("lookback_time"),
        "lookahead_time": model_inputs.pop("lookahead_time"),
        "warmup": warmup,
        "iters": iters,
        "active_prefix_bucket_size": active_prefix_bucket_size,
        "active_prefix_decode_length_override": active_prefix_decode_length,
        "native_q1_rope_cache_self_attention": bool(native_q1_rope_cache_self_attention),
        "compile_decoder_layer_variant": bool(compile_decoder_layer_variant),
        "candidate_decoder_runtime_island": bool(candidate_decoder_runtime_island),
        "candidate_native_decoder_layer_island": bool(candidate_native_decoder_layer_island),
        "candidate_native_self_cross_prefix": bool(candidate_native_self_cross_prefix),
        "cuda_graph_replay": bool(cuda_graph_replay),
        "full_song_decode_steps": int(full_song_decode_steps),
        "full_song_main_tokens": int(full_song_main_tokens),
        "full_song_model_time_s": float(full_song_model_time_s),
        "include_cache_write_fingerprint": bool(include_cache_write_fingerprint),
        "verify_cache_write_candidates": bool(verify_cache_write_candidates),
    }
    prompt = model_inputs["decoder_input_ids"]
    prompt_mask = model_inputs["decoder_attention_mask"]
    prompt_len = int(prompt.shape[-1])
    condition_kwargs = _condition_kwargs(model_inputs)
    logits_processors = _build_generation_logits_processors(
        args,
        tokenizer,
        model.device,
        lookback_time=float(metadata["lookback_time"]),
    )
    context_type = ContextType(metadata["context_type"])
    eos_token_ids = get_eos_token_id(
        tokenizer,
        lookback_time=float(metadata["lookback_time"]),
        lookahead_time=float(metadata["lookahead_time"]),
        context_type=context_type,
    )

    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=args.precision == "amp"), \
            generation_profile_context(
                sdpa_backend=args.profile_sdpa_backend,
                q1_bmm_cross_attention=True,
                native_q1_self_attention=True,
                native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
            ):
        hf_cache = get_cache(model, batch_size=1, num_beams=1, cfg_scale=1.0)
        hf_generate_outputs = model.generate(
            inputs=model_inputs["frames"],
            decoder_input_ids=prompt,
            decoder_attention_mask=prompt_mask,
            **condition_kwargs,
            do_sample=args.do_sample,
            num_beams=args.num_beams,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=1,
            use_cache=True,
            past_key_values=hf_cache,
            logits_processor=logits_processors,
            eos_token_id=eos_token_ids,
            return_dict_in_generate=True,
            output_logits=False,
        )
        probe_token = hf_generate_outputs.sequences[:, prompt_len:prompt_len + 1].to(torch.long)
        if probe_token.numel() != 1:
            raise RuntimeError(f"expected one generated probe token, got shape {list(probe_token.shape)}")
        full_prefix = torch.cat([prompt, probe_token], dim=-1)
        full_mask = torch.cat([prompt_mask, torch.ones_like(probe_token, dtype=prompt_mask.dtype)], dim=-1)

        state = prefill_static_cache(
            model,
            prompt=prompt,
            prompt_attention_mask=prompt_mask,
            frames=model_inputs["frames"],
            condition_kwargs=condition_kwargs,
            active_prefix_self_attention=False,
        )
        cache_position = torch.arange(state.prompt_length, state.prompt_length + 1, device=prompt.device)
        max_cache_len = int(state.cache.get_max_cache_shape())
        computed_active_prefix_length = _bucketed_prefix_length(
            int(full_prefix.shape[-1]),
            active_prefix_bucket_size,
            max_cache_len,
        )
        active_prefix_length = (
            int(active_prefix_decode_length)
            if active_prefix_decode_length is not None
            else computed_active_prefix_length
        )
        direct_result = decode_one_token_raw_logits(
            model,
            state,
            full_prefix=full_prefix,
            full_attention_mask=full_mask,
            condition_kwargs=condition_kwargs,
            cache_position=cache_position,
            active_prefix_self_attention=True,
            active_prefix_self_attention_length=active_prefix_length,
        )
        captures, replay_logits = _capture_decoder_layers(
            model,
            direct_result.prepared_inputs,
            active_prefix_length=active_prefix_length,
            native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
            sdpa_backend=args.profile_sdpa_backend,
            include_cache_write_fingerprint=include_cache_write_fingerprint or verify_cache_write_candidates,
        )

    if not captures:
        raise RuntimeError("did not capture any one-token decoder layer calls")

    signature_members: dict[str, list[str]] = defaultdict(list)
    for name, capture in captures.items():
        signature_members[_decoder_layer_signature(capture)].append(name)

    signature_reports: dict[str, Any] = {}
    projected_full_song: dict[str, Any] = {}
    for signature, names in sorted(signature_members.items()):
        representative = captures[sorted(names)[0]]
        benchmark = _benchmark_capture(
            representative,
            native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
            compile_decoder_layer_variant=compile_decoder_layer_variant,
            candidate_decoder_runtime_island=candidate_decoder_runtime_island,
            candidate_native_decoder_layer_island=candidate_native_decoder_layer_island,
            candidate_native_self_cross_prefix=candidate_native_self_cross_prefix,
            verify_cache_write_candidates=verify_cache_write_candidates,
            warmup=warmup,
            iters=iters,
            atol=atol,
            rtol=rtol,
            cuda_graph_replay=cuda_graph_replay,
        )
        member_count = len(names)
        signature_reports[signature] = {
            "representative": representative.name,
            "members": sorted(names),
            "member_count": member_count,
            "layer_idx": representative.layer_idx,
            "hidden_shape": list(representative.hidden_states.shape),
            "encoder_hidden_shape": (
                list(representative.encoder_hidden_states.shape)
                if isinstance(representative.encoder_hidden_states, torch.Tensor)
                else None
            ),
            "attention_mask_shape": (
                list(representative.attention_mask.shape)
                if isinstance(representative.attention_mask, torch.Tensor)
                else None
            ),
            "native_decoder_layer_abi": _decoder_layer_abi_manifest(representative),
            "results": benchmark,
        }
        repo_ms = float(benchmark["repo_decoder_layer"]["ms_per_call"])
        repo_seconds = repo_ms * member_count * full_song_decode_steps / 1000.0
        repo_graph_ms = benchmark["repo_decoder_layer"].get("cuda_graph_replay_ms_per_call")
        projected_full_song[signature] = {
            "member_count": member_count,
            "decode_steps": int(full_song_decode_steps),
            "main_tokens": int(full_song_main_tokens),
            "repo_decoder_layer_s": repo_seconds,
            "repo_decoder_layer_fraction_of_model_time": repo_seconds / full_song_model_time_s,
            "ungraphed_projection_valid": repo_seconds < full_song_model_time_s,
        }
        if isinstance(repo_graph_ms, float):
            repo_graph_seconds = repo_graph_ms * member_count * full_song_decode_steps / 1000.0
            graph_projection_valid = repo_graph_seconds < full_song_model_time_s
            graph_ideal_time_s = (
                full_song_model_time_s - repo_graph_seconds
                if graph_projection_valid
                else None
            )
            projected_full_song[signature]["cuda_graph_repo_decoder_layer_s"] = repo_graph_seconds
            projected_full_song[signature]["cuda_graph_repo_fraction_of_model_time"] = (
                repo_graph_seconds / full_song_model_time_s
            )
            projected_full_song[signature]["cuda_graph_projection_valid"] = graph_projection_valid
            projected_full_song[signature]["cuda_graph_ideal_free_decoder_layers_tps"] = (
                full_song_main_tokens / graph_ideal_time_s
                if graph_ideal_time_s is not None and graph_ideal_time_s > 0
                else None
            )
            segment_names = (
                "self_attn_residual_segment",
                "cross_attn_residual_segment",
                "mlp_residual_segment",
                "self_attn_norm_only",
                "cross_attn_norm_only",
                "mlp_norm_only",
                "mlp_input_norm_only",
                "mlp_fc1_only",
                "mlp_activation_only",
                "mlp_fc2_only",
                "mlp_residual_add_only",
                "mlp_fc2_residual_segment",
            )
            segment_seconds: dict[str, float] = {}
            for segment_name in segment_names:
                segment = benchmark.get(segment_name) or {}
                segment_graph_ms = segment.get("cuda_graph_replay_ms_per_call")
                if isinstance(segment_graph_ms, float):
                    segment_seconds[segment_name] = (
                        segment_graph_ms * member_count * full_song_decode_steps / 1000.0
                    )
            projected_full_song[signature]["cuda_graph_segment_seconds"] = segment_seconds
            segment_sum_names = (
                "self_attn_residual_segment",
                "cross_attn_residual_segment",
                "mlp_residual_segment",
            )
            segment_sum_s = sum(segment_seconds.get(name, 0.0) for name in segment_sum_names)
            projected_full_song[signature]["cuda_graph_residual_segment_sum_s"] = segment_sum_s
            projected_full_song[signature]["cuda_graph_residual_segment_unexplained_vs_layer_s"] = (
                repo_graph_seconds - segment_sum_s
            )
            mlp_component_sum_names = (
                "mlp_input_norm_only",
                "mlp_fc1_only",
                "mlp_activation_only",
                "mlp_fc2_only",
                "mlp_residual_add_only",
            )
            mlp_component_sum_s = sum(segment_seconds.get(name, 0.0) for name in mlp_component_sum_names)
            projected_full_song[signature]["cuda_graph_mlp_component_sum_s"] = mlp_component_sum_s
            projected_full_song[signature]["cuda_graph_mlp_component_unexplained_vs_mlp_residual_s"] = (
                segment_seconds.get("mlp_residual_segment", 0.0) - mlp_component_sum_s
            )
            mlp_fc2_residual_s = segment_seconds.get("mlp_fc2_residual_segment")
            if mlp_fc2_residual_s is not None:
                projected_full_song[signature]["cuda_graph_mlp_fc1_activation_prefix_s"] = (
                    segment_seconds.get("mlp_residual_segment", 0.0) - mlp_fc2_residual_s
                )
        compiled = benchmark.get("compiled_decoder_layer") or {}
        compiled_graph_ms = compiled.get("cuda_graph_replay_ms_per_call")
        if isinstance(compiled_graph_ms, float) and isinstance(repo_graph_ms, float):
            compiled_graph_seconds = compiled_graph_ms * member_count * full_song_decode_steps / 1000.0
            saved_seconds = repo_graph_seconds - compiled_graph_seconds
            projected_model_time_s = full_song_model_time_s - saved_seconds
            projected_full_song[signature]["cuda_graph_compiled_decoder_layer_s"] = compiled_graph_seconds
            projected_full_song[signature]["cuda_graph_compiled_saved_s"] = saved_seconds
            projected_full_song[signature]["cuda_graph_compiled_speedup"] = (
                repo_graph_ms / compiled_graph_ms
                if compiled_graph_ms > 0
                else None
            )
            projected_full_song[signature]["cuda_graph_compiled_projected_tps"] = (
                full_song_main_tokens / projected_model_time_s
                if projected_model_time_s > 0
                else None
            )
        candidate = benchmark.get("manual_decoder_runtime_island") or {}
        candidate_graph_ms = candidate.get("cuda_graph_replay_ms_per_call")
        if isinstance(candidate_graph_ms, float) and isinstance(repo_graph_ms, float):
            candidate_graph_seconds = candidate_graph_ms * member_count * full_song_decode_steps / 1000.0
            saved_seconds = repo_graph_seconds - candidate_graph_seconds
            projected_model_time_s = full_song_model_time_s - saved_seconds
            projected_full_song[signature]["cuda_graph_manual_decoder_runtime_island_s"] = (
                candidate_graph_seconds
            )
            projected_full_song[signature]["cuda_graph_manual_decoder_runtime_island_saved_s"] = (
                saved_seconds
            )
            projected_full_song[signature]["cuda_graph_manual_decoder_runtime_island_speedup"] = (
                repo_graph_ms / candidate_graph_ms
                if candidate_graph_ms > 0
                else None
            )
            projected_full_song[signature]["cuda_graph_manual_decoder_runtime_island_projected_tps"] = (
                full_song_main_tokens / projected_model_time_s
                if projected_model_time_s > 0
                else None
            )
        for native_name in NATIVE_DECODER_LAYER_CANDIDATES:
            native_candidate = benchmark.get(native_name) or {}
            native_graph_ms = native_candidate.get("cuda_graph_replay_ms_per_call")
            if (
                    isinstance(native_graph_ms, float)
                    and isinstance(repo_graph_ms, float)
                    and native_candidate.get("cuda_graph_replay_allclose")
            ):
                native_graph_seconds = native_graph_ms * member_count * full_song_decode_steps / 1000.0
                saved_seconds = repo_graph_seconds - native_graph_seconds
                projected_model_time_s = full_song_model_time_s - saved_seconds
                projected_full_song[signature][f"cuda_graph_{native_name}_s"] = native_graph_seconds
                projected_full_song[signature][f"cuda_graph_{native_name}_saved_s"] = saved_seconds
                projected_full_song[signature][f"cuda_graph_{native_name}_speedup"] = (
                    repo_graph_ms / native_graph_ms
                    if native_graph_ms > 0
                    else None
                )
                projected_full_song[signature][f"cuda_graph_{native_name}_projected_tps"] = (
                    full_song_main_tokens / projected_model_time_s
                    if projected_model_time_s > 0
                    else None
                )

    candidate_cache_write_checks: list[dict[str, Any]] = []
    for signature, report in signature_reports.items():
        results = report.get("results") or {}
        for variant_name, variant_result in results.items():
            if not isinstance(variant_result, dict):
                continue
            cache_check = variant_result.get("cache_write_check")
            if isinstance(cache_check, dict):
                candidate_cache_write_checks.append({
                    "signature": signature,
                    "variant": variant_name,
                    "pass": bool(cache_check.get("pass")),
                    "checked": bool(cache_check.get("checked")),
                    "keys_match": cache_check.get("keys_match"),
                    "values_match": cache_check.get("values_match"),
                    "keys_allclose": cache_check.get("keys_allclose"),
                    "values_allclose": cache_check.get("values_allclose"),
                    "keys_max_abs": cache_check.get("keys_max_abs"),
                    "values_max_abs": cache_check.get("values_max_abs"),
                    "output_allclose": cache_check.get("output_allclose"),
                    "output_max_abs": cache_check.get("output_max_abs"),
                    "error": cache_check.get("error"),
                    "reason": cache_check.get("reason"),
                })
    candidate_cache_write_checks_pass = (
        all(item["pass"] for item in candidate_cache_write_checks)
        if verify_cache_write_candidates and candidate_cache_write_checks
        else not verify_cache_write_candidates
    )
    logits_replay_allclose = bool(_allclose(direct_result.logits, replay_logits, atol=atol, rtol=rtol))
    return {
        "pass": bool(logits_replay_allclose and candidate_cache_write_checks_pass),
        "prompt_tokens": prompt_len,
        "probe_token_id": int(probe_token.item()),
        "full_prefix_tokens": int(full_prefix.shape[-1]),
        "cache_position": [int(item) for item in cache_position.detach().cpu().tolist()],
        "max_cache_len": max_cache_len,
        "computed_active_prefix_length": computed_active_prefix_length,
        "active_prefix_length": active_prefix_length,
        "captured_decoder_layer_count": len(captures),
        "signature_reports": signature_reports,
        "projected_full_song": projected_full_song,
        "candidate_cache_write_checks_pass": bool(candidate_cache_write_checks_pass),
        "candidate_cache_write_checks": candidate_cache_write_checks,
        "logits_replay_allclose": logits_replay_allclose,
        "logits_replay_max_abs": _max_abs(direct_result.logits, replay_logits),
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the real one-token decoder layer under active-prefix/native q1 settings. "
            "Diagnostic only; not an inference speed claim."
        )
    )
    parser.add_argument("--config-name", default="profile_salvalai_smoke15")
    parser.add_argument("--sequence-index", type=int, default=9)
    parser.add_argument("--active-prefix-bucket-size", type=int, default=64)
    parser.add_argument("--active-prefix-decode-length", type=int, default=None)
    parser.add_argument("--native-q1-rope-cache-self-attention", action="store_true")
    parser.add_argument(
        "--compile-decoder-layer-variant",
        action="store_true",
        help="Also benchmark a diagnostic torch.compile(mode='reduce-overhead') decoder-layer island variant.",
    )
    parser.add_argument(
        "--candidate-decoder-runtime-island",
        action="store_true",
        help=(
            "Also benchmark a verifier-only manual whole-layer runtime-island candidate. "
            "This is diagnostic sizing only, not a production inference path."
        ),
    )
    parser.add_argument(
        "--candidate-native-decoder-layer-island",
        action="store_true",
        help=(
            "Also benchmark verifier-only native decoder-layer MLP-tail candidates at the whole-layer "
            "boundary. This is diagnostic sizing only, not a production inference path."
        ),
    )
    parser.add_argument(
        "--candidate-native-self-cross-prefix",
        action="store_true",
        help=(
            "Also benchmark verifier-only native self-attention plus cross-attention residual prefix "
            "candidates at the full decoder-layer boundary. This is diagnostic sizing only, not a "
            "production inference path."
        ),
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--cuda-graph-replay", action="store_true")
    parser.add_argument("--full-song-decode-steps", type=int, default=7552)
    parser.add_argument("--full-song-main-tokens", type=int, default=7639)
    parser.add_argument("--full-song-model-time-s", type=float, default=28.243)
    parser.add_argument(
        "--include-cache-write-fingerprint",
        action="store_true",
        help=(
            "Include SHA256 fingerprints for the q_len=1 self-cache K/V slot written by each "
            "representative decoder layer. Diagnostic verifier metadata only."
        ),
    )
    parser.add_argument(
        "--verify-cache-write-candidates",
        action="store_true",
        help=(
            "Reset the q_len=1 self-cache K/V slot before cache-writing candidate variants, "
            "run each candidate once, and require the slot fingerprint to match the reference. "
            "Diagnostic verifier only."
        ),
    )
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. model_path=/path/to/model")
    cli_args = parser.parse_args()

    start = time.perf_counter()
    args = _load_args(cli_args.config_name, cli_args.overrides)
    result = profile_decode_decoder_layer_island(
        args,
        sequence_index=cli_args.sequence_index,
        active_prefix_bucket_size=cli_args.active_prefix_bucket_size,
        active_prefix_decode_length=cli_args.active_prefix_decode_length,
        native_q1_rope_cache_self_attention=cli_args.native_q1_rope_cache_self_attention,
        compile_decoder_layer_variant=cli_args.compile_decoder_layer_variant,
        candidate_decoder_runtime_island=cli_args.candidate_decoder_runtime_island,
        candidate_native_decoder_layer_island=cli_args.candidate_native_decoder_layer_island,
        candidate_native_self_cross_prefix=cli_args.candidate_native_self_cross_prefix,
        warmup=cli_args.warmup,
        iters=cli_args.iters,
        atol=cli_args.atol,
        rtol=cli_args.rtol,
        cuda_graph_replay=cli_args.cuda_graph_replay,
        full_song_decode_steps=cli_args.full_song_decode_steps,
        full_song_main_tokens=cli_args.full_song_main_tokens,
        full_song_model_time_s=cli_args.full_song_model_time_s,
        include_cache_write_fingerprint=cli_args.include_cache_write_fingerprint,
        verify_cache_write_candidates=cli_args.verify_cache_write_candidates,
    )
    result["metadata"]["config_name"] = cli_args.config_name
    result["wall_seconds"] = time.perf_counter() - start

    if cli_args.report_path is not None:
        cli_args.report_path.parent.mkdir(parents=True, exist_ok=True)
        cli_args.report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
