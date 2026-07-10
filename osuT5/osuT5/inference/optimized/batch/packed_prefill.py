"""Verifier-only packing of serial B1 prefills into stable merged slots."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import torch
from transformers.modeling_outputs import BaseModelOutput

from ...cache_utils import get_cache
from ..single.session import (
    DecodeSession,
    DecoderCacheState,
    EncoderState,
    GraphState,
    SamplerState,
    StaticInputBuffers,
)


def _require_initialized_layer(layer: Any, *, label: str) -> None:
    if not getattr(layer, "is_initialized", False):
        raise RuntimeError(f"{label} is uninitialized.")
    for name in ("keys", "values"):
        value = getattr(layer, name, None)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{label}.{name} must be a tensor.")
        if value.shape[0] != 1:
            raise ValueError(f"{label}.{name} must own exactly one B1 row.")


def _initialize_merged_layer(
        merged_layer: Any,
        reference_layer: Any,
        *,
        batch_size: int,
        label: str,
) -> None:
    _require_initialized_layer(reference_layer, label=f"reference {label}")
    reference_keys = reference_layer.keys
    if not getattr(merged_layer, "is_initialized", False):
        initializer = getattr(merged_layer, "lazy_initialization", None)
        if not callable(initializer):
            raise TypeError(f"merged {label} cannot be lazily initialized.")
        shape_probe = torch.empty(
            (batch_size, reference_keys.shape[1], 1, reference_keys.shape[-1]),
            dtype=reference_keys.dtype,
            device=reference_keys.device,
        )
        initializer(shape_probe)
    expected_shape = (batch_size, *reference_keys.shape[1:])
    for name in ("keys", "values"):
        value = getattr(merged_layer, name, None)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"merged {label}.{name} must be a tensor.")
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"merged {label}.{name} shape {tuple(value.shape)} != {expected_shape}."
            )


@torch.no_grad()
def pack_static_cache_rows(
        reference_caches: Sequence[Any],
        merged_cache: Any,
) -> dict[str, Any]:
    """Copy complete B1 static-cache backing rows and cross-update flags."""

    if not reference_caches:
        raise ValueError("reference_caches must be non-empty.")
    batch_size = len(reference_caches)
    reference_cfg_scales = [getattr(cache, "cfg_scale", None) for cache in reference_caches]
    if len(set(reference_cfg_scales)) != 1:
        raise ValueError("all B1 caches must use the same cfg_scale.")
    if getattr(merged_cache, "cfg_scale", None) != reference_cfg_scales[0]:
        raise ValueError("merged cache cfg_scale does not match B1 caches.")

    part_reports: dict[str, Any] = {}
    for part_name in ("self_attention_cache", "cross_attention_cache"):
        reference_parts = [getattr(cache, part_name) for cache in reference_caches]
        merged_part = getattr(merged_cache, part_name)
        layer_counts = [len(part.layers) for part in reference_parts]
        if len(set(layer_counts)) != 1 or len(merged_part.layers) != layer_counts[0]:
            raise ValueError(f"{part_name} layer counts do not match.")
        layer_reports: list[dict[str, Any]] = []
        for layer_index, merged_layer in enumerate(merged_part.layers):
            reference_layers = [part.layers[layer_index] for part in reference_parts]
            _initialize_merged_layer(
                merged_layer,
                reference_layers[0],
                batch_size=batch_size,
                label=f"{part_name}.layers[{layer_index}]",
            )
            merged_layer.keys.zero_()
            merged_layer.values.zero_()
            row_reports: list[dict[str, Any]] = []
            for row, reference_layer in enumerate(reference_layers):
                _require_initialized_layer(
                    reference_layer,
                    label=f"row {row} {part_name}.layers[{layer_index}]",
                )
                if reference_layer.keys.shape[1:] != merged_layer.keys.shape[1:]:
                    raise ValueError(f"row {row} {part_name} key shape is incompatible.")
                if reference_layer.values.shape[1:] != merged_layer.values.shape[1:]:
                    raise ValueError(f"row {row} {part_name} value shape is incompatible.")
                merged_layer.keys[row:row + 1].copy_(reference_layer.keys)
                merged_layer.values[row:row + 1].copy_(reference_layer.values)
                key_exact = bool(torch.equal(merged_layer.keys[row:row + 1], reference_layer.keys))
                value_exact = bool(torch.equal(merged_layer.values[row:row + 1], reference_layer.values))
                row_reports.append({
                    "row": row,
                    "keys_bitwise_equal_after_pack": key_exact,
                    "values_bitwise_equal_after_pack": value_exact,
                    "pass": bool(key_exact and value_exact),
                })
            layer_reports.append({
                "layer": layer_index,
                "rows": row_reports,
                "pass": all(row["pass"] for row in row_reports),
            })
        part_reports[part_name] = {
            "layer_count": len(layer_reports),
            "layers": layer_reports,
            "pass": all(layer["pass"] for layer in layer_reports),
        }

    flag_reports: list[dict[str, Any]] = []
    for layer_index in range(len(merged_cache.cross_attention_cache.layers)):
        row_flags = [bool(cache.is_updated.get(layer_index, False)) for cache in reference_caches]
        if len(set(row_flags)) != 1:
            raise ValueError(
                f"cross-attention is_updated flags differ for layer {layer_index}: {row_flags}."
            )
        merged_cache.is_updated[layer_index] = row_flags[0]
        flag_reports.append({
            "layer": layer_index,
            "reference_row_flags": row_flags,
            "merged_flag": bool(merged_cache.is_updated[layer_index]),
            "pass": bool(merged_cache.is_updated[layer_index]) == row_flags[0],
        })
    return {
        "batch_size": batch_size,
        "stable_slot_rows": list(range(batch_size)),
        "parts": part_reports,
        "cross_attention_update_flags": flag_reports,
        "pass": bool(
            all(part["pass"] for part in part_reports.values())
            and all(flag["pass"] for flag in flag_reports)
        ),
    }


def pack_b1_values(values: Sequence[Any], *, name: str) -> Any:
    """Pack one request-local value per row without sharing mutable tensors."""

    if not values:
        raise ValueError(f"{name} values must be non-empty.")
    if all(isinstance(value, torch.Tensor) for value in values):
        tensors = list(values)
        if any(value.ndim == 0 or value.shape[0] != 1 for value in tensors):
            raise ValueError(f"{name} tensors must have a B1 leading dimension.")
        if len({tuple(value.shape[1:]) for value in tensors}) != 1:
            raise ValueError(f"{name} tensor row shapes must match.")
        if len({(value.dtype, value.device) for value in tensors}) != 1:
            raise ValueError(f"{name} tensor dtype/device must match.")
        return torch.cat(tensors, dim=0).contiguous()
    if any(isinstance(value, torch.Tensor) for value in values):
        raise TypeError(f"{name} cannot mix tensor and non-tensor values.")
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"{name} non-tensor values must match across rows.")
    return first


def _pack_condition_kwargs(sessions: Sequence[DecodeSession]) -> dict[str, Any]:
    key_sets = [set(session.condition_kwargs) for session in sessions]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("condition_kwargs keys must match across B1 sessions.")
    return {
        key: pack_b1_values(
            [session.condition_kwargs[key] for session in sessions],
            name=f"condition_kwargs.{key}",
        )
        for key in sorted(key_sets[0])
    }


def _packed_rows_equal(packed: Any, values: Sequence[Any]) -> bool:
    if isinstance(packed, torch.Tensor):
        return all(torch.equal(packed[row:row + 1], value) for row, value in enumerate(values))
    return all(value == packed for value in values)


@torch.no_grad()
def pack_b1_prefill_sessions(
        model: Any,
        reference_sessions: Sequence[DecodeSession],
) -> tuple[DecodeSession, dict[str, Any]]:
    """Allocate one B8 session and pack serial B1 prefill state into its slots."""

    if len(reference_sessions) != 8:
        raise ValueError("the first packed-prefill gate requires exactly eight B1 sessions.")
    if any(session.model is not model for session in reference_sessions):
        raise ValueError("all B1 sessions must share the supplied immutable model.")
    prompt_lengths = [session.cache_state.prompt_length for session in reference_sessions]
    if len(set(prompt_lengths)) != 1:
        raise ValueError("B1 prompt lengths must match for fixed-slot packing.")
    reference_caches = [session.cache_state.cache for session in reference_sessions]
    if any(cache is None for cache in reference_caches):
        raise RuntimeError("every B1 session must own a cache.")
    cfg_scale = float(reference_caches[0].cfg_scale)
    if cfg_scale != 1.0:
        raise ValueError("the first packed-prefill gate supports cfg_scale=1 only.")

    device = torch.device(model.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cuda_start.record()
    else:
        cuda_start = None
        cuda_end = None
    wall_started = time.perf_counter()
    merged_cache = get_cache(model, batch_size=8, num_beams=1, cfg_scale=cfg_scale)
    cache_report = pack_static_cache_rows(reference_caches, merged_cache)
    encoder_rows = []
    prefill_logit_rows = []
    position_rows = []
    for row, session in enumerate(reference_sessions):
        encoder_outputs = session.encoder_state.encoder_outputs
        if encoder_outputs is None or not isinstance(encoder_outputs.last_hidden_state, torch.Tensor):
            raise RuntimeError(f"B1 row {row} has no encoder last_hidden_state.")
        prefill_logits = session.cache_state.prefill_logits
        position = session.cache_state.prefill_cache_position
        if not isinstance(prefill_logits, torch.Tensor):
            raise RuntimeError(f"B1 row {row} has no prefill logits.")
        if not isinstance(position, torch.Tensor):
            raise RuntimeError(f"B1 row {row} has no prefill cache position.")
        encoder_rows.append(encoder_outputs.last_hidden_state)
        prefill_logit_rows.append(prefill_logits)
        position_rows.append(position)
    if any(not torch.equal(position_rows[0], value) for value in position_rows[1:]):
        raise ValueError("B1 prefill cache positions must match.")

    packed_encoder = pack_b1_values(encoder_rows, name="encoder_outputs.last_hidden_state")
    packed_logits = pack_b1_values(prefill_logit_rows, name="prefill_logits")
    packed_prompt = pack_b1_values(
        [session.static_inputs.prompt for session in reference_sessions],
        name="prompt",
    )
    packed_mask = pack_b1_values(
        [session.static_inputs.prompt_attention_mask for session in reference_sessions],
        name="prompt_attention_mask",
    )
    packed_frames = pack_b1_values(
        [session.static_inputs.frames for session in reference_sessions],
        name="frames",
    )
    packed_condition_kwargs = _pack_condition_kwargs(reference_sessions)
    packed_position = position_rows[0].clone(memory_format=torch.contiguous_format)
    merged_session = DecodeSession(
        model=model,
        encoder_state=EncoderState(
            encoder_outputs=BaseModelOutput(last_hidden_state=packed_encoder),
        ),
        cache_state=DecoderCacheState(
            cache=merged_cache,
            prompt_length=prompt_lengths[0],
            prefill_cache_position=packed_position,
            prefill_prepared_inputs={},
            prefill_logits=packed_logits,
            current_cache_position=packed_position,
        ),
        static_inputs=StaticInputBuffers(
            prompt=packed_prompt,
            prompt_attention_mask=packed_mask,
            frames=packed_frames,
        ),
        condition_kwargs=packed_condition_kwargs,
        sampler_state=SamplerState(),
        graph_state=GraphState(graphs={}),
    )
    if cuda_end is not None and cuda_start is not None:
        cuda_end.record()
        cuda_end.synchronize()
        cuda_seconds = cuda_start.elapsed_time(cuda_end) / 1000.0
    else:
        cuda_seconds = None
    wall_seconds = time.perf_counter() - wall_started

    encoder_exact = all(
        torch.equal(packed_encoder[row:row + 1], encoder_rows[row])
        for row in range(8)
    )
    logits_exact = all(
        torch.equal(packed_logits[row:row + 1], prefill_logit_rows[row])
        for row in range(8)
    )
    prompt_exact = _packed_rows_equal(
        packed_prompt,
        [session.static_inputs.prompt for session in reference_sessions],
    )
    mask_exact = _packed_rows_equal(
        packed_mask,
        [session.static_inputs.prompt_attention_mask for session in reference_sessions],
    )
    frames_exact = _packed_rows_equal(
        packed_frames,
        [session.static_inputs.frames for session in reference_sessions],
    )
    condition_exact = all(
        _packed_rows_equal(
            packed_condition_kwargs[key],
            [session.condition_kwargs[key] for session in reference_sessions],
        )
        for key in packed_condition_kwargs
    )
    return merged_session, {
        "strategy": "serial_B1_prefill_then_stable_B8_pack",
        "wall_seconds": wall_seconds,
        "cuda_seconds": cuda_seconds,
        "cache": cache_report,
        "encoder_rows_bitwise_equal_after_pack": encoder_exact,
        "prefill_logits_bitwise_equal_after_pack": logits_exact,
        "prefill_cache_positions_equal": True,
        "prompt_rows_bitwise_equal_after_pack": prompt_exact,
        "prompt_attention_mask_rows_bitwise_equal_after_pack": mask_exact,
        "frame_rows_bitwise_equal_after_pack": frames_exact,
        "condition_rows_bitwise_equal_after_pack": condition_exact,
        "prompt_rows_packed": int(packed_prompt.shape[0]),
        "condition_keys": sorted(packed_condition_kwargs),
        "pass": bool(
            cache_report["pass"]
            and encoder_exact
            and logits_exact
            and prompt_exact
            and mask_exact
            and frames_exact
            and condition_exact
        ),
    }
