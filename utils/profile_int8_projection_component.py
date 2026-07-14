"""Profile incremental INT8 storage for the remaining one-token projections."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager, nullcontext
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from utils.profile_native_prefix_dtype_scout import (  # noqa: E402
    LayerCapture,
    SENTINEL_BUCKETS,
    _accepted_main_session_run,
    _all_cache_snapshots,
    _assert_scout_args,
    _cache_from_static_inputs,
    _capture_cuda_graph,
    _capture_representative_layer,
    _load_args,
    _max_abs,
    _reciprocal_graph_rounds,
    _restore_all_cache,
)


DECODER_LAYERS = 12
REGIONS = (
    "self_norm_qkv",
    "self_out_residual",
    "cross_norm_q",
    "cross_out_residual",
    "final_norm_logits",
)
CURRENT_BASELINE = {
    "self_norm_qkv": "fp16_weight",
    "self_out_residual": "fp16_weight",
    "cross_norm_q": "fp32_weight",
    "cross_out_residual": "fp32_weight",
    "final_norm_logits": "fp16_weight",
}
EXCLUDED_ALREADY_INT8_REGIONS = ("mlp_fc1", "mlp_fc2")
RETAINED_COMPOSITION_VERSION = (
    "k4-split-kv-mixed-weight-shared-rope-k1-remainder-int8-mlp-v1"
)
REQUIRED_DISPATCHES = (
    "weight_only_self_attention_block",
    "native_q1_rope_cache_self_attention",
    "q1_bmm_cross_attention",
    "weight_only_mlp_tail",
    "int8_weight_mlp_tail",
    "weight_only_final_projection",
)
OUTPUTS_PER_BLOCK = (2, 4, 8)
FIXED_MAIN_SAVING_GATE_SECONDS = 0.5
MEASURED_INT8_VS_FP16_MLP_SPEEDUP = 1.3666607805044033


class _DiagnosticRecorder:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def record(self, event: str, **payload: Any) -> None:
        row = {"event": event, "time_unix": time.time(), **payload}
        with self.path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()


def _cuda_sync_checkpoint(
    diagnostics: _DiagnosticRecorder,
    checkpoint: str,
) -> None:
    started = time.perf_counter()
    try:
        torch.cuda.synchronize()
    except Exception as exc:
        diagnostics.record(
            "cuda_sync",
            checkpoint=checkpoint,
            passed=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise RuntimeError(
            f"CUDA synchronization failed at checkpoint {checkpoint}"
        ) from exc
    diagnostics.record(
        "cuda_sync",
        checkpoint=checkpoint,
        passed=True,
        elapsed_seconds=time.perf_counter() - started,
    )


def _tensor_layout(value: torch.Tensor) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor):
        raise TypeError("tensor layout requires a tensor")
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "data_ptr": int(value.data_ptr()),
        "storage_data_ptr": int(value.untyped_storage().data_ptr()),
        "storage_offset": int(value.storage_offset()),
        "numel": int(value.numel()),
        "element_size": int(value.element_size()),
        "contiguous": bool(value.is_contiguous()),
    }


def _validate_tensor_layout(
    value: Any,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype | None = None,
    shape: tuple[int, ...] | None = None,
    contiguous: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.device != device:
        raise RuntimeError(f"{name} must use device {device}, got {value.device}")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}, got {value.dtype}")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if contiguous and not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    layout = _tensor_layout(value)
    if value.numel() > 0 and (
        layout["data_ptr"] <= 0 or layout["storage_data_ptr"] <= 0
    ):
        raise RuntimeError(f"{name} has invalid CUDA storage pointers")
    if layout["storage_offset"] < 0 or any(stride < 0 for stride in value.stride()):
        raise RuntimeError(f"{name} has invalid storage offset or strides")
    return layout


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return value


def _positive_int(value: Any, *, name: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _validate_current_projection_scope(
    *,
    regions: Any,
    baseline_by_region: Any,
    excluded_regions: Any,
) -> None:
    if tuple(regions) != REGIONS:
        raise RuntimeError(
            f"projection scout region map changed: expected {REGIONS}, got {regions}"
        )
    if dict(baseline_by_region) != CURRENT_BASELINE:
        raise RuntimeError("projection scout current-baseline dispatch map changed")
    if tuple(excluded_regions) != EXCLUDED_ALREADY_INT8_REGIONS:
        raise RuntimeError("projection scout must exclude the retained INT8 MLP regions")
    overlap = set(regions) & set(excluded_regions)
    if overlap:
        raise RuntimeError(f"projection scout double-counts INT8 MLP regions: {overlap}")


def _validate_weight_state_metadata(metadata: Any) -> dict[str, Any]:
    value = _object(metadata, name="weight_state")
    exact = {
        "version": "approximate-fp16-weights-fp32-state-v2",
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
        "fp32_selected_decode_matrix_regions": ["cross_query", "cross_output"],
        "outputs_per_block": {
            "self_qkv": 4,
            "self_output": 4,
            "mlp_fc1": 2,
            "mlp_fc2": 4,
            "final_logits": 2,
        },
    }
    for key, expected in exact.items():
        if value.get(key) != expected:
            raise RuntimeError(f"retained weight state changed {key}")
    overlay = _object(value.get("int8_mlp_overlay"), name="int8_mlp_overlay")
    overlay_exact = {
        "version": "per-row-symmetric-int8-mlp-v1",
        "result_class": "documented-drift",
        "exactness_claim": False,
        "scope": "main-model-decoder-mlp-only",
        "composition_control": "k4-split-kv-mixed-weight-shared-rope-v1",
        "replaces_effective_regions": list(EXCLUDED_ALREADY_INT8_REGIONS),
        "retained_unused_fp16_regions": list(EXCLUDED_ALREADY_INT8_REGIONS),
        "fp32_activations_norm_bias_reductions_residual_outputs": True,
        "quantization": "symmetric-per-output-row",
        "dispatch_counter": "int8_weight_mlp_tail",
    }
    for key, expected in overlay_exact.items():
        if overlay.get(key) != expected:
            raise RuntimeError(f"retained INT8 MLP overlay changed {key}")
    return value


def _validate_dispatch_policy(
    record: dict[str, Any],
    *,
    index: int,
) -> dict[str, int]:
    if record.get("optimized_dispatch_mode") != "approximate_weight_only_batch1":
        raise RuntimeError(f"main generation record {index} changed dispatch mode")
    policy = _object(
        record.get("optimized_dispatch_policy"),
        name=f"main_generation[{index}].optimized_dispatch_policy",
    )
    expected_policy = {
        "q1_bmm_cross_attention": {"enabled": True, "disabled_reason": None},
        "native_q1_self_attention": {
            "enabled": False,
            "disabled_reason": "approximate_weight_only",
        },
        "native_q1_rope_cache_self_attention": {
            "enabled": False,
            "disabled_reason": "approximate_weight_only",
        },
        "effective_native_q1_rope_cache_self_attention": {
            "enabled": True,
            "owner": "approximate_weight_only",
            "kernel": "native_q1_rope_cache_attention",
            "standard_attention_hook_enabled": False,
            "split_kv_selector_enabled": True,
        },
        "native_cross_mlp_tail": {
            "enabled": False,
            "disabled_reason": "approximate_weight_only",
        },
    }
    for name, fields in expected_policy.items():
        entry = _object(policy.get(name), name=f"dispatch_policy.{name}")
        for field, expected in fields.items():
            if entry.get(field) != expected:
                raise RuntimeError(
                    f"main generation record {index} changed {name}.{field}"
                )

    hits = _object(
        record.get("optimized_dispatch_capture_hits"),
        name=f"main_generation[{index}].optimized_dispatch_capture_hits",
    )
    values = {
        name: _positive_int(
            hits.get(name),
            name=f"dispatch_hits.{name}",
            allow_zero=True,
        )
        for name in REQUIRED_DISPATCHES
    }
    if values["weight_only_self_attention_block"] != values[
        "native_q1_rope_cache_self_attention"
    ]:
        raise RuntimeError("weight-only self-attention dispatch ownership diverged")
    tail = values["q1_bmm_cross_attention"]
    if not (
        tail
        == values["weight_only_mlp_tail"]
        == values["int8_weight_mlp_tail"]
    ):
        raise RuntimeError("cross/INT8-MLP dispatch ownership diverged")
    for forbidden in ("native_q1_self_attention", "native_cross_mlp_tail"):
        if int(hits.get(forbidden, 0) or 0) != 0:
            raise RuntimeError(f"unexpected retained dispatch {forbidden}")
    split_prefixes = {
        name: int(count)
        for name, count in hits.items()
        if name.startswith(
            "native_q1_rope_cache_self_attention_split_kv_8_prefix_"
        )
    }
    split_total = int(
        hits.get("native_q1_rope_cache_self_attention_split_kv_8", 0) or 0
    )
    if split_total != sum(split_prefixes.values()):
        raise RuntimeError("split-KV aggregate dispatch count diverged")
    if split_total > values["weight_only_self_attention_block"]:
        raise RuntimeError("split-KV dispatch count exceeds self-attention ownership")
    return values


def _validate_k1_captured_remainder(
    record: dict[str, Any],
    *,
    index: int,
) -> dict[str, int]:
    graphs = _object(
        record.get("optimized_cuda_graphs"),
        name=f"main_generation[{index}].optimized_cuda_graphs",
    )
    runtime = _object(
        graphs.get("k8_candidate"),
        name=f"main_generation[{index}].optimized_cuda_graphs.k8_candidate",
    )
    if runtime.get("block_size") != 4:
        raise RuntimeError(f"main generation record {index} did not retain K4")
    if runtime.get("remainder_backend") != "cuda_graph":
        raise RuntimeError(f"main generation record {index} did not retain captured K1")
    counts = {
        name: _positive_int(
            runtime.get(name),
            name=f"k1.{name}",
            allow_zero=True,
        )
        for name in (
            "prefill_steps",
            "eligible_steps",
            "block_replays",
            "remainder_steps",
            "remainder_graph_replays",
            "eager_remainder_steps",
            "physical_steps",
            "logical_steps",
            "wasted_steps",
        )
    }
    if counts["eager_remainder_steps"] != 0:
        raise RuntimeError("captured K1 fell back to eager remainder steps")
    if counts["remainder_graph_replays"] != counts["remainder_steps"]:
        raise RuntimeError("captured K1 did not own every remainder step")
    if counts["physical_steps"] - counts["logical_steps"] != counts["wasted_steps"]:
        raise RuntimeError("K4 physical/logical/padding accounting diverged")
    if counts["wasted_steps"] >= 4:
        raise RuntimeError("K4 post-EOS physical padding exceeds one partial block")
    if counts["wasted_steps"] > 0 and counts["block_replays"] <= 0:
        raise RuntimeError("K4 post-EOS padding occurred without a K4 block replay")
    if counts["eligible_steps"] != 4 * counts["block_replays"]:
        raise RuntimeError("K4 block replay accounting diverged")
    if counts["physical_steps"] != (
        counts["prefill_steps"]
        + counts["eligible_steps"]
        + counts["remainder_steps"]
    ):
        raise RuntimeError("K4/K1 physical-step accounting diverged")
    generated_tokens = _positive_int(
        record.get("generated_tokens"),
        name=f"main_generation[{index}].generated_tokens",
    )
    if counts["logical_steps"] != generated_tokens:
        raise RuntimeError("K4/K1 logical steps diverge from generated tokens")
    expected_rng = {
        "rng_policy": "counter_request_seed_window_prompt_v2",
        "rng_exact": False,
        "rng_early_eos_isolation": True,
        "capture_state_restore_synchronized": True,
    }
    for name, expected in expected_rng.items():
        if runtime.get(name) != expected:
            raise RuntimeError(f"captured K1 changed {name}")
    return counts


def _validate_retained_composition_manifest(manifest: Any) -> dict[str, Any]:
    value = _object(manifest, name="retained_composition")
    if value.get("composition_version") != RETAINED_COMPOSITION_VERSION:
        raise RuntimeError("retained K1+INT8/shared-RoPE composition version changed")
    _validate_current_projection_scope(
        regions=value.get("scouted_regions"),
        baseline_by_region=value.get("baseline_by_region"),
        excluded_regions=value.get("excluded_already_int8_regions"),
    )
    weight_state = _validate_weight_state_metadata(value.get("weight_state"))
    shared_rope = _object(value.get("shared_rope"), name="shared_rope")
    shared_exact = {
        "version": "shared-decoder-rope-v1",
        "scope": "main-model-only",
        "incremental_exactness_claim": True,
        "original_decoder_forward_required": True,
    }
    for name, expected in shared_exact.items():
        if shared_rope.get(name) != expected:
            raise RuntimeError(f"retained shared RoPE changed {name}")
    rope_stats = _object(shared_rope.get("stats"), name="shared_rope.stats")
    if rope_stats.get("module_count") != DECODER_LAYERS or rope_stats.get("group_count") != 1:
        raise RuntimeError("shared RoPE no longer covers the complete decoder")
    forwards = _positive_int(rope_stats.get("forwards"), name="shared_rope.forwards")
    if rope_stats.get("computes") != forwards or rope_stats.get("expected_computes") != forwards:
        raise RuntimeError("shared RoPE compute accounting diverged")
    expected_reuses = (DECODER_LAYERS - 1) * forwards
    if rope_stats.get("reuses") != expected_reuses or rope_stats.get(
        "expected_reuses"
    ) != expected_reuses:
        raise RuntimeError("shared RoPE reuse accounting diverged")

    records = value.get("main_generation_records")
    if not isinstance(records, list) or not records:
        raise RuntimeError("retained composition has no main generation records")
    totals = {
        "records": len(records),
        "remainder_steps": 0,
        "remainder_graph_replays": 0,
        "eager_remainder_steps": 0,
        "physical_steps": 0,
        "logical_steps": 0,
        "k4_padding_steps": 0,
    }
    dispatch_totals = {name: 0 for name in REQUIRED_DISPATCHES}
    for index, raw in enumerate(records):
        record = _object(raw, name=f"main_generation[{index}]")
        dispatch = _validate_dispatch_policy(record, index=index)
        for name, count in dispatch.items():
            dispatch_totals[name] += count
        counts = _validate_k1_captured_remainder(record, index=index)
        totals["remainder_steps"] += counts["remainder_steps"]
        totals["remainder_graph_replays"] += counts["remainder_graph_replays"]
        totals["eager_remainder_steps"] += counts["eager_remainder_steps"]
        totals["physical_steps"] += counts["physical_steps"]
        totals["logical_steps"] += counts["logical_steps"]
        totals["k4_padding_steps"] += counts["wasted_steps"]
    if totals["remainder_steps"] <= 0 or totals["physical_steps"] <= 0:
        raise RuntimeError("retained K1 evidence contains no real decode work")
    missing_dispatch = [
        name for name, count in dispatch_totals.items() if count <= 0
    ]
    if missing_dispatch:
        raise RuntimeError(
            f"retained composition never captured dispatches {missing_dispatch}"
        )
    return {
        "composition_version": RETAINED_COMPOSITION_VERSION,
        "baseline_by_region": dict(CURRENT_BASELINE),
        "excluded_already_int8_regions": list(EXCLUDED_ALREADY_INT8_REGIONS),
        "weight_state": weight_state,
        "shared_rope": shared_rope,
        "k1": totals,
        "dispatch_totals": dispatch_totals,
        "pass": True,
    }


def _eps(module: torch.nn.Module) -> float:
    value = getattr(module, "eps", None)
    return float(torch.finfo(torch.float32).eps if value is None else value)


def _locate_decoder(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Module], torch.nn.Module, torch.nn.Module]:
    from osuT5.osuT5.model.custom_transformers.modeling_varwhisper import (
        VarWhisperDecoderLayer,
    )

    layers = [
        module for module in model.modules() if isinstance(module, VarWhisperDecoderLayer)
    ]
    if len(layers) != DECODER_LAYERS:
        raise RuntimeError(
            f"INT8 projection scout requires {DECODER_LAYERS} decoder layers, "
            f"found {len(layers)}"
        )
    named = dict(model.named_modules())
    layer_names = [name for name, value in named.items() if value is layers[0]]
    if len(layer_names) != 1 or ".layers." not in layer_names[0]:
        raise RuntimeError("could not uniquely locate the decoder parent")
    decoder = named[layer_names[0].rsplit(".layers.", 1)[0]]
    final_norm = getattr(decoder, "layer_norm", None)
    projections = {
        id(module): module
        for name, module in named.items()
        if name == "proj_out" or name.endswith(".proj_out")
    }
    if not isinstance(final_norm, torch.nn.Module) or len(projections) != 1:
        raise RuntimeError("could not uniquely locate final norm and vocabulary projection")
    return layers, final_norm, next(iter(projections.values()))


def _validate_restored_capture_inputs(
    model: torch.nn.Module,
    static_inputs: dict[str, Any],
    *,
    prefix: int,
    diagnostics: _DiagnosticRecorder,
) -> dict[str, Any]:
    from osuT5.osuT5.inference.optimized.scout.native_prefix import (
        _cache_tensors,
    )

    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("capture model has no parameters") from exc
    if device.type != "cuda":
        raise RuntimeError("capture model must remain on CUDA")
    layouts = {
        "decoder_input_ids": _validate_tensor_layout(
            static_inputs.get("decoder_input_ids"),
            name="decoder_input_ids",
            device=device,
            dtype=torch.long,
            shape=(1, 1),
            contiguous=True,
        ),
        "cache_position": _validate_tensor_layout(
            static_inputs.get("cache_position"),
            name="cache_position",
            device=device,
            dtype=torch.long,
            shape=(1,),
            contiguous=True,
        ),
        "decoder_position_ids": _validate_tensor_layout(
            static_inputs.get("decoder_position_ids"),
            name="decoder_position_ids",
            device=device,
            dtype=torch.long,
            shape=(1, 1),
            contiguous=True,
        ),
    }
    mask = static_inputs.get("decoder_attention_mask")
    mask_layout = _validate_tensor_layout(
        mask,
        name="decoder_attention_mask",
        device=device,
        dtype=torch.float32,
    )
    if mask.ndim != 4 or tuple(mask.shape[:3]) != (1, 1, 1):
        raise ValueError("decoder_attention_mask must have shape [1, 1, 1, length]")
    if mask.shape[-1] < prefix:
        raise ValueError("decoder_attention_mask is shorter than the active prefix")
    layouts["decoder_attention_mask"] = mask_layout
    position = int(static_inputs["cache_position"].detach().item())
    position_id = int(static_inputs["decoder_position_ids"].detach().item())
    if position != position_id:
        raise RuntimeError("cache and decoder positions disagree")
    if not 0 <= position < prefix:
        raise RuntimeError(
            f"cache position {position} is outside active prefix {prefix}"
        )
    cache = _cache_from_static_inputs(static_inputs)
    cache_layouts: dict[str, Any] = {}
    for layer_idx in range(DECODER_LAYERS):
        for kind in ("self", "cross"):
            keys, values = _cache_tensors(cache, kind, layer_idx)
            key_name = f"{kind}.{layer_idx}.keys"
            value_name = f"{kind}.{layer_idx}.values"
            key_layout = _validate_tensor_layout(
                keys,
                name=key_name,
                device=device,
                dtype=torch.float32,
                contiguous=True,
            )
            value_layout = _validate_tensor_layout(
                values,
                name=value_name,
                device=device,
                dtype=torch.float32,
                contiguous=True,
            )
            if keys.ndim != 4 or values.shape != keys.shape:
                raise ValueError(f"{kind} layer {layer_idx} cache shapes are invalid")
            if kind == "self" and keys.shape[-2] < prefix:
                raise ValueError(
                    f"self layer {layer_idx} cache is shorter than prefix {prefix}"
                )
            if keys.data_ptr() == values.data_ptr():
                raise RuntimeError(f"{kind} layer {layer_idx} key/value storage aliases")
            cache_layouts[key_name] = key_layout
            cache_layouts[value_name] = value_layout
    encoder_outputs = static_inputs.get("encoder_outputs")
    encoder_hidden = getattr(encoder_outputs, "last_hidden_state", None)
    if isinstance(encoder_hidden, torch.Tensor):
        layouts["encoder_outputs.last_hidden_state"] = _validate_tensor_layout(
            encoder_hidden,
            name="encoder_outputs.last_hidden_state",
            device=device,
            dtype=torch.float32,
        )
    report = {
        "prefix": prefix,
        "cache_position": position,
        "static_tensors": layouts,
        "cache_tensors": cache_layouts,
    }
    diagnostics.record("restored_capture_inputs", **report)
    return report


def _capture_final_input(
    model: torch.nn.Module,
    final_decoder_layer: torch.nn.Module,
    static_inputs: dict[str, Any],
) -> torch.Tensor:
    captured: list[torch.Tensor] = []

    def hook(module, args, output):
        del module, args
        if captured:
            raise RuntimeError("final decoder layer ran more than once")
        hidden = output[0] if isinstance(output, tuple) and output else None
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("final decoder layer did not return hidden states")
        captured.append(hidden.detach().clone())

    handle = final_decoder_layer.register_forward_hook(hook)
    try:
        model(**static_inputs, return_dict=True)
        torch.cuda.synchronize()
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("did not capture one decoder final-norm input")
    return captured[0]


def _prepare_region_inputs(
    capture: LayerCapture,
    fp16_packs: dict[str, Any],
) -> dict[str, torch.Tensor]:
    from osuT5.osuT5.inference.optimized.kernels.decoder_layer import (
        native_one_token_linear_residual,
        native_one_token_rmsnorm_linear,
    )
    from osuT5.osuT5.inference.optimized.kernels.q1_attention import (
        native_q1_rope_cache_attention,
    )
    from osuT5.osuT5.inference.optimized.kernels.dispatch import (
        _q1_bmm_cross_attention,
    )
    from osuT5.osuT5.inference.optimized.kernels.weight_only import (
        weight_only_linear_residual,
        weight_only_rmsnorm_linear,
    )
    from osuT5.osuT5.inference.optimized.scout.native_prefix import (
        _cache_tensors,
        _trim_mask,
    )

    module = capture.module
    self_attn = module.self_attn
    hidden = capture.hidden_states
    qkv = weight_only_rmsnorm_linear(
        hidden,
        module.self_attn_layer_norm.weight,
        fp16_packs["self_norm_qkv"],
        eps=_eps(module.self_attn_layer_norm),
        outputs_per_block=4,
    ).view(1, 1, 3, self_attn.num_heads, self_attn.head_dim)
    self_keys, self_values = _cache_tensors(
        capture.past_key_value, "self", capture.layer_idx
    )
    cos, sin = self_attn.rotary_emb(qkv, position_ids=capture.position_ids)
    self_output = native_q1_rope_cache_attention(
        qkv,
        self_keys,
        self_values,
        cos,
        sin,
        capture.cache_position,
        _trim_mask(capture.attention_mask, capture.active_prefix_length),
        capture.active_prefix_length,
    ).transpose(1, 2).contiguous().view(1, 1, self_attn.all_head_size)
    after_self = weight_only_linear_residual(
        self_output,
        hidden,
        fp16_packs["self_out_residual"],
        outputs_per_block=4,
    )

    cross_attn = module.cross_attn
    query = native_one_token_rmsnorm_linear(
        after_self,
        module.cross_attn_layer_norm.weight,
        cross_attn.Wq.weight,
        cross_attn.Wq.bias,
        eps=_eps(module.cross_attn_layer_norm),
        outputs_per_block=8,
    ).view(1, 1, cross_attn.num_heads, cross_attn.head_dim).transpose(1, 2)
    cross_keys, cross_values = _cache_tensors(
        capture.past_key_value, "cross", capture.layer_idx
    )
    cross_output = _q1_bmm_cross_attention(
        query,
        cross_keys,
        cross_values,
        expected_dtype=torch.float32,
    ).transpose(1, 2).contiguous().view(1, 1, cross_attn.all_head_size)
    after_cross = native_one_token_linear_residual(
        cross_output,
        after_self,
        cross_attn.Wo.weight,
        cross_attn.Wo.bias,
        outputs_per_block=8,
    )
    return {
        "layer_input": hidden,
        "self_attention_output": self_output,
        "after_self": after_self,
        "cross_attention_output": cross_output,
        "after_cross": after_cross,
    }


def _validate_projection_call_inputs(
    inputs: dict[str, torch.Tensor],
    final_input: torch.Tensor,
    fp16_packs: dict[str, Any],
    int8_packs: dict[str, Any],
    *,
    prefix: int,
    diagnostics: _DiagnosticRecorder,
) -> None:
    device = final_input.device
    tensor_layouts = {
        name: _validate_tensor_layout(
            value,
            name=f"prefix {prefix} {name}",
            device=device,
            dtype=torch.float32,
            contiguous=True,
        )
        for name, value in {**inputs, "final_input": final_input}.items()
    }
    pack_layouts: dict[str, Any] = {}
    for name, pack in fp16_packs.items():
        pack_layouts[f"fp16.{name}.weight"] = _validate_tensor_layout(
            getattr(pack, "weight", None),
            name=f"fp16 {name} weight",
            device=device,
            dtype=torch.float16,
            contiguous=True,
        )
        bias = getattr(pack, "bias", None)
        if bias is not None:
            pack_layouts[f"fp16.{name}.bias"] = _validate_tensor_layout(
                bias,
                name=f"fp16 {name} bias",
                device=device,
                dtype=torch.float32,
                contiguous=True,
            )
    for name, pack in int8_packs.items():
        pack_layouts[f"int8.{name}.weight"] = _validate_tensor_layout(
            getattr(pack, "weight", None),
            name=f"int8 {name} weight",
            device=device,
            dtype=torch.int8,
            contiguous=True,
        )
        pack_layouts[f"int8.{name}.scale"] = _validate_tensor_layout(
            getattr(pack, "scale", None),
            name=f"int8 {name} scale",
            device=device,
            dtype=torch.float32,
            contiguous=True,
        )
        bias = getattr(pack, "bias", None)
        if bias is not None:
            pack_layouts[f"int8.{name}.bias"] = _validate_tensor_layout(
                bias,
                name=f"int8 {name} bias",
                device=device,
                dtype=torch.float32,
                contiguous=True,
            )
    diagnostics.record(
        "projection_call_inputs",
        prefix=prefix,
        tensors=tensor_layouts,
        packs=pack_layouts,
    )


def _baseline_functions(
    capture: LayerCapture,
    inputs: dict[str, torch.Tensor],
    final_input: torch.Tensor,
    final_norm: torch.nn.Module,
    final_projection: torch.nn.Module,
    fp16_packs: dict[str, Any],
) -> dict[str, Callable[[], torch.Tensor]]:
    from osuT5.osuT5.inference.optimized.kernels.decoder_layer import (
        native_one_token_linear_residual,
        native_one_token_rmsnorm_linear,
    )
    from osuT5.osuT5.inference.optimized.kernels.weight_only import (
        weight_only_linear_residual,
        weight_only_rmsnorm_linear,
    )

    module = capture.module
    return {
        "self_norm_qkv": lambda: weight_only_rmsnorm_linear(
            inputs["layer_input"],
            module.self_attn_layer_norm.weight,
            fp16_packs["self_norm_qkv"],
            eps=_eps(module.self_attn_layer_norm),
            outputs_per_block=4,
        ),
        "self_out_residual": lambda: weight_only_linear_residual(
            inputs["self_attention_output"],
            inputs["layer_input"],
            fp16_packs["self_out_residual"],
            outputs_per_block=4,
        ),
        "cross_norm_q": lambda: native_one_token_rmsnorm_linear(
            inputs["after_self"],
            module.cross_attn_layer_norm.weight,
            module.cross_attn.Wq.weight,
            module.cross_attn.Wq.bias,
            eps=_eps(module.cross_attn_layer_norm),
            outputs_per_block=8,
        ),
        "cross_out_residual": lambda: native_one_token_linear_residual(
            inputs["cross_attention_output"],
            inputs["after_self"],
            module.cross_attn.Wo.weight,
            module.cross_attn.Wo.bias,
            outputs_per_block=8,
        ),
        "final_norm_logits": lambda: weight_only_rmsnorm_linear(
            final_input,
            final_norm.weight,
            fp16_packs["final_norm_logits"],
            eps=_eps(final_norm),
            outputs_per_block=2,
        ),
    }


def _candidate_functions(
    capture: LayerCapture,
    inputs: dict[str, torch.Tensor],
    final_input: torch.Tensor,
    final_norm: torch.nn.Module,
    int8_packs: dict[str, Any],
    *,
    outputs_per_block: int,
) -> dict[str, Callable[[], torch.Tensor]]:
    from osuT5.osuT5.inference.optimized.scout.int8_mlp import (
        int8_weight_linear_residual,
        int8_weight_rmsnorm_linear,
    )

    module = capture.module
    return {
        "self_norm_qkv": lambda: int8_weight_rmsnorm_linear(
            inputs["layer_input"],
            module.self_attn_layer_norm.weight,
            int8_packs["self_norm_qkv"],
            eps=_eps(module.self_attn_layer_norm),
            outputs_per_block=outputs_per_block,
        ),
        "self_out_residual": lambda: int8_weight_linear_residual(
            inputs["self_attention_output"],
            inputs["layer_input"],
            int8_packs["self_out_residual"],
            outputs_per_block=outputs_per_block,
        ),
        "cross_norm_q": lambda: int8_weight_rmsnorm_linear(
            inputs["after_self"],
            module.cross_attn_layer_norm.weight,
            int8_packs["cross_norm_q"],
            eps=_eps(module.cross_attn_layer_norm),
            outputs_per_block=outputs_per_block,
        ),
        "cross_out_residual": lambda: int8_weight_linear_residual(
            inputs["cross_attention_output"],
            inputs["after_self"],
            int8_packs["cross_out_residual"],
            outputs_per_block=outputs_per_block,
        ),
        "final_norm_logits": lambda: int8_weight_rmsnorm_linear(
            final_input,
            final_norm.weight,
            int8_packs["final_norm_logits"],
            eps=_eps(final_norm),
            outputs_per_block=outputs_per_block,
        ),
    }


def _observe(graph) -> tuple[torch.Tensor, torch.Tensor]:
    graph.graph.replay()
    torch.cuda.synchronize()
    first = graph.outputs.detach().float().cpu().clone()
    graph.graph.replay()
    torch.cuda.synchronize()
    second = graph.outputs.detach().float().cpu().clone()
    return first, second


def summarize_component(
    buckets: dict[str, dict[str, Any]],
    *,
    total_replays: int,
) -> dict[str, Any]:
    if not buckets:
        raise ValueError("INT8 projection summary requires bucket evidence")
    measured_replays = sum(int(entry["decode_replays"]) for entry in buckets.values())
    if measured_replays <= 0 or total_replays < measured_replays:
        raise ValueError("invalid measured/total replay counts")
    region_summary: dict[str, Any] = {}
    selected_fixed_saving = 0.0
    current_projection_family_seconds = 0.0
    failures: dict[str, list[str]] = {}
    for region in REGIONS:
        multiplier = 1 if region == "final_norm_logits" else DECODER_LAYERS
        baseline_ms = sum(
            int(entry["decode_replays"])
            * multiplier
            * float(entry["regions"][region]["baseline_ms_per_call"])
            for entry in buckets.values()
        )
        for prefix, entry in buckets.items():
            baseline = entry["regions"][region]
            if not all(
                bool(baseline[name])
                for name in (
                    "baseline_finite",
                    "baseline_repeat_deterministic",
                    "baseline_memory_stable",
                )
            ):
                failures.setdefault(prefix, []).append(f"{region}:baseline")
        candidate_seconds: dict[str, float] = {}
        valid: dict[str, bool] = {}
        for opb in OUTPUTS_PER_BLOCK:
            key = str(opb)
            candidate_ms = 0.0
            candidate_valid = True
            for prefix, entry in buckets.items():
                candidate = entry["regions"][region]["candidates"][key]
                candidate_ms += (
                    int(entry["decode_replays"])
                    * multiplier
                    * float(candidate["ms_per_call"])
                )
                candidate_valid &= all(
                    bool(candidate[name])
                    for name in ("finite", "repeat_deterministic", "memory_stable")
                )
            candidate_seconds[key] = candidate_ms / 1000.0
            valid[key] = candidate_valid
        valid_keys = [key for key, passed in valid.items() if passed]
        if not valid_keys:
            raise RuntimeError(f"no valid INT8 configuration for {region}")
        selected_key = min(valid_keys, key=lambda key: candidate_seconds[key])
        baseline_seconds = baseline_ms / 1000.0
        fixed_baseline_seconds = baseline_seconds * total_replays / measured_replays
        current_projection_family_seconds += fixed_baseline_seconds
        measured_saving = baseline_seconds - candidate_seconds[selected_key]
        fixed_saving = measured_saving * total_replays / measured_replays
        retained = fixed_saving > 0
        if retained:
            selected_fixed_saving += fixed_saving
        region_summary[region] = {
            "current_baseline": CURRENT_BASELINE[region],
            "selected_outputs_per_block": int(selected_key),
            "measured_baseline_seconds": baseline_seconds,
            "fixed_work_current_baseline_seconds": fixed_baseline_seconds,
            "measured_candidate_seconds": candidate_seconds[selected_key],
            "measured_saving_seconds": measured_saving,
            "fixed_work_saving_seconds": fixed_saving,
            "retained_by_selective_policy": retained,
            "candidate_seconds_by_outputs_per_block": candidate_seconds,
        }
    modeled_fraction = 1.0 - 1.0 / MEASURED_INT8_VS_FP16_MLP_SPEEDUP
    modeled_live_ceiling = current_projection_family_seconds * modeled_fraction
    return {
        "measured_replays": measured_replays,
        "total_replays": int(total_replays),
        "coverage_fraction": measured_replays / total_replays,
        "regions": region_summary,
        "current_projection_family_seconds": current_projection_family_seconds,
        "modeled_int8_vs_fp16_speedup": MEASURED_INT8_VS_FP16_MLP_SPEEDUP,
        "live_modeled_realistic_ceiling_seconds": modeled_live_ceiling,
        "live_modeled_ceiling_clears_gate": (
            modeled_live_ceiling >= FIXED_MAIN_SAVING_GATE_SECONDS
        ),
        "selective_fixed_work_main_saving_seconds": selected_fixed_saving,
        "saving_gate_seconds": FIXED_MAIN_SAVING_GATE_SECONDS,
        "invariant_failures": failures,
        "invariants_pass": not failures,
        "sizing_pass": selected_fixed_saving >= FIXED_MAIN_SAVING_GATE_SECONDS,
        "promotion_pass": (
            not failures and selected_fixed_saving >= FIXED_MAIN_SAVING_GATE_SECONDS
        ),
    }


def _snapshot_pre_replay_entry(
    runtime_entry: Any,
    prefix: int,
    diagnostics: _DiagnosticRecorder,
) -> dict[str, Any] | None:
    if prefix not in SENTINEL_BUCKETS:
        return None
    static_inputs = getattr(runtime_entry, "static_inputs", None)
    if not isinstance(static_inputs, dict) or not static_inputs:
        raise RuntimeError(f"prefix {prefix} pre-replay static inputs are missing")
    position = static_inputs.get("cache_position")
    if not isinstance(position, torch.Tensor) or position.numel() != 1:
        raise RuntimeError(f"prefix {prefix} pre-replay cache position is invalid")
    _cuda_sync_checkpoint(diagnostics, f"prefix_{prefix}.pre_replay.before_snapshot")
    position_value = int(position.detach().item())
    if not 0 <= position_value < prefix:
        raise RuntimeError(
            f"prefix {prefix} pre-replay cache position {position_value} is invalid"
        )
    cache = _cache_from_static_inputs(static_inputs)
    cache_snapshots = _all_cache_snapshots(cache, position)
    static_tensors = {
        name: value.detach().clone()
        for name, value in static_inputs.items()
        if isinstance(value, torch.Tensor)
    }
    static_ownership = {
        name: _tensor_layout(value)
        for name, value in static_inputs.items()
        if isinstance(value, torch.Tensor)
    }
    _cuda_sync_checkpoint(diagnostics, f"prefix_{prefix}.pre_replay.after_snapshot")
    diagnostics.record(
        "pre_replay_snapshot",
        prefix=prefix,
        cache_position=position_value,
        static_tensors=static_ownership,
        cache_layers=len(cache_snapshots),
    )
    return {
        "prefix": prefix,
        "cache_position": position_value,
        "static_tensors": static_tensors,
        "static_ownership": static_ownership,
        "cache_snapshots": cache_snapshots,
    }


def _restore_pre_replay_entry(
    entry: dict[str, Any],
    diagnostics: _DiagnosticRecorder,
) -> dict[str, Any]:
    prefix = int(entry["active_prefix_length"])
    snapshot = _object(
        entry.get("profile_pre_replay_snapshot"),
        name=f"prefix {prefix} profile_pre_replay_snapshot",
    )
    if snapshot.get("prefix") != prefix:
        raise RuntimeError(f"prefix {prefix} pre-replay snapshot ownership changed")
    static_inputs = entry["static_inputs"]
    static_tensors = _object(
        snapshot.get("static_tensors"),
        name=f"prefix {prefix} pre-replay static tensors",
    )
    ownership = _object(
        snapshot.get("static_ownership"),
        name=f"prefix {prefix} pre-replay static ownership",
    )
    if set(static_tensors) != set(ownership):
        raise RuntimeError(f"prefix {prefix} pre-replay tensor evidence diverged")
    with torch.no_grad():
        for name, saved in static_tensors.items():
            current = static_inputs.get(name)
            if not isinstance(saved, torch.Tensor) or not isinstance(
                current, torch.Tensor
            ):
                raise RuntimeError(
                    f"prefix {prefix} pre-replay tensor {name} is missing"
                )
            if _tensor_layout(current) != ownership[name]:
                raise RuntimeError(
                    f"prefix {prefix} pre-replay tensor {name} changed ownership"
                )
            if (
                current.shape != saved.shape
                or current.dtype != saved.dtype
                or current.device != saved.device
            ):
                raise RuntimeError(
                    f"prefix {prefix} pre-replay tensor {name} changed signature"
                )
            current.copy_(saved)
    cache_snapshots = snapshot.get("cache_snapshots")
    if not isinstance(cache_snapshots, list) or len(cache_snapshots) != DECODER_LAYERS:
        raise RuntimeError(f"prefix {prefix} pre-replay cache snapshot is incomplete")
    _restore_all_cache(_cache_from_static_inputs(static_inputs), cache_snapshots)
    _cuda_sync_checkpoint(diagnostics, f"prefix_{prefix}.pre_replay.restore")
    position = static_inputs.get("cache_position")
    if not isinstance(position, torch.Tensor) or position.numel() != 1:
        raise RuntimeError(f"prefix {prefix} restored cache position is invalid")
    position_value = int(position.detach().item())
    if position_value != snapshot.get("cache_position"):
        raise RuntimeError(f"prefix {prefix} restored the wrong cache position")
    if not 0 <= position_value < prefix:
        raise RuntimeError(
            f"prefix {prefix} restored cache position {position_value} is outside prefix"
        )
    position_ids = static_inputs.get("decoder_position_ids")
    if (
        not isinstance(position_ids, torch.Tensor)
        or position_ids.shape != (1, 1)
        or position_ids.dtype != torch.long
        or int(position_ids.detach().item()) != position_value
    ):
        raise RuntimeError(f"prefix {prefix} restored decoder position is inconsistent")
    diagnostics.record(
        "pre_replay_restore",
        prefix=prefix,
        cache_position=position_value,
        static_tensors={
            name: _tensor_layout(value)
            for name, value in static_inputs.items()
            if isinstance(value, torch.Tensor)
        },
    )
    return snapshot


def _validate_live_graph_cache(
    graph_cache: Any,
) -> dict[int, list[dict[str, Any]]]:
    if not isinstance(graph_cache, dict) or not graph_cache:
        raise TypeError("retained ProductionDecodeSession.graph_cache must be non-empty")
    entries: dict[int, list[dict[str, Any]]] = {}
    for cache_key, raw in graph_cache.items():
        entry = _object(raw, name="retained composed graph entry")
        if not isinstance(cache_key, tuple) or len(cache_key) != 7:
            raise RuntimeError("retained composed graph has an invalid cache key")
        if cache_key[0] != "k8_candidate":
            raise RuntimeError("retained graph cache contains a non-K4 entry")
        if cache_key[1] != 4:
            raise RuntimeError("retained composed graph is not the accepted K4 block")
        if cache_key[2] is not True:
            raise RuntimeError("retained composed graph did not capture K1 remainders")
        if not isinstance(cache_key[5], bool):
            raise TypeError("retained composed graph sampling key must be a bool")
        prefix = entry.get("active_prefix_length")
        if isinstance(prefix, bool) or not isinstance(prefix, int) or prefix <= 0:
            raise ValueError("retained graph entry has an invalid active prefix")
        if cache_key[4] != prefix:
            raise RuntimeError(f"retained graph prefix {prefix} disagrees with its key")
        if entry.get("k8_candidate") is not True:
            raise RuntimeError(f"retained graph prefix {prefix} is not a K4 candidate")
        runtime_entry = entry.get("runtime_entry")
        if runtime_entry is None:
            raise RuntimeError(f"retained graph prefix {prefix} is missing runtime_entry")
        state = getattr(runtime_entry, "state", None)
        if state is None or cache_key[3] != id(state):
            raise RuntimeError(
                f"retained graph prefix {prefix} resolved the wrong persistent state"
            )
        if getattr(runtime_entry, "processor", None) is None:
            raise RuntimeError(f"retained graph prefix {prefix} is missing its processor")
        parent = getattr(runtime_entry, "parent", None)
        remainder_parent = getattr(runtime_entry, "remainder_parent", None)
        if parent is None:
            raise RuntimeError(f"retained graph prefix {prefix} is missing its K4 parent")
        if remainder_parent is None:
            raise RuntimeError(
                f"retained graph prefix {prefix} is missing its captured K1 parent"
            )
        model_graph = getattr(parent, "model_graph", None)
        tail_graph = getattr(parent, "tail_graph", None)
        remainder_model_graph = getattr(remainder_parent, "model_graph", None)
        remainder_tail_graph = getattr(remainder_parent, "tail_graph", None)
        if model_graph is None or tail_graph is None:
            raise RuntimeError(f"retained graph prefix {prefix} has invalid K4 children")
        if model_graph is not remainder_model_graph or tail_graph is not remainder_tail_graph:
            raise RuntimeError(
                f"retained graph prefix {prefix} K4/K1 parents do not own the same children"
            )
        static_inputs = getattr(runtime_entry, "static_inputs", None)
        if not isinstance(static_inputs, dict) or not static_inputs:
            raise RuntimeError(f"retained graph prefix {prefix} has invalid static inputs")
        for required in ("past_key_values", "cache_position"):
            if required not in static_inputs:
                raise RuntimeError(
                    f"retained graph prefix {prefix} static inputs are missing {required}"
                )
        expected_signature = tuple(
            (name, tuple(value.shape), str(value.dtype), str(value.device))
            if isinstance(value, torch.Tensor)
            else (name, type(value).__name__, id(value))
            for name, value in sorted(static_inputs.items())
        )
        if cache_key[6] != expected_signature:
            raise RuntimeError(
                f"retained graph prefix {prefix} static-input signature changed"
            )
        outputs = getattr(runtime_entry, "outputs", None)
        if outputs is None:
            raise RuntimeError(f"retained graph prefix {prefix} is missing model outputs")
        capture_seconds = entry.get("capture_seconds")
        if (
            isinstance(capture_seconds, bool)
            or not isinstance(capture_seconds, (int, float))
            or capture_seconds <= 0
            or capture_seconds != getattr(runtime_entry, "capture_seconds", None)
        ):
            raise RuntimeError(
                f"retained graph prefix {prefix} has invalid capture ownership"
            )
        stats = _object(entry.get("k8_stats"), name=f"prefix {prefix} k8_stats")
        expected_stats = {
            "k8_candidate": True,
            "block_size": 4,
            "remainder_backend": "cuda_graph",
            "eager_remainder_steps": 0,
        }
        for name, expected in expected_stats.items():
            if stats.get(name) != expected:
                raise RuntimeError(
                    f"retained graph prefix {prefix} changed K4/K1 {name}"
                )
        sampling_mode = stats.get("sampling_mode")
        if sampling_mode not in {"sample", "greedy"}:
            raise RuntimeError(
                f"retained graph prefix {prefix} has invalid sampling metadata"
            )
        if cache_key[5] != (sampling_mode == "sample"):
            raise RuntimeError(
                f"retained graph prefix {prefix} sampling ownership changed"
            )
        profile_snapshot = entry.get("profile_pre_replay_snapshot")
        if prefix in SENTINEL_BUCKETS and not isinstance(profile_snapshot, dict):
            raise RuntimeError(
                f"retained graph prefix {prefix} is missing its pre-replay snapshot"
            )
        if profile_snapshot is not None and (
            not isinstance(profile_snapshot, dict)
            or profile_snapshot.get("prefix") != prefix
        ):
            raise RuntimeError(
                f"retained graph prefix {prefix} has invalid pre-replay ownership"
            )
        _positive_int(entry["decode_replays"], name=f"prefix {prefix} decode_replays")
        entries.setdefault(prefix, []).append(
            {
                "graph": remainder_model_graph,
                "outputs": outputs,
                "static_inputs": static_inputs,
                "decode_replays": entry["decode_replays"],
                "active_prefix_length": prefix,
                "capture_seconds": float(capture_seconds),
                "graph_source": "captured_k1_remainder_parent.model_graph",
                "profile_pre_replay_snapshot": profile_snapshot,
            }
        )
    missing = sorted(set(SENTINEL_BUCKETS) - set(entries))
    if missing:
        raise RuntimeError(f"retained graph cache is missing sentinel buckets {missing}")
    return dict(sorted(entries.items()))


def _select_live_buckets(
    entries: dict[int, list[dict[str, Any]]],
    mode: str,
) -> dict[int, dict[str, Any]]:
    if mode == "sentinel":
        selected = SENTINEL_BUCKETS
    elif mode == "all":
        selected = tuple(entries)
    else:
        raise ValueError("bucket mode must be 'sentinel' or 'all'")
    ambiguous = {
        prefix: len(entries[prefix])
        for prefix in selected
        if len(entries[prefix]) != 1
    }
    if ambiguous:
        raise RuntimeError(
            "retained graph selection has ambiguous input signatures: "
            f"{ambiguous}"
        )
    return {prefix: entries[prefix][0] for prefix in selected}


def _retained_main_session_run(
    args,
    *,
    output_path: Path,
    diagnostics: _DiagnosticRecorder,
) -> dict[str, Any]:
    """Run and validate the exact retained main composition before component work."""

    import inference

    from osuT5.osuT5.inference.optimized.scout.shared_rope import (
        SharedRopeStats,
        shared_decoder_rope_context,
    )
    from osuT5.osuT5.inference.optimized.single.k8_runtime import (
        install_k8_candidate,
    )
    from utils.run_k4_shared_rope_approximate_weight_only import (
        _validated_shared_rope_evidence,
    )
    from utils.run_k4_shared_rope_k1_remainder_int8_mlp_weight_only import (
        COMPOSITION_VERSION,
    )

    if COMPOSITION_VERSION != RETAINED_COMPOSITION_VERSION:
        raise RuntimeError("component scout composition constant is stale")
    original_loader = inference.load_model_with_engine
    stack = ExitStack()
    shared_stats = SharedRopeStats()
    captured: dict[str, Any] = {}
    loaded_bindings = 0

    def retained_loader(*loader_args, **loader_kwargs):
        nonlocal loaded_bindings
        binding, tokenizer = original_loader(*loader_args, **loader_kwargs)
        loaded_bindings += 1
        if loaded_bindings == 1:
            model = binding.raw_model
            metadata = binding.runtime.initialize_approximate_int8_mlp_weight_only(model)
            state = getattr(binding.runtime, "_approximate_weight_only_state", None)
            if state is None:
                raise RuntimeError("retained runtime did not expose initialized weight state")
            state.validate_owner(model)
            captured.update(model=model, state=state, weight_state=metadata)
            stack.enter_context(shared_decoder_rope_context(model, stats=shared_stats))
        return binding, tokenizer

    inference.load_model_with_engine = retained_loader
    try:
        with install_k8_candidate(
            block_size=4,
            graph_remainders=True,
            entry_snapshotter=lambda entry, prefix: _snapshot_pre_replay_entry(
                entry,
                prefix,
                diagnostics,
            ),
        ):
            run = _accepted_main_session_run(args, output_path=output_path)
    finally:
        inference.load_model_with_engine = original_loader
        stack.close()
    if loaded_bindings < 2:
        raise RuntimeError("retained full-song capture expected separate main/timing bindings")
    if run["model"] is not captured.get("model"):
        raise RuntimeError("retained full-song capture returned the wrong main model")
    records = [
        record
        for record in run["processor"].profiler.generation
        if record.get("profile_label") == "main_generation"
    ]
    raw_manifest = {
        "composition_version": RETAINED_COMPOSITION_VERSION,
        "scouted_regions": list(REGIONS),
        "baseline_by_region": dict(CURRENT_BASELINE),
        "excluded_already_int8_regions": list(EXCLUDED_ALREADY_INT8_REGIONS),
        "weight_state": captured["weight_state"],
        "shared_rope": _validated_shared_rope_evidence(shared_stats),
        "main_generation_records": records,
    }
    run.update(
        state=captured["state"],
        retained_composition=_validate_retained_composition_manifest(raw_manifest),
    )
    return run


@contextmanager
def _retained_real_tensor_context(
    model: torch.nn.Module,
    state: Any,
    *,
    prefix: int,
) -> Any:
    from osuT5.osuT5.inference.optimized.scout.shared_rope import (
        SharedRopeStats,
        shared_decoder_rope_context,
    )
    from osuT5.osuT5.runtime_profiling import generation_profile_context

    state.validate_owner(model)
    dispatch_counts = {name: 0 for name in REQUIRED_DISPATCHES}
    rope_stats = SharedRopeStats()
    with (
        shared_decoder_rope_context(model, stats=rope_stats),
        generation_profile_context(
            active_prefix_self_attention_length=prefix,
            q1_bmm_cross_attention=True,
            native_q1_self_attention=True,
            native_q1_rope_cache_self_attention=True,
            approximate_weight_only_state=state,
            optimized_expected_dtype=torch.float32,
            optimized_dispatch_counts=dispatch_counts,
        ),
    ):
        yield {"dispatch_counts": dispatch_counts, "shared_rope_stats": rope_stats}
    if rope_stats.forwards <= 0:
        raise RuntimeError("retained real-tensor capture did not execute a model forward")
    for name in REQUIRED_DISPATCHES:
        if dispatch_counts.get(name, 0) <= 0:
            raise RuntimeError(f"retained real-tensor capture missed dispatch {name}")


def _pack_projection_family(
    layers: list[torch.nn.Module],
    final_projection: torch.nn.Module,
    state: Any,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any], dict[str, int]]:
    from osuT5.osuT5.inference.optimized.scout.int8_mlp import Int8PackedLinear

    torch.cuda.synchronize()
    allocated_before = int(torch.cuda.memory_allocated())
    started = time.perf_counter()
    by_layer: dict[int, dict[str, Any]] = {}
    source_bytes = current_mixed_bytes = int8_bytes = 0
    for layer in layers:
        modules = {
            "self_norm_qkv": layer.self_attn.Wqkv,
            "self_out_residual": layer.self_attn.Wo,
            "cross_norm_q": layer.cross_attn.Wq,
            "cross_out_residual": layer.cross_attn.Wo,
        }
        retained_pack = state.pack_for_layer(layer)
        fp16 = {
            "self_norm_qkv": retained_pack.self_qkv,
            "self_out_residual": retained_pack.self_out,
        }
        int8 = {name: Int8PackedLinear.from_module(module) for name, module in modules.items()}
        by_layer[id(layer)] = {"fp16": fp16, "int8": int8}
        source_bytes += sum(value.source_weight_bytes for value in int8.values())
        current_mixed_bytes += (
            fp16["self_norm_qkv"].packed_weight_bytes
            + fp16["self_out_residual"].packed_weight_bytes
            + int8["cross_norm_q"].source_weight_bytes
            + int8["cross_out_residual"].source_weight_bytes
        )
        int8_bytes += sum(value.packed_weight_bytes for value in int8.values())
    final_fp16 = state.final_projection_pack
    final_int8 = Int8PackedLinear.from_module(final_projection)
    source_bytes += final_int8.source_weight_bytes
    current_mixed_bytes += final_fp16.packed_weight_bytes
    int8_bytes += final_int8.packed_weight_bytes
    torch.cuda.synchronize()
    return by_layer, {"fp16": final_fp16, "int8": final_int8}, {
        "setup_seconds": time.perf_counter() - started,
        "allocated_before_bytes": allocated_before,
        "allocated_after_bytes": int(torch.cuda.memory_allocated()),
        "retained_fp32_source_weight_bytes": source_bytes,
        "current_mixed_active_weight_bytes": current_mixed_bytes,
        "candidate_int8_packed_bytes": int8_bytes,
        "candidate_active_storage_reduction_bytes": current_mixed_bytes - int8_bytes,
    }


@torch.no_grad()
def profile_component(
    args,
    *,
    output_path: Path,
    diagnostic_path: Path,
    bucket_mode: str,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("INT8 projection profiler requires CUDA")
    if warmup < 1 or iters < 1:
        raise ValueError("warmup and iters must be positive")
    _assert_scout_args(args)
    output_path.mkdir(parents=True, exist_ok=True)
    diagnostics = _DiagnosticRecorder(diagnostic_path)
    diagnostics.record("component_scout_start")
    run = _retained_main_session_run(
        args,
        output_path=output_path,
        diagnostics=diagnostics,
    )
    _cuda_sync_checkpoint(diagnostics, "retained_full_song.complete")
    model = run["model"]
    state = run["state"]
    model.eval()
    entries = _validate_live_graph_cache(run["session"].graph_cache)
    selected = _select_live_buckets(entries, bucket_mode)
    total_replays = sum(
        int(entry["decode_replays"])
        for prefix_entries in entries.values()
        for entry in prefix_entries
    )
    layers, final_norm, final_projection = _locate_decoder(model)

    from osuT5.osuT5.inference.optimized.kernels.weight_only import (
        preload_weight_only_extension,
    )
    from osuT5.osuT5.inference.optimized.scout.int8_mlp import (
        preload_int8_mlp_extension,
    )

    preload_times: dict[str, float] = {}
    for name, preload in (
        ("current_fp16_projection", preload_weight_only_extension),
        ("candidate_int8_projection", preload_int8_mlp_extension),
    ):
        _cuda_sync_checkpoint(diagnostics, f"extension.{name}.before_preload")
        started = time.perf_counter()
        preload()
        _cuda_sync_checkpoint(diagnostics, f"extension.{name}.after_preload")
        preload_times[name] = time.perf_counter() - started
    layer_packs, final_packs, pack_report = _pack_projection_family(
        layers, final_projection, state
    )
    _cuda_sync_checkpoint(diagnostics, "projection_family.pack.complete")

    buckets: dict[str, dict[str, Any]] = {}
    for prefix, accepted in selected.items():
        static_inputs = accepted["static_inputs"]
        _restore_pre_replay_entry(accepted, diagnostics)
        _validate_restored_capture_inputs(
            model,
            static_inputs,
            prefix=prefix,
            diagnostics=diagnostics,
        )
        cache = _cache_from_static_inputs(static_inputs)
        position = static_inputs.get("cache_position")
        if not isinstance(position, torch.Tensor) or position.numel() != 1:
            raise RuntimeError(f"prefix {prefix} has invalid cache position")
        snapshots = _all_cache_snapshots(cache, position)
        _restore_all_cache(cache, snapshots)
        _cuda_sync_checkpoint(diagnostics, f"prefix_{prefix}.before_model_capture")
        with _retained_real_tensor_context(
            model,
            state,
            prefix=prefix,
        ) as capture_evidence:
            try:
                capture = _capture_representative_layer(
                    model,
                    static_inputs,
                    prefix=prefix,
                )
            except Exception as exc:
                diagnostics.record(
                    "retained_model_capture",
                    prefix=prefix,
                    stage="representative_layer",
                    passed=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise RuntimeError(
                    f"retained representative-layer capture failed at prefix {prefix}"
                ) from exc
            _cuda_sync_checkpoint(
                diagnostics,
                f"prefix_{prefix}.representative_layer_capture",
            )
            _restore_all_cache(cache, snapshots)
            _cuda_sync_checkpoint(
                diagnostics,
                f"prefix_{prefix}.before_final_input_capture",
            )
            try:
                final_input = _capture_final_input(
                    model,
                    layers[-1],
                    static_inputs,
                )
            except Exception as exc:
                diagnostics.record(
                    "retained_model_capture",
                    prefix=prefix,
                    stage="final_input",
                    passed=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise RuntimeError(
                    f"retained final-input capture failed at prefix {prefix}"
                ) from exc
            _cuda_sync_checkpoint(
                diagnostics,
                f"prefix_{prefix}.final_input_capture",
            )
        _restore_all_cache(cache, snapshots)
        _cuda_sync_checkpoint(diagnostics, f"prefix_{prefix}.capture_restore")

        packs = layer_packs.get(id(capture.module))
        if packs is None:
            raise RuntimeError("captured decoder layer has no owned weight pack")
        fp16_packs = dict(packs["fp16"])
        fp16_packs["final_norm_logits"] = final_packs["fp16"]
        inputs = _prepare_region_inputs(capture, fp16_packs)
        _cuda_sync_checkpoint(diagnostics, f"prefix_{prefix}.region_inputs")
        _restore_all_cache(cache, snapshots)
        int8_packs = dict(packs["int8"])
        int8_packs["final_norm_logits"] = final_packs["int8"]
        _validate_projection_call_inputs(
            inputs,
            final_input,
            fp16_packs,
            int8_packs,
            prefix=prefix,
            diagnostics=diagnostics,
        )
        baselines = _baseline_functions(
            capture,
            inputs,
            final_input,
            final_norm,
            final_projection,
            fp16_packs,
        )
        candidates = {
            str(opb): _candidate_functions(
                capture,
                inputs,
                final_input,
                final_norm,
                int8_packs,
                outputs_per_block=opb,
            )
            for opb in OUTPUTS_PER_BLOCK
        }
        region_results: dict[str, Any] = {}
        for region in REGIONS:
            callables = {"baseline": baselines[region]}
            callables.update(
                {f"int8_{opb}": candidates[str(opb)][region] for opb in OUTPUTS_PER_BLOCK}
            )
            graphs = {}
            for name, fn in callables.items():
                checkpoint = f"prefix_{prefix}.{region}.{name}"
                _cuda_sync_checkpoint(diagnostics, f"{checkpoint}.before_capture")
                try:
                    graphs[name] = _capture_cuda_graph(
                        fn,
                        context=nullcontext,
                        warmup=0,
                    )
                except Exception as exc:
                    diagnostics.record(
                        "component_graph_capture",
                        checkpoint=checkpoint,
                        passed=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    raise RuntimeError(
                        f"component graph capture failed at {checkpoint}"
                    ) from exc
                _cuda_sync_checkpoint(diagnostics, f"{checkpoint}.after_capture")
                diagnostics.record(
                    "component_graph_capture",
                    checkpoint=checkpoint,
                    passed=True,
                )
            timings, rounds, memory = _reciprocal_graph_rounds(
                {name: graph.graph for name, graph in graphs.items()},
                restore=lambda: None,
                warmup=warmup,
                iters=iters,
            )
            observations = {name: _observe(graph) for name, graph in graphs.items()}
            reference = observations["baseline"][0]
            region_results[region] = {
                "current_baseline": CURRENT_BASELINE[region],
                "baseline_ms_per_call": float(timings["baseline"]),
                "baseline_finite": bool(torch.isfinite(reference).all()),
                "baseline_repeat_deterministic": torch.equal(
                    *observations["baseline"]
                ),
                "baseline_memory_stable": bool(memory["baseline"]),
                "candidates": {
                    str(opb): {
                        "ms_per_call": float(timings[f"int8_{opb}"]),
                        "finite": bool(
                            torch.isfinite(observations[f"int8_{opb}"][0]).all()
                        ),
                        "repeat_deterministic": torch.equal(
                            *observations[f"int8_{opb}"]
                        ),
                        "memory_stable": bool(memory[f"int8_{opb}"]),
                        "output_max_abs_drift_vs_current_baseline": _max_abs(
                            reference, observations[f"int8_{opb}"][0]
                        ),
                    }
                    for opb in OUTPUTS_PER_BLOCK
                },
                "rounds": rounds,
            }
            del graphs
            torch.cuda.empty_cache()
        buckets[str(prefix)] = {
            "decode_replays": int(accepted["decode_replays"]),
            "retained_capture_dispatches": dict(capture_evidence["dispatch_counts"]),
            "retained_capture_shared_rope": capture_evidence[
                "shared_rope_stats"
            ].as_dict(),
            "regions": region_results,
        }
    summary = summarize_component(buckets, total_replays=total_replays)
    return {
        "schema_version": 1,
        "metadata": {
            "candidate": "per_row_symmetric_int8_remaining_projection_family",
            "result_class": "component_scout_documented_drift",
            "production_wiring": False,
            "comparison": "retained_k1_int8_mlp_shared_rope_projection_policy",
            "retained_composition": run["retained_composition"],
            "excluded_already_int8_regions": list(EXCLUDED_ALREADY_INT8_REGIONS),
            "current_baseline_by_region": CURRENT_BASELINE,
            "live_component_model": {
                "measured_int8_vs_fp16_mlp_speedup": (
                    MEASURED_INT8_VS_FP16_MLP_SPEEDUP
                ),
                "current_projection_family_seconds": summary[
                    "current_projection_family_seconds"
                ],
                "realistic_ceiling_seconds": summary[
                    "live_modeled_realistic_ceiling_seconds"
                ],
                "required_ceiling_seconds": FIXED_MAIN_SAVING_GATE_SECONDS,
            },
            "precision_contract": {
                "weight_storage": "int8_per_output_row_symmetric",
                "scale": "fp32_per_output_row",
                "input_output_norm_bias_residual": "fp32",
                "accumulation": "fp32",
            },
            "bucket_mode": bucket_mode,
            "measured_buckets": list(selected),
            "live_bucket_counts": {
                str(prefix): sum(
                    int(entry["decode_replays"]) for entry in prefix_entries
                )
                for prefix, prefix_entries in entries.items()
            },
            "live_graph_counts": {
                str(prefix): len(prefix_entries)
                for prefix, prefix_entries in entries.items()
            },
            "selected_graph_sources": {
                str(prefix): entry["graph_source"]
                for prefix, entry in selected.items()
            },
            "warmup": warmup,
            "iters": iters,
            "extension_preload_seconds": preload_times,
        },
        "pack_report": pack_report,
        "diagnostic_path": str(diagnostic_path),
        "buckets": buckets,
        "summary": summary,
    }


def _text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "INT8 remaining projection family component scout",
        f"measured_replays={summary['measured_replays']}",
        f"total_replays={summary['total_replays']}",
    ]
    for region, entry in summary["regions"].items():
        lines.append(
            f"{region}: baseline={entry['current_baseline']} "
            f"opb={entry['selected_outputs_per_block']} "
            f"fixed_saving_seconds={entry['fixed_work_saving_seconds']:.6f} "
            f"retained={entry['retained_by_selective_policy']}"
        )
    lines.extend(
        [
            "current_projection_family_seconds="
            f"{summary['current_projection_family_seconds']:.6f}",
            "live_modeled_realistic_ceiling_seconds="
            f"{summary['live_modeled_realistic_ceiling_seconds']:.6f}",
            "selective_fixed_work_main_saving_seconds="
            f"{summary['selective_fixed_work_main_saving_seconds']:.6f}",
            f"saving_gate_seconds={summary['saving_gate_seconds']:.6f}",
            f"invariants_pass={summary['invariants_pass']}",
            f"sizing_pass={summary['sizing_pass']}",
            f"promotion_pass={summary['promotion_pass']}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--text-path", type=Path, required=True)
    parser.add_argument("--diagnostic-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--bucket-mode", choices=("sentinel", "all"), default="sentinel")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parsed, overrides = parser.parse_known_args()
    args = _load_args(parsed.config_name, overrides)
    report = profile_component(
        args,
        output_path=parsed.output_path,
        diagnostic_path=parsed.diagnostic_path,
        bucket_mode=parsed.bucket_mode,
        warmup=parsed.warmup,
        iters=parsed.iters,
    )
    parsed.report_path.parent.mkdir(parents=True, exist_ok=True)
    parsed.report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    parsed.text_path.write_text(_text(report))
    print(_text(report), end="")
    if not report["summary"]["promotion_pass"]:
        raise SystemExit("STOP_INT8_PROJECTION_COMPONENT")


if __name__ == "__main__":
    main()
