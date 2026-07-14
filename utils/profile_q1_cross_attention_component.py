from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from utils.profile_native_prefix_dtype_scout import (  # noqa: E402
    CapturedGraph,
    _all_cache_snapshots,
    _assert_scout_args,
    _cache_from_static_inputs,
    _capture_cuda_graph,
    _load_args,
    _max_abs,
    _reciprocal_graph_rounds,
    _restore_all_cache,
    validate_accepted_graph_cache,
)


LAYERS = 12
HEADS = 12
KV_LENGTH = 1024
HEAD_DIM = 64
DEFAULT_CAPTURE_PREFIX = 640
MAX_FP32_DRIFT = 1e-3
MAIN_SAVING_TARGET_SECONDS = 1.503
FIXED_WORK_MAIN_TOKENS = 8_294
FIXED_WORK_MAIN_SECONDS = 30.069
FIXED_WORK_500_TPS_SECONDS = FIXED_WORK_MAIN_TOKENS / 500.0
FIXED_WORK_500_TPS_GAP_SECONDS = (
    FIXED_WORK_MAIN_SECONDS - FIXED_WORK_500_TPS_SECONDS
)
SESSION_PROFILE_LABELS = {
    "timing_context": "timing_generation",
    "main_generation": "main_generation",
}


@dataclass(frozen=True)
class CrossInputs:
    layer_idx: int
    query: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    residual: torch.Tensor
    cross_norm_weight: torch.Tensor
    cross_norm_eps: float
    query_weight: torch.Tensor
    query_bias: torch.Tensor | None
    output_weight: torch.Tensor
    output_bias: torch.Tensor | None


def _tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _run_and_capture_sessions(args, *, output_path: Path) -> dict[str, Any]:
    from inference import (
        compile_args,
        generate,
        get_config,
        load_model_with_engine,
        setup_inference_environment,
    )
    from osuT5.osuT5.inference import Processor

    compile_args(args, verbose=False)
    setup_inference_environment(args.seed)
    binding, tokenizer = load_model_with_engine(
        args.model_path,
        args.train,
        args.device,
        max_batch_size=args.max_batch_size,
        use_server=args.use_server,
        precision=args.precision,
        attn_implementation=args.attn_implementation,
        lora_path=args.lora_path,
        gamemode=args.gamemode,
        auto_select_gamemode_model=args.auto_select_gamemode_model,
        inference_engine=args.inference_engine,
    )
    generation_config, beatmap_config = get_config(args)
    captures: dict[str, Any] = {}
    original_generate = Processor.generate

    def wrapped_generate(processor, *positional, **kwargs):
        result = original_generate(processor, *positional, **kwargs)
        profile_label = kwargs.get("profile_label")
        label = SESSION_PROFILE_LABELS.get(profile_label)
        if label is not None:
            if label in captures:
                raise RuntimeError(f"captured more than one {label} Processor")
            if processor.decode_session_state is None:
                raise RuntimeError(f"{label} did not expose a decode session")
            captures[label] = {
                "processor": processor,
                "session": processor.decode_session_state,
            }
        return result

    Processor.generate = wrapped_generate
    try:
        generated, result_path = generate(
            args,
            output_path=str(output_path),
            generation_config=generation_config,
            beatmap_config=beatmap_config,
            model=binding,
            tokenizer=tokenizer,
            timing_model=binding,
            timing_tokenizer=tokenizer,
            verbose=False,
        )
    finally:
        Processor.generate = original_generate
    missing = {"timing_generation", "main_generation"} - set(captures)
    if missing:
        raise RuntimeError(f"full SALVALAI run missed decode sessions: {sorted(missing)}")
    main = captures["main_generation"]
    return {
        "model": main["processor"].model,
        "timing_session": captures["timing_generation"]["session"],
        "main_session": main["session"],
        "generated": generated,
        "result_path": str(result_path),
    }


def _capture_real_cross_inputs(
    model: torch.nn.Module,
    static_inputs: dict[str, Any],
    *,
    prefix: int,
) -> list[CrossInputs]:
    from osuT5.osuT5.runtime_profiling import generation_profile_context
    from osuT5.osuT5.model.custom_transformers.modeling_varwhisper import (
        VarWhisperAttention,
        VarWhisperDecoderLayer,
    )

    cache = _cache_from_static_inputs(static_inputs)
    cache_position = static_inputs.get("cache_position")
    if not isinstance(cache_position, torch.Tensor) or cache_position.numel() != 1:
        raise RuntimeError("representative graph has invalid cache_position")
    snapshots = _all_cache_snapshots(cache, cache_position)
    found: dict[int, CrossInputs] = {}
    residuals: dict[int, torch.Tensor] = {}
    handles = []

    decoder_layers = [
        module for module in model.modules() if isinstance(module, VarWhisperDecoderLayer)
    ]
    if len(decoder_layers) != LAYERS:
        raise RuntimeError(f"expected {LAYERS} decoder layers, got {len(decoder_layers)}")
    norm_owner = {
        id(layer.cross_attn_layer_norm): int(layer.cross_attn.layer_idx)
        for layer in decoder_layers
    }

    def norm_hook(module, positional):
        layer_idx = norm_owner[id(module)]
        if layer_idx in residuals:
            raise RuntimeError(f"cross norm layer {layer_idx} executed twice")
        if len(positional) != 1 or not isinstance(positional[0], torch.Tensor):
            raise RuntimeError(f"cross norm layer {layer_idx} missed its residual input")
        residuals[layer_idx] = positional[0].detach().clone()

    def hook(module, positional, kwargs):
        hidden_states = kwargs.get(
            "hidden_states",
            positional[0] if positional else None,
        )
        past_key_value = kwargs.get("past_key_value")
        if not isinstance(hidden_states, torch.Tensor) or past_key_value is None:
            raise RuntimeError("cross-attention hook missed hidden/cache inputs")
        layer_idx = int(module.layer_idx)
        if layer_idx in found:
            raise RuntimeError(f"cross-attention layer {layer_idx} executed twice")
        cross_cache = getattr(past_key_value, "cross_attention_cache", None)
        if cross_cache is None or layer_idx >= len(cross_cache.layers):
            raise RuntimeError(f"cross-attention layer {layer_idx} missed its cache")
        cache_layer = cross_cache.layers[layer_idx]
        if not getattr(cache_layer, "is_initialized", False):
            raise RuntimeError(f"cross-attention cache layer {layer_idx} is uninitialized")
        query = (
            module.Wq(hidden_states)
            .view(1, 1, module.num_heads, module.head_dim)
            .transpose(1, 2)
            .contiguous()
            .detach()
            .clone()
        )
        if layer_idx not in residuals:
            raise RuntimeError(f"cross-attention layer {layer_idx} missed residual capture")
        decoder_layer = decoder_layers[layer_idx]
        found[layer_idx] = CrossInputs(
            layer_idx=layer_idx,
            query=query,
            keys=cache_layer.keys.detach(),
            values=cache_layer.values.detach(),
            residual=residuals[layer_idx],
            cross_norm_weight=decoder_layer.cross_attn_layer_norm.weight.detach(),
            cross_norm_eps=float(
                getattr(
                    decoder_layer.cross_attn_layer_norm,
                    "eps",
                    torch.finfo(torch.float32).eps,
                )
                or torch.finfo(torch.float32).eps
            ),
            query_weight=module.Wq.weight.detach(),
            query_bias=(None if module.Wq.bias is None else module.Wq.bias.detach()),
            output_weight=module.Wo.weight.detach(),
            output_bias=(None if module.Wo.bias is None else module.Wo.bias.detach()),
        )

    for layer in decoder_layers:
        handles.append(layer.cross_attn_layer_norm.register_forward_pre_hook(norm_hook))
    for module in model.modules():
        if isinstance(module, VarWhisperAttention) and module.is_cross_attention:
            handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
    if len(handles) != 2 * LAYERS:
        for handle in handles:
            handle.remove()
        raise RuntimeError(
            f"expected {LAYERS} cross norm and attention hook pairs, got {len(handles)} hooks"
        )
    try:
        _restore_all_cache(cache, snapshots)
        with generation_profile_context(
            active_prefix_self_attention_length=prefix,
            q1_bmm_cross_attention=True,
            native_q1_self_attention=True,
            native_q1_rope_cache_self_attention=True,
            native_cross_mlp_tail=False,
            optimized_expected_dtype=torch.float32,
        ):
            model(**static_inputs, return_dict=True)
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
        _restore_all_cache(cache, snapshots)
    if set(found) != set(range(LAYERS)):
        raise RuntimeError(
            f"expected cross inputs for layers 0..{LAYERS - 1}, got {sorted(found)}"
        )
    captures = [found[layer_idx] for layer_idx in range(LAYERS)]
    expected_query = (1, HEADS, 1, HEAD_DIM)
    expected_kv = (1, HEADS, KV_LENGTH, HEAD_DIM)
    for capture in captures:
        if capture.query.dtype != torch.float32 or tuple(capture.query.shape) != expected_query:
            raise RuntimeError(
                f"layer {capture.layer_idx} query is not FP32 {expected_query}: "
                f"{capture.query.dtype} {tuple(capture.query.shape)}"
            )
        for name, tensor in (("keys", capture.keys), ("values", capture.values)):
            if tensor.dtype != torch.float32 or tuple(tensor.shape) != expected_kv:
                raise RuntimeError(
                    f"layer {capture.layer_idx} {name} is not FP32 {expected_kv}: "
                    f"{tensor.dtype} {tuple(tensor.shape)}"
                )
            if not tensor.is_contiguous():
                raise RuntimeError(f"layer {capture.layer_idx} {name} is not contiguous")
        if tuple(capture.residual.shape) != (1, 1, HEADS * HEAD_DIM):
            raise RuntimeError(
                f"layer {capture.layer_idx} residual has invalid shape "
                f"{tuple(capture.residual.shape)}"
            )
    return captures


def _accepted_callable(captures: list[CrossInputs]) -> Callable[[], tuple[torch.Tensor, ...]]:
    def run() -> tuple[torch.Tensor, ...]:
        from osuT5.osuT5.inference.optimized.kernels.dispatch import (
            _q1_bmm_cross_attention,
        )

        return tuple(
            _q1_bmm_cross_attention(
                capture.query,
                capture.keys,
                capture.values,
                expected_dtype=torch.float32,
            )
            for capture in captures
        )

    return run


def _candidate_callable(
    captures: list[CrossInputs],
    *,
    splits: int | None,
    fp16_kv: bool,
) -> tuple[Callable[[], tuple[torch.Tensor, ...]], list[CrossInputs]]:
    from osuT5.osuT5.inference.optimized.scout.cross_attention import (
        cross_attention_one_pass,
        cross_attention_split,
    )

    selected = captures
    if fp16_kv:
        selected = [
            CrossInputs(
                layer_idx=capture.layer_idx,
                query=capture.query,
                keys=capture.keys.half(),
                values=capture.values.half(),
                residual=capture.residual,
                cross_norm_weight=capture.cross_norm_weight,
                cross_norm_eps=capture.cross_norm_eps,
                query_weight=capture.query_weight,
                query_bias=capture.query_bias,
                output_weight=capture.output_weight,
                output_bias=capture.output_bias,
            )
            for capture in captures
        ]

    def run() -> tuple[torch.Tensor, ...]:
        if splits is None:
            return tuple(
                cross_attention_one_pass(capture.query, capture.keys, capture.values)
                for capture in selected
            )
        return tuple(
            cross_attention_split(
                capture.query,
                capture.keys,
                capture.values,
                splits,
            )
            for capture in selected
        )

    return run, selected


def _observe(
    captured: CapturedGraph,
    *,
    expected_shape: tuple[int, ...] = (1, HEADS, 1, HEAD_DIM),
) -> tuple[torch.Tensor, ...]:
    captured.graph.replay()
    torch.cuda.synchronize()
    if not isinstance(captured.outputs, tuple) or len(captured.outputs) != LAYERS:
        raise RuntimeError("cross-attention graph must return one tensor per decoder layer")
    if any(tuple(output.shape) != expected_shape for output in captured.outputs):
        raise RuntimeError(
            f"cross component graph returned a shape other than {expected_shape}"
        )
    return tuple(output.detach().float().cpu().clone() for output in captured.outputs)


def _input_hashes(captures: list[CrossInputs]) -> dict[str, str]:
    result: dict[str, str] = {}
    for capture in captures:
        for name, tensor in (
            ("query", capture.query),
            ("keys", capture.keys),
            ("values", capture.values),
            ("residual", capture.residual),
            ("cross_norm_weight", capture.cross_norm_weight),
            ("query_weight", capture.query_weight),
            ("output_weight", capture.output_weight),
        ):
            result[f"layer{capture.layer_idx}.{name}"] = _tensor_sha256(tensor)
        for name, tensor in (
            ("query_bias", capture.query_bias),
            ("output_bias", capture.output_bias),
        ):
            if tensor is not None:
                result[f"layer{capture.layer_idx}.{name}"] = _tensor_sha256(tensor)
    return result


def _fp32_cross_block_callable(
    captures: list[CrossInputs],
) -> Callable[[], tuple[torch.Tensor, ...]]:
    def run() -> tuple[torch.Tensor, ...]:
        from osuT5.osuT5.inference.optimized.kernels.decoder_layer import (
            native_one_token_linear_residual,
            native_one_token_rmsnorm_linear,
        )
        from osuT5.osuT5.inference.optimized.kernels.dispatch import (
            _q1_bmm_cross_attention,
        )

        outputs = []
        for capture in captures:
            query = native_one_token_rmsnorm_linear(
                capture.residual,
                capture.cross_norm_weight,
                capture.query_weight,
                capture.query_bias,
                eps=capture.cross_norm_eps,
                outputs_per_block=8,
            ).view(1, 1, HEADS, HEAD_DIM).transpose(1, 2)
            attention = _q1_bmm_cross_attention(
                query,
                capture.keys,
                capture.values,
                expected_dtype=torch.float32,
            )
            outputs.append(
                native_one_token_linear_residual(
                    attention.transpose(1, 2).contiguous().view(1, 1, HEADS * HEAD_DIM),
                    capture.residual,
                    capture.output_weight,
                    capture.output_bias,
                    outputs_per_block=8,
                )
            )
        return tuple(outputs)

    return run


def _packed_cross_block_callable(
    captures: list[CrossInputs],
) -> tuple[Callable[[], tuple[torch.Tensor, ...]], dict[str, Any]]:
    from osuT5.osuT5.inference.optimized.kernels.weight_only import (
        PackedLinear,
        weight_only_linear_residual,
        weight_only_rmsnorm_linear,
    )

    class LinearView:
        def __init__(self, weight, bias):
            self.weight = weight
            self.bias = bias

    packed = [
        (
            PackedLinear.from_module(LinearView(capture.query_weight, capture.query_bias)),
            PackedLinear.from_module(LinearView(capture.output_weight, capture.output_bias)),
        )
        for capture in captures
    ]

    def run() -> tuple[torch.Tensor, ...]:
        from osuT5.osuT5.inference.optimized.kernels.dispatch import (
            _q1_bmm_cross_attention,
        )

        outputs = []
        for capture, (query_pack, output_pack) in zip(captures, packed, strict=True):
            query = weight_only_rmsnorm_linear(
                capture.residual,
                capture.cross_norm_weight,
                query_pack,
                eps=capture.cross_norm_eps,
                outputs_per_block=8,
            ).view(1, 1, HEADS, HEAD_DIM).transpose(1, 2)
            attention = _q1_bmm_cross_attention(
                query,
                capture.keys,
                capture.values,
                expected_dtype=torch.float32,
            )
            outputs.append(
                weight_only_linear_residual(
                    attention.transpose(1, 2).contiguous().view(1, 1, HEADS * HEAD_DIM),
                    capture.residual,
                    output_pack,
                    outputs_per_block=8,
                )
            )
        return tuple(outputs)

    source_bytes = sum(
        query_pack.source_weight_bytes + output_pack.source_weight_bytes
        for query_pack, output_pack in packed
    )
    packed_bytes = sum(
        query_pack.packed_weight_bytes + output_pack.packed_weight_bytes
        for query_pack, output_pack in packed
    )
    return run, {
        "fp32_source_weight_bytes": source_bytes,
        "fp16_packed_weight_bytes": packed_bytes,
        "resident_increment_bytes": packed_bytes,
        "hypothetical_saving_if_sources_replaced_bytes": source_bytes - packed_bytes,
        "layers": LAYERS,
        "packed_regions": ["cross_query", "cross_output"],
        "fp32_activations_bias_residual_attention_output": True,
    }


def _launch_metadata(name: str) -> dict[str, Any]:
    if name == "accepted_q1_bmm":
        return {
            "kernel_launches_per_layer": 4,
            "cta_count": None,
            "cta_note": "cuBLAS/softmax launch geometry is implementation-owned",
            "workspace_bytes_per_layer": 0,
        }
    if name == "one_pass_fp32":
        return {
            "kernel_launches_per_layer": 1,
            "partial_grid": [12, 1, 1],
            "partial_block": [128, 1, 1],
            "merge_grid": None,
            "workspace_bytes_per_layer": 0,
            "underfill_control": True,
        }
    splits = int(name.split("split", 1)[1].split("_", 1)[0])
    workspace = (HEADS * splits * HEAD_DIM + 2 * HEADS * splits) * 4
    return {
        "kernel_launches_per_layer": 2,
        "partial_grid": [HEADS * splits, 1, 1],
        "partial_block": [128, 1, 1],
        "merge_grid": [HEADS, 1, 1],
        "merge_block": [64, 1, 1],
        "workspace_bytes_per_layer": workspace,
        "underfill_control": False,
    }


def summarize_component(
    variants: dict[str, dict[str, Any]],
    *,
    timing_decode_replays: int,
    main_decode_replays: int,
) -> dict[str, Any]:
    if "accepted_q1_bmm" not in variants:
        raise ValueError("component report requires accepted_q1_bmm")
    if timing_decode_replays <= 0 or main_decode_replays <= 0:
        raise ValueError("timing/main decode replay counts must be positive")
    accepted_ms = float(variants["accepted_q1_bmm"]["ms_per_decode_step"])
    if not math.isfinite(accepted_ms) or accepted_ms <= 0:
        raise ValueError("accepted timing must be finite and positive")
    accepted_main_seconds = main_decode_replays * accepted_ms / 1000.0
    accepted_timing_seconds = timing_decode_replays * accepted_ms / 1000.0
    candidates: dict[str, Any] = {}
    for name, entry in variants.items():
        if name == "accepted_q1_bmm":
            continue
        candidate_ms = float(entry["ms_per_decode_step"])
        if not math.isfinite(candidate_ms) or candidate_ms <= 0:
            raise ValueError(f"{name} timing must be finite and positive")
        delta_ms = accepted_ms - candidate_ms
        main_saving = main_decode_replays * delta_ms / 1000.0
        timing_saving = timing_decode_replays * delta_ms / 1000.0
        fp32_candidate = entry["kv_storage_dtype"] == "torch.float32"
        promotion_eligible = fp32_candidate and name.startswith("split")
        correctness_pass = bool(entry["checks_pass"]) and (
            not fp32_candidate or float(entry["max_abs_drift"]) <= MAX_FP32_DRIFT
        )
        projected_seconds = FIXED_WORK_MAIN_SECONDS - main_saving
        candidates[name] = {
            "local_speedup": accepted_ms / candidate_ms,
            "main_saving_seconds": main_saving,
            "timing_saving_seconds": timing_saving,
            "timing_plus_main_saving_seconds": main_saving + timing_saving,
            "main_saving_target_seconds": MAIN_SAVING_TARGET_SECONDS,
            "sizing_pass": main_saving >= MAIN_SAVING_TARGET_SECONDS,
            "correctness_pass": correctness_pass,
            "promotion_eligible": promotion_eligible,
            "promotion_pass": (
                promotion_eligible
                and correctness_pass
                and main_saving >= MAIN_SAVING_TARGET_SECONDS
            ),
            "projected_fixed_work_main_seconds": projected_seconds,
            "projected_fixed_work_main_tps": (
                FIXED_WORK_MAIN_TOKENS / projected_seconds
                if projected_seconds > 0
                else math.inf
            ),
            "fraction_of_500_tps_gap_closed": (
                main_saving / FIXED_WORK_500_TPS_GAP_SECONDS
            ),
        }
    best_fp32 = max(
        (
            (name, report)
            for name, report in candidates.items()
            if report["promotion_eligible"] and report["correctness_pass"]
        ),
        key=lambda item: item[1]["main_saving_seconds"],
        default=None,
    )
    ideal_seconds = FIXED_WORK_MAIN_SECONDS - accepted_main_seconds
    return {
        "timing_decode_replays": timing_decode_replays,
        "main_decode_replays": main_decode_replays,
        "accepted_timing_seconds": accepted_timing_seconds,
        "accepted_main_seconds": accepted_main_seconds,
        "perfect_cross_removal_main_ceiling_seconds": accepted_main_seconds,
        "perfect_cross_removal_projected_fixed_work_tps": (
            FIXED_WORK_MAIN_TOKENS / ideal_seconds if ideal_seconds > 0 else math.inf
        ),
        "perfect_cross_removal_fraction_of_500_gap": (
            accepted_main_seconds / FIXED_WORK_500_TPS_GAP_SECONDS
        ),
        "fixed_work_reference": {
            "tokens": FIXED_WORK_MAIN_TOKENS,
            "seconds": FIXED_WORK_MAIN_SECONDS,
            "tps": FIXED_WORK_MAIN_TOKENS / FIXED_WORK_MAIN_SECONDS,
            "target_tps": 500.0,
            "target_seconds": FIXED_WORK_500_TPS_SECONDS,
            "gap_seconds": FIXED_WORK_500_TPS_GAP_SECONDS,
            "evidence_class": "calibrated_current-main_fixed-work_reference",
        },
        "candidates": candidates,
        "best_correct_fp32_candidate": None if best_fp32 is None else best_fp32[0],
        "any_fp32_promotion_pass": any(
            report["promotion_pass"] for report in candidates.values()
        ),
    }


def summarize_projection_component(
    *,
    baseline_ms: float,
    candidate_ms: float,
    max_abs_drift: float,
    checks_pass: bool,
    timing_decode_replays: int,
    main_decode_replays: int,
) -> dict[str, Any]:
    if min(baseline_ms, candidate_ms) <= 0 or not all(
        math.isfinite(value) for value in (baseline_ms, candidate_ms, max_abs_drift)
    ):
        raise ValueError("projection component timings/drift must be finite and valid")
    delta_ms = baseline_ms - candidate_ms
    main_saving = main_decode_replays * delta_ms / 1000.0
    timing_saving = timing_decode_replays * delta_ms / 1000.0
    projected_seconds = FIXED_WORK_MAIN_SECONDS - main_saving
    return {
        "baseline_ms_per_decode_step": baseline_ms,
        "candidate_ms_per_decode_step": candidate_ms,
        "local_speedup": baseline_ms / candidate_ms,
        "max_abs_drift": max_abs_drift,
        "checks_pass": bool(checks_pass),
        "result_class": "documented_drift_fp16_weights_fp32_state",
        "main_saving_seconds": main_saving,
        "timing_saving_seconds": timing_saving,
        "timing_plus_main_saving_seconds": main_saving + timing_saving,
        "main_saving_target_seconds": MAIN_SAVING_TARGET_SECONDS,
        "sizing_pass": main_saving >= MAIN_SAVING_TARGET_SECONDS,
        "projected_fixed_work_main_tps": (
            FIXED_WORK_MAIN_TOKENS / projected_seconds
            if projected_seconds > 0
            else math.inf
        ),
        "fraction_of_500_tps_gap_closed": (
            main_saving / FIXED_WORK_500_TPS_GAP_SECONDS
        ),
        "component_retention_pass": bool(checks_pass) and main_saving > 0,
        "production_promotion_pass": False,
    }


def _write_text_report(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    lines = [
        "q1 cross-attention fixed-shape component scout",
        f"commit={result['metadata']['commit']}",
        f"timing_decode_replays={summary['timing_decode_replays']}",
        f"main_decode_replays={summary['main_decode_replays']}",
        f"accepted_timing_seconds={summary['accepted_timing_seconds']:.9f}",
        f"accepted_main_seconds={summary['accepted_main_seconds']:.9f}",
        (
            "perfect_cross_removal_projected_fixed_work_tps="
            f"{summary['perfect_cross_removal_projected_fixed_work_tps']:.6f}"
        ),
        (
            "perfect_cross_removal_fraction_of_500_gap="
            f"{summary['perfect_cross_removal_fraction_of_500_gap']:.6f}"
        ),
    ]
    for name, report in summary["candidates"].items():
        raw = result["variants"][name]
        lines.extend(
            (
                f"candidate={name}",
                f"candidate_ms_per_decode_step={raw['ms_per_decode_step']:.9f}",
                f"candidate_max_abs_drift={raw['max_abs_drift']:.9g}",
                f"candidate_local_speedup={report['local_speedup']:.6f}",
                f"candidate_main_saving_seconds={report['main_saving_seconds']:.9f}",
                f"candidate_timing_saving_seconds={report['timing_saving_seconds']:.9f}",
                f"candidate_projected_fixed_work_tps={report['projected_fixed_work_main_tps']:.6f}",
                f"candidate_fraction_of_500_gap_closed={report['fraction_of_500_tps_gap_closed']:.6f}",
                f"candidate_promotion_pass={str(report['promotion_pass']).lower()}",
            )
        )
    lines.append(
        f"best_correct_fp32_candidate={summary['best_correct_fp32_candidate']}"
    )
    lines.append(
        f"any_fp32_promotion_pass={str(summary['any_fp32_promotion_pass']).lower()}"
    )
    projection = result["cross_projection_weight_scout"]["summary"]
    lines.extend(
        (
            "cross_projection_candidate=fp16_packed_wq_wo_accepted_bmm",
            (
                "cross_projection_baseline_ms_per_decode_step="
                f"{projection['baseline_ms_per_decode_step']:.9f}"
            ),
            (
                "cross_projection_candidate_ms_per_decode_step="
                f"{projection['candidate_ms_per_decode_step']:.9f}"
            ),
            f"cross_projection_max_abs_drift={projection['max_abs_drift']:.9g}",
            f"cross_projection_main_saving_seconds={projection['main_saving_seconds']:.9f}",
            f"cross_projection_timing_saving_seconds={projection['timing_saving_seconds']:.9f}",
            (
                "cross_projection_projected_fixed_work_tps="
                f"{projection['projected_fixed_work_main_tps']:.6f}"
            ),
            (
                "cross_projection_component_retention_pass="
                f"{str(projection['component_retention_pass']).lower()}"
            ),
            "cross_projection_production_promotion_pass=false",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def profile_component(
    args,
    *,
    output_path: Path,
    capture_prefix: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("q1 cross-attention component scout requires CUDA")
    if warmup < 1 or iters < 1:
        raise ValueError("warmup and iters must be positive")
    _assert_scout_args(args)
    output_path.mkdir(parents=True, exist_ok=True)
    run = _run_and_capture_sessions(args, output_path=output_path)
    timing_summary = run["timing_session"].graph_profile_summary()
    main_summary = run["main_session"].graph_profile_summary()
    entries = validate_accepted_graph_cache(run["main_session"].graph_cache)
    matching = [
        entry
        for entry in entries.values()
        if int(entry["active_prefix_length"]) == capture_prefix
    ]
    if len(matching) != 1:
        raise RuntimeError(
            f"expected one main graph for prefix {capture_prefix}, got {len(matching)}"
        )
    static_inputs = matching[0]["static_inputs"]
    captures = _capture_real_cross_inputs(
        run["model"],
        static_inputs,
        prefix=capture_prefix,
    )
    original_hashes = _input_hashes(captures)

    from osuT5.osuT5.inference.optimized.scout.cross_attention import (
        preload_cross_attention_scout,
    )
    from osuT5.osuT5.inference.optimized.kernels.decoder_layer import (
        preload_native_decoder_layer,
    )
    from osuT5.osuT5.inference.optimized.kernels.weight_only import (
        preload_weight_only_extension,
    )

    torch.cuda.synchronize()
    extension_started = time.perf_counter()
    preload_cross_attention_scout()
    preload_native_decoder_layer()
    preload_weight_only_extension()
    torch.cuda.synchronize()
    extension_preload_seconds = time.perf_counter() - extension_started

    callables: dict[str, Callable[[], tuple[torch.Tensor, ...]]] = {
        "accepted_q1_bmm": _accepted_callable(captures),
    }
    variant_inputs: dict[str, list[CrossInputs]] = {
        "accepted_q1_bmm": captures,
    }
    for name, splits, fp16_kv in (
        ("one_pass_fp32", None, False),
        ("split2_fp32", 2, False),
        ("split4_fp32", 4, False),
        ("split8_fp32", 8, False),
        ("split8_fp16_kv", 8, True),
    ):
        candidate, selected_inputs = _candidate_callable(
            captures,
            splits=splits,
            fp16_kv=fp16_kv,
        )
        callables[name] = candidate
        variant_inputs[name] = selected_inputs

    graphs: dict[str, CapturedGraph] = {}
    for name, callable_ in callables.items():
        graphs[name] = _capture_cuda_graph(
            callable_,
            context=nullcontext,
            warmup=0,
        )
    timings, rounds, memory_stable = _reciprocal_graph_rounds(
        {name: captured.graph for name, captured in graphs.items()},
        restore=lambda: None,
        warmup=warmup,
        iters=iters,
    )
    reference_first = _observe(graphs["accepted_q1_bmm"])
    reference_second = _observe(graphs["accepted_q1_bmm"])
    if not all(
        torch.equal(first, second)
        for first, second in zip(reference_first, reference_second, strict=True)
    ):
        raise RuntimeError("accepted q1 BMM graph was not deterministic")

    variants: dict[str, dict[str, Any]] = {}
    for name, captured in graphs.items():
        first = _observe(captured)
        second = _observe(captured)
        repeat_exact = all(
            torch.equal(left, right)
            for left, right in zip(first, second, strict=True)
        )
        finite = all(torch.isfinite(output).all().item() for output in first)
        shapes_valid = all(
            tuple(output.shape) == (1, HEADS, 1, HEAD_DIM)
            for output in first
        )
        max_drift = max(
            _max_abs(reference, candidate)
            for reference, candidate in zip(reference_first, first, strict=True)
        )
        inputs_unchanged = _input_hashes(captures) == original_hashes
        kv_dtype = variant_inputs[name][0].keys.dtype
        checks = {
            "finite_outputs": bool(finite),
            "output_shapes_valid": bool(shapes_valid),
            "repeat_deterministic": bool(repeat_exact),
            "source_cross_cache_unchanged": bool(inputs_unchanged),
            "memory_stable": bool(memory_stable[name]),
        }
        variants[name] = {
            "ms_per_decode_step": float(timings[name]),
            "ms_per_layer": float(timings[name]) / LAYERS,
            "max_abs_drift": max_drift,
            "checks": checks,
            "checks_pass": all(checks.values()),
            "kv_storage_dtype": str(kv_dtype),
            "capture_setup_seconds": captured.setup_seconds,
            "capture_peak_vram_bytes": captured.peak_vram_bytes,
            "launch": _launch_metadata(name),
            "rounds": [row for row in rounds if row["variant"] == name],
        }
    summary = summarize_component(
        variants,
        timing_decode_replays=int(timing_summary["decode_replays"]),
        main_decode_replays=int(main_summary["decode_replays"]),
    )
    fp32_cross_block = _capture_cuda_graph(
        _fp32_cross_block_callable(captures),
        context=nullcontext,
        warmup=0,
    )
    packed_callable, packed_memory = _packed_cross_block_callable(captures)
    packed_cross_block = _capture_cuda_graph(
        packed_callable,
        context=nullcontext,
        warmup=0,
    )
    projection_graphs = {
        "fp32_wq_wo_accepted_bmm": fp32_cross_block,
        "fp16_packed_wq_wo_accepted_bmm": packed_cross_block,
    }
    projection_timings, projection_rounds, projection_memory_stable = (
        _reciprocal_graph_rounds(
            {name: graph.graph for name, graph in projection_graphs.items()},
            restore=lambda: None,
            warmup=warmup,
            iters=iters,
        )
    )
    fp32_projection_first = _observe(
        fp32_cross_block,
        expected_shape=(1, 1, HEADS * HEAD_DIM),
    )
    fp32_projection_second = _observe(
        fp32_cross_block,
        expected_shape=(1, 1, HEADS * HEAD_DIM),
    )
    packed_projection_first = _observe(
        packed_cross_block,
        expected_shape=(1, 1, HEADS * HEAD_DIM),
    )
    packed_projection_second = _observe(
        packed_cross_block,
        expected_shape=(1, 1, HEADS * HEAD_DIM),
    )
    projection_repeat = all(
        torch.equal(first, second)
        for first, second in zip(
            packed_projection_first,
            packed_projection_second,
            strict=True,
        )
    )
    fp32_projection_repeat = all(
        torch.equal(first, second)
        for first, second in zip(
            fp32_projection_first,
            fp32_projection_second,
            strict=True,
        )
    )
    projection_drift = max(
        _max_abs(reference, candidate)
        for reference, candidate in zip(
            fp32_projection_first,
            packed_projection_first,
            strict=True,
        )
    )
    projection_checks = {
        "fp32_baseline_repeat_deterministic": fp32_projection_repeat,
        "candidate_repeat_deterministic": projection_repeat,
        "candidate_finite": all(
            torch.isfinite(value).all().item() for value in packed_projection_first
        ),
        "source_cross_cache_and_weights_unchanged": (
            _input_hashes(captures) == original_hashes
        ),
        "fp32_baseline_memory_stable": projection_memory_stable[
            "fp32_wq_wo_accepted_bmm"
        ],
        "candidate_memory_stable": projection_memory_stable[
            "fp16_packed_wq_wo_accepted_bmm"
        ],
    }
    projection_summary = summarize_projection_component(
        baseline_ms=projection_timings["fp32_wq_wo_accepted_bmm"],
        candidate_ms=projection_timings["fp16_packed_wq_wo_accepted_bmm"],
        max_abs_drift=projection_drift,
        checks_pass=all(projection_checks.values()),
        timing_decode_replays=int(timing_summary["decode_replays"]),
        main_decode_replays=int(main_summary["decode_replays"]),
    )
    return {
        "schema_version": 1,
        "metadata": {
            "commit": __import__("subprocess").check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                text=True,
            ).strip(),
            "candidate_scope": "component_only_no_production_wiring",
            "precision": "fp32_query_output_and_accumulation",
            "shape": [1, HEADS, 1, HEAD_DIM],
            "cross_kv_shape": [1, HEADS, KV_LENGTH, HEAD_DIM],
            "capture_prefix": capture_prefix,
            "layers": LAYERS,
            "warmup": warmup,
            "iters": iters,
            "extension_preload_seconds": extension_preload_seconds,
            "one_pass_is_rejected_12_cta_underfill_control": True,
            "split_variants_fix_underfill_with_head_times_split_grid": True,
        },
        "workload": {
            "timing_graph_profile": timing_summary,
            "main_graph_profile": main_summary,
        },
        "summary": summary,
        "variants": variants,
        "cross_projection_weight_scout": {
            "summary": projection_summary,
            "checks": projection_checks,
            "packed_memory": packed_memory,
            "capture": {
                "fp32_peak_vram_bytes": fp32_cross_block.peak_vram_bytes,
                "candidate_peak_vram_bytes": packed_cross_block.peak_vram_bytes,
                "fp32_setup_seconds": fp32_cross_block.setup_seconds,
                "candidate_setup_seconds": packed_cross_block.setup_seconds,
            },
            "rounds": projection_rounds,
            "scope": (
                "component_only; current mixed runtime keeps cross Wq/Wo FP32"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--text-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--capture-prefix", type=int, default=DEFAULT_CAPTURE_PREFIX)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("overrides", nargs="*")
    cli = parser.parse_args()
    overrides = list(cli.overrides)
    if not any(value.startswith("output_path=") for value in overrides):
        overrides.append(f"output_path={cli.output_path}")
    args = _load_args(cli.config_name, overrides)
    result = profile_component(
        args,
        output_path=cli.output_path,
        capture_prefix=cli.capture_prefix,
        warmup=cli.warmup,
        iters=cli.iters,
    )
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.text_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_text_report(cli.text_path, result)
    print(json.dumps(result["summary"], indent=2))
    if any(
        not entry["checks_pass"]
        for entry in result["variants"].values()
        if entry["kv_storage_dtype"] == "torch.float32"
    ):
        raise SystemExit(1)
    if not result["cross_projection_weight_scout"]["summary"][
        "component_retention_pass"
    ]:
        # This independent documented-drift subvariant may lose without making
        # the FP32 attention kernel evidence invalid. Its decision is recorded.
        pass
    if not result["summary"]["any_fp32_promotion_pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
