from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
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
)


LAYERS = 12
HEADS = 12
KV_LENGTH = 1024
HEAD_DIM = 64
SENTINEL_PREFIXES = (128, 576, 640)
SESSION_LABELS = ("timing_context", "main_generation")
FIXED_WORK_MAIN_TOKENS = 8_294
SELECTED_MAIN_SECONDS = 17.597493572930155
TARGET_500_SECONDS = FIXED_WORK_MAIN_TOKENS / 500.0
TARGET_500_GAP_SECONDS = SELECTED_MAIN_SECONDS - TARGET_500_SECONDS
COMPONENT_SAVING_GATE_SECONDS = 0.3
RELAXED_MAX_ABS_DRIFT = 1e-2


@dataclass(frozen=True)
class CrossInputs:
    layer_idx: int
    keys: torch.Tensor
    values: torch.Tensor
    residual: torch.Tensor
    cross_norm_weight: torch.Tensor
    cross_norm_eps: float
    query_weight: torch.Tensor
    query_bias: torch.Tensor | None
    output_weight: torch.Tensor
    output_bias: torch.Tensor | None


class _LinearView:
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None):
        self.weight = weight
        self.bias = bias


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().contiguous().cpu().numpy().tobytes()
    ).hexdigest()


def _input_hashes(captures: list[CrossInputs]) -> dict[str, str]:
    result: dict[str, str] = {}
    for capture in captures:
        for name, tensor in (
            ("keys", capture.keys),
            ("values", capture.values),
            ("residual", capture.residual),
            ("norm", capture.cross_norm_weight),
            ("wq", capture.query_weight),
            ("wo", capture.output_weight),
        ):
            result[f"{capture.layer_idx}.{name}"] = _tensor_sha256(tensor)
    return result


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
    sessions: dict[str, Any] = {}
    original_generate = Processor.generate

    def wrapped_generate(processor, *positional, **kwargs):
        result = original_generate(processor, *positional, **kwargs)
        label = kwargs.get("profile_label")
        if label in SESSION_LABELS:
            if label in sessions or processor.decode_session_state is None:
                raise RuntimeError(f"invalid decode-session capture for {label}")
            sessions[label] = {
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
    missing = set(SESSION_LABELS) - set(sessions)
    if missing:
        raise RuntimeError(f"full run missed decode sessions: {sorted(missing)}")
    return {
        "model": sessions["main_generation"]["processor"].model,
        "timing_session": sessions["timing_context"]["session"],
        "main_session": sessions["main_generation"]["session"],
        "generated": generated,
        "result_path": str(result_path),
    }


def _graph_entries(
    session: Any,
    *,
    required_prefixes: tuple[int, ...],
) -> dict[int, dict[str, Any]]:
    cache = getattr(session, "graph_cache", None)
    if not isinstance(cache, dict):
        raise TypeError("decode session graph_cache must be a dict")
    entries: dict[int, dict[str, Any]] = {}
    for entry in cache.values():
        if not isinstance(entry, dict):
            raise TypeError("decode graph entry must be a dict")
        prefix = entry.get("active_prefix_length")
        count = entry.get("decode_replays")
        if not isinstance(prefix, int) or prefix <= 0:
            raise ValueError(f"decode graph has invalid prefix {prefix!r}")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"decode graph prefix {prefix} has invalid replay count")
        if prefix in entries:
            raise RuntimeError(f"decode graph repeats prefix {prefix}")
        for field in ("graph", "outputs", "static_inputs"):
            if field not in entry:
                raise RuntimeError(f"decode graph prefix {prefix} is missing {field}")
        entries[prefix] = entry
    missing = sorted(set(required_prefixes) - set(entries))
    if missing:
        raise RuntimeError(f"decode graph is missing sentinel prefixes {missing}")
    return dict(sorted(entries.items()))


def _capture_real_cross_inputs(
    model: torch.nn.Module,
    static_inputs: dict[str, Any],
    *,
    prefix: int,
) -> list[CrossInputs]:
    from osuT5.osuT5.model.custom_transformers.modeling_varwhisper import (
        VarWhisperAttention,
        VarWhisperDecoderLayer,
    )
    from osuT5.osuT5.runtime_profiling import generation_profile_context

    cache = _cache_from_static_inputs(static_inputs)
    cache_position = static_inputs.get("cache_position")
    if not isinstance(cache_position, torch.Tensor) or cache_position.numel() != 1:
        raise RuntimeError("representative graph has invalid cache_position")
    snapshots = _all_cache_snapshots(cache, cache_position)
    layers = [
        module for module in model.modules() if isinstance(module, VarWhisperDecoderLayer)
    ]
    if len(layers) != LAYERS:
        raise RuntimeError(f"expected {LAYERS} decoder layers, got {len(layers)}")
    norm_owner = {
        id(layer.cross_attn_layer_norm): int(layer.cross_attn.layer_idx)
        for layer in layers
    }
    residuals: dict[int, torch.Tensor] = {}
    found: dict[int, CrossInputs] = {}
    handles = []

    def norm_hook(module, positional):
        layer_idx = norm_owner[id(module)]
        if layer_idx in residuals or len(positional) != 1:
            raise RuntimeError(f"invalid cross norm capture for layer {layer_idx}")
        residuals[layer_idx] = positional[0].detach().clone()

    def attention_hook(module, positional, kwargs):
        hidden_states = kwargs.get("hidden_states", positional[0] if positional else None)
        past_key_value = kwargs.get("past_key_value")
        layer_idx = int(module.layer_idx)
        if not isinstance(hidden_states, torch.Tensor) or past_key_value is None:
            raise RuntimeError(f"cross layer {layer_idx} missed hidden/cache inputs")
        if layer_idx in found or layer_idx not in residuals:
            raise RuntimeError(f"invalid cross attention capture for layer {layer_idx}")
        cross_cache = getattr(past_key_value, "cross_attention_cache", None)
        if cross_cache is None or layer_idx >= len(cross_cache.layers):
            raise RuntimeError(f"cross layer {layer_idx} missed its cache")
        cache_layer = cross_cache.layers[layer_idx]
        if not getattr(cache_layer, "is_initialized", False):
            raise RuntimeError(f"cross cache layer {layer_idx} is uninitialized")
        layer = layers[layer_idx]
        found[layer_idx] = CrossInputs(
            layer_idx=layer_idx,
            keys=cache_layer.keys.detach(),
            values=cache_layer.values.detach(),
            residual=residuals[layer_idx],
            cross_norm_weight=layer.cross_attn_layer_norm.weight.detach(),
            cross_norm_eps=float(
                getattr(layer.cross_attn_layer_norm, "eps", torch.finfo(torch.float32).eps)
                or torch.finfo(torch.float32).eps
            ),
            query_weight=module.Wq.weight.detach(),
            query_bias=None if module.Wq.bias is None else module.Wq.bias.detach(),
            output_weight=module.Wo.weight.detach(),
            output_bias=None if module.Wo.bias is None else module.Wo.bias.detach(),
        )

    for layer in layers:
        handles.append(layer.cross_attn_layer_norm.register_forward_pre_hook(norm_hook))
    for module in model.modules():
        if isinstance(module, VarWhisperAttention) and module.is_cross_attention:
            handles.append(module.register_forward_pre_hook(attention_hook, with_kwargs=True))
    if len(handles) != 2 * LAYERS:
        for handle in handles:
            handle.remove()
        raise RuntimeError(f"expected {2 * LAYERS} cross hooks, got {len(handles)}")
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
        raise RuntimeError(f"cross capture got layers {sorted(found)}")
    captures = [found[index] for index in range(LAYERS)]
    for capture in captures:
        if tuple(capture.keys.shape) != (1, HEADS, KV_LENGTH, HEAD_DIM):
            raise RuntimeError(f"layer {capture.layer_idx} has invalid key shape")
        if capture.keys.dtype != torch.float32 or capture.values.dtype != torch.float32:
            raise TypeError("source cross key/value caches must remain FP32")
        if not capture.keys.is_contiguous() or not capture.values.is_contiguous():
            raise ValueError("source cross key/value caches must be contiguous")
        if tuple(capture.residual.shape) != (1, 1, HEADS * HEAD_DIM):
            raise RuntimeError(f"layer {capture.layer_idx} has invalid residual shape")
    return captures


def _pack_linears(captures: list[CrossInputs]):
    from osuT5.osuT5.inference.optimized.kernels.weight_only import PackedLinear

    return [
        (
            PackedLinear.from_module(_LinearView(capture.query_weight, capture.query_bias)),
            PackedLinear.from_module(_LinearView(capture.output_weight, capture.output_bias)),
        )
        for capture in captures
    ]


def _fp32_block(captures: list[CrossInputs]):
    def run():
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
            ).view(1, 1, HEADS, HEAD_DIM).transpose(1, 2).contiguous()
            attention = _q1_bmm_cross_attention(
                query, capture.keys, capture.values, expected_dtype=torch.float32
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


def _selected_block(captures: list[CrossInputs], packs):
    def run():
        from osuT5.osuT5.inference.optimized.kernels.dispatch import (
            _q1_bmm_cross_attention,
        )
        from osuT5.osuT5.inference.optimized.kernels.weight_only import (
            weight_only_linear_residual,
            weight_only_rmsnorm_linear,
        )

        outputs = []
        for capture, (query_pack, output_pack) in zip(captures, packs, strict=True):
            query = weight_only_rmsnorm_linear(
                capture.residual,
                capture.cross_norm_weight,
                query_pack,
                eps=capture.cross_norm_eps,
                outputs_per_block=8,
            ).view(1, 1, HEADS, HEAD_DIM).transpose(1, 2).contiguous()
            attention = _q1_bmm_cross_attention(
                query, capture.keys, capture.values, expected_dtype=torch.float32
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

    return run


def _hybrid_block(captures: list[CrossInputs], packs, packed_keys):
    def run():
        from osuT5.osuT5.inference.optimized.kernels.decoder_layer import (
            native_one_token_linear_residual,
        )
        from osuT5.osuT5.inference.optimized.kernels.weight_only import (
            weight_only_rmsnorm_linear,
        )
        from osuT5.osuT5.inference.optimized.scout.hybrid_qk_cross import (
            hybrid_qk_fp32_value_attention,
        )

        outputs = []
        for capture, (query_pack, _), key in zip(
            captures, packs, packed_keys, strict=True
        ):
            query = weight_only_rmsnorm_linear(
                capture.residual,
                capture.cross_norm_weight,
                query_pack,
                eps=capture.cross_norm_eps,
                outputs_per_block=8,
            ).view(1, 1, HEADS, HEAD_DIM).transpose(1, 2).contiguous()
            attention = hybrid_qk_fp32_value_attention(query, key, capture.values)
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


def _observe(graph: CapturedGraph) -> tuple[torch.Tensor, ...]:
    graph.graph.replay()
    torch.cuda.synchronize()
    outputs = graph.outputs
    if not isinstance(outputs, tuple) or len(outputs) != LAYERS:
        raise RuntimeError("cross block graph must return one tensor per decoder layer")
    copied = tuple(value.detach().float().cpu().clone() for value in outputs)
    if any(tuple(value.shape) != (1, 1, HEADS * HEAD_DIM) for value in copied):
        raise RuntimeError("cross block graph returned an invalid output shape")
    return copied


def summarize_hybrid_component(
    buckets: dict[int, dict[str, Any]],
    *,
    main_counts: dict[int, int],
    timing_counts: dict[int, int],
    main_key_pack_setup_seconds: float = 0.0,
) -> dict[str, Any]:
    if set(buckets) != set(SENTINEL_PREFIXES):
        raise ValueError(f"bucket evidence must be sentinel prefixes {SENTINEL_PREFIXES}")
    if not set(SENTINEL_PREFIXES).issubset(main_counts):
        raise ValueError("main counts are missing sentinel prefixes")
    for label, counts in (("main", main_counts), ("timing", timing_counts)):
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in counts.values()):
            raise ValueError(f"{label} counts must be positive integers")
    if not timing_counts:
        raise ValueError("timing counts must be non-empty")
    if not math.isfinite(main_key_pack_setup_seconds) or main_key_pack_setup_seconds < 0:
        raise ValueError("main key-pack setup must be finite and non-negative")

    main_saving = 0.0
    timing_saving = 0.0
    checks_pass = True
    worst_selected_drift = 0.0
    worst_fp32_drift = 0.0
    per_bucket: dict[str, Any] = {}
    for prefix in SENTINEL_PREFIXES:
        entry = buckets[prefix]
        variants = entry.get("variants")
        if not isinstance(variants, dict):
            raise ValueError(f"bucket {prefix} is missing variants")
        current = variants["selected_packed_wq_wo_fp32_kv"]
        hybrid = variants["hybrid_packed_wq_fp16_k_fp32_v_fp32_wo"]
        current_ms = float(current["ms_per_decode_step"])
        hybrid_ms = float(hybrid["ms_per_decode_step"])
        selected_drift = float(hybrid["max_abs_drift_vs_selected"])
        fp32_drift = float(hybrid["max_abs_drift_vs_fp32"])
        numeric = (current_ms, hybrid_ms, selected_drift, fp32_drift)
        if not all(math.isfinite(value) for value in numeric) or min(current_ms, hybrid_ms) <= 0:
            raise ValueError(f"bucket {prefix} has invalid timing/drift")
        delta_ms = current_ms - hybrid_ms
        bucket_main = main_counts[prefix] * delta_ms / 1000.0
        bucket_timing = timing_counts.get(prefix, 0) * delta_ms / 1000.0
        main_saving += bucket_main
        timing_saving += bucket_timing
        worst_selected_drift = max(worst_selected_drift, selected_drift)
        worst_fp32_drift = max(worst_fp32_drift, fp32_drift)
        bucket_checks = bool(hybrid.get("checks_pass")) and selected_drift <= RELAXED_MAX_ABS_DRIFT
        checks_pass &= bucket_checks
        per_bucket[str(prefix)] = {
            "current_ms_per_decode_step": current_ms,
            "hybrid_ms_per_decode_step": hybrid_ms,
            "local_speedup": current_ms / hybrid_ms,
            "main_replays": main_counts[prefix],
            "timing_replays": timing_counts.get(prefix, 0),
            "main_saving_seconds": bucket_main,
            "timing_saving_seconds": bucket_timing,
            "checks_pass": bucket_checks,
        }
    raw_main_saving = main_saving
    main_saving -= main_key_pack_setup_seconds
    projected_seconds = SELECTED_MAIN_SECONDS - main_saving
    return {
        "candidate": "hybrid_packed_wq_fp16_k_fp32_v_fp32_wo",
        "selected_control": "selected_packed_wq_wo_fp32_kv",
        "measured_buckets": list(SENTINEL_PREFIXES),
        "unmeasured_main_buckets": sorted(set(main_counts) - set(SENTINEL_PREFIXES)),
        "unmeasured_bucket_delta_policy": "zero_delta",
        "raw_main_replay_saving_seconds": raw_main_saving,
        "main_key_pack_setup_seconds": main_key_pack_setup_seconds,
        "main_saving_seconds": main_saving,
        "timing_saving_seconds": timing_saving,
        "timing_plus_main_saving_seconds": main_saving + timing_saving,
        "saving_gate_seconds": COMPONENT_SAVING_GATE_SECONDS,
        "sizing_pass": main_saving >= COMPONENT_SAVING_GATE_SECONDS,
        "checks_pass": checks_pass,
        "worst_max_abs_drift_vs_selected": worst_selected_drift,
        "worst_max_abs_drift_vs_fp32": worst_fp32_drift,
        "relaxed_max_abs_drift": RELAXED_MAX_ABS_DRIFT,
        "selected_fixed_work_seconds": SELECTED_MAIN_SECONDS,
        "selected_fixed_work_tps": FIXED_WORK_MAIN_TOKENS / SELECTED_MAIN_SECONDS,
        "projected_fixed_work_seconds": projected_seconds,
        "projected_fixed_work_tps": FIXED_WORK_MAIN_TOKENS / projected_seconds,
        "fraction_of_remaining_500_gap": main_saving / TARGET_500_GAP_SECONDS,
        "component_retention_pass": checks_pass and main_saving >= COMPONENT_SAVING_GATE_SECONDS,
        "production_promotion_pass": False,
        "per_bucket": per_bucket,
    }


def _write_text(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    lines = [
        "hybrid qk cross-attention real-tensor sentinel component scout",
        f"commit={result['metadata']['commit']}",
        f"candidate={summary['candidate']}",
        f"main_saving_seconds={summary['main_saving_seconds']:.9f}",
        f"timing_saving_seconds={summary['timing_saving_seconds']:.9f}",
        f"projected_fixed_work_seconds={summary['projected_fixed_work_seconds']:.9f}",
        f"projected_fixed_work_tps={summary['projected_fixed_work_tps']:.6f}",
        f"worst_max_abs_drift_vs_selected={summary['worst_max_abs_drift_vs_selected']:.9g}",
        f"component_retention_pass={str(summary['component_retention_pass']).lower()}",
        "production_promotion_pass=false",
    ]
    for prefix, entry in summary["per_bucket"].items():
        lines.append(
            f"bucket={prefix} current_ms={entry['current_ms_per_decode_step']:.9f} "
            f"hybrid_ms={entry['hybrid_ms_per_decode_step']:.9f} "
            f"main_saving_seconds={entry['main_saving_seconds']:.9f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def profile_component(args, *, output_path: Path, warmup: int, iters: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("hybrid qk cross component scout requires CUDA")
    if warmup < 1 or iters < 1:
        raise ValueError("warmup and iters must be positive")
    _assert_scout_args(args)
    output_path.mkdir(parents=True, exist_ok=True)
    run = _run_and_capture_sessions(args, output_path=output_path)
    main_entries = _graph_entries(
        run["main_session"], required_prefixes=SENTINEL_PREFIXES
    )
    timing_entries = _graph_entries(run["timing_session"], required_prefixes=(128,))
    main_counts = {prefix: int(entry["decode_replays"]) for prefix, entry in main_entries.items()}
    timing_counts = {prefix: int(entry["decode_replays"]) for prefix, entry in timing_entries.items()}

    from osuT5.osuT5.inference.optimized.kernels.decoder_layer import preload_native_decoder_layer
    from osuT5.osuT5.inference.optimized.kernels.weight_only import preload_weight_only_extension

    torch.cuda.synchronize()
    preload_started = time.perf_counter()
    preload_native_decoder_layer()
    preload_weight_only_extension()
    torch.cuda.synchronize()
    preload_seconds = time.perf_counter() - preload_started

    bucket_reports: dict[int, dict[str, Any]] = {}
    total_key_pack_bytes = 0
    total_query_pack_bytes = 0
    total_selected_output_pack_bytes = 0
    for prefix in SENTINEL_PREFIXES:
        captures = _capture_real_cross_inputs(
            run["model"], main_entries[prefix]["static_inputs"], prefix=prefix
        )
        source_hashes = _input_hashes(captures)
        packs = _pack_linears(captures)
        key_pack_start = torch.cuda.Event(enable_timing=True)
        key_pack_end = torch.cuda.Event(enable_timing=True)
        key_pack_start.record()
        packed_keys = [capture.keys.to(dtype=torch.float16).contiguous() for capture in captures]
        key_pack_end.record()
        torch.cuda.synchronize()
        key_pack_ms = float(key_pack_start.elapsed_time(key_pack_end))
        if any(key.dtype != torch.float16 or not key.is_contiguous() for key in packed_keys):
            raise RuntimeError("hybrid key packing did not produce contiguous FP16 storage")

        callables: dict[str, Callable[[], tuple[torch.Tensor, ...]]] = {
            "fp32_wq_wo_fp32_kv": _fp32_block(captures),
            "selected_packed_wq_wo_fp32_kv": _selected_block(captures, packs),
            "hybrid_packed_wq_fp16_k_fp32_v_fp32_wo": _hybrid_block(
                captures, packs, packed_keys
            ),
        }
        graphs = {
            name: _capture_cuda_graph(callable_, context=nullcontext, warmup=0)
            for name, callable_ in callables.items()
        }
        timings, rounds, stable = _reciprocal_graph_rounds(
            {name: graph.graph for name, graph in graphs.items()},
            restore=lambda: None,
            warmup=warmup,
            iters=iters,
        )
        observations = {name: _observe(graph) for name, graph in graphs.items()}
        variants: dict[str, Any] = {}
        for name, graph in graphs.items():
            first = observations[name]
            second = _observe(graph)
            repeat = all(torch.equal(left, right) for left, right in zip(first, second, strict=True))
            finite = all(torch.isfinite(value).all().item() for value in first)
            drift_selected = max(
                _max_abs(reference, candidate)
                for reference, candidate in zip(
                    observations["selected_packed_wq_wo_fp32_kv"], first, strict=True
                )
            )
            drift_fp32 = max(
                _max_abs(reference, candidate)
                for reference, candidate in zip(
                    observations["fp32_wq_wo_fp32_kv"], first, strict=True
                )
            )
            checks = {
                "finite_outputs": bool(finite),
                "repeat_deterministic": bool(repeat),
                "memory_stable": bool(stable[name]),
                "source_fp32_cache_weights_unchanged": _input_hashes(captures) == source_hashes,
            }
            variants[name] = {
                "ms_per_decode_step": float(timings[name]),
                "ms_per_layer": float(timings[name]) / LAYERS,
                "max_abs_drift_vs_selected": float(drift_selected),
                "max_abs_drift_vs_fp32": float(drift_fp32),
                "checks": checks,
                "checks_pass": all(checks.values()),
                "capture_setup_seconds": graph.setup_seconds,
                "capture_peak_vram_bytes": graph.peak_vram_bytes,
                "rounds": [row for row in rounds if row["variant"] == name],
            }
        if _input_hashes(captures) != source_hashes:
            raise RuntimeError("component candidate mutated source cache or weights")
        bucket_reports[prefix] = {
            "main_decode_replays": main_counts[prefix],
            "timing_decode_replays": timing_counts.get(prefix, 0),
            "key_pack_setup_ms": key_pack_ms,
            "variants": variants,
        }
        total_key_pack_bytes += sum(key.numel() * key.element_size() for key in packed_keys)
        total_query_pack_bytes += sum(query.packed_weight_bytes for query, _ in packs)
        total_selected_output_pack_bytes += sum(output.packed_weight_bytes for _, output in packs)

    state_holders = getattr(run["main_session"], "stable_encoder_holders", None)
    if not isinstance(state_holders, dict) or not state_holders:
        raise RuntimeError("main decode session has no stable encoder-state manifest")
    main_state_count = len(state_holders)
    worst_key_pack_ms = max(
        float(report["key_pack_setup_ms"]) for report in bucket_reports.values()
    )
    charged_key_pack_seconds = main_state_count * worst_key_pack_ms / 1000.0
    summary = summarize_hybrid_component(
        bucket_reports,
        main_counts=main_counts,
        timing_counts=timing_counts,
        main_key_pack_setup_seconds=charged_key_pack_seconds,
    )
    return {
        "schema_version": 1,
        "metadata": {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "candidate_scope": "component_only_no_production_wiring",
            "selected_topology_base": "k4_shared_rope_k1_remainder_int8_mlp_fp16_packed_cross",
            "candidate_storage": {
                "query_weight": "fp16_packed",
                "query_activation_for_qk": "fp16",
                "key_cache": "fp16_packed",
                "score_accumulation": "fp32",
                "softmax": "fp32",
                "value_cache_and_reduction": "fp32",
                "output_projection_weight_and_activation": "fp32",
            },
            "sentinel_prefixes": list(SENTINEL_PREFIXES),
            "warmup": warmup,
            "iters": iters,
            "extension_preload_seconds": preload_seconds,
        },
        "workload": {
            "main_counts": {str(key): value for key, value in main_counts.items()},
            "timing_counts": {str(key): value for key, value in timing_counts.items()},
            "main_graph_profile": run["main_session"].graph_profile_summary(),
            "timing_graph_profile": run["timing_session"].graph_profile_summary(),
        },
        "memory": {
            "sentinel_key_pack_bytes": total_key_pack_bytes,
            "sentinel_query_weight_pack_bytes": total_query_pack_bytes,
            "selected_output_weight_pack_bytes_not_retained_by_hybrid": total_selected_output_pack_bytes,
            "main_encoder_state_count": main_state_count,
            "worst_sentinel_key_pack_ms_per_state": worst_key_pack_ms,
            "charged_main_key_pack_seconds": charged_key_pack_seconds,
            "key_pack_setup_is_recorded_not_hidden": True,
        },
        "buckets": {str(prefix): report for prefix, report in bucket_reports.items()},
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--text-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("overrides", nargs="*")
    cli = parser.parse_args()
    overrides = list(cli.overrides)
    if not any(value.startswith("output_path=") for value in overrides):
        overrides.append(f"output_path={cli.output_path}")
    args = _load_args(cli.config_name, overrides)
    result = profile_component(
        args, output_path=cli.output_path, warmup=cli.warmup, iters=cli.iters
    )
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.text_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_text(cli.text_path, result)
    print(json.dumps(result["summary"], indent=2))
    if not result["summary"]["checks_pass"]:
        raise SystemExit(1)
    if not result["summary"]["component_retention_pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
