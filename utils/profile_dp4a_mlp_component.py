"""Real-tensor reciprocal hot/cold gate for the opt-in SM75 DP4A MLP."""

from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from utils.profile_native_prefix_dtype_scout import (  # noqa: E402
    _accepted_main_session_run,
    _all_cache_snapshots,
    _assert_scout_args,
    _cache_from_static_inputs,
    _capture_cuda_graph,
    _load_args,
    _reciprocal_graph_rounds,
    _restore_all_cache,
)


DECODER_LAYERS = 12
FIXED_MAIN_STEPS = 8_294
FIXED_MAIN_SAVING_GATE_SECONDS = 0.300
TARGET_LAYER_INDEX = 6
SENTINEL_BUCKETS = (128, 576, 640)
CONFIGS = {
    "w4_c32": (4, 32),
    "w4_c48": (4, 48),
    "w4_c68": (4, 68),
    "w8_c32": (8, 32),
    "w8_c48": (8, 48),
    "w8_c68": (8, 68),
}
QUANTIZED_REFERENCE_MAX_ABS = 1e-4


@dataclass(frozen=True)
class MlpCapture:
    module: torch.nn.Module
    hidden_states: torch.Tensor


def _eps(module: torch.nn.Module) -> float:
    value = getattr(module, "eps", None)
    return float(torch.finfo(torch.float32).eps if value is None else value)


def _live_entries(graph_cache: dict[Any, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    if not isinstance(graph_cache, dict):
        raise TypeError("ProductionDecodeSession.graph_cache must be a dict")
    entries: dict[int, dict[str, Any]] = {}
    for value in graph_cache.values():
        if not isinstance(value, dict):
            raise TypeError("graph cache entries must be dictionaries")
        prefix = value.get("active_prefix_length")
        if not isinstance(prefix, int) or prefix <= 0:
            raise ValueError(f"graph cache has invalid active prefix {prefix!r}")
        if prefix in entries:
            raise RuntimeError(f"graph cache repeats active prefix {prefix}")
        missing = [
            name
            for name in ("graph", "outputs", "static_inputs", "decode_replays")
            if name not in value
        ]
        if missing:
            raise RuntimeError(f"prefix {prefix} graph entry is missing {missing}")
        if int(value["decode_replays"]) <= 0:
            raise RuntimeError(f"prefix {prefix} decode replay count must be positive")
        entries[prefix] = value
    missing = sorted(set(SENTINEL_BUCKETS) - set(entries))
    if missing:
        raise RuntimeError(f"live graph cache is missing sentinel buckets {missing}")
    return dict(sorted(entries.items()))


def _select(entries: dict[int, dict[str, Any]], mode: str) -> dict[int, dict[str, Any]]:
    if mode == "sentinel":
        return {prefix: entries[prefix] for prefix in SENTINEL_BUCKETS}
    if mode == "all":
        return dict(entries)
    raise ValueError("bucket mode must be 'sentinel' or 'all'")


def _selected_main_session_run(args, *, output_path: Path) -> dict[str, Any]:
    """Capture live graph state with the selected INT8-MLP/FP16-cross runtime."""

    import inference

    from osuT5.osuT5.inference.optimized.kernels.weight_only_runtime import (
        CROSS_FP16_PACKED,
    )

    original_loader = inference.load_model_with_engine
    captured: dict[str, Any] = {}
    loads = 0

    def selected_loader(*positional, **kwargs):
        nonlocal loads
        binding, tokenizer = original_loader(*positional, **kwargs)
        loads += 1
        if loads == 1:
            initializer = getattr(
                binding.runtime,
                "initialize_approximate_int8_mlp_weight_only_cross",
                None,
            )
            if initializer is None:
                raise RuntimeError("optimized runtime lacks selected composition initializer")
            metadata = initializer(binding.raw_model, mode=CROSS_FP16_PACKED)
            state = getattr(binding.runtime, "_approximate_weight_only_state", None)
            if state is None or not state.int8_mlp_enabled:
                raise RuntimeError("selected composition did not retain the INT8 MLP state")
            if state.cross_mode != CROSS_FP16_PACKED:
                raise RuntimeError("selected composition did not retain FP16 cross weights")
            captured.update(
                binding=binding,
                model=binding.raw_model,
                state=state,
                initialization_metadata=metadata,
            )
        return binding, tokenizer

    inference.load_model_with_engine = selected_loader
    try:
        run = _accepted_main_session_run(args, output_path=output_path)
    finally:
        inference.load_model_with_engine = original_loader
    if loads < 2:
        raise RuntimeError("selected capture expected separate main and timing bindings")
    if not captured:
        raise RuntimeError("selected capture did not initialize the main binding")
    run.update(captured)
    return run


def _capture_real_mlp_input(
    model: torch.nn.Module,
    state: Any,
    static_inputs: dict[str, Any],
    *,
    prefix: int,
) -> tuple[MlpCapture, dict[str, int]]:
    from osuT5.osuT5.inference.optimized.kernels.weight_only_runtime import (
        approximate_weight_only_runtime_context,
    )
    from osuT5.osuT5.inference.optimized.scout import int8_mlp
    from osuT5.osuT5.model.custom_transformers.modeling_varwhisper import (
        VarWhisperDecoderLayer,
    )
    from osuT5.osuT5.runtime_profiling import generation_profile_context

    layers = [
        module for module in model.modules() if isinstance(module, VarWhisperDecoderLayer)
    ]
    if len(layers) != DECODER_LAYERS:
        raise RuntimeError(
            f"DP4A MLP scout requires {DECODER_LAYERS} decoder layers, found {len(layers)}"
        )
    target = layers[TARGET_LAYER_INDEX]
    target_pack = state.int8_mlp_pack_for_layer(target)
    original = int8_mlp.int8_weight_mlp_residual
    found: dict[str, torch.Tensor] = {}
    dispatch_counts: dict[str, int] = {
        "native_q1_rope_cache_self_attention": 0,
        "native_q1_self_attention": 0,
        "q1_bmm_cross_attention": 0,
        "native_cross_mlp_tail": 0,
        "weight_only_self_attention_block": 0,
        "weight_only_mlp_tail": 0,
        "weight_only_final_projection": 0,
        "int8_weight_mlp_tail": 0,
        "fp16_packed_cross_projection_candidate": 0,
    }

    def capture(input, norm_weight, pack, **kwargs):
        if pack is target_pack:
            if found:
                raise RuntimeError("captured target MLP more than once in one replay")
            if norm_weight is not target.final_layer_norm.weight:
                raise RuntimeError("target MLP capture observed the wrong norm owner")
            found["hidden_states"] = input.detach().clone()
        return original(input, norm_weight, pack, **kwargs)

    int8_mlp.int8_weight_mlp_residual = capture
    try:
        with generation_profile_context(
            active_prefix_self_attention_length=prefix,
            q1_bmm_cross_attention=True,
            native_q1_self_attention=True,
            native_q1_rope_cache_self_attention=True,
        ), approximate_weight_only_runtime_context(
            state,
            dispatch_counts=dispatch_counts,
        ):
            model(**static_inputs, return_dict=True)
        torch.cuda.synchronize()
    finally:
        int8_mlp.int8_weight_mlp_residual = original
    hidden = found.get("hidden_states")
    if hidden is None:
        raise RuntimeError(f"did not capture selected MLP input for prefix {prefix}")
    if tuple(hidden.shape) != (1, 1, 768) or hidden.dtype != torch.float32:
        raise RuntimeError(
            f"captured MLP input must be [1,1,768] FP32, got {hidden.shape} {hidden.dtype}"
        )
    return MlpCapture(target, hidden), dispatch_counts


def _time_eviction_graph(graph: torch.cuda.CUDAGraph, *, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iters


def _cache_cold_rounds(
    graphs: dict[str, torch.cuda.CUDAGraph],
    eviction_graph: torch.cuda.CUDAGraph,
    *,
    warmup: int,
    iters: int,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, bool], float]:
    if warmup < 1 or iters < 1:
        raise ValueError("cache-cold warmup and iters must be positive")
    names = list(graphs)
    rounds: list[dict[str, Any]] = []
    stable = {name: True for name in names}
    eviction_ms = _time_eviction_graph(eviction_graph, iters=iters)
    for order in (names, list(reversed(names))):
        for name in order:
            graph = graphs[name]
            for _ in range(warmup):
                eviction_graph.replay()
                graph.replay()
            torch.cuda.synchronize()
            allocated_before = int(torch.cuda.memory_allocated())
            raw_starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            component_starts = [
                torch.cuda.Event(enable_timing=True) for _ in range(iters)
            ]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            for index in range(iters):
                raw_starts[index].record()
                eviction_graph.replay()
                component_starts[index].record()
                graph.replay()
                ends[index].record()
            torch.cuda.synchronize()
            allocated_after = int(torch.cuda.memory_allocated())
            isolated = sum(
                start.elapsed_time(end)
                for start, end in zip(component_starts, ends, strict=True)
            ) / iters
            raw = sum(
                start.elapsed_time(end)
                for start, end in zip(raw_starts, ends, strict=True)
            ) / iters
            subtracted = max(0.0, raw - eviction_ms)
            conservative = max(float(isolated), float(subtracted))
            memory_stable = allocated_before == allocated_after
            stable[name] &= memory_stable
            rounds.append(
                {
                    "variant": name,
                    "order": list(order),
                    "isolated_ms_per_call": float(isolated),
                    "raw_evict_plus_component_ms_per_call": float(raw),
                    "eviction_ms_per_call": float(eviction_ms),
                    "subtracted_ms_per_call": float(subtracted),
                    "conservative_ms_per_call": conservative,
                    "subtraction_disagreement_ms": abs(float(isolated) - subtracted),
                    "memory_stable": memory_stable,
                }
            )
    timings = {
        name: max(
            float(row["conservative_ms_per_call"])
            for row in rounds
            if row["variant"] == name
        )
        for name in names
    }
    return timings, rounds, stable, eviction_ms


def _drift(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference = reference.detach().float().cpu()
    candidate = candidate.detach().float().cpu()
    if reference.shape != candidate.shape:
        return {"max_abs": math.inf, "mean_abs": math.inf, "relative_l2": math.inf}
    difference = candidate - reference
    relative = torch.linalg.vector_norm(difference) / torch.clamp(
        torch.linalg.vector_norm(reference), min=1e-12
    )
    return {
        "max_abs": float(difference.abs().max().item()),
        "mean_abs": float(difference.abs().mean().item()),
        "relative_l2": float(relative.item()),
    }


def _integer_linear_reference(
    quantized: torch.Tensor,
    activation_scale: torch.Tensor,
    pack: Any,
) -> torch.Tensor:
    integer_dot = torch.mv(
        pack.weight.detach().cpu().to(torch.float64),
        quantized.detach().cpu().to(torch.float64),
    ).to(torch.float32)
    value = integer_dot * (
        activation_scale.detach().cpu() * pack.scale.detach().cpu()
    )
    if pack.bias is not None:
        value = value + pack.bias.detach().cpu()
    return value


def _dynamic_quantization_reference(
    values: torch.Tensor,
    quantized: torch.Tensor,
    activation_scale: torch.Tensor,
    *,
    norm_weight: torch.Tensor | None = None,
    eps: float | None = None,
) -> dict[str, Any]:
    reference = values.detach().float().cpu().view(-1)
    if norm_weight is not None:
        if eps is None:
            raise ValueError("normalized dynamic-quantization reference requires eps")
        weight = norm_weight.detach().float().cpu().view(-1)
        if reference.shape != weight.shape:
            raise ValueError("dynamic-quantization norm/input shape mismatch")
        reference = reference * weight * torch.rsqrt(
            reference.square().mean() + float(eps)
        )
    observed_scale = float(activation_scale.detach().float().cpu().item())
    maximum = float(reference.abs().max().item())
    expected_scale = maximum / 127.0 if maximum > 0.0 else 1.0
    scale_error = abs(observed_scale - expected_scale)
    scale_tolerance = max(1e-7, abs(expected_scale) * 2e-5)
    quantized_cpu = quantized.detach().cpu().view(-1)
    reconstructed = quantized_cpu.float() * observed_scale
    reconstruction_max_abs = float((reconstructed - reference).abs().max().item())
    rounding_tolerance = max(observed_scale, expected_scale) * 0.501 + 2e-6
    contains_minus_128 = bool((quantized_cpu == -128).any().item())
    passed = (
        math.isfinite(observed_scale)
        and observed_scale > 0.0
        and scale_error <= scale_tolerance
        and reconstruction_max_abs <= rounding_tolerance
        and not contains_minus_128
    )
    return {
        "observed_scale": observed_scale,
        "reference_scale": expected_scale,
        "scale_abs_error": scale_error,
        "scale_tolerance": scale_tolerance,
        "reconstruction_max_abs": reconstruction_max_abs,
        "rounding_tolerance": rounding_tolerance,
        "contains_minus_128": contains_minus_128,
        "pass": passed,
    }


def _quantized_reference(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    pack: Any,
    state: tuple[torch.Tensor, ...],
    *,
    eps: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    output, activated, fc1_quantized, fc1_scale, fc2_quantized, fc2_scale = state
    fc1_pre_gelu = _integer_linear_reference(fc1_quantized, fc1_scale, pack.fc1)
    fc1_reference = 0.5 * fc1_pre_gelu * (
        1.0 + torch.erf(fc1_pre_gelu * 0.7071067811865475244)
    )
    fc2_reference = _integer_linear_reference(fc2_quantized, fc2_scale, pack.fc2)
    final_reference = fc2_reference.view_as(input.cpu()) + input.detach().cpu()
    activated_cpu = activated.detach().float().cpu().view(-1)
    fc1_quantization = _dynamic_quantization_reference(
        input,
        fc1_quantized,
        fc1_scale,
        norm_weight=norm_weight,
        eps=eps,
    )
    fc2_quantization = _dynamic_quantization_reference(
        activated,
        fc2_quantized,
        fc2_scale,
    )
    return final_reference, {
        "fc1_kernel_vs_quantized_reference": _drift(fc1_reference, activated_cpu),
        "fc2_kernel_vs_quantized_reference": _drift(final_reference, output),
        "fc1_activation_scale": float(fc1_scale.item()),
        "fc2_activation_scale": float(fc2_scale.item()),
        "fc1_saturation_count": int((fc1_quantized.abs() == 127).sum().item()),
        "fc2_saturation_count": int((fc2_quantized.abs() == 127).sum().item()),
        "fc1_dynamic_quantization": fc1_quantization,
        "fc2_dynamic_quantization": fc2_quantization,
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
        raise ValueError("DP4A MLP component summary requires non-empty buckets")
    measured_replays = sum(int(entry["decode_replays"]) for entry in buckets.values())
    if measured_replays <= 0 or total_replays < measured_replays:
        raise ValueError("invalid DP4A MLP replay counts")
    fixed_scale = FIXED_MAIN_STEPS / measured_replays
    measured_by_mode: dict[str, dict[str, Any]] = {}
    for mode in ("l2_hot", "cache_cold"):
        measured_baseline = sum(
            int(entry["decode_replays"])
            * DECODER_LAYERS
            * float(entry[mode]["baseline_ms_per_call"])
            for entry in buckets.values()
        ) / 1000.0
        candidates = {
            config: sum(
                int(entry["decode_replays"])
                * DECODER_LAYERS
                * float(entry[mode]["candidates"][config]["ms_per_call"])
                for entry in buckets.values()
            )
            / 1000.0
            for config in CONFIGS
        }
        measured_by_mode[mode] = {
            "baseline": measured_baseline * fixed_scale,
            "candidates": {
                config: value * fixed_scale for config, value in candidates.items()
            },
        }
    selected = min(
        measured_by_mode["l2_hot"]["candidates"],
        key=measured_by_mode["l2_hot"]["candidates"].get,
    )
    modes: dict[str, Any] = {}
    for mode in ("l2_hot", "cache_cold"):
        baseline = measured_by_mode[mode]["baseline"]
        fixed_candidates = measured_by_mode[mode]["candidates"]
        candidate = fixed_candidates[selected]
        saving = baseline - candidate
        modes[mode] = {
            "selected_config": selected,
            "unconstrained_fastest_config": min(
                fixed_candidates, key=fixed_candidates.get
            ),
            "fixed_8294_baseline_seconds": baseline,
            "fixed_8294_candidate_seconds": candidate,
            "fixed_8294_saving_seconds": saving,
            "fixed_8294_speedup": baseline / candidate,
            "candidate_seconds_by_config": fixed_candidates,
        }
    invariant_failures: dict[str, list[str]] = {}
    for prefix, entry in buckets.items():
        failures = [name for name, passed in entry["checks"].items() if not passed]
        if failures:
            invariant_failures[prefix] = failures
    invariants_pass = not invariant_failures
    hot_pass = (
        modes["l2_hot"]["fixed_8294_saving_seconds"]
        >= FIXED_MAIN_SAVING_GATE_SECONDS
    )
    cold_pass = (
        modes["cache_cold"]["fixed_8294_saving_seconds"]
        >= FIXED_MAIN_SAVING_GATE_SECONDS
    )
    return {
        "measured_replays": measured_replays,
        "total_replays": int(total_replays),
        "coverage_fraction": measured_replays / total_replays,
        "fixed_work_steps": FIXED_MAIN_STEPS,
        "modes": modes,
        "saving_gate_seconds": FIXED_MAIN_SAVING_GATE_SECONDS,
        "l2_hot_saving_pass": hot_pass,
        "cache_cold_saving_pass": cold_pass,
        "invariant_failures": invariant_failures,
        "invariants_pass": invariants_pass,
        "component_pass": invariants_pass and hot_pass and cold_pass,
        "production_wiring": False,
    }


@torch.no_grad()
def profile_component(
    args,
    *,
    output_path: Path,
    bucket_mode: str,
    warmup: int,
    iters: int,
    cold_warmup: int,
    cold_iters: int,
    eviction_bytes: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("DP4A MLP profiler requires CUDA")
    if min(warmup, iters, cold_warmup, cold_iters) < 1:
        raise ValueError("all warmup/iteration counts must be positive")
    if eviction_bytes < 32 * 1024 * 1024 or eviction_bytes % 4:
        raise ValueError("eviction_bytes must be a multiple of four and at least 32 MiB")
    _assert_scout_args(args)
    output_path.mkdir(parents=True, exist_ok=True)
    run = _selected_main_session_run(args, output_path=output_path)
    model = run["model"]
    state = run["state"]
    model.eval()
    entries = _live_entries(run["session"].graph_cache)
    selected = _select(entries, bucket_mode)
    total_replays = sum(int(entry["decode_replays"]) for entry in entries.values())

    from osuT5.osuT5.inference.optimized.scout.int8_dp4a_mlp import (
        dp4a_mlp_residual,
        dp4a_mlp_residual_with_state,
        preload_dp4a_mlp_extension,
        validate_dp4a_mlp_pack,
    )
    from osuT5.osuT5.inference.optimized.scout.int8_mlp import (
        int8_weight_mlp_residual,
        preload_int8_mlp_extension,
    )

    preload_seconds: dict[str, float] = {}
    for name, preload in (
        ("current_scalar_int8_mlp", preload_int8_mlp_extension),
        ("candidate_sm75_dp4a_mlp", preload_dp4a_mlp_extension),
    ):
        torch.cuda.synchronize()
        started = time.perf_counter()
        preload()
        torch.cuda.synchronize()
        preload_seconds[name] = time.perf_counter() - started

    eviction = torch.zeros(
        eviction_bytes // 4,
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )
    eviction_capture = _capture_cuda_graph(
        lambda: eviction.add_(1.0),
        context=nullcontext,
        warmup=1,
    )

    buckets: dict[str, Any] = {}
    pack_validation: dict[str, Any] = {}
    for prefix, entry in selected.items():
        static_inputs = entry["static_inputs"]
        cache = _cache_from_static_inputs(static_inputs)
        cache_position = static_inputs.get("cache_position")
        if not isinstance(cache_position, torch.Tensor) or cache_position.numel() != 1:
            raise RuntimeError(f"prefix {prefix} has invalid cache position")
        snapshots = _all_cache_snapshots(cache, cache_position)
        _restore_all_cache(cache, snapshots)
        capture, dispatch_counts = _capture_real_mlp_input(
            model,
            state,
            static_inputs,
            prefix=prefix,
        )
        _restore_all_cache(cache, snapshots)
        layer = capture.module
        pack = state.int8_mlp_pack_for_layer(layer)
        pack_validation[str(prefix)] = validate_dp4a_mlp_pack(pack)
        eps = _eps(layer.final_layer_norm)
        callables = {
            "baseline": lambda: int8_weight_mlp_residual(
                capture.hidden_states,
                layer.final_layer_norm.weight,
                pack,
                eps=eps,
                fc1_outputs_per_block=2,
                fc2_outputs_per_block=4,
            )
        }
        callables.update(
            {
                config: (
                    lambda warps=warps, ctas=ctas: dp4a_mlp_residual(
                        capture.hidden_states,
                        layer.final_layer_norm.weight,
                        pack,
                        eps=eps,
                        active_warps=warps,
                        persistent_ctas=ctas,
                    )
                )
                for config, (warps, ctas) in CONFIGS.items()
            }
        )
        graphs = {
            name: _capture_cuda_graph(fn, context=nullcontext, warmup=0)
            for name, fn in callables.items()
        }
        hot, hot_rounds, hot_memory = _reciprocal_graph_rounds(
            {name: graph.graph for name, graph in graphs.items()},
            restore=lambda: None,
            warmup=warmup,
            iters=iters,
        )
        cold, cold_rounds, cold_memory, eviction_ms = _cache_cold_rounds(
            {name: graph.graph for name, graph in graphs.items()},
            eviction_capture.graph,
            warmup=cold_warmup,
            iters=cold_iters,
        )
        observations = {name: _observe(graph) for name, graph in graphs.items()}
        baseline = observations["baseline"][0]
        candidate_rows: dict[str, Any] = {}
        first_candidate: torch.Tensor | None = None
        checks = {
            "baseline_finite": bool(torch.isfinite(baseline).all()),
            "baseline_repeat_bitwise_equal": torch.equal(*observations["baseline"]),
            "baseline_memory_stable": bool(
                hot_memory["baseline"] and cold_memory["baseline"]
            ),
            "selected_weight_only_self_dispatch_observed": (
                dispatch_counts.get("weight_only_self_attention_block")
                == DECODER_LAYERS
            ),
            "selected_native_rope_cache_self_dispatch_observed": (
                dispatch_counts.get("native_q1_rope_cache_self_attention")
                == DECODER_LAYERS
            ),
            "selected_q1_bmm_cross_dispatch_observed": (
                dispatch_counts.get("q1_bmm_cross_attention") == DECODER_LAYERS
            ),
            "selected_weight_only_mlp_dispatch_observed": (
                dispatch_counts.get("weight_only_mlp_tail") == DECODER_LAYERS
            ),
            "selected_int8_dispatch_observed": (
                dispatch_counts.get("int8_weight_mlp_tail") == DECODER_LAYERS
            ),
            "selected_fp16_cross_dispatch_observed": (
                dispatch_counts.get("fp16_packed_cross_projection_candidate")
                == DECODER_LAYERS
            ),
            "selected_weight_only_final_projection_observed": (
                dispatch_counts.get("weight_only_final_projection") == 1
            ),
            "rejected_native_cross_dispatch_absent": (
                dispatch_counts.get("native_cross_mlp_tail") == 0
            ),
            "nonfused_native_self_fallback_absent": (
                dispatch_counts.get("native_q1_self_attention") == 0
            ),
        }
        for config, (warps, ctas) in CONFIGS.items():
            direct_state = dp4a_mlp_residual_with_state(
                capture.hidden_states,
                layer.final_layer_norm.weight,
                pack,
                eps=eps,
                active_warps=warps,
                persistent_ctas=ctas,
            )
            torch.cuda.synchronize()
            reference, quantized = _quantized_reference(
                capture.hidden_states,
                layer.final_layer_norm.weight,
                pack,
                direct_state,
                eps=eps,
            )
            observed = observations[config][0]
            output_reference_drift = _drift(reference, observed)
            fc1_reference_drift = quantized["fc1_kernel_vs_quantized_reference"]
            candidate_rows[config] = {
                "active_warps": warps,
                "persistent_ctas": ctas,
                "finite": bool(torch.isfinite(observed).all()),
                "repeat_bitwise_equal": torch.equal(*observations[config]),
                "memory_stable": bool(hot_memory[config] and cold_memory[config]),
                "kernel_vs_quantized_reference": output_reference_drift,
                "fc1_kernel_vs_quantized_reference": fc1_reference_drift,
                "kernel_vs_quantized_reference_pass": (
                    output_reference_drift["max_abs"] <= QUANTIZED_REFERENCE_MAX_ABS
                    and fc1_reference_drift["max_abs"]
                    <= QUANTIZED_REFERENCE_MAX_ABS
                    and quantized["fc1_dynamic_quantization"]["pass"]
                    and quantized["fc2_dynamic_quantization"]["pass"]
                ),
                "candidate_vs_current_scalar_int8": _drift(baseline, observed),
                "quantized_activation": quantized,
                "l2_hot_ms_per_call": float(hot[config]),
                "cache_cold_ms_per_call": float(cold[config]),
            }
            checks[f"{config}_finite"] = candidate_rows[config]["finite"]
            checks[f"{config}_repeat_bitwise_equal"] = candidate_rows[config][
                "repeat_bitwise_equal"
            ]
            checks[f"{config}_memory_stable"] = candidate_rows[config][
                "memory_stable"
            ]
            checks[f"{config}_quantized_reference"] = candidate_rows[config][
                "kernel_vs_quantized_reference_pass"
            ]
            if first_candidate is None:
                first_candidate = observed
            else:
                checks[f"{config}_cross_config_bitwise_equal"] = torch.equal(
                    first_candidate, observed
                )
        buckets[str(prefix)] = {
            "decode_replays": int(entry["decode_replays"]),
            "target_layer_index": TARGET_LAYER_INDEX,
            "hidden_shape": list(capture.hidden_states.shape),
            "selected_dispatch_counts": dispatch_counts,
            "checks": checks,
            "candidates": candidate_rows,
            "l2_hot": {
                "baseline_ms_per_call": float(hot["baseline"]),
                "candidates": {
                    config: {"ms_per_call": float(hot[config])} for config in CONFIGS
                },
                "rounds": hot_rounds,
            },
            "cache_cold": {
                "baseline_ms_per_call": float(cold["baseline"]),
                "candidates": {
                    config: {"ms_per_call": float(cold[config])} for config in CONFIGS
                },
                "eviction_ms_per_call": float(eviction_ms),
                "rounds": cold_rounds,
            },
        }
        del graphs
        torch.cuda.empty_cache()

    summary = summarize_component(buckets, total_replays=total_replays)
    return {
        "schema_version": 1,
        "metadata": {
            "candidate": "two_stage_sm75_dynamic_activation_int8_dp4a_mlp",
            "result_class": "component_scout_documented_drift",
            "production_wiring": False,
            "comparison": "selected_scalar_fp32_accumulation_int8_mlp",
            "selected_base_commit": "7569d50b26bada95c4dafcadbdb68b15f628fc16",
            "selected_initialization_metadata": run["initialization_metadata"],
            "precision_contract": {
                "weight_storage": "int8_per_output_row_symmetric",
                "activation_storage": "dynamic_int8_per_token_symmetric",
                "accumulation": "signed_dp4a_int32",
                "norm_bias_gelu_residual_output": "fp32",
                "exactness_claim": False,
            },
            "configs": {
                name: {"active_warps": warps, "persistent_ctas": ctas}
                for name, (warps, ctas) in CONFIGS.items()
            },
            "bucket_mode": bucket_mode,
            "measured_buckets": list(selected),
            "live_bucket_counts": {
                str(prefix): int(entry["decode_replays"])
                for prefix, entry in entries.items()
            },
            "warmup": warmup,
            "iters": iters,
            "cold_warmup": cold_warmup,
            "cold_iters": cold_iters,
            "eviction_bytes": eviction_bytes,
            "extension_preload_seconds": preload_seconds,
        },
        "pack_validation": pack_validation,
        "buckets": buckets,
        "summary": summary,
    }


def _text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "two-stage SM75 dynamic-activation INT8 DP4A MLP component scout",
        f"measured_replays={summary['measured_replays']}",
        f"total_replays={summary['total_replays']}",
    ]
    for mode, row in summary["modes"].items():
        lines.append(
            f"{mode}: config={row['selected_config']} "
            f"baseline_seconds={row['fixed_8294_baseline_seconds']:.6f} "
            f"candidate_seconds={row['fixed_8294_candidate_seconds']:.6f} "
            f"saving_seconds={row['fixed_8294_saving_seconds']:.6f}"
        )
    lines.extend(
        [
            f"saving_gate_seconds={summary['saving_gate_seconds']:.6f}",
            f"l2_hot_saving_pass={summary['l2_hot_saving_pass']}",
            f"cache_cold_saving_pass={summary['cache_cold_saving_pass']}",
            f"invariants_pass={summary['invariants_pass']}",
            f"component_pass={summary['component_pass']}",
            "production_wiring=False",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "prefix",
        "decode_replays",
        "mode",
        "variant",
        "active_warps",
        "persistent_ctas",
        "ms_per_call",
        "candidate_vs_baseline_max_abs",
        "quantized_reference_max_abs",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for prefix, bucket in report["buckets"].items():
            for mode in ("l2_hot", "cache_cold"):
                writer.writerow(
                    {
                        "prefix": prefix,
                        "decode_replays": bucket["decode_replays"],
                        "mode": mode,
                        "variant": "baseline",
                        "active_warps": "",
                        "persistent_ctas": "",
                        "ms_per_call": bucket[mode]["baseline_ms_per_call"],
                        "candidate_vs_baseline_max_abs": 0.0,
                        "quantized_reference_max_abs": 0.0,
                    }
                )
                for config, timing in bucket[mode]["candidates"].items():
                    candidate = bucket["candidates"][config]
                    writer.writerow(
                        {
                            "prefix": prefix,
                            "decode_replays": bucket["decode_replays"],
                            "mode": mode,
                            "variant": config,
                            "active_warps": candidate["active_warps"],
                            "persistent_ctas": candidate["persistent_ctas"],
                            "ms_per_call": timing["ms_per_call"],
                            "candidate_vs_baseline_max_abs": candidate[
                                "candidate_vs_current_scalar_int8"
                            ]["max_abs"],
                            "quantized_reference_max_abs": candidate[
                                "kernel_vs_quantized_reference"
                            ]["max_abs"],
                        }
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--text-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--bucket-mode", choices=("sentinel", "all"), default="sentinel")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--cold-warmup", type=int, default=10)
    parser.add_argument("--cold-iters", type=int, default=100)
    parser.add_argument("--eviction-bytes", type=int, default=64 * 1024 * 1024)
    parsed, overrides = parser.parse_known_args()
    if not any(value.startswith("output_path=") for value in overrides):
        overrides.append(f"output_path={parsed.output_path}")
    args = _load_args(parsed.config_name, overrides)
    report = profile_component(
        args,
        output_path=parsed.output_path,
        bucket_mode=parsed.bucket_mode,
        warmup=parsed.warmup,
        iters=parsed.iters,
        cold_warmup=parsed.cold_warmup,
        cold_iters=parsed.cold_iters,
        eviction_bytes=parsed.eviction_bytes,
    )
    parsed.report_path.parent.mkdir(parents=True, exist_ok=True)
    parsed.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    parsed.text_path.parent.mkdir(parents=True, exist_ok=True)
    parsed.text_path.write_text(_text(report), encoding="utf-8")
    _write_csv(report, parsed.csv_path)
    print(_text(report), end="")
    if not report["summary"]["component_pass"]:
        raise SystemExit("STOP_DP4A_MLP_COMPONENT")
    raise SystemExit("REQUIRE_SELECTED_FULL_GRAPH_AB")


if __name__ == "__main__":
    main()
