"""Real-tensor hot/cold component gate for the CUTLASS-seeking SM75 w8a8 projections."""

from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import nullcontext
from pathlib import Path
import sys
import time
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from utils import profile_int8_projection_component as retained  # noqa: E402
from utils.profile_native_prefix_dtype_scout import (  # noqa: E402
    _all_cache_snapshots,
    _assert_scout_args,
    _cache_from_static_inputs,
    _capture_cuda_graph,
    _capture_representative_layer,
    _load_args,
    _reciprocal_graph_rounds,
    _restore_all_cache,
)


REGIONS = ("self_norm_qkv", "cross_norm_q")
DECODER_LAYERS = 12
FIXED_MAIN_SAVING_GATE_SECONDS = 0.3
CONFIGS = {
    "w4_c32": (4, 32),
    "w4_c48": (4, 48),
    "w4_c68": (4, 68),
    "w8_c32": (8, 32),
    "w8_c48": (8, 48),
    "w8_c68": (8, 68),
}


def _eps(module: torch.nn.Module) -> float:
    value = getattr(module, "eps", None)
    return float(torch.finfo(torch.float32).eps if value is None else value)


def _pack_layers(
    layers: list[torch.nn.Module],
    state: Any,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    from osuT5.osuT5.inference.optimized.scout.cutlass_w8a8_projection import (
        validate_w8a8_pack,
    )
    from osuT5.osuT5.inference.optimized.scout.int8_mlp import Int8PackedLinear

    torch.cuda.synchronize()
    allocated_before = int(torch.cuda.memory_allocated())
    started = time.perf_counter()
    packs: dict[int, dict[str, Any]] = {}
    current_bytes = candidate_bytes = source_bytes = 0
    scale_min = math.inf
    scale_max = 0.0
    for layer in layers:
        retained_pack = state.pack_for_layer(layer)
        candidates = {
            "self_norm_qkv": Int8PackedLinear.from_module(layer.self_attn.Wqkv),
            "cross_norm_q": Int8PackedLinear.from_module(layer.cross_attn.Wq),
        }
        validation = {
            name: validate_w8a8_pack(pack) for name, pack in candidates.items()
        }
        packs[id(layer)] = {
            "self_fp16": retained_pack.self_qkv,
            "self_out_fp16": retained_pack.self_out,
            "candidates": candidates,
            "validation": validation,
        }
        current_bytes += retained_pack.self_qkv.packed_weight_bytes
        current_bytes += candidates["cross_norm_q"].source_weight_bytes
        candidate_bytes += sum(pack.packed_weight_bytes for pack in candidates.values())
        source_bytes += sum(pack.source_weight_bytes for pack in candidates.values())
        scale_min = min(scale_min, *(row["scale_min"] for row in validation.values()))
        scale_max = max(scale_max, *(row["scale_max"] for row in validation.values()))
    torch.cuda.synchronize()
    return packs, {
        "setup_seconds": time.perf_counter() - started,
        "allocated_before_bytes": allocated_before,
        "allocated_after_bytes": int(torch.cuda.memory_allocated()),
        "retained_fp32_source_weight_bytes": source_bytes,
        "current_mixed_active_weight_bytes": current_bytes,
        "candidate_int8_packed_bytes": candidate_bytes,
        "candidate_active_storage_reduction_bytes": current_bytes - candidate_bytes,
        "weight_scale_min": scale_min,
        "weight_scale_max": scale_max,
        "weight_contains_minus_128": False,
    }


def _baseline_functions(
    capture,
    inputs: dict[str, torch.Tensor],
    packs: dict[str, Any],
) -> dict[str, Callable[[], torch.Tensor]]:
    from osuT5.osuT5.inference.optimized.kernels.decoder_layer import (
        native_one_token_rmsnorm_linear,
    )
    from osuT5.osuT5.inference.optimized.kernels.weight_only import (
        weight_only_rmsnorm_linear,
    )

    module = capture.module
    return {
        "self_norm_qkv": lambda: weight_only_rmsnorm_linear(
            inputs["layer_input"],
            module.self_attn_layer_norm.weight,
            packs["self_fp16"],
            eps=_eps(module.self_attn_layer_norm),
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
    }


def _candidate_functions(
    capture,
    inputs: dict[str, torch.Tensor],
    packs: dict[str, Any],
    *,
    active_warps: int,
    persistent_ctas: int,
) -> dict[str, Callable[[], torch.Tensor]]:
    from osuT5.osuT5.inference.optimized.scout.cutlass_w8a8_projection import (
        w8a8_rmsnorm_linear,
    )

    module = capture.module
    return {
        "self_norm_qkv": lambda: w8a8_rmsnorm_linear(
            inputs["layer_input"],
            module.self_attn_layer_norm.weight,
            packs["candidates"]["self_norm_qkv"],
            eps=_eps(module.self_attn_layer_norm),
            active_warps=active_warps,
            persistent_ctas=persistent_ctas,
        ),
        "cross_norm_q": lambda: w8a8_rmsnorm_linear(
            inputs["after_self"],
            module.cross_attn_layer_norm.weight,
            packs["candidates"]["cross_norm_q"],
            eps=_eps(module.cross_attn_layer_norm),
            active_warps=active_warps,
            persistent_ctas=persistent_ctas,
        ),
    }


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
            projection_starts = [
                torch.cuda.Event(enable_timing=True) for _ in range(iters)
            ]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            for index in range(iters):
                raw_starts[index].record()
                eviction_graph.replay()
                projection_starts[index].record()
                graph.replay()
                ends[index].record()
            torch.cuda.synchronize()
            allocated_after = int(torch.cuda.memory_allocated())
            isolated = sum(
                start.elapsed_time(end)
                for start, end in zip(projection_starts, ends, strict=True)
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
                    "raw_evict_plus_projection_ms_per_call": float(raw),
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
    ref = reference.detach().float()
    value = candidate.detach().float()
    if ref.shape != value.shape:
        return {
            "max_abs": math.inf,
            "mean_abs": math.inf,
            "rmse": math.inf,
            "relative_l2": math.inf,
            "cosine": -1.0,
        }
    difference = value - ref
    ref_norm = torch.linalg.vector_norm(ref)
    relative = torch.linalg.vector_norm(difference) / torch.clamp(ref_norm, min=1e-12)
    cosine = F.cosine_similarity(ref.flatten(), value.flatten(), dim=0)
    return {
        "max_abs": float(difference.abs().max().item()),
        "mean_abs": float(difference.abs().mean().item()),
        "rmse": float(torch.sqrt(difference.square().mean()).item()),
        "relative_l2": float(relative.item()),
        "cosine": float(cosine.item()),
    }


def _quantized_reference(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    pack: Any,
    *,
    eps: float,
    quantized: torch.Tensor,
    activation_scale: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    values = input.float()
    normalized = values * norm_weight.float() * torch.rsqrt(
        values.square().mean(dim=-1, keepdim=True) + eps
    )
    reconstructed = quantized.float().view_as(normalized) * activation_scale.float()
    # Use CPU float64 only as an exact integer-dot oracle.  Every multiplicand is
    # an integer with magnitude <=127 and the 768-term sum is far below 2**53,
    # so this avoids TF32 or CUDA GEMM accumulation changing the reference.
    integer_dot = torch.mv(
        pack.weight.detach().cpu().to(torch.float64),
        quantized.detach().cpu().to(torch.float64),
    ).to(torch.float32)
    reference = integer_dot.view(1, 1, -1) * (
        activation_scale.detach().cpu() * pack.scale.detach().cpu()
    ).view(1, 1, -1)
    if pack.bias is not None:
        reference = reference + pack.bias.detach().cpu().view(1, 1, -1)
    activation_difference = reconstructed - normalized
    saturation_count = int((quantized.abs() == 127).sum().item())
    return reference, {
        "activation_absmax": float(normalized.abs().max().item()),
        "activation_scale": float(activation_scale.item()),
        "activation_saturation_count": saturation_count,
        "activation_saturation_fraction": saturation_count / quantized.numel(),
        "activation_zero_scale_fallback": bool(
            normalized.abs().max().item() == 0 and activation_scale.item() == 1
        ),
        "activation_reconstruction_max_abs": float(
            activation_difference.abs().max().item()
        ),
        "activation_reconstruction_mean_abs": float(
            activation_difference.abs().mean().item()
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
    measured_replays = sum(int(entry["decode_replays"]) for entry in buckets.values())
    if measured_replays <= 0 or total_replays < measured_replays:
        raise ValueError("invalid w8a8 replay counts")
    scale = total_replays / measured_replays
    mode_reports: dict[str, Any] = {}
    invariants_pass = True
    for mode in ("l2_hot", "cache_cold"):
        regions: dict[str, Any] = {}
        total_saving = 0.0
        for region in REGIONS:
            baseline = sum(
                int(entry["decode_replays"])
                * DECODER_LAYERS
                * float(entry["regions"][region][mode]["baseline_ms_per_call"])
                for entry in buckets.values()
            ) / 1000.0
            candidate_by_config = {
                config: sum(
                    int(entry["decode_replays"])
                    * DECODER_LAYERS
                    * float(
                        entry["regions"][region][mode]["candidates"][config][
                            "ms_per_call"
                        ]
                    )
                    for entry in buckets.values()
                ) / 1000.0
                for config in CONFIGS
            }
            selected = min(candidate_by_config, key=candidate_by_config.get)
            fixed_baseline = baseline * scale
            fixed_candidate = candidate_by_config[selected] * scale
            saving = fixed_baseline - fixed_candidate
            total_saving += saving
            regions[region] = {
                "selected_config": selected,
                "fixed_baseline_seconds": fixed_baseline,
                "fixed_candidate_seconds": fixed_candidate,
                "fixed_saving_seconds": saving,
                "candidate_seconds_by_config": candidate_by_config,
            }
        mode_reports[mode] = {
            "regions": regions,
            "fixed_main_saving_seconds": total_saving,
            "passes_saving_gate": total_saving >= FIXED_MAIN_SAVING_GATE_SECONDS,
        }
    for entry in buckets.values():
        for region in REGIONS:
            row = entry["regions"][region]
            invariants_pass &= bool(row["baseline_finite"])
            invariants_pass &= bool(row["baseline_repeat_bitwise_equal"])
            invariants_pass &= bool(row["baseline_memory_stable"])
            for candidate in row["candidates"].values():
                invariants_pass &= bool(candidate["finite"])
                invariants_pass &= bool(candidate["repeat_bitwise_equal"])
                invariants_pass &= bool(candidate["memory_stable"])
                invariants_pass &= bool(candidate["kernel_vs_quantized_reference_pass"])
            invariants_pass &= bool(row["cross_configuration_bitwise_equal"])
    conservative_saving = min(
        mode_reports[mode]["fixed_main_saving_seconds"] for mode in mode_reports
    )
    component_pass = invariants_pass and all(
        mode_reports[mode]["passes_saving_gate"] for mode in mode_reports
    )
    return {
        "measured_replays": measured_replays,
        "total_replays": total_replays,
        "coverage_fraction": measured_replays / total_replays,
        "modes": mode_reports,
        "conservative_fixed_main_saving_seconds": conservative_saving,
        "saving_gate_seconds": FIXED_MAIN_SAVING_GATE_SECONDS,
        "invariants_pass": invariants_pass,
        "component_pass": component_pass,
        "retained_full_graph_required": component_pass,
        "promotion_pass": False,
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
    cold_warmup: int,
    cold_iters: int,
    eviction_bytes: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("w8a8 projection profiler requires CUDA")
    if min(warmup, iters, cold_warmup, cold_iters) < 1:
        raise ValueError("all warmup/iteration counts must be positive")
    if eviction_bytes < 32 * 1024 * 1024 or eviction_bytes % 4:
        raise ValueError("eviction_bytes must be a multiple of four and at least 32 MiB")
    _assert_scout_args(args)
    output_path.mkdir(parents=True, exist_ok=True)
    diagnostics = retained._DiagnosticRecorder(diagnostic_path)
    diagnostics.record("cutlass_w8a8_component_scout_start")
    run = retained._retained_main_session_run(
        args,
        output_path=output_path,
        diagnostics=diagnostics,
    )
    retained._cuda_sync_checkpoint(diagnostics, "retained_full_song.complete")
    model = run["model"]
    state = run["state"]
    model.eval()
    entries = retained._validate_live_graph_cache(run["session"].graph_cache)
    selected = retained._select_live_buckets(entries, bucket_mode)
    total_replays = sum(
        int(entry["decode_replays"])
        for prefix_entries in entries.values()
        for entry in prefix_entries
    )
    layers, _, _ = retained._locate_decoder(model)

    from osuT5.osuT5.inference.optimized.kernels.decoder_layer import (
        preload_native_decoder_layer,
    )
    from osuT5.osuT5.inference.optimized.kernels.weight_only import (
        preload_weight_only_extension,
    )
    from osuT5.osuT5.inference.optimized.scout.cutlass_w8a8_projection import (
        select_w8a8_backend,
        w8a8_rmsnorm_linear_with_state,
        preload_w8a8_projection_extension,
    )
    backend = select_w8a8_backend()

    preload_times: dict[str, float] = {}
    for name, preload in (
        ("fp16_self_baseline", preload_weight_only_extension),
        ("fp32_cross_baseline", preload_native_decoder_layer),
        ("sm75_w8a8_candidate", preload_w8a8_projection_extension),
    ):
        retained._cuda_sync_checkpoint(diagnostics, f"extension.{name}.before")
        started = time.perf_counter()
        preload()
        retained._cuda_sync_checkpoint(diagnostics, f"extension.{name}.after")
        preload_times[name] = time.perf_counter() - started

    layer_packs, pack_report = _pack_layers(layers, state)
    retained._cuda_sync_checkpoint(diagnostics, "w8a8_projection.pack.complete")
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
    for prefix, accepted in selected.items():
        retained._restore_pre_replay_entry(accepted, diagnostics)
        static_inputs = accepted["static_inputs"]
        retained._validate_restored_capture_inputs(
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
        with retained._retained_real_tensor_context(
            model,
            state,
            prefix=prefix,
        ) as capture_evidence:
            capture = _capture_representative_layer(
                model,
                static_inputs,
                prefix=prefix,
            )
        _restore_all_cache(cache, snapshots)
        retained._cuda_sync_checkpoint(
            diagnostics, f"prefix_{prefix}.representative_layer_capture"
        )
        packs = layer_packs.get(id(capture.module))
        if packs is None:
            raise RuntimeError("captured layer has no owned w8a8 pack")
        inputs = retained._prepare_region_inputs(
            capture,
            {
                "self_norm_qkv": packs["self_fp16"],
                "self_out_residual": packs["self_out_fp16"],
            },
        )
        _restore_all_cache(cache, snapshots)
        device = inputs["layer_input"].device
        for name in ("layer_input", "after_self"):
            retained._validate_tensor_layout(
                inputs[name],
                name=f"prefix {prefix} {name}",
                device=device,
                dtype=torch.float32,
                shape=(1, 1, 768),
                contiguous=True,
            )
        baselines = _baseline_functions(capture, inputs, packs)
        candidates = {
            name: _candidate_functions(
                capture,
                inputs,
                packs,
                active_warps=warps,
                persistent_ctas=ctas,
            )
            for name, (warps, ctas) in CONFIGS.items()
        }
        region_results: dict[str, Any] = {}
        for region in REGIONS:
            callables = {"baseline": baselines[region]}
            callables.update(
                {config: candidate[region] for config, candidate in candidates.items()}
            )
            graphs = {
                name: _capture_cuda_graph(fn, context=nullcontext, warmup=0)
                for name, fn in callables.items()
            }
            hot_timings, hot_rounds, hot_memory = _reciprocal_graph_rounds(
                {name: graph.graph for name, graph in graphs.items()},
                restore=lambda: None,
                warmup=warmup,
                iters=iters,
            )
            cold_timings, cold_rounds, cold_memory, eviction_ms = _cache_cold_rounds(
                {name: graph.graph for name, graph in graphs.items()},
                eviction_capture.graph,
                warmup=cold_warmup,
                iters=cold_iters,
            )
            observations = {name: _observe(graph) for name, graph in graphs.items()}
            baseline_output = observations["baseline"][0]
            module = capture.module
            if region == "self_norm_qkv":
                candidate_input = inputs["layer_input"]
                norm_weight = module.self_attn_layer_norm.weight
                pack = packs["candidates"][region]
                eps = _eps(module.self_attn_layer_norm)
            else:
                candidate_input = inputs["after_self"]
                norm_weight = module.cross_attn_layer_norm.weight
                pack = packs["candidates"][region]
                eps = _eps(module.cross_attn_layer_norm)
            candidate_rows: dict[str, Any] = {}
            first_output = None
            for config, (warps, ctas) in CONFIGS.items():
                output, quantized, activation_scale = w8a8_rmsnorm_linear_with_state(
                    candidate_input,
                    norm_weight,
                    pack,
                    eps=eps,
                    active_warps=warps,
                    persistent_ctas=ctas,
                )
                torch.cuda.synchronize()
                reference, activation = _quantized_reference(
                    candidate_input,
                    norm_weight,
                    pack,
                    eps=eps,
                    quantized=quantized,
                    activation_scale=activation_scale,
                )
                torch.cuda.synchronize()
                observed = observations[config][0]
                implementation_drift = _drift(reference.cpu(), observed)
                baseline_drift = _drift(baseline_output, observed)
                if first_output is None:
                    first_output = observed
                candidate_rows[config] = {
                    "active_warps": warps,
                    "persistent_ctas": ctas,
                    "finite": bool(torch.isfinite(observed).all()),
                    "repeat_bitwise_equal": torch.equal(*observations[config]),
                    "memory_stable": bool(hot_memory[config] and cold_memory[config]),
                    "kernel_vs_quantized_reference": implementation_drift,
                    "kernel_vs_quantized_reference_pass": (
                        implementation_drift["max_abs"] <= 1e-4
                    ),
                    "candidate_vs_current_baseline": baseline_drift,
                    "activation": activation,
                    "l2_hot_ms_per_call": float(hot_timings[config]),
                    "cache_cold_ms_per_call": float(cold_timings[config]),
                }
            cross_configuration_equal = all(
                torch.equal(first_output, observations[config][0]) for config in CONFIGS
            )
            region_results[region] = {
                "current_baseline": (
                    "fp16_weight" if region == "self_norm_qkv" else "fp32_weight"
                ),
                "baseline_finite": bool(torch.isfinite(baseline_output).all()),
                "baseline_repeat_bitwise_equal": torch.equal(*observations["baseline"]),
                "baseline_memory_stable": bool(
                    hot_memory["baseline"] and cold_memory["baseline"]
                ),
                "cross_configuration_bitwise_equal": cross_configuration_equal,
                "candidates": candidate_rows,
                "l2_hot": {
                    "baseline_ms_per_call": float(hot_timings["baseline"]),
                    "candidates": {
                        config: {"ms_per_call": float(hot_timings[config])}
                        for config in CONFIGS
                    },
                    "rounds": hot_rounds,
                },
                "cache_cold": {
                    "baseline_ms_per_call": float(cold_timings["baseline"]),
                    "candidates": {
                        config: {"ms_per_call": float(cold_timings[config])}
                        for config in CONFIGS
                    },
                    "eviction_ms_per_call": float(eviction_ms),
                    "rounds": cold_rounds,
                },
            }
            del graphs
            torch.cuda.empty_cache()
        buckets[str(prefix)] = {
            "decode_replays": int(accepted["decode_replays"]),
            "retained_capture_dispatches": dict(capture_evidence["dispatch_counts"]),
            "regions": region_results,
        }
    summary = summarize_component(buckets, total_replays=total_replays)
    return {
        "schema_version": 1,
        "metadata": {
            "candidate": "cutlass_seeking_sm75_w8a8_projection_self_qkv_cross_q",
            "result_class": "component_scout_documented_drift",
            "cutlass_probe": backend,
            "cutlass_available": bool(backend["cutlass_available"]),
            "scout_backend": backend["backend"],
            "cutlass_gemm_wired": bool(backend["cutlass_gemm_wired"]),
            "revisit_condition": backend["revisit_condition"],
            "nsight_target_note": ("Largest Nsight hotspot ~2.44s projection GEMMs; ranking #3 after compiled-cross"),
            "production_wiring": False,
            "comparison": retained.RETAINED_COMPOSITION_VERSION,
            "retained_composition": run["retained_composition"],
            "regions": list(REGIONS),
            "excluded_regions": [
                "self_out_residual",
                "cross_out_residual",
                "final_norm_logits",
                "mlp_fc1",
                "mlp_fc2",
            ],
            "precision_contract": {
                "weight_storage": "int8_per_output_row_symmetric",
                "activation_storage": "dynamic_int8_per_token_symmetric",
                "accumulation": "signed_dp4a_int32_fallback_or_cutlass_when_wired",
                "output_dequant_bias": "fp32",
                "exactness_claim": False,
            },
            "configs": {
                name: {"active_warps": warps, "persistent_ctas": ctas}
                for name, (warps, ctas) in CONFIGS.items()
            },
            "bucket_mode": bucket_mode,
            "measured_buckets": list(selected),
            "live_bucket_counts": {
                str(prefix): sum(
                    int(entry["decode_replays"]) for entry in prefix_entries
                )
                for prefix, prefix_entries in entries.items()
            },
            "selected_graph_sources": {
                str(prefix): entry["graph_source"] for prefix, entry in selected.items()
            },
            "hot_warmup": warmup,
            "hot_iters": iters,
            "cold_warmup": cold_warmup,
            "cold_iters": cold_iters,
            "eviction_bytes": eviction_bytes,
            "extension_preload_seconds": preload_times,
            "retained_full_graph_policy": (
                "required only when both component timing modes clear 0.3s"
            ),
        },
        "pack_report": pack_report,
        "diagnostic_path": str(diagnostic_path),
        "buckets": buckets,
        "summary": summary,
    }


def _text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    meta = report["metadata"]
    lines = [
        "CUTLASS-seeking SM75 w8a8 projection component scout",
        f"cutlass_available={meta.get('cutlass_available')}",
        f"scout_backend={meta.get('scout_backend')}",
        f"cutlass_gemm_wired={meta.get('cutlass_gemm_wired')}",
        f"revisit_condition={meta.get('revisit_condition')}",
        f"measured_replays={summary['measured_replays']}",
        f"total_replays={summary['total_replays']}",
    ]
    for mode, row in summary["modes"].items():
        lines.append(
            f"{mode}: fixed_main_saving_seconds="
            f"{row['fixed_main_saving_seconds']:.6f} pass={row['passes_saving_gate']}"
        )
        for region, region_row in row["regions"].items():
            lines.append(
                f"  {region}: config={region_row['selected_config']} "
                f"saving_seconds={region_row['fixed_saving_seconds']:.6f}"
            )
    lines.extend(
        [
            "conservative_fixed_main_saving_seconds="
            f"{summary['conservative_fixed_main_saving_seconds']:.6f}",
            f"saving_gate_seconds={summary['saving_gate_seconds']:.6f}",
            f"invariants_pass={summary['invariants_pass']}",
            f"component_pass={summary['component_pass']}",
            f"retained_full_graph_required={summary['retained_full_graph_required']}",
            "promotion_pass=False",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "prefix",
        "decode_replays",
        "region",
        "mode",
        "variant",
        "active_warps",
        "persistent_ctas",
        "ms_per_call",
        "baseline_max_abs_drift",
        "quantized_reference_max_abs_drift",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for prefix, bucket in report["buckets"].items():
            for region, row in bucket["regions"].items():
                for mode in ("l2_hot", "cache_cold"):
                    writer.writerow(
                        {
                            "prefix": prefix,
                            "decode_replays": bucket["decode_replays"],
                            "region": region,
                            "mode": mode,
                            "variant": "baseline",
                            "active_warps": "",
                            "persistent_ctas": "",
                            "ms_per_call": row[mode]["baseline_ms_per_call"],
                            "baseline_max_abs_drift": 0.0,
                            "quantized_reference_max_abs_drift": 0.0,
                        }
                    )
                    for config, timing in row[mode]["candidates"].items():
                        candidate = row["candidates"][config]
                        writer.writerow(
                            {
                                "prefix": prefix,
                                "decode_replays": bucket["decode_replays"],
                                "region": region,
                                "mode": mode,
                                "variant": config,
                                "active_warps": candidate["active_warps"],
                                "persistent_ctas": candidate["persistent_ctas"],
                                "ms_per_call": timing["ms_per_call"],
                                "baseline_max_abs_drift": candidate[
                                    "candidate_vs_current_baseline"
                                ]["max_abs"],
                                "quantized_reference_max_abs_drift": candidate[
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
    parser.add_argument("--diagnostic-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--bucket-mode", choices=("sentinel", "all"), default="sentinel")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--cold-warmup", type=int, default=10)
    parser.add_argument("--cold-iters", type=int, default=100)
    parser.add_argument("--eviction-bytes", type=int, default=64 * 1024 * 1024)
    parsed, overrides = parser.parse_known_args()
    args = _load_args(parsed.config_name, overrides)
    report = profile_component(
        args,
        output_path=parsed.output_path,
        diagnostic_path=parsed.diagnostic_path,
        bucket_mode=parsed.bucket_mode,
        warmup=parsed.warmup,
        iters=parsed.iters,
        cold_warmup=parsed.cold_warmup,
        cold_iters=parsed.cold_iters,
        eviction_bytes=parsed.eviction_bytes,
    )
    parsed.report_path.parent.mkdir(parents=True, exist_ok=True)
    parsed.report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    parsed.text_path.write_text(_text(report))
    _write_csv(report, parsed.csv_path)
    print(_text(report), end="")
    if not report["summary"]["component_pass"]:
        raise SystemExit("STOP_CUTLASS_W8A8_PROJECTION_COMPONENT")
    raise SystemExit("REQUIRE_RETAINED_FULL_GRAPH_AB")


if __name__ == "__main__":
    main()
