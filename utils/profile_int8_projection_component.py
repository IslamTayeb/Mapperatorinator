"""Profile incremental INT8 storage for the remaining one-token projections."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from utils.profile_native_prefix_dtype_scout import (  # noqa: E402
    LayerCapture,
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
    select_buckets,
    validate_accepted_graph_cache,
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
OUTPUTS_PER_BLOCK = (2, 4, 8)
FIXED_MAIN_SAVING_GATE_SECONDS = 0.5
AUDITED_CURRENT_REGION_SECONDS = {
    "self_norm_qkv": 0.6724584794483185,
    "self_out_residual": 0.4069147370853424,
    "cross_norm_q": 0.4132181438140869,
    "cross_out_residual": 0.40312816597938533,
    "final_norm_logits": 0.08943501430416106,
}
MEASURED_INT8_VS_FP16_MLP_SPEEDUP = 1.3666607805044033
AUDITED_REALISTIC_CEILING_SECONDS = sum(AUDITED_CURRENT_REGION_SECONDS.values()) * (
    1.0 - 1.0 / MEASURED_INT8_VS_FP16_MLP_SPEEDUP
)


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


def _capture_final_input(
    model: torch.nn.Module,
    final_norm: torch.nn.Module,
    static_inputs: dict[str, Any],
) -> torch.Tensor:
    captured: list[torch.Tensor] = []

    def hook(module, args):
        del module
        if captured:
            raise RuntimeError("decoder final norm ran more than once")
        if not args or not isinstance(args[0], torch.Tensor):
            raise TypeError("decoder final norm did not receive a tensor")
        captured.append(args[0].detach().clone())

    handle = final_norm.register_forward_pre_hook(hook)
    try:
        model(**static_inputs, return_dict=True)
        torch.cuda.synchronize()
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("did not capture one decoder final-norm input")
    return captured[0]


def _prepare_region_inputs(capture: LayerCapture) -> dict[str, torch.Tensor]:
    from osuT5.osuT5.inference.optimized.kernels.decoder_layer import (
        native_one_token_linear_residual,
        native_one_token_rmsnorm_linear,
    )
    from osuT5.osuT5.inference.optimized.kernels.q1_attention import (
        native_q1_rope_cache_attention,
    )
    from osuT5.osuT5.inference.optimized.scout.native_prefix import (
        _cache_tensors,
        _trim_mask,
        fp32_rms_norm,
        framework_q1_attention,
    )

    module = capture.module
    self_attn = module.self_attn
    hidden = capture.hidden_states
    qkv = F.linear(
        fp32_rms_norm(
            hidden,
            module.self_attn_layer_norm.weight,
            getattr(module.self_attn_layer_norm, "eps", None),
        ),
        self_attn.Wqkv.weight,
        self_attn.Wqkv.bias,
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
    after_self = hidden + F.linear(
        self_output,
        self_attn.Wo.weight,
        self_attn.Wo.bias,
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
    cross_output = framework_q1_attention(
        query, cross_keys, cross_values, None
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
        measured_saving = baseline_seconds - candidate_seconds[selected_key]
        fixed_saving = measured_saving * total_replays / measured_replays
        retained = fixed_saving > 0
        if retained:
            selected_fixed_saving += fixed_saving
        region_summary[region] = {
            "current_baseline": CURRENT_BASELINE[region],
            "selected_outputs_per_block": int(selected_key),
            "measured_baseline_seconds": baseline_seconds,
            "measured_candidate_seconds": candidate_seconds[selected_key],
            "measured_saving_seconds": measured_saving,
            "fixed_work_saving_seconds": fixed_saving,
            "retained_by_selective_policy": retained,
            "candidate_seconds_by_outputs_per_block": candidate_seconds,
        }
    return {
        "measured_replays": measured_replays,
        "total_replays": int(total_replays),
        "coverage_fraction": measured_replays / total_replays,
        "regions": region_summary,
        "selective_fixed_work_main_saving_seconds": selected_fixed_saving,
        "saving_gate_seconds": FIXED_MAIN_SAVING_GATE_SECONDS,
        "invariant_failures": failures,
        "invariants_pass": not failures,
        "sizing_pass": selected_fixed_saving >= FIXED_MAIN_SAVING_GATE_SECONDS,
        "promotion_pass": (
            not failures and selected_fixed_saving >= FIXED_MAIN_SAVING_GATE_SECONDS
        ),
    }


def _pack_projection_family(
    layers: list[torch.nn.Module],
    final_projection: torch.nn.Module,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any], dict[str, int]]:
    from osuT5.osuT5.inference.optimized.kernels.weight_only import PackedLinear
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
        fp16 = {name: PackedLinear.from_module(module) for name, module in modules.items()}
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
    final_fp16 = PackedLinear.from_module(final_projection)
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
    run = _accepted_main_session_run(args, output_path=output_path)
    model = run["model"]
    model.eval()
    entries = validate_accepted_graph_cache(run["session"].graph_cache)
    selected = select_buckets(entries, bucket_mode)
    total_replays = sum(int(entry["decode_replays"]) for entry in entries.values())
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
        torch.cuda.synchronize()
        started = time.perf_counter()
        preload()
        torch.cuda.synchronize()
        preload_times[name] = time.perf_counter() - started
    layer_packs, final_packs, pack_report = _pack_projection_family(
        layers, final_projection
    )

    buckets: dict[str, dict[str, Any]] = {}
    for prefix, accepted in selected.items():
        static_inputs = accepted["static_inputs"]
        cache = _cache_from_static_inputs(static_inputs)
        position = static_inputs.get("cache_position")
        if not isinstance(position, torch.Tensor) or position.numel() != 1:
            raise RuntimeError(f"prefix {prefix} has invalid cache position")
        snapshots = _all_cache_snapshots(cache, position)
        _restore_all_cache(cache, snapshots)
        capture = _capture_representative_layer(model, static_inputs, prefix=prefix)
        _restore_all_cache(cache, snapshots)
        inputs = _prepare_region_inputs(capture)
        _restore_all_cache(cache, snapshots)
        final_input = _capture_final_input(model, final_norm, static_inputs)
        _restore_all_cache(cache, snapshots)

        packs = layer_packs.get(id(capture.module))
        if packs is None:
            raise RuntimeError("captured decoder layer has no owned weight pack")
        fp16_packs = dict(packs["fp16"])
        fp16_packs["final_norm_logits"] = final_packs["fp16"]
        int8_packs = dict(packs["int8"])
        int8_packs["final_norm_logits"] = final_packs["int8"]
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
            graphs = {
                name: _capture_cuda_graph(fn, context=nullcontext, warmup=0)
                for name, fn in callables.items()
            }
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
            "regions": region_results,
        }
    summary = summarize_component(buckets, total_replays=total_replays)
    return {
        "schema_version": 1,
        "metadata": {
            "candidate": "per_row_symmetric_int8_remaining_projection_family",
            "result_class": "component_scout_documented_drift",
            "production_wiring": False,
            "comparison": "mixed_weight_plus_int8_mlp_current_projection_policy",
            "excluded_already_int8_regions": ["mlp_fc1", "mlp_fc2"],
            "current_baseline_by_region": CURRENT_BASELINE,
            "preimplementation_audit": {
                "current_region_seconds": AUDITED_CURRENT_REGION_SECONDS,
                "measured_int8_vs_fp16_mlp_speedup": (
                    MEASURED_INT8_VS_FP16_MLP_SPEEDUP
                ),
                "realistic_ceiling_seconds": AUDITED_REALISTIC_CEILING_SECONDS,
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
                str(prefix): int(entry["decode_replays"])
                for prefix, entry in entries.items()
            },
            "warmup": warmup,
            "iters": iters,
            "extension_preload_seconds": preload_times,
        },
        "pack_report": pack_report,
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
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--bucket-mode", choices=("sentinel", "all"), default="sentinel")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parsed, overrides = parser.parse_known_args()
    args = _load_args(parsed.config_name, overrides)
    report = profile_component(
        args,
        output_path=parsed.output_path,
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
