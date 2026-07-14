from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from osuT5.osuT5.runtime_profiling import generation_profile_context  # noqa: E402
from utils.profile_native_prefix_dtype_scout import (  # noqa: E402
    _accepted_main_session_run,
    _all_cache_snapshots,
    _assert_scout_args,
    _cache_from_static_inputs,
    _capture_cuda_graph,
    _load_args,
    _max_abs,
    _reciprocal_graph_rounds,
    _restore_all_cache,
    select_buckets,
    validate_accepted_graph_cache,
)


DECODER_LAYERS = 12
MIN_LOCAL_SPEEDUP = 1.4
FIXED_MAIN_STEPS = 8_294
FIXED_TIMING_STEPS = 821
TARGET_LAYER_INDEX = 6


@dataclass(frozen=True)
class MlpCapture:
    module: torch.nn.Module
    hidden_states: torch.Tensor


def _accepted_context(prefix: int):
    return generation_profile_context(
        active_prefix_self_attention_length=prefix,
        q1_bmm_cross_attention=True,
        native_q1_self_attention=True,
        native_q1_rope_cache_self_attention=True,
        native_cross_mlp_tail=True,
    )


def _capture_real_mlp_input(
    model: torch.nn.Module,
    static_inputs: dict[str, Any],
    *,
    prefix: int,
) -> MlpCapture:
    """Capture layer-six post-cross-attention state from a real decoder replay."""

    from osuT5.osuT5.inference.optimized.kernels import cross_mlp
    from osuT5.osuT5.model.custom_transformers.modeling_varwhisper import (
        VarWhisperDecoderLayer,
    )

    layers = [
        module for module in model.modules() if isinstance(module, VarWhisperDecoderLayer)
    ]
    if len(layers) != DECODER_LAYERS:
        raise RuntimeError(
            f"INT8 MLP scout requires {DECODER_LAYERS} decoder layers, found {len(layers)}"
        )
    target = layers[TARGET_LAYER_INDEX]
    original = cross_mlp.native_one_token_mlp_residual
    found: dict[str, torch.Tensor] = {}

    def capture(*args, **kwargs):
        hidden_states = args[0] if args else kwargs.get("input")
        fc1_weight = args[2] if len(args) > 2 else kwargs.get("fc1_weight")
        if fc1_weight is target.fc1.weight:
            if found:
                raise RuntimeError("captured target MLP more than once in one decoder replay")
            if not isinstance(hidden_states, torch.Tensor):
                raise RuntimeError("target MLP input was not a tensor")
            found["hidden_states"] = hidden_states.detach().clone()
        return original(*args, **kwargs)

    cross_mlp.native_one_token_mlp_residual = capture
    try:
        with _accepted_context(prefix):
            model(**static_inputs, return_dict=True)
        torch.cuda.synchronize()
    finally:
        cross_mlp.native_one_token_mlp_residual = original
    if "hidden_states" not in found:
        raise RuntimeError(f"did not capture target MLP input for prefix {prefix}")
    hidden = found["hidden_states"]
    if hidden.shape[:2] != (1, 1) or hidden.dtype != torch.float32:
        raise RuntimeError(
            f"captured MLP input must be [1, 1, H] FP32, got {hidden.shape} {hidden.dtype}"
        )
    return MlpCapture(module=target, hidden_states=hidden)


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
        raise ValueError("INT8 MLP component summary requires at least one bucket")
    if total_replays <= 0:
        raise ValueError("total replay count must be positive")
    totals_ms = {name: 0.0 for name in ("fp32", "fp16_weight", "int8_weight")}
    measured_replays = 0
    failures: dict[str, list[str]] = {}
    for prefix, entry in sorted(buckets.items(), key=lambda item: int(item[0])):
        count = int(entry["decode_replays"])
        if count <= 0:
            raise ValueError(f"bucket {prefix} decode replay count must be positive")
        measured_replays += count
        for name in totals_ms:
            value = float(entry["ms_per_call"][name])
            if value <= 0:
                raise ValueError(f"bucket {prefix} {name} timing must be positive")
            totals_ms[name] += DECODER_LAYERS * count * value
        bucket_failures = [
            name for name, passed in entry["checks"].items() if not bool(passed)
        ]
        if bucket_failures:
            failures[prefix] = bucket_failures
    if total_replays < measured_replays:
        raise ValueError("total replay count cannot be smaller than measured count")
    weighted_seconds = {name: value / 1000.0 for name, value in totals_ms.items()}
    speedup = weighted_seconds["fp16_weight"] / weighted_seconds["int8_weight"]
    main_saving = weighted_seconds["fp16_weight"] - weighted_seconds["int8_weight"]
    request_scale = (FIXED_MAIN_STEPS + FIXED_TIMING_STEPS) / FIXED_MAIN_STEPS
    return {
        "measured_replays": measured_replays,
        "total_replays": int(total_replays),
        "coverage_fraction": measured_replays / total_replays,
        "weighted_mlp_seconds": weighted_seconds,
        "int8_vs_fp16_local_speedup": speedup,
        "minimum_local_speedup": MIN_LOCAL_SPEEDUP,
        "fixed_work_main_saving_seconds": main_saving,
        "warm_request_saving_seconds_estimate": main_saving * request_scale,
        "request_estimate_timing_to_main_step_ratio": FIXED_TIMING_STEPS / FIXED_MAIN_STEPS,
        "invariant_failures": failures,
        "invariants_pass": not failures,
        "sizing_pass": speedup >= MIN_LOCAL_SPEEDUP,
        "promotion_pass": not failures and speedup >= MIN_LOCAL_SPEEDUP,
    }


@torch.no_grad()
def profile_int8_mlp_component(
    args,
    *,
    output_path: Path,
    bucket_mode: str,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("INT8 MLP component profiler requires CUDA")
    if warmup < 1 or iters < 1:
        raise ValueError("warmup and iters must be positive")
    _assert_scout_args(args)
    output_path.mkdir(parents=True, exist_ok=True)
    run = _accepted_main_session_run(args, output_path=output_path)
    model = run["model"]
    model.eval()
    accepted_entries = validate_accepted_graph_cache(run["session"].graph_cache)
    selected = select_buckets(accepted_entries, bucket_mode)
    total_replays = sum(
        int(entry["decode_replays"]) for entry in accepted_entries.values()
    )

    from osuT5.osuT5.inference.optimized.kernels.decoder_layer import (
        native_one_token_mlp_residual,
        preload_native_decoder_layer,
    )
    from osuT5.osuT5.inference.optimized.kernels.weight_only import (
        PackedLinear,
        preload_weight_only_extension,
        weight_only_mlp_residual,
    )
    from osuT5.osuT5.inference.optimized.scout.int8_mlp import (
        Int8MlpPack,
        int8_weight_mlp_residual,
        preload_int8_mlp_extension,
    )

    extension_times: dict[str, float] = {}
    for name, preload in (
        ("fp32_native_mlp", preload_native_decoder_layer),
        ("fp16_weight_mlp", preload_weight_only_extension),
        ("int8_weight_mlp", preload_int8_mlp_extension),
    ):
        torch.cuda.synchronize()
        started = time.perf_counter()
        preload()
        torch.cuda.synchronize()
        extension_times[name] = time.perf_counter() - started

    buckets: dict[str, dict[str, Any]] = {}
    pack_reports: dict[str, dict[str, Any]] = {}
    for prefix, accepted in selected.items():
        inputs = accepted["static_inputs"]
        cache = _cache_from_static_inputs(inputs)
        cache_position = inputs.get("cache_position")
        if not isinstance(cache_position, torch.Tensor) or cache_position.numel() != 1:
            raise RuntimeError(f"prefix {prefix} has invalid cache position")
        snapshots = _all_cache_snapshots(cache, cache_position)
        _restore_all_cache(cache, snapshots)
        capture = _capture_real_mlp_input(model, inputs, prefix=prefix)
        _restore_all_cache(cache, snapshots)
        layer = capture.module
        eps = float(getattr(layer.final_layer_norm, "eps", torch.finfo(torch.float32).eps))

        torch.cuda.synchronize()
        allocated_before = int(torch.cuda.memory_allocated())
        reserved_before = int(torch.cuda.memory_reserved())
        pack_started = time.perf_counter()
        fp16_fc1 = PackedLinear.from_module(layer.fc1)
        fp16_fc2 = PackedLinear.from_module(layer.fc2)
        int8_pack = Int8MlpPack.from_layer(layer)
        torch.cuda.synchronize()
        pack_seconds = time.perf_counter() - pack_started
        allocated_after = int(torch.cuda.memory_allocated())
        reserved_after = int(torch.cuda.memory_reserved())

        callables = {
            "fp32": lambda: native_one_token_mlp_residual(
                capture.hidden_states,
                layer.final_layer_norm.weight,
                layer.fc1.weight,
                layer.fc1.bias,
                layer.fc2.weight,
                layer.fc2.bias,
                eps=eps,
                outputs_per_block=8,
            ),
            "fp16_weight": lambda: weight_only_mlp_residual(
                capture.hidden_states,
                layer.final_layer_norm.weight,
                fp16_fc1,
                fp16_fc2,
                eps=eps,
                fc1_outputs_per_block=2,
                fc2_outputs_per_block=4,
            ),
            "int8_weight": lambda: int8_weight_mlp_residual(
                capture.hidden_states,
                layer.final_layer_norm.weight,
                int8_pack,
                eps=eps,
                fc1_outputs_per_block=2,
                fc2_outputs_per_block=4,
            ),
        }
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
        reference = observations["fp32"][0]
        fp16_reference = observations["fp16_weight"][0]
        checks = {
            "fp32_finite": bool(torch.isfinite(reference).all()),
            "fp16_finite": bool(torch.isfinite(fp16_reference).all()),
            "int8_finite": bool(torch.isfinite(observations["int8_weight"][0]).all()),
            "fp32_repeat_deterministic": torch.equal(*observations["fp32"]),
            "fp16_repeat_deterministic": torch.equal(*observations["fp16_weight"]),
            "int8_repeat_deterministic": torch.equal(*observations["int8_weight"]),
            "fp32_memory_stable": bool(memory["fp32"]),
            "fp16_memory_stable": bool(memory["fp16_weight"]),
            "int8_memory_stable": bool(memory["int8_weight"]),
            "output_shape_preserved": (
                reference.shape
                == fp16_reference.shape
                == observations["int8_weight"][0].shape
                == capture.hidden_states.cpu().shape
            ),
        }
        int8_output = observations["int8_weight"][0]
        buckets[str(prefix)] = {
            "decode_replays": int(accepted["decode_replays"]),
            "target_layer_index": TARGET_LAYER_INDEX,
            "hidden_shape": list(capture.hidden_states.shape),
            "ms_per_call": timings,
            "local_speedup": timings["fp16_weight"] / timings["int8_weight"],
            "capture_setup_seconds": {
                name: graph.setup_seconds for name, graph in graphs.items()
            },
            "capture_peak_vram_bytes": {
                name: graph.peak_vram_bytes for name, graph in graphs.items()
            },
            "checks": checks,
            "drift": {
                "fp16_vs_fp32_max_abs": _max_abs(reference, fp16_reference),
                "int8_vs_fp32_max_abs": _max_abs(reference, int8_output),
                "int8_vs_fp16_max_abs": _max_abs(fp16_reference, int8_output),
                "int8_vs_fp16_mean_abs": float(
                    (int8_output - fp16_reference).abs().mean().item()
                ),
            },
            "rounds": rounds,
        }
        pack_reports[str(prefix)] = {
            "pack_seconds": pack_seconds,
            "allocated_bytes_delta": allocated_after - allocated_before,
            "reserved_bytes_delta": reserved_after - reserved_before,
            "fp16_packed_weight_bytes": (
                fp16_fc1.packed_weight_bytes + fp16_fc2.packed_weight_bytes
            ),
            "int8": int8_pack.memory_report(),
        }

    summary = summarize_component(buckets, total_replays=total_replays)
    return {
        "schema_version": 1,
        "metadata": {
            "candidate": "per_row_symmetric_int8_mlp_weights",
            "result_class": "component_scout_documented_drift",
            "production_wiring": False,
            "precision_contract": {
                "weight_storage": "int8_per_output_row_symmetric",
                "scale": "fp32_per_output_row",
                "input_output_norm_bias_residual": "fp32",
                "accumulation": "fp32",
            },
            "comparison": "current_fp16_weight_mlp",
            "bucket_mode": bucket_mode,
            "target_layer_index": TARGET_LAYER_INDEX,
            "extension_preload_seconds": extension_times,
            "warmup": warmup,
            "iters": iters,
        },
        "summary": summary,
        "pack_reports": pack_reports,
        "buckets": buckets,
    }


def _summary_text(result: dict[str, Any]) -> str:
    summary = result["summary"]
    return "\n".join(
        (
            f"int8_vs_fp16_local_speedup={summary['int8_vs_fp16_local_speedup']:.6f}",
            f"fixed_work_main_saving_seconds={summary['fixed_work_main_saving_seconds']:.6f}",
            f"warm_request_saving_seconds_estimate={summary['warm_request_saving_seconds_estimate']:.6f}",
            f"coverage_fraction={summary['coverage_fraction']:.6f}",
            f"invariants_pass={str(summary['invariants_pass']).lower()}",
            f"sizing_pass={str(summary['sizing_pass']).lower()}",
            f"promotion_pass={str(summary['promotion_pass']).lower()}",
        )
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--text-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--bucket-mode", choices=("sentinel", "all"), default="sentinel")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("overrides", nargs="*")
    cli = parser.parse_args()
    overrides = list(cli.overrides)
    if not any(value.startswith("output_path=") for value in overrides):
        overrides.append(f"output_path={cli.output_path}")
    args = _load_args(cli.config_name, overrides)
    result = profile_int8_mlp_component(
        args,
        output_path=cli.output_path,
        bucket_mode=cli.bucket_mode,
        warmup=cli.warmup,
        iters=cli.iters,
    )
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    cli.text_path.parent.mkdir(parents=True, exist_ok=True)
    cli.text_path.write_text(_summary_text(result), encoding="utf-8")
    print(_summary_text(result), end="")
    summary = result["summary"]
    if not summary["invariants_pass"]:
        raise SystemExit(1)
    if not summary["sizing_pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
