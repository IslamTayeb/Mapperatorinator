from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.summarize_decoder_layer_roofline import (
    _attention_bytes,
    _attention_flops,
    _component_table,
    _first_signature_report,
    _gelu_bytes,
    _gelu_flops,
    _linear_bytes,
    _linear_flops,
    _rmsnorm_bytes,
    _rmsnorm_flops,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _segment_components(
        *,
        model_dim: int,
        ffn_dim: int,
        self_prefix_len: int,
        encoder_len: int,
        dtype_bytes: int,
        cross_kv_recomputed: bool,
) -> dict[str, dict[str, dict[str, int]]]:
    components = _component_table(
        model_dim=model_dim,
        ffn_dim=ffn_dim,
        self_prefix_len=self_prefix_len,
        encoder_len=encoder_len,
        dtype_bytes=dtype_bytes,
        cross_kv_recomputed=cross_kv_recomputed,
    )
    one_norm = {
        "flops": _rmsnorm_flops(model_dim),
        "bytes": _rmsnorm_bytes(model_dim, dtype_bytes=dtype_bytes),
    }
    one_residual = {
        "flops": model_dim,
        "bytes": 2 * model_dim * dtype_bytes,
    }
    return {
        "self_attn_residual_segment": {
            "self_attn_layer_norm": one_norm,
            "self_qkv_linear": components["self_qkv_linear"],
            "self_rope_cache_attention": components["self_rope_cache_attention"],
            "self_out_linear": components["self_out_linear"],
            "self_residual_add": one_residual,
        },
        "cross_attn_residual_segment": {
            "cross_attn_layer_norm": one_norm,
            "cross_q_linear": components["cross_q_linear"],
            "cross_q1_attention": components["cross_q1_attention"],
            "cross_out_linear": components["cross_out_linear"],
            "cross_residual_add": one_residual,
            **(
                {"cross_kv_linear": components["cross_kv_linear"]}
                if cross_kv_recomputed and "cross_kv_linear" in components
                else {}
            ),
        },
        "mlp_residual_segment": {
            "final_layer_norm": one_norm,
            "mlp_fc1": components["mlp_fc1"],
            "mlp_gelu": components["mlp_gelu"],
            "mlp_fc2": components["mlp_fc2"],
            "mlp_residual_add": one_residual,
        },
        "mlp_fc1_activation_prefix": {
            "final_layer_norm": one_norm,
            "mlp_fc1": components["mlp_fc1"],
            "mlp_gelu": components["mlp_gelu"],
        },
        "mlp_fc2_residual_segment": {
            "mlp_fc2": components["mlp_fc2"],
            "mlp_residual_add": one_residual,
        },
        "decoder_layer_total": components,
        "self_attention_only": {
            "q_dot_k_and_weighted_v": {
                "flops": _attention_flops(model_dim, self_prefix_len),
                "bytes": _attention_bytes(model_dim, self_prefix_len, dtype_bytes=dtype_bytes),
            },
        },
        "cross_attention_only": {
            "q_dot_k_and_weighted_v": {
                "flops": _attention_flops(model_dim, encoder_len),
                "bytes": _attention_bytes(model_dim, encoder_len, dtype_bytes=dtype_bytes),
            },
        },
        "mlp_two_linears_activation": {
            "mlp_fc1": {
                "flops": _linear_flops(model_dim, ffn_dim),
                "bytes": _linear_bytes(model_dim, ffn_dim, dtype_bytes=dtype_bytes),
            },
            "mlp_gelu": {
                "flops": _gelu_flops(ffn_dim),
                "bytes": _gelu_bytes(ffn_dim, dtype_bytes=dtype_bytes),
            },
            "mlp_fc2": {
                "flops": _linear_flops(ffn_dim, model_dim),
                "bytes": _linear_bytes(ffn_dim, model_dim, dtype_bytes=dtype_bytes),
            },
        },
    }


def _sum_components(components: dict[str, dict[str, int]]) -> tuple[int, int]:
    return (
        sum(int(item["flops"]) for item in components.values()),
        sum(int(item["bytes"]) for item in components.values()),
    )


def _time_at_saving(*, baseline_model_s: float, full_song_tokens: int, saved_s: float) -> float | None:
    model_s = baseline_model_s - saved_s
    if model_s <= 0:
        return None
    return full_song_tokens / model_s


def summarize_decoder_layer_segment_pressure(
        decoder_layer_report: dict[str, Any],
        *,
        ffn_dim: int | None,
        decoder_heads: int | None,
        dtype_bytes: int,
        peak_fp32_tflops: float,
        peak_memory_bandwidth_gb_s: float,
        target_tps: float,
        cross_kv_recomputed: bool,
) -> dict[str, Any]:
    signature, signature_report, projection = _first_signature_report(decoder_layer_report)
    metadata = decoder_layer_report.get("metadata") or {}
    hidden_shape = signature_report.get("hidden_shape")
    encoder_shape = signature_report.get("encoder_hidden_shape")
    if not isinstance(hidden_shape, list) or len(hidden_shape) < 3:
        raise ValueError("signature report is missing hidden_shape [B, T, D]")
    if not isinstance(encoder_shape, list) or len(encoder_shape) < 2:
        raise ValueError("signature report is missing encoder_hidden_shape")

    model_dim = int(hidden_shape[-1])
    encoder_len = int(encoder_shape[1])
    inferred_ffn_dim = 4 * model_dim
    ffn_dim = int(ffn_dim) if ffn_dim is not None else inferred_ffn_dim
    decoder_heads = int(decoder_heads) if decoder_heads is not None else max(model_dim // 64, 1)
    head_dim = model_dim // decoder_heads if decoder_heads > 0 else None
    active_prefix_len = int(decoder_layer_report.get("active_prefix_length") or 0)
    decode_steps = int(metadata.get("full_song_decode_steps") or projection.get("decode_steps"))
    layer_count = int(signature_report.get("member_count") or decoder_layer_report.get("captured_decoder_layer_count"))
    full_song_tokens = int(metadata.get("full_song_main_tokens") or projection.get("main_tokens"))
    baseline_model_s = float(metadata.get("full_song_model_time_s"))
    target_model_s = full_song_tokens / target_tps
    required_saving_s = baseline_model_s - target_model_s
    keep_5_s = baseline_model_s * 0.05
    keep_10_s = baseline_model_s * 0.10

    segment_seconds = dict(projection.get("cuda_graph_segment_seconds") or {})
    measured_total = projection.get("cuda_graph_repo_decoder_layer_s")
    if isinstance(measured_total, (int, float)):
        segment_seconds["decoder_layer_total"] = float(measured_total)
    manual_saving = projection.get("cuda_graph_manual_decoder_runtime_island_saved_s")

    segment_component_map = _segment_components(
        model_dim=model_dim,
        ffn_dim=ffn_dim,
        self_prefix_len=active_prefix_len,
        encoder_len=encoder_len,
        dtype_bytes=dtype_bytes,
        cross_kv_recomputed=cross_kv_recomputed,
    )

    rows: dict[str, dict[str, Any]] = {}
    total_replays = layer_count * decode_steps
    for segment_name, components in segment_component_map.items():
        measured_s = segment_seconds.get(segment_name)
        if not isinstance(measured_s, (int, float)):
            continue
        flops_per_layer_step, bytes_per_layer_step = _sum_components(components)
        total_flops = flops_per_layer_step * total_replays
        total_bytes = bytes_per_layer_step * total_replays
        compute_floor_s = total_flops / (peak_fp32_tflops * 1e12)
        bandwidth_floor_s = total_bytes / (peak_memory_bandwidth_gb_s * 1e9)
        roofline_floor_s = max(compute_floor_s, bandwidth_floor_s)
        above_floor_s = max(float(measured_s) - roofline_floor_s, 0.0)
        floor_model_s = baseline_model_s - above_floor_s
        free_model_s = baseline_model_s - float(measured_s)
        component_floor_rows = {}
        for component_name, item in components.items():
            component_flops = int(item["flops"]) * total_replays
            component_bytes = int(item["bytes"]) * total_replays
            component_floor_rows[component_name] = {
                "compute_floor_s": component_flops / (peak_fp32_tflops * 1e12),
                "bandwidth_floor_s": component_bytes / (peak_memory_bandwidth_gb_s * 1e9),
            }
        rows[segment_name] = {
            "measured_s": float(measured_s),
            "fraction_of_model_time": float(measured_s) / baseline_model_s,
            "flops_per_layer_step": flops_per_layer_step,
            "bytes_per_layer_step": bytes_per_layer_step,
            "compute_floor_s": compute_floor_s,
            "bandwidth_floor_s": bandwidth_floor_s,
            "roofline_floor_s": roofline_floor_s,
            "roofline_limiter": "compute" if compute_floor_s >= bandwidth_floor_s else "bandwidth",
            "above_roofline_floor_s": above_floor_s,
            "above_floor_fraction_of_measured": above_floor_s / float(measured_s) if measured_s else None,
            "tps_if_segment_free": full_song_tokens / free_model_s if free_model_s > 0 else None,
            "tps_if_segment_at_roofline_floor": full_song_tokens / floor_model_s if floor_model_s > 0 else None,
            "reaches_target_if_at_floor": floor_model_s <= target_model_s,
            "clears_5pct_bar_if_at_floor": above_floor_s >= keep_5_s,
            "clears_10pct_bar_if_at_floor": above_floor_s >= keep_10_s,
            "component_floors": component_floor_rows,
        }

    candidate_order = sorted(
        (
            {
                "segment": name,
                "measured_s": row["measured_s"],
                "above_roofline_floor_s": row["above_roofline_floor_s"],
                "roofline_floor_s": row["roofline_floor_s"],
                "tps_if_segment_at_roofline_floor": row["tps_if_segment_at_roofline_floor"],
                "reaches_target_if_at_floor": row["reaches_target_if_at_floor"],
                "clears_5pct_bar_if_at_floor": row["clears_5pct_bar_if_at_floor"],
                "clears_10pct_bar_if_at_floor": row["clears_10pct_bar_if_at_floor"],
            }
            for name, row in rows.items()
            if name != "decoder_layer_total"
        ),
        key=lambda item: item["above_roofline_floor_s"],
        reverse=True,
    )

    return {
        "source_signature": signature,
        "source_pass": bool(decoder_layer_report.get("pass", False)),
        "assumptions": {
            "model_dim": model_dim,
            "ffn_dim": ffn_dim,
            "inferred_ffn_dim": inferred_ffn_dim,
            "decoder_heads": decoder_heads,
            "head_dim": head_dim,
            "active_prefix_len": active_prefix_len,
            "encoder_len": encoder_len,
            "layer_count": layer_count,
            "decode_steps": decode_steps,
            "dtype_bytes": dtype_bytes,
            "peak_fp32_tflops": peak_fp32_tflops,
            "peak_memory_bandwidth_gb_s": peak_memory_bandwidth_gb_s,
            "cross_kv_recomputed": cross_kv_recomputed,
        },
        "accepted_baseline": {
            "full_song_main_tokens": full_song_tokens,
            "full_song_model_time_s": baseline_model_s,
            "tokens_per_second": full_song_tokens / baseline_model_s,
        },
        "target": {
            "tokens_per_second": target_tps,
            "required_model_time_s": target_model_s,
            "required_saving_s": required_saving_s,
            "keep_5pct_s": keep_5_s,
            "keep_10pct_s": keep_10_s,
        },
        "manual_decoder_runtime_island_saved_s": (
            float(manual_saving)
            if isinstance(manual_saving, (int, float))
            else None
        ),
        "segments": rows,
        "candidate_order_by_above_floor": candidate_order,
        "decision": {
            "verifier_only_go_segments": [
                item["segment"]
                for item in candidate_order
                if item["clears_5pct_bar_if_at_floor"]
            ],
            "segments_reaching_target_at_floor": [
                item["segment"]
                for item in candidate_order
                if item["reaches_target_if_at_floor"]
            ],
            "recommended_next_scope": (
                "multi_segment_or_whole_layer_native_math_memory_island"
                if any(item["clears_5pct_bar_if_at_floor"] for item in candidate_order)
                else "stop_no_segment_clears_keep_bar"
            ),
            "production_candidate_requires_full_equivalence": True,
            "note": (
                "Segment roofline headroom is a target-sizing diagnostic, not a speed claim. "
                "It can justify a bounded native/CUDA/CUTLASS verifier only when the measured "
                "segment is exclusive enough and above-floor saving clears the keep bar. "
                "A single segment at its optimistic floor is not a production plan unless it "
                "also passes exact end-to-end gates and clears full-song throughput bars."
            ),
        },
    }


def print_summary(report: dict[str, Any]) -> None:
    def fmt(value: Any, digits: int = 3) -> str:
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"

    baseline = report["accepted_baseline"]
    target = report["target"]
    print("Decoder Layer Segment Pressure Summary")
    print(
        "  accepted baseline: "
        f"{baseline['full_song_main_tokens']} tokens, "
        f"{baseline['full_song_model_time_s']:.3f}s, "
        f"{baseline['tokens_per_second']:.3f} tok/s"
    )
    print(
        "  target: "
        f"{target['tokens_per_second']:.1f} tok/s requires "
        f"{target['required_model_time_s']:.3f}s "
        f"({target['required_saving_s']:.3f}s saved); "
        f"5% bar={target['keep_5pct_s']:.3f}s"
    )
    print("  segment order by above-floor headroom:")
    for item in report["candidate_order_by_above_floor"]:
        print(
            "    "
            f"{item['segment']}: measured={fmt(item['measured_s'])}s, "
            f"floor={fmt(item['roofline_floor_s'])}s, "
            f"above_floor={fmt(item['above_roofline_floor_s'])}s, "
            f"tps_at_floor={fmt(item['tps_if_segment_at_roofline_floor'])}, "
            f"clears5={item['clears_5pct_bar_if_at_floor']}, "
            f"reaches_target={item['reaches_target_if_at_floor']}"
        )
    print(
        "  manual decoder runtime island saved: "
        f"{fmt(report.get('manual_decoder_runtime_island_saved_s'))}s"
    )
    print(
        "  verifier-only go segments: "
        + ", ".join(report["decision"]["verifier_only_go_segments"])
    )
    print(f"  recommended next scope: {report['decision']['recommended_next_scope']}")
    print(f"  note: {report['decision']['note']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare measured decoder-layer segment replay time with optimistic "
            "compute/bandwidth roofline floors. Diagnostic only; not a throughput claim."
        )
    )
    parser.add_argument("decoder_layer_report", type=Path)
    parser.add_argument("--ffn-dim", type=int, default=None)
    parser.add_argument("--decoder-heads", type=int, default=None)
    parser.add_argument("--dtype-bytes", type=int, default=4)
    parser.add_argument("--peak-fp32-tflops", type=float, default=13.45)
    parser.add_argument("--peak-memory-bandwidth-gb-s", type=float, default=616.0)
    parser.add_argument("--target-tps", type=float, default=500.0)
    parser.add_argument(
        "--cross-kv-recomputed",
        action="store_true",
        help="Include cross-attention Wkv projection for encoder states. Normal one-token decode reuses cross K/V.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    report = summarize_decoder_layer_segment_pressure(
        _load_json(args.decoder_layer_report),
        ffn_dim=args.ffn_dim,
        decoder_heads=args.decoder_heads,
        dtype_bytes=args.dtype_bytes,
        peak_fp32_tflops=args.peak_fp32_tflops,
        peak_memory_bandwidth_gb_s=args.peak_memory_bandwidth_gb_s,
        target_tps=args.target_tps,
        cross_kv_recomputed=args.cross_kv_recomputed,
    )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_summary(report)


if __name__ == "__main__":
    main()
