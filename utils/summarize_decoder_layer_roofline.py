from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_signature_report(report: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    signatures = report.get("signature_reports")
    projected = report.get("projected_full_song")
    if not isinstance(signatures, dict) or not signatures:
        raise ValueError("decoder-layer report has no signature_reports")
    if not isinstance(projected, dict) or not projected:
        raise ValueError("decoder-layer report has no projected_full_song")
    signature = sorted(signatures)[0]
    signature_report = signatures[signature]
    projection = projected.get(signature)
    if not isinstance(signature_report, dict) or not isinstance(projection, dict):
        raise ValueError(f"decoder-layer signature {signature!r} is malformed")
    return signature, signature_report, projection


def _linear_flops(in_dim: int, out_dim: int) -> int:
    # Count multiply and add as two fp32 operations.
    return 2 * in_dim * out_dim


def _linear_bytes(in_dim: int, out_dim: int, *, dtype_bytes: int, has_bias: bool = True) -> int:
    # Lower-bound DRAM traffic for one token: input, weight, optional bias, output.
    values = in_dim + (in_dim * out_dim) + out_dim
    if has_bias:
        values += out_dim
    return values * dtype_bytes


def _attention_flops(model_dim: int, kv_len: int) -> int:
    # q dot k plus softmax-weighted v accumulation. Softmax exp/reduce is not included.
    return 4 * model_dim * kv_len


def _attention_bytes(model_dim: int, kv_len: int, *, dtype_bytes: int) -> int:
    # q, K cache, V cache, scores/weights lower-bound scratch, and output.
    values = model_dim + (2 * model_dim * kv_len) + kv_len + model_dim
    return values * dtype_bytes


def _rmsnorm_flops(model_dim: int) -> int:
    # Approximate square/sum/rsqrt/scale. This is intentionally conservative.
    return 5 * model_dim


def _rmsnorm_bytes(model_dim: int, *, dtype_bytes: int) -> int:
    # Input, weight, output lower-bound traffic.
    return 3 * model_dim * dtype_bytes


def _gelu_flops(ffn_dim: int) -> int:
    # Rough tanh-GELU equivalent operation count. Used only for roofline order-of-magnitude.
    return 10 * ffn_dim


def _gelu_bytes(ffn_dim: int, *, dtype_bytes: int) -> int:
    return 2 * ffn_dim * dtype_bytes


def _component_table(
        *,
        model_dim: int,
        ffn_dim: int,
        self_prefix_len: int,
        encoder_len: int,
        dtype_bytes: int,
        cross_kv_recomputed: bool,
) -> dict[str, dict[str, int]]:
    components: dict[str, dict[str, int]] = {
        "self_qkv_linear": {
            "flops": _linear_flops(model_dim, 3 * model_dim),
            "bytes": _linear_bytes(model_dim, 3 * model_dim, dtype_bytes=dtype_bytes),
        },
        "self_rope_cache_attention": {
            "flops": _attention_flops(model_dim, self_prefix_len),
            "bytes": (
                _attention_bytes(model_dim, self_prefix_len, dtype_bytes=dtype_bytes)
                + 2 * model_dim * dtype_bytes
            ),
        },
        "self_out_linear": {
            "flops": _linear_flops(model_dim, model_dim),
            "bytes": _linear_bytes(model_dim, model_dim, dtype_bytes=dtype_bytes),
        },
        "cross_q_linear": {
            "flops": _linear_flops(model_dim, model_dim),
            "bytes": _linear_bytes(model_dim, model_dim, dtype_bytes=dtype_bytes),
        },
        "cross_q1_attention": {
            "flops": _attention_flops(model_dim, encoder_len),
            "bytes": _attention_bytes(model_dim, encoder_len, dtype_bytes=dtype_bytes),
        },
        "cross_out_linear": {
            "flops": _linear_flops(model_dim, model_dim),
            "bytes": _linear_bytes(model_dim, model_dim, dtype_bytes=dtype_bytes),
        },
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
        "three_rmsnorms": {
            "flops": 3 * _rmsnorm_flops(model_dim),
            "bytes": 3 * _rmsnorm_bytes(model_dim, dtype_bytes=dtype_bytes),
        },
        "residual_adds": {
            "flops": 3 * model_dim,
            "bytes": 6 * model_dim * dtype_bytes,
        },
    }
    if cross_kv_recomputed:
        components["cross_kv_linear"] = {
            "flops": _linear_flops(model_dim, 2 * model_dim) * encoder_len,
            "bytes": (
                (encoder_len * model_dim)
                + (2 * model_dim * model_dim)
                + (2 * encoder_len * model_dim)
                + (2 * model_dim)
            ) * dtype_bytes,
        }
    return components


def summarize_decoder_layer_roofline(
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
    model_dim = int(hidden_shape[-1])
    if not isinstance(encoder_shape, list) or len(encoder_shape) < 2:
        raise ValueError("signature report is missing encoder_hidden_shape")
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
    measured_layer_s = projection.get("cuda_graph_repo_decoder_layer_s")
    if not isinstance(measured_layer_s, (int, float)):
        raise ValueError("projection is missing cuda_graph_repo_decoder_layer_s")
    measured_layer_s = float(measured_layer_s)

    components = _component_table(
        model_dim=model_dim,
        ffn_dim=ffn_dim,
        self_prefix_len=active_prefix_len,
        encoder_len=encoder_len,
        dtype_bytes=dtype_bytes,
        cross_kv_recomputed=cross_kv_recomputed,
    )
    flops_per_layer_step = sum(item["flops"] for item in components.values())
    bytes_per_layer_step = sum(item["bytes"] for item in components.values())
    total_replays = layer_count * decode_steps
    total_flops = flops_per_layer_step * total_replays
    total_bytes = bytes_per_layer_step * total_replays
    compute_floor_s = total_flops / (peak_fp32_tflops * 1e12)
    bandwidth_floor_s = total_bytes / (peak_memory_bandwidth_gb_s * 1e9)
    roofline_floor_s = max(compute_floor_s, bandwidth_floor_s)
    removable_above_roofline_s = max(measured_layer_s - roofline_floor_s, 0.0)
    model_time_at_layer_roofline_s = baseline_model_s - removable_above_roofline_s
    layer_only_required_fraction = required_saving_s / measured_layer_s if measured_layer_s > 0 else None
    above_floor_required_fraction = (
        required_saving_s / removable_above_roofline_s
        if removable_above_roofline_s > 0
        else None
    )

    component_rows = {}
    for name, item in components.items():
        full_flops = item["flops"] * total_replays
        full_bytes = item["bytes"] * total_replays
        component_rows[name] = {
            "flops_per_layer_step": item["flops"],
            "bytes_per_layer_step": item["bytes"],
            "projected_full_song_compute_floor_s": full_flops / (peak_fp32_tflops * 1e12),
            "projected_full_song_bandwidth_floor_s": full_bytes / (peak_memory_bandwidth_gb_s * 1e9),
        }

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
        },
        "measured_decoder_layer": {
            "cuda_graph_repo_decoder_layer_s": measured_layer_s,
            "fraction_of_model_time": measured_layer_s / baseline_model_s,
            "layer_only_required_fraction_to_hit_target": layer_only_required_fraction,
            "layer_time_left_if_layer_only_hit_target_s": measured_layer_s - required_saving_s,
        },
        "roofline": {
            "flops_per_layer_step": flops_per_layer_step,
            "bytes_per_layer_step": bytes_per_layer_step,
            "total_flops": total_flops,
            "total_bytes": total_bytes,
            "compute_floor_s": compute_floor_s,
            "bandwidth_floor_s": bandwidth_floor_s,
            "roofline_floor_s": roofline_floor_s,
            "roofline_limiter": "compute" if compute_floor_s >= bandwidth_floor_s else "bandwidth",
            "removable_above_roofline_s": removable_above_roofline_s,
            "above_floor_required_fraction_to_hit_target": above_floor_required_fraction,
            "model_time_at_layer_roofline_s": model_time_at_layer_roofline_s,
            "tokens_per_second_at_layer_roofline": (
                full_song_tokens / model_time_at_layer_roofline_s
                if model_time_at_layer_roofline_s > 0
                else None
            ),
            "reaches_target_at_layer_roofline": model_time_at_layer_roofline_s <= target_model_s,
        },
        "components": component_rows,
        "decision": {
            "verifier_only_go": bool(removable_above_roofline_s >= baseline_model_s * 0.05),
            "layer_roofline_reaches_target": model_time_at_layer_roofline_s <= target_model_s,
            "production_candidate_requires_verifier": True,
            "note": (
                "This is an optimistic roofline lower bound, not a speed claim. It can justify a bounded "
                "verifier-only whole-layer experiment when headroom clears the keep bar, but production work "
                "still needs exact logits, token, RNG, full-song throughput, and output-byte gates."
            ),
        },
    }


def print_summary(report: dict[str, Any]) -> None:
    def fmt(value: Any, digits: int = 3) -> str:
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"

    baseline = report["accepted_baseline"]
    target = report["target"]
    measured = report["measured_decoder_layer"]
    roofline = report["roofline"]
    decision = report["decision"]
    print("Decoder Layer Roofline Summary")
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
        f"({target['required_saving_s']:.3f}s saved)"
    )
    print(
        "  measured decoder-layer replay: "
        f"{measured['cuda_graph_repo_decoder_layer_s']:.3f}s "
        f"({measured['fraction_of_model_time'] * 100:.1f}% of model time)"
    )
    print(
        "  layer-only target pressure: "
        f"{measured['layer_only_required_fraction_to_hit_target'] * 100:.1f}% "
        f"of measured layer replay would need to disappear"
    )
    print(
        "  roofline floors: "
        f"compute={fmt(roofline['compute_floor_s'])}s, "
        f"bandwidth={fmt(roofline['bandwidth_floor_s'])}s, "
        f"floor={fmt(roofline['roofline_floor_s'])}s "
        f"({roofline['roofline_limiter']})"
    )
    print(
        "  above-floor headroom: "
        f"{fmt(roofline['removable_above_roofline_s'])}s, "
        f"model_at_floor={fmt(roofline['model_time_at_layer_roofline_s'])}s, "
        f"tps_at_floor={fmt(roofline['tokens_per_second_at_layer_roofline'])}"
    )
    print(
        "  decision: "
        f"verifier_only_go={decision['verifier_only_go']} "
        f"layer_roofline_reaches_target={decision['layer_roofline_reaches_target']}"
    )
    print(f"  note: {decision['note']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate an optimistic RTX 2080 Ti roofline lower bound for the captured "
            "one-token decoder-layer replay bucket. Diagnostic only; not a throughput claim."
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

    report = summarize_decoder_layer_roofline(
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
