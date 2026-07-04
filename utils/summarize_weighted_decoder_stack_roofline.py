from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.summarize_decoder_layer_roofline import _component_table


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_signature_report(report: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
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
    return signature_report, projection


def _model_contract(
        decoder_layer_report: dict[str, Any] | None,
        *,
        model_dim: int | None,
        ffn_dim: int | None,
        encoder_len: int | None,
        decoder_layers: int | None,
) -> dict[str, int]:
    if decoder_layer_report is not None:
        signature_report, projection = _first_signature_report(decoder_layer_report)
        hidden_shape = signature_report.get("hidden_shape")
        encoder_shape = signature_report.get("encoder_hidden_shape")
        if model_dim is None:
            if not isinstance(hidden_shape, list) or len(hidden_shape) < 3:
                raise ValueError("decoder-layer report is missing hidden_shape [B, T, D]")
            model_dim = int(hidden_shape[-1])
        if encoder_len is None:
            if not isinstance(encoder_shape, list) or len(encoder_shape) < 2:
                raise ValueError("decoder-layer report is missing encoder_hidden_shape")
            encoder_len = int(encoder_shape[1])
        if decoder_layers is None:
            decoder_layers = int(signature_report.get("member_count") or projection.get("member_count") or 0)
    if model_dim is None or model_dim <= 0:
        raise ValueError("model_dim must be provided or inferred from --decoder-layer-report")
    if ffn_dim is None:
        ffn_dim = 4 * model_dim
    if encoder_len is None or encoder_len <= 0:
        raise ValueError("encoder_len must be provided or inferred from --decoder-layer-report")
    if decoder_layers is None or decoder_layers <= 0:
        raise ValueError("decoder_layers must be provided or inferred from --decoder-layer-report")
    return {
        "model_dim": int(model_dim),
        "ffn_dim": int(ffn_dim),
        "encoder_len": int(encoder_len),
        "decoder_layers": int(decoder_layers),
    }


def _tps(tokens: int, seconds: float) -> float | None:
    return tokens / seconds if tokens > 0 and seconds > 0 else None


def summarize_weighted_decoder_stack_roofline(
        weighted_stack_summary: dict[str, Any],
        *,
        decoder_layer_report: dict[str, Any] | None,
        model_dim: int | None,
        ffn_dim: int | None,
        encoder_len: int | None,
        decoder_layers: int | None,
        decoder_heads: int | None,
        dtype_bytes: int,
        peak_fp32_tflops: float,
        peak_memory_bandwidth_gb_s: float,
        target_tps: float,
        cross_kv_recomputed: bool,
) -> dict[str, Any]:
    rows = weighted_stack_summary.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("weighted stack summary has no rows")
    contract = _model_contract(
        decoder_layer_report,
        model_dim=model_dim,
        ffn_dim=ffn_dim,
        encoder_len=encoder_len,
        decoder_layers=decoder_layers,
    )
    model_dim = contract["model_dim"]
    ffn_dim = contract["ffn_dim"]
    encoder_len = contract["encoder_len"]
    decoder_layers = contract["decoder_layers"]
    decoder_heads = int(decoder_heads) if decoder_heads is not None else max(model_dim // 64, 1)
    head_dim = model_dim // decoder_heads if decoder_heads > 0 else None

    weighted_seconds = weighted_stack_summary.get("weighted_seconds")
    if not isinstance(weighted_seconds, dict):
        raise ValueError("weighted stack summary is missing weighted_seconds")
    measured_stack_s = weighted_seconds.get("decoder_stack_hidden")
    if not isinstance(measured_stack_s, (int, float)):
        raise ValueError("weighted_seconds.decoder_stack_hidden is missing")
    measured_stack_s = float(measured_stack_s)
    measured_full_forward_s = weighted_seconds.get("full_model_forward_logits")
    measured_full_forward_s = float(measured_full_forward_s) if isinstance(measured_full_forward_s, (int, float)) else None

    full_song_tokens = int(weighted_stack_summary.get("full_song_main_tokens") or 0)
    baseline_model_s = float(weighted_stack_summary.get("full_song_model_time_s") or 0.0)
    if full_song_tokens <= 0 or baseline_model_s <= 0:
        raise ValueError("weighted stack summary is missing full-song token/model-time metadata")
    target_model_s = full_song_tokens / target_tps
    required_saving_s = baseline_model_s - target_model_s

    component_totals: dict[str, dict[str, float]] = {}
    bucket_rows: list[dict[str, Any]] = []
    total_flops = 0
    total_bytes = 0
    total_replays = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        prefix = int(row.get("active_prefix_length") or row.get("prefix") or 0)
        decode_replays = int(row.get("decode_replays") or 0)
        if prefix <= 0 or decode_replays <= 0:
            continue
        components = _component_table(
            model_dim=model_dim,
            ffn_dim=ffn_dim,
            self_prefix_len=prefix,
            encoder_len=encoder_len,
            dtype_bytes=dtype_bytes,
            cross_kv_recomputed=cross_kv_recomputed,
        )
        bucket_flops_per_layer_step = sum(item["flops"] for item in components.values())
        bucket_bytes_per_layer_step = sum(item["bytes"] for item in components.values())
        bucket_replays = decoder_layers * decode_replays
        bucket_flops = bucket_flops_per_layer_step * bucket_replays
        bucket_bytes = bucket_bytes_per_layer_step * bucket_replays
        total_flops += bucket_flops
        total_bytes += bucket_bytes
        total_replays += decode_replays
        bucket_rows.append({
            "prefix": prefix,
            "decode_replays": decode_replays,
            "layer_replays": bucket_replays,
            "flops": bucket_flops,
            "bytes": bucket_bytes,
            "compute_floor_s": bucket_flops / (peak_fp32_tflops * 1e12),
            "bandwidth_floor_s": bucket_bytes / (peak_memory_bandwidth_gb_s * 1e9),
        })
        for name, item in components.items():
            totals = component_totals.setdefault(name, {"flops": 0.0, "bytes": 0.0})
            totals["flops"] += float(item["flops"] * bucket_replays)
            totals["bytes"] += float(item["bytes"] * bucket_replays)

    if total_replays <= 0:
        raise ValueError("weighted stack summary has no usable decode replay rows")

    compute_floor_s = total_flops / (peak_fp32_tflops * 1e12)
    bandwidth_floor_s = total_bytes / (peak_memory_bandwidth_gb_s * 1e9)
    roofline_floor_s = max(compute_floor_s, bandwidth_floor_s)
    removable_above_roofline_s = max(measured_stack_s - roofline_floor_s, 0.0)
    model_time_at_stack_roofline_s = baseline_model_s - removable_above_roofline_s
    full_forward_gap_s = (
        measured_full_forward_s - measured_stack_s
        if measured_full_forward_s is not None
        else None
    )
    component_rows = {}
    for name, totals in component_totals.items():
        component_rows[name] = {
            "total_flops": totals["flops"],
            "total_bytes": totals["bytes"],
            "compute_floor_s": totals["flops"] / (peak_fp32_tflops * 1e12),
            "bandwidth_floor_s": totals["bytes"] / (peak_memory_bandwidth_gb_s * 1e9),
        }

    return {
        "source_pass": bool(weighted_stack_summary.get("pass", False)),
        "source_commit": weighted_stack_summary.get("commit"),
        "source_dcc_job_id": weighted_stack_summary.get("dcc_job_id"),
        "assumptions": {
            **contract,
            "decoder_heads": decoder_heads,
            "head_dim": head_dim,
            "dtype_bytes": int(dtype_bytes),
            "peak_fp32_tflops": float(peak_fp32_tflops),
            "peak_memory_bandwidth_gb_s": float(peak_memory_bandwidth_gb_s),
            "cross_kv_recomputed": bool(cross_kv_recomputed),
            "weighted_decode_replays": int(weighted_stack_summary.get("weighted_decode_replays") or total_replays),
            "bucket_count": len(bucket_rows),
        },
        "accepted_baseline": {
            "full_song_main_tokens": full_song_tokens,
            "full_song_model_time_s": baseline_model_s,
            "tokens_per_second": full_song_tokens / baseline_model_s,
        },
        "target": {
            "tokens_per_second": float(target_tps),
            "required_model_time_s": target_model_s,
            "required_saving_s": required_saving_s,
        },
        "measured_weighted_stack": {
            "decoder_stack_hidden_s": measured_stack_s,
            "full_model_forward_logits_s": measured_full_forward_s,
            "full_forward_minus_decoder_stack_s": full_forward_gap_s,
            "fraction_of_model_time": measured_stack_s / baseline_model_s,
            "stack_only_required_fraction_to_hit_target": (
                required_saving_s / measured_stack_s if measured_stack_s > 0 else None
            ),
        },
        "roofline": {
            "total_flops": total_flops,
            "total_bytes": total_bytes,
            "compute_floor_s": compute_floor_s,
            "bandwidth_floor_s": bandwidth_floor_s,
            "roofline_floor_s": roofline_floor_s,
            "roofline_limiter": "compute" if compute_floor_s >= bandwidth_floor_s else "bandwidth",
            "removable_above_roofline_s": removable_above_roofline_s,
            "above_floor_required_fraction_to_hit_target": (
                required_saving_s / removable_above_roofline_s
                if removable_above_roofline_s > 0
                else None
            ),
            "model_time_at_stack_roofline_s": model_time_at_stack_roofline_s,
            "tokens_per_second_at_stack_roofline": _tps(full_song_tokens, model_time_at_stack_roofline_s),
            "reaches_target_at_stack_roofline": model_time_at_stack_roofline_s <= target_model_s,
        },
        "components": dict(sorted(
            component_rows.items(),
            key=lambda item: item[1]["bandwidth_floor_s"],
            reverse=True,
        )),
        "buckets": bucket_rows,
        "decision": {
            "verifier_only_go": bool(removable_above_roofline_s >= baseline_model_s * 0.05),
            "stack_roofline_reaches_target": model_time_at_stack_roofline_s <= target_model_s,
            "production_candidate_requires_verifier": True,
            "note": (
                "This is an optimistic weighted roofline lower bound, not a speed claim. "
                "It separates potentially removable above-floor stack time from necessary "
                "math/data movement under the roofline assumptions."
            ),
        },
    }


def print_summary(report: dict[str, Any]) -> None:
    def fmt(value: Any, digits: int = 3) -> str:
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"

    baseline = report["accepted_baseline"]
    target = report["target"]
    measured = report["measured_weighted_stack"]
    roofline = report["roofline"]
    decision = report["decision"]
    print("Weighted Decoder Stack Roofline Summary")
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
        "  measured weighted decoder stack: "
        f"{measured['decoder_stack_hidden_s']:.3f}s "
        f"({measured['fraction_of_model_time'] * 100:.1f}% of model time)"
    )
    print(
        "  stack-only target pressure: "
        f"{measured['stack_only_required_fraction_to_hit_target'] * 100:.1f}% "
        "of measured stack replay would need to disappear"
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
        f"model_at_floor={fmt(roofline['model_time_at_stack_roofline_s'])}s, "
        f"tps_at_floor={fmt(roofline['tokens_per_second_at_stack_roofline'])}"
    )
    print(
        "  decision: "
        f"verifier_only_go={decision['verifier_only_go']} "
        f"stack_roofline_reaches_target={decision['stack_roofline_reaches_target']}"
    )
    print(f"  note: {decision['note']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate an optimistic RTX 2080 Ti roofline lower bound for the weighted "
            "one-token decoder-stack replay bucket. Diagnostic only; not a throughput claim."
        )
    )
    parser.add_argument("weighted_stack_summary", type=Path)
    parser.add_argument("--decoder-layer-report", type=Path, default=None)
    parser.add_argument("--model-dim", type=int, default=None)
    parser.add_argument("--ffn-dim", type=int, default=None)
    parser.add_argument("--encoder-len", type=int, default=None)
    parser.add_argument("--decoder-layers", type=int, default=None)
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

    report = summarize_weighted_decoder_stack_roofline(
        _load_json(args.weighted_stack_summary),
        decoder_layer_report=(
            _load_json(args.decoder_layer_report)
            if args.decoder_layer_report is not None
            else None
        ),
        model_dim=args.model_dim,
        ffn_dim=args.ffn_dim,
        encoder_len=args.encoder_len,
        decoder_layers=args.decoder_layers,
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
