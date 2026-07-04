from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SIGNATURE_RE = re.compile(r"in(?P<in>\d+)_out(?P<out>\d+)_bias(?P<bias>[01])")


def _parse_signature(signature: str) -> tuple[int, int, bool]:
    match = SIGNATURE_RE.search(signature)
    if match is None:
        raise ValueError(f"could not parse linear signature: {signature}")
    return int(match.group("in")), int(match.group("out")), bool(int(match.group("bias")))


def _linear_flops(in_features: int, out_features: int) -> int:
    return 2 * in_features * out_features


def _linear_min_bytes(in_features: int, out_features: int, bias: bool, *, dtype_bytes: int) -> int:
    weight = in_features * out_features * dtype_bytes
    source = in_features * dtype_bytes
    output = out_features * dtype_bytes
    bias_bytes = out_features * dtype_bytes if bias else 0
    return weight + source + output + bias_bytes


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _project_seconds(ms_per_call: float, member_count: int, decode_steps: int) -> float:
    return float(ms_per_call) * int(member_count) * int(decode_steps) / 1000.0


def summarize_linear_roofline(
        report: dict[str, Any],
        *,
        full_song_decode_steps: int,
        full_song_main_tokens: int,
        full_song_model_time_s: float,
        dtype_bytes: int,
        peak_memory_bandwidth_gb_s: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    totals: dict[str, float] = defaultdict(float)
    op_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for signature, benchmark in sorted(report.get("unique_linear_benchmarks", {}).items()):
        in_features, out_features, bias = _parse_signature(signature)
        member_count = int(benchmark.get("member_count", 0) or 0)
        operation_kinds = list(benchmark.get("operation_kinds", []))
        functional = (benchmark.get("results") or {}).get("functional_linear") or {}
        graph_ms = functional.get("cuda_graph_replay_ms_per_call")
        eager_ms = functional.get("ms_per_call")
        selected_ms = graph_ms if isinstance(graph_ms, (int, float)) else eager_ms
        if not isinstance(selected_ms, (int, float)):
            continue

        flops = _linear_flops(in_features, out_features)
        min_bytes = _linear_min_bytes(in_features, out_features, bias, dtype_bytes=dtype_bytes)
        projected_s = _project_seconds(float(selected_ms), member_count, full_song_decode_steps)
        lower_bound_ms = (min_bytes / (peak_memory_bandwidth_gb_s * 1e9)) * 1000.0
        lower_bound_s = _project_seconds(lower_bound_ms, member_count, full_song_decode_steps)
        achieved_gflops = _safe_div(flops, float(selected_ms) * 1e-3) / 1e9
        achieved_gb_s = _safe_div(min_bytes, float(selected_ms) * 1e-3) / 1e9
        row = {
            "signature": signature,
            "operation_kinds": operation_kinds,
            "member_count": member_count,
            "in_features": in_features,
            "out_features": out_features,
            "bias": bias,
            "ms_per_call": float(selected_ms),
            "source_timing": "cuda_graph_replay" if isinstance(graph_ms, (int, float)) else "eager_cuda_event",
            "flops_per_call": flops,
            "min_bytes_per_call": min_bytes,
            "achieved_gflops": achieved_gflops,
            "achieved_min_bytes_gb_s": achieved_gb_s,
            "peak_bandwidth_lower_bound_ms_per_call": lower_bound_ms,
            "projected_full_song_seconds": projected_s,
            "peak_bandwidth_lower_bound_full_song_seconds": lower_bound_s,
            "ideal_zero_component_tps": (
                full_song_main_tokens / (full_song_model_time_s - projected_s)
                if full_song_model_time_s > projected_s
                else None
            ),
            "peak_bandwidth_floor_tps": (
                full_song_main_tokens / (full_song_model_time_s - projected_s + lower_bound_s)
                if (full_song_model_time_s - projected_s + lower_bound_s) > 0
                else None
            ),
        }
        rows.append(row)
        totals["projected_full_song_seconds"] += projected_s
        totals["peak_bandwidth_lower_bound_full_song_seconds"] += lower_bound_s
        totals["weighted_flops"] += flops * member_count * full_song_decode_steps
        totals["weighted_min_bytes"] += min_bytes * member_count * full_song_decode_steps
        for operation_kind in operation_kinds or ["unknown"]:
            op_totals[operation_kind]["projected_full_song_seconds"] += projected_s
            op_totals[operation_kind]["peak_bandwidth_lower_bound_full_song_seconds"] += lower_bound_s
            op_totals[operation_kind]["weighted_flops"] += flops * member_count * full_song_decode_steps
            op_totals[operation_kind]["weighted_min_bytes"] += min_bytes * member_count * full_song_decode_steps

    rows.sort(key=lambda row: row["projected_full_song_seconds"], reverse=True)
    projected = totals["projected_full_song_seconds"]
    lower_bound = totals["peak_bandwidth_lower_bound_full_song_seconds"]
    removable_above_peak_floor = max(projected - lower_bound, 0.0)
    model_time_if_zero = full_song_model_time_s - projected
    model_time_at_peak_floor = full_song_model_time_s - projected + lower_bound
    summary = {
        "linear_projected_full_song_seconds": projected,
        "linear_fraction_of_model_time": projected / full_song_model_time_s,
        "peak_bandwidth_lower_bound_full_song_seconds": lower_bound,
        "removable_above_peak_bandwidth_floor_seconds": removable_above_peak_floor,
        "zero_linear_tps": (
            full_song_main_tokens / model_time_if_zero
            if model_time_if_zero > 0
            else None
        ),
        "peak_bandwidth_floor_tps": (
            full_song_main_tokens / model_time_at_peak_floor
            if model_time_at_peak_floor > 0
            else None
        ),
        "achieved_weighted_min_bytes_gb_s": (
            totals["weighted_min_bytes"] / projected / 1e9
            if projected > 0
            else None
        ),
        "peak_memory_bandwidth_gb_s": peak_memory_bandwidth_gb_s,
        "five_percent_full_song_seconds": full_song_model_time_s * 0.05,
        "ten_percent_full_song_seconds": full_song_model_time_s * 0.10,
        "seconds_needed_for_500_tps": full_song_model_time_s - (full_song_main_tokens / 500.0),
    }
    return {
        "pass": bool(report.get("pass", False)),
        "source_metadata": report.get("metadata", {}),
        "source_prompt_tokens": report.get("prompt_tokens"),
        "source_active_prefix_length": report.get("active_prefix_length"),
        "full_song_decode_steps": int(full_song_decode_steps),
        "full_song_main_tokens": int(full_song_main_tokens),
        "full_song_model_time_s": float(full_song_model_time_s),
        "dtype_bytes": int(dtype_bytes),
        "summary": summary,
        "operation_totals": {
            key: dict(value)
            for key, value in sorted(
                op_totals.items(),
                key=lambda item: item[1].get("projected_full_song_seconds", 0.0),
                reverse=True,
            )
        },
        "linear_rows": rows,
        "notes": [
            "Diagnostic-only roofline summary; not an inference speed claim.",
            "Minimum bytes include weight, input, output, and optional bias bytes once per linear call.",
            "Peak-bandwidth floor is a fantasy lower bound; real kernels may be slower because of launch, cache, layout, and dependency costs.",
        ],
    }


def _print_summary(summary: dict[str, Any], *, limit: int) -> None:
    root = summary["summary"]
    print("Decode Linear Roofline Summary")
    print(f"  pass: {summary['pass']}")
    print(f"  active_prefix_length: {summary['source_active_prefix_length']}")
    print(f"  linear projected: {root['linear_projected_full_song_seconds']:.3f}s")
    print(f"  peak bandwidth floor: {root['peak_bandwidth_lower_bound_full_song_seconds']:.3f}s")
    print(f"  removable above floor: {root['removable_above_peak_bandwidth_floor_seconds']:.3f}s")
    print(f"  zero-linear TPS: {root['zero_linear_tps']:.1f}")
    print(f"  peak-floor TPS: {root['peak_bandwidth_floor_tps']:.1f}")
    print(f"  achieved min-byte bandwidth: {root['achieved_weighted_min_bytes_gb_s']:.1f} GB/s")
    print()
    print("Top Signatures")
    for row in summary["linear_rows"][:limit]:
        print(
            f"  {row['projected_full_song_seconds']:7.3f}s "
            f"floor={row['peak_bandwidth_lower_bound_full_song_seconds']:6.3f}s "
            f"{row['achieved_min_bytes_gb_s']:7.1f} GB/s "
            f"members={row['member_count']:2d} "
            f"{','.join(row['operation_kinds'])} "
            f"{row['signature']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize one-token decoder linear roofline from profile_decode_linear_kernels JSON."
    )
    parser.add_argument("report", type=Path, help="profile_decode_linear_kernels JSON report.")
    parser.add_argument("--full-song-decode-steps", type=int, default=7552)
    parser.add_argument("--full-song-main-tokens", type=int, default=7639)
    parser.add_argument("--full-song-model-time-s", type=float, default=28.243)
    parser.add_argument("--dtype-bytes", type=int, default=4)
    parser.add_argument(
        "--peak-memory-bandwidth-gb-s",
        type=float,
        default=616.0,
        help="Theoretical memory bandwidth used for a lower-bound floor. Default is RTX 2080 Ti nominal bandwidth.",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    summary = summarize_linear_roofline(
        report,
        full_song_decode_steps=args.full_song_decode_steps,
        full_song_main_tokens=args.full_song_main_tokens,
        full_song_model_time_s=args.full_song_model_time_s,
        dtype_bytes=args.dtype_bytes,
        peak_memory_bandwidth_gb_s=args.peak_memory_bandwidth_gb_s,
    )
    _print_summary(summary, limit=args.limit)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
