from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_profile(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _records_for_label(profile: dict[str, Any], label: str) -> list[dict[str, Any]]:
    return [record for record in profile.get("generation", []) if record.get("profile_label") == label]


def _add_numeric(target: dict[str, float], key: str, value: Any) -> None:
    if isinstance(value, (int, float)):
        target[key] = float(target.get(key, 0.0)) + float(value)


def _record_tokens(records: list[dict[str, Any]]) -> int:
    return sum(int(record.get("generated_tokens", 0) or 0) for record in records)


def _record_model_seconds(records: list[dict[str, Any]]) -> float:
    return sum(float(record.get("model_elapsed_seconds", 0.0) or 0.0) for record in records)


def _event_single_range_ceilings(
        *,
        cuda_event_ms: dict[str, float],
        cuda_event_calls: dict[str, int],
        generated_tokens: int,
        model_seconds: float,
        decode_steps: int,
) -> list[dict[str, Any]]:
    ceilings: list[dict[str, Any]] = []
    baseline_tps = generated_tokens / model_seconds if model_seconds > 0 else 0.0
    keep_bar_seconds = model_seconds * 0.05
    for name, milliseconds in cuda_event_ms.items():
        event_seconds = float(milliseconds) / 1000.0
        remaining_seconds = model_seconds - event_seconds
        estimated_tps = (
            generated_tokens / remaining_seconds
            if generated_tokens > 0 and remaining_seconds > 0
            else None
        )
        ceilings.append({
            "name": name,
            "event_seconds": event_seconds,
            "event_ms": float(milliseconds),
            "calls": int(cuda_event_calls.get(name, 0)),
            "per_decode_step_us": (
                float(milliseconds) * 1000.0 / decode_steps
                if decode_steps > 0
                else None
            ),
            "model_time_pct": (
                event_seconds / model_seconds * 100.0
                if model_seconds > 0
                else 0.0
            ),
            "estimated_tokens_per_second_if_removed": estimated_tps,
            "estimated_tokens_per_second_delta": (
                estimated_tps - baseline_tps
                if estimated_tps is not None
                else None
            ),
            "above_5pct_keep_bar": event_seconds >= keep_bar_seconds if model_seconds > 0 else False,
        })
    return sorted(ceilings, key=lambda value: value["event_seconds"], reverse=True)


def _normalized_graph_signature(graph: dict[str, Any]) -> tuple[Any, ...]:
    try:
        prefix = int(graph.get("active_prefix_length"))
    except (TypeError, ValueError):
        prefix = None
    shapes = graph.get("static_input_shapes") or {}
    normalized_shapes = []
    for key, value in shapes.items():
        try:
            shape = tuple(int(dim) for dim in value)
        except (TypeError, ValueError):
            shape = tuple(value) if isinstance(value, (list, tuple)) else (str(value),)
        normalized_shapes.append((str(key), shape))
    return prefix, tuple(sorted(normalized_shapes))


def summarize_active_prefix_diagnostics(profile: dict[str, Any], *, label: str) -> dict[str, Any]:
    records = _records_for_label(profile, label)
    records_with_diagnostics = [
        record for record in records if isinstance(record.get("active_prefix_decode_diagnostics"), dict)
    ]

    numeric_totals: dict[str, float] = {}
    processor_wall: dict[str, float] = {}
    processor_calls: dict[str, int] = {}
    cuda_event_ms: dict[str, float] = {}
    cuda_event_calls: dict[str, int] = {}
    bucket_lengths: Counter[int] = Counter()
    cuda_graph_by_prefix: dict[int, dict[str, float | int]] = defaultdict(
        lambda: {
            "graphs": 0,
            "capture_seconds": 0.0,
            "decode_replays": 0,
        }
    )
    cuda_graph_totals = {
        "records_with_cuda_graph": 0,
        "graphs": 0,
        "capture_seconds": 0.0,
        "decode_replays": 0,
    }
    normalized_graphs: dict[tuple[Any, ...], dict[str, float | int | Any]] = defaultdict(
        lambda: {
            "prefix": None,
            "graphs": 0,
            "capture_seconds": 0.0,
            "min_capture_seconds": None,
            "decode_replays": 0,
        }
    )

    for record in records_with_diagnostics:
        diagnostics = record.get("active_prefix_decode_diagnostics") or {}
        for key, value in diagnostics.items():
            _add_numeric(numeric_totals, key, value)

        for key, value in (diagnostics.get("cuda_event_ms") or {}).items():
            _add_numeric(cuda_event_ms, key, value)

        for key, value in (diagnostics.get("cuda_event_calls") or {}).items():
            if isinstance(value, int):
                cuda_event_calls[key] = int(cuda_event_calls.get(key, 0)) + value

        for key, value in (diagnostics.get("logits_processor_detail_wall_cpu_s") or {}).items():
            _add_numeric(processor_wall, key, value)

        for key, value in (diagnostics.get("logits_processor_detail_calls") or {}).items():
            if isinstance(value, int):
                processor_calls[key] = int(processor_calls.get(key, 0)) + value

        for length in diagnostics.get("bucket_lengths_seen") or []:
            try:
                bucket_lengths[int(length)] += 1
            except (TypeError, ValueError):
                continue

        cuda_graph = diagnostics.get("cuda_graph") or {}
        graphs = cuda_graph.get("graphs") or []
        if graphs:
            cuda_graph_totals["records_with_cuda_graph"] += 1
        for graph in graphs:
            try:
                prefix = int(graph.get("active_prefix_length"))
            except (TypeError, ValueError):
                continue
            capture_seconds = float(graph.get("capture_seconds", 0.0) or 0.0)
            decode_replays = int(graph.get("decode_replays", 0) or 0)
            bucket = cuda_graph_by_prefix[prefix]
            bucket["graphs"] = int(bucket["graphs"]) + 1
            bucket["capture_seconds"] = float(bucket["capture_seconds"]) + capture_seconds
            bucket["decode_replays"] = int(bucket["decode_replays"]) + decode_replays
            cuda_graph_totals["graphs"] += 1
            cuda_graph_totals["capture_seconds"] += capture_seconds
            cuda_graph_totals["decode_replays"] += decode_replays
            normalized_key = _normalized_graph_signature(graph)
            normalized = normalized_graphs[normalized_key]
            normalized["prefix"] = prefix
            normalized["graphs"] = int(normalized["graphs"]) + 1
            normalized["capture_seconds"] = float(normalized["capture_seconds"]) + capture_seconds
            normalized["decode_replays"] = int(normalized["decode_replays"]) + decode_replays
            current_min = normalized["min_capture_seconds"]
            if current_min is None or capture_seconds < float(current_min):
                normalized["min_capture_seconds"] = capture_seconds

    generated_tokens = _record_tokens(records)
    model_seconds = _record_model_seconds(records)
    decode_steps = int(numeric_totals.get("decode_steps", 0))
    first_capture_seconds = sum(
        float(value["min_capture_seconds"] or 0.0)
        for value in normalized_graphs.values()
    )
    duplicate_capture_seconds = max(float(cuda_graph_totals["capture_seconds"]) - first_capture_seconds, 0.0)
    model_seconds_without_duplicate_capture = model_seconds - duplicate_capture_seconds
    duplicate_by_prefix: dict[int, dict[str, float | int]] = defaultdict(
        lambda: {
            "normalized_graphs": 0,
            "graphs": 0,
            "duplicate_graphs": 0,
            "capture_seconds": 0.0,
            "first_capture_seconds": 0.0,
            "duplicate_capture_seconds": 0.0,
            "decode_replays": 0,
        }
    )
    for value in normalized_graphs.values():
        prefix = value.get("prefix")
        if prefix is None:
            continue
        bucket = duplicate_by_prefix[int(prefix)]
        graphs = int(value["graphs"])
        capture_seconds = float(value["capture_seconds"])
        first_seconds = float(value["min_capture_seconds"] or 0.0)
        bucket["normalized_graphs"] = int(bucket["normalized_graphs"]) + 1
        bucket["graphs"] = int(bucket["graphs"]) + graphs
        bucket["duplicate_graphs"] = int(bucket["duplicate_graphs"]) + max(graphs - 1, 0)
        bucket["capture_seconds"] = float(bucket["capture_seconds"]) + capture_seconds
        bucket["first_capture_seconds"] = float(bucket["first_capture_seconds"]) + first_seconds
        bucket["duplicate_capture_seconds"] = (
            float(bucket["duplicate_capture_seconds"]) + max(capture_seconds - first_seconds, 0.0)
        )
        bucket["decode_replays"] = int(bucket["decode_replays"]) + int(value["decode_replays"])

    return {
        "label": label,
        "records": len(records),
        "records_with_diagnostics": len(records_with_diagnostics),
        "generated_tokens": generated_tokens,
        "model_elapsed_seconds": model_seconds,
        "tokens_per_second": generated_tokens / model_seconds if model_seconds > 0 else 0.0,
        "decode_steps": decode_steps,
        "numeric_totals": dict(sorted(numeric_totals.items())),
        "wall_totals": {
            key: value for key, value in sorted(numeric_totals.items()) if key.endswith("_wall_cpu_s")
        },
        "processor_wall_cpu_s": dict(sorted(processor_wall.items(), key=lambda item: item[1], reverse=True)),
        "processor_calls": dict(sorted(processor_calls.items())),
        "cuda_event_ms": dict(sorted(cuda_event_ms.items())),
        "cuda_event_calls": dict(sorted(cuda_event_calls.items())),
        "cuda_event_ms_ranked": sorted(cuda_event_ms.items(), key=lambda item: item[1], reverse=True),
        "cuda_event_weighted_per_decode_step_us": {
            key: value * 1000.0 / decode_steps
            for key, value in sorted(cuda_event_ms.items())
        } if decode_steps > 0 else {},
        "cuda_event_single_range_ceilings": _event_single_range_ceilings(
            cuda_event_ms=cuda_event_ms,
            cuda_event_calls=cuda_event_calls,
            generated_tokens=generated_tokens,
            model_seconds=model_seconds,
            decode_steps=decode_steps,
        ),
        "bucket_lengths_seen_counts": {
            str(key): value for key, value in sorted(bucket_lengths.items())
        },
        "cuda_graph_totals": cuda_graph_totals,
        "cuda_graph_by_prefix": {
            str(key): value for key, value in sorted(cuda_graph_by_prefix.items())
        },
        "cuda_graph_duplicate_capture_ceiling": {
            "normalized_graphs": len(normalized_graphs),
            "graph_captures": int(cuda_graph_totals["graphs"]),
            "duplicate_graph_captures": max(int(cuda_graph_totals["graphs"]) - len(normalized_graphs), 0),
            "capture_seconds": float(cuda_graph_totals["capture_seconds"]),
            "first_capture_seconds": first_capture_seconds,
            "duplicate_capture_seconds": duplicate_capture_seconds,
            "duplicate_capture_model_pct": (
                duplicate_capture_seconds / model_seconds * 100.0 if model_seconds > 0 else 0.0
            ),
            "estimated_tokens_per_second_without_duplicate_capture": (
                generated_tokens / model_seconds_without_duplicate_capture
                if model_seconds_without_duplicate_capture > 0
                else 0.0
            ),
            "by_prefix": {
                str(key): value for key, value in sorted(duplicate_by_prefix.items())
            },
        },
    }


def _fmt_seconds(value: float) -> str:
    return f"{value:.6f}s"


def print_summary(summary: dict[str, Any], *, limit: int) -> None:
    print(f"Label: {summary['label']}")
    print(f"Records: {summary['records']} ({summary['records_with_diagnostics']} with diagnostics)")
    print(
        "Main aggregate: "
        f"{summary['generated_tokens']} tokens, "
        f"{summary['model_elapsed_seconds']:.3f}s model, "
        f"{summary['tokens_per_second']:.3f} tok/s"
    )
    print()

    wall_totals = sorted(
        summary["wall_totals"].items(),
        key=lambda item: item[1],
        reverse=True,
    )
    print("Diagnostic Wall Totals")
    for key, value in wall_totals[:limit]:
        print(f"  {_fmt_seconds(value)}  {key}")
    if not wall_totals:
        print("  n/a")
    print()

    print("Logits Processor Wall Totals")
    for key, value in list(summary["processor_wall_cpu_s"].items())[:limit]:
        calls = summary["processor_calls"].get(key, "n/a")
        print(f"  {_fmt_seconds(value)}  {key}  calls={calls}")
    if not summary["processor_wall_cpu_s"]:
        print("  n/a")
    print()

    print("CUDA Event Totals")
    for key, value in summary["cuda_event_ms_ranked"][:limit]:
        calls = summary["cuda_event_calls"].get(key, "n/a")
        per_decode_us = summary["cuda_event_weighted_per_decode_step_us"].get(key, 0.0)
        print(f"  {value:.3f}ms  {key}  {per_decode_us:.3f}us/decode-step  calls={calls}")
    if not summary["cuda_event_ms_ranked"]:
        print("  n/a")
    print()

    print("CUDA Event Single-Range Ceilings")
    for item in summary["cuda_event_single_range_ceilings"][:limit]:
        estimated_tps = item["estimated_tokens_per_second_if_removed"]
        estimated_tps_text = "n/a" if estimated_tps is None else f"{estimated_tps:.3f}"
        print(
            f"  {_fmt_seconds(item['event_seconds'])}  "
            f"{item['model_time_pct']:.3f}% model  "
            f"est_tok/s_if_removed={estimated_tps_text}  "
            f"calls={item['calls']}  "
            f"{item['name']}"
        )
    if not summary["cuda_event_single_range_ceilings"]:
        print("  n/a")
    print()

    cuda_graph_totals = summary["cuda_graph_totals"]
    print("CUDA Graph Totals")
    print(f"  records_with_cuda_graph: {cuda_graph_totals['records_with_cuda_graph']}")
    print(f"  graphs: {cuda_graph_totals['graphs']}")
    print(f"  capture_seconds: {_fmt_seconds(cuda_graph_totals['capture_seconds'])}")
    print(f"  decode_replays: {cuda_graph_totals['decode_replays']}")
    print()

    duplicate = summary["cuda_graph_duplicate_capture_ceiling"]
    print("CUDA Graph Duplicate Capture Ceiling")
    print(f"  normalized_graphs: {duplicate['normalized_graphs']}")
    print(f"  graph_captures: {duplicate['graph_captures']}")
    print(f"  duplicate_graph_captures: {duplicate['duplicate_graph_captures']}")
    print(f"  duplicate_capture_seconds: {_fmt_seconds(duplicate['duplicate_capture_seconds'])}")
    print(f"  duplicate_capture_model_pct: {duplicate['duplicate_capture_model_pct']:.3f}%")
    print(
        "  estimated_tok/s_without_duplicate_capture: "
        f"{duplicate['estimated_tokens_per_second_without_duplicate_capture']:.3f}"
    )
    print()

    print("CUDA Graphs By Prefix")
    for prefix, values in summary["cuda_graph_by_prefix"].items():
        print(
            f"  {prefix}: graphs={values['graphs']}, "
            f"capture={_fmt_seconds(values['capture_seconds'])}, "
            f"replays={values['decode_replays']}"
        )
    if not summary["cuda_graph_by_prefix"]:
        print("  n/a")
    print()

    print("Bucket Lengths Seen")
    for prefix, count in summary["bucket_lengths_seen_counts"].items():
        print(f"  {prefix}: {count} records")
    if not summary["bucket_lengths_seen_counts"]:
        print("  n/a")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize active-prefix decode diagnostics from a profile JSON.")
    parser.add_argument("profile", type=Path, help="Path to profile JSON.")
    parser.add_argument("--label", default="main_generation", help="Generation label to summarize.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum rows per printed section.")
    parser.add_argument("--json-output", type=Path, default=None, help="Write machine-readable summary JSON.")
    args = parser.parse_args()

    summary = summarize_active_prefix_diagnostics(_load_profile(args.profile), label=args.label)
    print_summary(summary, limit=args.limit)
    if args.json_output is not None:
        args.json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
