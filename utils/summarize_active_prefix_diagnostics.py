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


def summarize_active_prefix_diagnostics(profile: dict[str, Any], *, label: str) -> dict[str, Any]:
    records = _records_for_label(profile, label)
    records_with_diagnostics = [
        record for record in records if isinstance(record.get("active_prefix_decode_diagnostics"), dict)
    ]

    numeric_totals: dict[str, float] = {}
    processor_wall: dict[str, float] = {}
    processor_calls: dict[str, int] = {}
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

    for record in records_with_diagnostics:
        diagnostics = record.get("active_prefix_decode_diagnostics") or {}
        for key, value in diagnostics.items():
            _add_numeric(numeric_totals, key, value)

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

    generated_tokens = _record_tokens(records)
    model_seconds = _record_model_seconds(records)
    return {
        "label": label,
        "records": len(records),
        "records_with_diagnostics": len(records_with_diagnostics),
        "generated_tokens": generated_tokens,
        "model_elapsed_seconds": model_seconds,
        "tokens_per_second": generated_tokens / model_seconds if model_seconds > 0 else 0.0,
        "numeric_totals": dict(sorted(numeric_totals.items())),
        "wall_totals": {
            key: value for key, value in sorted(numeric_totals.items()) if key.endswith("_wall_cpu_s")
        },
        "processor_wall_cpu_s": dict(sorted(processor_wall.items(), key=lambda item: item[1], reverse=True)),
        "processor_calls": dict(sorted(processor_calls.items())),
        "bucket_lengths_seen_counts": {
            str(key): value for key, value in sorted(bucket_lengths.items())
        },
        "cuda_graph_totals": cuda_graph_totals,
        "cuda_graph_by_prefix": {
            str(key): value for key, value in sorted(cuda_graph_by_prefix.items())
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

    cuda_graph_totals = summary["cuda_graph_totals"]
    print("CUDA Graph Totals")
    print(f"  records_with_cuda_graph: {cuda_graph_totals['records_with_cuda_graph']}")
    print(f"  graphs: {cuda_graph_totals['graphs']}")
    print(f"  capture_seconds: {_fmt_seconds(cuda_graph_totals['capture_seconds'])}")
    print(f"  decode_replays: {cuda_graph_totals['decode_replays']}")
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
