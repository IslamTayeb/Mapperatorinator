"""Summarize process-isolated batched super-timing reports."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__:
    from .profile_batched_super_timing_graph import compare_exact
else:
    from profile_batched_super_timing_graph import compare_exact


def _reports(root: Path, *, precision: str, phase: str) -> list[dict[str, Any]]:
    reports = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metadata = payload.get("metadata") or {}
        if (
            payload.get("schema_version") == 1
            and metadata.get("precision") == precision
            and metadata.get("phase") == phase
            and metadata.get("mode") in {"eager", "graph"}
        ):
            payload["_path"] = str(path)
            reports.append(payload)
    if not reports:
        raise ValueError(
            f"no {phase} {precision} batched super-timing reports under {root}"
        )
    return reports


def summarize(
    reports: list[dict[str, Any]],
    *,
    required_repetitions: int,
) -> dict[str, Any]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        metadata = report["metadata"]
        grouped[(int(metadata["batch_size"]), metadata["mode"])].append(report)

    batch_sizes = sorted({batch for batch, _ in grouped})
    variants: dict[str, Any] = {}
    eligible_graphs: list[tuple[int, float]] = []
    eager_medians: list[tuple[int, float]] = []
    for batch_size in batch_sizes:
        eager_rows = grouped.get((batch_size, "eager"), [])
        graph_rows = grouped.get((batch_size, "graph"), [])
        expected_repetitions = set(range(1, required_repetitions + 1))
        eager_by_repetition = {
            int(row["metadata"]["repetition"]): row for row in eager_rows
        }
        graph_by_repetition = {
            int(row["metadata"]["repetition"]): row for row in graph_rows
        }
        counts_pass = (
            len(eager_rows) == required_repetitions
            and len(graph_rows) == required_repetitions
            and set(eager_by_repetition) == expected_repetitions
            and set(graph_by_repetition) == expected_repetitions
        )
        eager = (
            [eager_by_repetition[index] for index in sorted(expected_repetitions)]
            if counts_pass
            else []
        )
        graph = (
            [graph_by_repetition[index] for index in sorted(expected_repetitions)]
            if counts_pass
            else []
        )
        order_pass = counts_pass and all(
            int(eager_row["metadata"].get("order", -1))
            == (1 if repetition % 2 else 2)
            and int(graph_row["metadata"].get("order", -1))
            == (2 if repetition % 2 else 1)
            for repetition, eager_row, graph_row in zip(
                sorted(expected_repetitions),
                eager,
                graph,
                strict=True,
            )
        )
        parity = [
            compare_exact(eager_row, graph_row)
            for eager_row, graph_row in zip(eager, graph, strict=True)
        ] if counts_pass else []
        eager_deterministic = bool(eager) and all(
            eager[0]["workload"]["windows"] == row["workload"]["windows"]
            and eager[0]["output"] == row["output"]
            for row in eager[1:]
        )
        graph_deterministic = bool(graph) and all(
            graph[0]["workload"]["windows"] == row["workload"]["windows"]
            and graph[0]["output"] == row["output"]
            for row in graph[1:]
        )
        eager_seconds = [
            float(row["timing"]["complete_super_timing_seconds"])
            for row in eager
        ]
        graph_seconds = [
            float(row["timing"]["complete_super_timing_seconds"])
            for row in graph
        ]
        eager_median = statistics.median(eager_seconds) if eager_seconds else None
        graph_median = statistics.median(graph_seconds) if graph_seconds else None
        eager_gates_pass = counts_pass and all(
            bool(row.get("gates", {}).get("pass")) for row in eager
        )
        gates_pass = counts_pass and all(
            bool(row.get("gates", {}).get("pass")) for row in eager + graph
        )
        exact_pass = bool(parity) and all(row["pass"] for row in parity)
        row = {
            "batch_size": batch_size,
            "required_repetitions": required_repetitions,
            "eager_report_paths": [item["_path"] for item in eager],
            "graph_report_paths": [item["_path"] for item in graph],
            "eager_complete_seconds": eager_seconds,
            "graph_complete_seconds": graph_seconds,
            "eager_median_seconds": eager_median,
            "graph_median_seconds": graph_median,
            "same_batch_speedup_pct": (
                (eager_median - graph_median) / eager_median * 100.0
                if eager_median and graph_median
                else None
            ),
            "parity": parity,
            "exact_pass": exact_pass,
            "eager_deterministic": eager_deterministic,
            "graph_deterministic": graph_deterministic,
            "reciprocal_order_pass": order_pass,
            "gates_pass": gates_pass,
        }
        row["survives"] = all((
            counts_pass,
            exact_pass,
            eager_deterministic,
            graph_deterministic,
            order_pass,
            gates_pass,
        ))
        variants[str(batch_size)] = row
        if all((
            eager_median is not None,
            eager_deterministic,
            eager_gates_pass,
            order_pass,
        )):
            eager_medians.append((batch_size, eager_median))
        if row["survives"] and graph_median is not None:
            eligible_graphs.append((batch_size, graph_median))

    fastest_eager = min(eager_medians, key=lambda item: item[1]) if eager_medians else None
    fastest_graph = min(eligible_graphs, key=lambda item: item[1]) if eligible_graphs else None
    global_speedup = None
    if fastest_eager is not None and fastest_graph is not None:
        global_speedup = (
            (fastest_eager[1] - fastest_graph[1])
            / fastest_eager[1]
            * 100.0
        )
    all_exact_pass = bool(variants) and all(
        row["exact_pass"] and row["gates_pass"]
        for row in variants.values()
    )
    return {
        "schema_version": 1,
        "required_repetitions": required_repetitions,
        "variants": variants,
        "fastest_eager": (
            {"batch_size": fastest_eager[0], "median_seconds": fastest_eager[1]}
            if fastest_eager is not None
            else None
        ),
        "fastest_graph": (
            {"batch_size": fastest_graph[0], "median_seconds": fastest_graph[1]}
            if fastest_graph is not None
            else None
        ),
        "global_complete_wall_speedup_pct": global_speedup,
        "promotion_threshold_pct": 5.0,
        "promotion_pass": (
            global_speedup is not None and global_speedup >= 5.0
        ),
        "winning_batch_size": fastest_graph[0] if fastest_graph is not None else None,
        "all_exact_pass": all_exact_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), required=True)
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--required-repetitions", type=int, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    cli = parser.parse_args()
    if cli.required_repetitions <= 0:
        raise ValueError("required repetitions must be positive")
    result = summarize(
        _reports(cli.report_root, precision=cli.precision, phase=cli.phase),
        required_repetitions=cli.required_repetitions,
    )
    cli.output_path.parent.mkdir(parents=True, exist_ok=True)
    cli.output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["promotion_pass"] and cli.phase == "full":
        raise SystemExit(4)


if __name__ == "__main__":
    main()
