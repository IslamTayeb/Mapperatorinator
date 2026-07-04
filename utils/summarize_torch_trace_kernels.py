from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_trace(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _duration_us(event: dict[str, Any]) -> float:
    value = event.get("dur", 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _normalize_kernel_name(name: str) -> str:
    normalized = re.sub(r"<.*", "<...>", name)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized[:240]


def _kernel_category(event: dict[str, Any]) -> str:
    category = str(event.get("cat", "")).lower()
    name = str(event.get("name", ""))
    lower = name.lower()

    if category == "gpu_memcpy" or "memcpy" in lower:
        return "copy_index_cat_memcpy"
    if category == "gpu_memset" or "memset" in lower:
        return "elementwise_reduce_gelu_fill"
    if "fmha" in lower or "flash" in lower or "scaled_dot_product" in lower:
        return "attention_fmha_sdpa"
    if "q1" in lower and "attention" in lower:
        return "attention_native_q1"
    if "attention" in lower and "kernel" in lower:
        return "attention_other"
    if "gemm" in lower or "gemv" in lower or "cublas" in lower or "sgemm" in lower:
        return "gemv_gemm_linear"
    if "layer_norm" in lower or "layernorm" in lower or "rms_norm" in lower:
        return "layernorm"
    if "sort" in lower or "radix" in lower or "softmax" in lower or "multinomial" in lower:
        return "sampling_sort_softmax"
    if "cat" in lower or "index" in lower or "gather" in lower or "scatter" in lower or "copy" in lower:
        return "copy_index_cat_memcpy"
    if (
            "elementwise" in lower
            or "gelu" in lower
            or "fill" in lower
            or "reduce" in lower
            or "where" in lower
            or "masked" in lower
            or "unary" in lower
            or "binary" in lower
            or "functor" in lower
    ):
        return "elementwise_reduce_gelu_fill"
    return "other_kernel"


def _is_gpu_work_event(event: dict[str, Any]) -> bool:
    return event.get("ph") == "X" and event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}


def summarize_trace(path: Path, *, top_k: int) -> dict[str, Any]:
    trace = _load_trace(path)
    events = trace.get("traceEvents", [])
    category_totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"duration_us": 0.0, "count": 0}
    )
    kernel_totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"duration_us": 0.0, "count": 0, "category": None}
    )
    raw_event_categories: dict[str, int] = defaultdict(int)

    total_gpu_work_us = 0.0
    gpu_work_events = 0
    for event in events:
        raw_event_categories[str(event.get("cat", ""))] += 1
        if not _is_gpu_work_event(event):
            continue
        duration = _duration_us(event)
        if duration <= 0:
            continue
        gpu_work_events += 1
        total_gpu_work_us += duration
        category = _kernel_category(event)
        category_totals[category]["duration_us"] += duration
        category_totals[category]["count"] += 1
        kernel_name = _normalize_kernel_name(str(event.get("name", "unknown")))
        kernel_totals[kernel_name]["duration_us"] += duration
        kernel_totals[kernel_name]["count"] += 1
        kernel_totals[kernel_name]["category"] = category

    categories = []
    for category, values in category_totals.items():
        duration_us = float(values["duration_us"])
        categories.append({
            "category": category,
            "duration_us": duration_us,
            "duration_ms": duration_us / 1000.0,
            "share_of_gpu_work": (
                duration_us / total_gpu_work_us
                if total_gpu_work_us > 0
                else None
            ),
            "count": int(values["count"]),
        })
    categories.sort(key=lambda item: item["duration_us"], reverse=True)

    top_kernels = []
    for name, values in kernel_totals.items():
        duration_us = float(values["duration_us"])
        top_kernels.append({
            "name": name,
            "category": values["category"],
            "duration_us": duration_us,
            "duration_ms": duration_us / 1000.0,
            "share_of_gpu_work": (
                duration_us / total_gpu_work_us
                if total_gpu_work_us > 0
                else None
            ),
            "count": int(values["count"]),
        })
    top_kernels.sort(key=lambda item: item["duration_us"], reverse=True)

    return {
        "trace_path": str(path),
        "trace_events": len(events) if isinstance(events, list) else None,
        "gpu_work_events": gpu_work_events,
        "gpu_work_duration_us": total_gpu_work_us,
        "gpu_work_duration_ms": total_gpu_work_us / 1000.0,
        "categories": categories,
        "top_kernels": top_kernels[:top_k],
        "raw_event_categories": dict(sorted(raw_event_categories.items())),
        "notes": [
            "Diagnostic-only torch profiler Chrome trace summary; not a throughput claim.",
            "Durations are Chrome-trace event durations for GPU kernel/memcpy/memset events.",
            "Categories are regex buckets; inspect top_kernels before making kernel-work decisions.",
        ],
    }


def _print_summary(summary: dict[str, Any], *, top_k: int) -> None:
    print("Torch Trace Kernel Summary")
    print(f"  trace: {summary['trace_path']}")
    print(f"  gpu work: {summary['gpu_work_duration_ms']:.3f} ms over {summary['gpu_work_events']} events")
    print()
    print("Categories")
    for row in summary["categories"]:
        share = row["share_of_gpu_work"]
        share_text = "n/a" if share is None else f"{share * 100.0:.1f}%"
        print(
            f"  {row['duration_ms']:10.3f} ms  "
            f"{share_text:>6}  "
            f"count={row['count']:7d}  "
            f"{row['category']}"
        )
    print()
    print(f"Top Kernels ({top_k})")
    for row in summary["top_kernels"][:top_k]:
        share = row["share_of_gpu_work"]
        share_text = "n/a" if share is None else f"{share * 100.0:.1f}%"
        print(
            f"  {row['duration_ms']:10.3f} ms  "
            f"{share_text:>6}  "
            f"count={row['count']:7d}  "
            f"{row['category']}  "
            f"{row['name']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize GPU kernel categories from a torch.profiler Chrome trace."
    )
    parser.add_argument("trace", type=Path, help="Path to torch.profiler Chrome trace JSON.")
    parser.add_argument("--top-k", type=int, default=30, help="Number of top kernels to include.")
    parser.add_argument("--json-output", type=Path, default=None, help="Write machine-readable summary JSON.")
    args = parser.parse_args()

    summary = summarize_trace(args.trace, top_k=args.top_k)
    _print_summary(summary, top_k=args.top_k)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
