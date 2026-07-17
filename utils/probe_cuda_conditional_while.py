"""Run a bounded CUDA conditional-WHILE capability and lifecycle probe."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from osuT5.osuT5.inference.optimized.single.conditional_while_scout import (  # noqa: E402
    ConditionalGraph,
    MINIMUM_WHILE_CUDA_VERSION,
)


SCHEMA_VERSION = 1


def _parse_limits(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("limits must be comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("limits must be positive")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("limits must not repeat")
    return values


def _validate_report(report: dict) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected conditional probe schema")
    metadata = report.get("metadata")
    probes = report.get("probes")
    if not isinstance(metadata, dict) or not isinstance(probes, list) or not probes:
        raise TypeError("conditional probe report is incomplete")
    if int(metadata["runtime_version"]) < MINIMUM_WHILE_CUDA_VERSION:
        raise RuntimeError("conditional probe used an unsupported CUDA runtime")
    if int(metadata["driver_version"]) < MINIMUM_WHILE_CUDA_VERSION:
        raise RuntimeError("conditional probe used an unsupported CUDA driver")
    for row in probes:
        if int(row["observed_counter"]) != int(row["limit"]):
            raise RuntimeError("conditional WHILE counter did not reach its limit")
        if int(row["replays"]) < 2:
            raise RuntimeError("probe must exercise launch-default reset")
        times = row.get("cuda_event_ms")
        if not isinstance(times, list) or len(times) != int(row["replays"]):
            raise ValueError("conditional probe timing count changed")
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in times):
            raise ValueError("conditional probe recorded an invalid timing")
        if not bool(row["memory_stable"]):
            raise RuntimeError("conditional probe allocated during replay")


@torch.no_grad()
def run_probe(
    *,
    limits: tuple[int, ...],
    replays: int,
    required_gpu_substring: str,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("conditional WHILE probe requires CUDA")
    if replays < 2:
        raise ValueError("replays must be at least two to test default reset")
    device = torch.device("cuda", torch.cuda.current_device())
    gpu_name = torch.cuda.get_device_name(device)
    if required_gpu_substring not in gpu_name:
        raise RuntimeError(
            f"expected GPU containing {required_gpu_substring!r}; got {gpu_name!r}"
        )
    rows = []
    versions = None
    for limit_value in limits:
        counter = torch.zeros((1,), dtype=torch.long, device=device)
        limit = torch.full((1,), limit_value, dtype=torch.long, device=device)
        graph = ConditionalGraph.counter_probe(counter, limit)
        try:
            current_versions = graph.versions()
            if versions is None:
                versions = current_versions
            elif current_versions != versions:
                raise RuntimeError("CUDA versions changed during one probe")
            event_times = []
            host_times = []
            allocated = []
            for _ in range(replays):
                counter.zero_()
                torch.cuda.synchronize(device)
                allocated.append(torch.cuda.memory_allocated(device))
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                host_start = time.perf_counter()
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                host_times.append(time.perf_counter() - host_start)
                event_times.append(float(start.elapsed_time(end)))
                if int(counter.item()) != limit_value:
                    raise RuntimeError(
                        f"WHILE counter expected {limit_value}, got {int(counter.item())}"
                    )
                allocated.append(torch.cuda.memory_allocated(device))
            rows.append({
                "limit": limit_value,
                "replays": replays,
                "observed_counter": int(counter.item()),
                "cuda_event_ms": event_times,
                "cuda_event_median_ms": statistics.median(event_times),
                "host_wall_seconds": host_times,
                "host_wall_median_seconds": statistics.median(host_times),
                "memory_allocated_bytes": allocated,
                "memory_stable": min(allocated) == max(allocated),
                "assign_default_reset_verified": True,
            })
        finally:
            graph.close()
    assert versions is not None
    report = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "result_class": "cuda-conditional-while-capability-probe",
            "production_wiring": False,
            "gpu": gpu_name,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "runtime_version": versions["runtime"],
            "driver_version": versions["driver"],
            "minimum_while_version": MINIMUM_WHILE_CUDA_VERSION,
            "conditional_type": "WHILE",
            "condition_owner": "device_kernel",
            "parent_launches_per_replay": 1,
        },
        "probes": rows,
        "decision": {
            "pass": True,
            "next_gate": "real-prefix fixed-work child-graph comparison",
        },
    }
    _validate_report(report)
    return report


def _text_report(report: dict) -> str:
    metadata = report["metadata"]
    lines = [
        "conditional_while_probe=PASS",
        f"gpu={metadata['gpu']}",
        f"runtime_version={metadata['runtime_version']}",
        f"driver_version={metadata['driver_version']}",
    ]
    for row in report["probes"]:
        lines.append(
            "limit={limit} replays={replays} counter={observed_counter} "
            "median_cuda_ms={cuda_event_median_ms:.6f} memory_stable={memory_stable}".format(
                **row
            )
        )
    lines.append(f"next_gate={report['decision']['next_gate']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limits", type=_parse_limits, default=(1, 3, 10))
    parser.add_argument("--replays", type=int, default=3)
    parser.add_argument("--required-gpu-substring", default="2080 Ti")
    args = parser.parse_args()
    report = run_probe(
        limits=args.limits,
        replays=args.replays,
        required_gpu_substring=args.required_gpu_substring,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "probe.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    text = _text_report(report)
    (args.output_dir / "probe.txt").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
