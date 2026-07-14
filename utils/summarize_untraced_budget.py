"""Validate and aggregate an opt-in untraced inference budget profile."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


BUDGET_SCHEMA_VERSION = 1
RECONCILIATION_LIMIT_FRACTION = 0.02
REGION_NAMES = (
    "prefill",
    "graph_input_copies",
    "graph_capture",
    "decode_replay",
    "cache_update",
    "logits_processors",
    "sampling",
    "append_stopping",
    "sync",
)
COPY_DIRECTIONS = ("h2d", "d2h", "d2d", "h2h", "other")
EXTERNAL_TRANSFER_NAMES = (
    "model_input_to_device",
    "model_input_dtype_conversion",
    "final_device_to_host",
)
CUDA_EVENT_REGION_NAMES = frozenset(
    {
        "prefill",
        "graph_input_copies",
        "decode_replay",
        "cache_update",
        "logits_processors",
        "sampling",
        "append_stopping",
    }
)


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _number(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        qualifier = "positive finite" if positive else "finite non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _count(value: Any, *, name: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0 or (positive and value <= 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def _parse_region(value: Any, *, name: str) -> dict[str, Any]:
    raw = _object(value, name=name)
    count_by_direction = _object(
        raw.get("copy_count_by_direction"),
        name=f"{name}.copy_count_by_direction",
    )
    bytes_by_direction = _object(
        raw.get("copy_bytes_by_direction"),
        name=f"{name}.copy_bytes_by_direction",
    )
    if set(count_by_direction) != set(COPY_DIRECTIONS):
        raise ValueError(f"{name}.copy_count_by_direction has invalid directions")
    if set(bytes_by_direction) != set(COPY_DIRECTIONS):
        raise ValueError(f"{name}.copy_bytes_by_direction has invalid directions")
    parsed = {
        "calls": _count(raw.get("calls"), name=f"{name}.calls"),
        "host_wall_seconds": _number(
            raw.get("host_wall_seconds"),
            name=f"{name}.host_wall_seconds",
        ),
        "cuda_event_samples": _count(
            raw.get("cuda_event_samples"),
            name=f"{name}.cuda_event_samples",
        ),
        "cuda_event_seconds": _number(
            raw.get("cuda_event_seconds"),
            name=f"{name}.cuda_event_seconds",
        ),
        "copy_count": _count(raw.get("copy_count"), name=f"{name}.copy_count"),
        "copy_bytes": _count(raw.get("copy_bytes"), name=f"{name}.copy_bytes"),
        "copy_count_by_direction": {
            direction: _count(
                count_by_direction[direction],
                name=f"{name}.copy_count_by_direction.{direction}",
            )
            for direction in COPY_DIRECTIONS
        },
        "copy_bytes_by_direction": {
            direction: _count(
                bytes_by_direction[direction],
                name=f"{name}.copy_bytes_by_direction.{direction}",
            )
            for direction in COPY_DIRECTIONS
        },
    }
    if parsed["copy_count"] != sum(parsed["copy_count_by_direction"].values()):
        raise ValueError(f"{name}.copy_count does not match direction totals")
    if parsed["copy_bytes"] != sum(parsed["copy_bytes_by_direction"].values()):
        raise ValueError(f"{name}.copy_bytes does not match direction totals")
    if parsed["cuda_event_samples"] > parsed["calls"]:
        raise ValueError(f"{name}.cuda_event_samples exceeds calls")
    return parsed


def _parse_external_transfer(value: Any, *, name: str) -> dict[str, Any]:
    raw = _object(value, name=name)
    # External transfers do not own CUDA events but otherwise share the copy
    # accounting schema with measured model regions.
    return _parse_region(
        {
            **raw,
            "cuda_event_samples": 0,
            "cuda_event_seconds": 0.0,
        },
        name=name,
    )


def _parse_budget(
    value: Any,
    *,
    name: str,
    expected_generated_tokens: int,
) -> dict[str, Any]:
    raw = _object(value, name=name)
    if raw.get("schema_version") != BUDGET_SCHEMA_VERSION:
        raise ValueError(f"{name}.schema_version must be {BUDGET_SCHEMA_VERSION}")
    generated_tokens = _count(
        raw.get("generated_tokens"),
        name=f"{name}.generated_tokens",
        positive=True,
    )
    if generated_tokens != expected_generated_tokens:
        raise ValueError(
            f"{name}.generated_tokens does not match generation record"
        )
    regions_raw = _object(raw.get("regions"), name=f"{name}.regions")
    if set(regions_raw) != set(REGION_NAMES):
        raise ValueError(f"{name}.regions must contain the complete budget schema")
    regions = {
        region: _parse_region(
            regions_raw[region],
            name=f"{name}.regions.{region}",
        )
        for region in REGION_NAMES
    }
    cuda_available = raw.get("cuda_available")
    if not isinstance(cuda_available, bool):
        raise ValueError(f"{name}.cuda_available must be boolean")
    for region_name, region in regions.items():
        expected_samples = (
            region["calls"]
            if cuda_available and region_name in CUDA_EVENT_REGION_NAMES
            else 0
        )
        if region["cuda_event_samples"] != expected_samples:
            raise ValueError(
                f"{name}.regions.{region_name}.cuda_event_samples must be "
                f"{expected_samples}"
            )
    external_raw = _object(
        raw.get("external_transfers"),
        name=f"{name}.external_transfers",
    )
    if set(external_raw) != set(EXTERNAL_TRANSFER_NAMES):
        raise ValueError(
            f"{name}.external_transfers must contain the complete schema"
        )
    external_transfers = {
        transfer_name: _parse_external_transfer(
            external_raw[transfer_name],
            name=f"{name}.external_transfers.{transfer_name}",
        )
        for transfer_name in EXTERNAL_TRANSFER_NAMES
    }
    model_seconds = _number(
        raw.get("model_elapsed_seconds"),
        name=f"{name}.model_elapsed_seconds",
        positive=True,
    )
    measured_host_seconds = sum(
        region["host_wall_seconds"] for region in regions.values()
    )
    host_unattributed_seconds = _number(
        raw.get("host_unattributed_seconds"),
        name=f"{name}.host_unattributed_seconds",
    )
    accounted_host_seconds = measured_host_seconds + host_unattributed_seconds
    error_seconds = abs(model_seconds - accounted_host_seconds)
    error_fraction = error_seconds / model_seconds
    cuda_event_measured_seconds = sum(
        region["cuda_event_seconds"] for region in regions.values()
    )
    for field, computed in (
        ("measured_host_seconds", measured_host_seconds),
        ("accounted_host_seconds", accounted_host_seconds),
        ("reconciliation_error_seconds", error_seconds),
        ("reconciliation_error_fraction", error_fraction),
        ("cuda_event_measured_seconds", cuda_event_measured_seconds),
    ):
        if not _close(
            _number(raw.get(field), name=f"{name}.{field}"),
            computed,
        ):
            raise ValueError(f"{name}.{field} does not match recomputed value")
    limit = _number(
        raw.get("reconciliation_limit_fraction"),
        name=f"{name}.reconciliation_limit_fraction",
        positive=True,
    )
    if not _close(limit, RECONCILIATION_LIMIT_FRACTION):
        raise ValueError(f"{name} changed the reconciliation limit")
    reconciliation_pass = error_fraction <= limit
    if raw.get("reconciliation_pass") is not reconciliation_pass:
        raise ValueError(f"{name}.reconciliation_pass is inconsistent")
    return {
        "cuda_available": cuda_available,
        "generated_tokens": generated_tokens,
        "model_elapsed_seconds": model_seconds,
        "regions": regions,
        "external_transfers": external_transfers,
        "host_unattributed_seconds": host_unattributed_seconds,
        "reconciliation_error_seconds": error_seconds,
        "reconciliation_error_fraction": error_fraction,
        "reconciliation_pass": reconciliation_pass,
    }


def _empty_aggregate() -> dict[str, Any]:
    return {
        "records": 0,
        "generated_tokens": 0,
        "model_elapsed_seconds": 0.0,
        "host_unattributed_seconds": 0.0,
        "regions": {
            name: {
                "calls": 0,
                "host_wall_seconds": 0.0,
                "cuda_event_samples": 0,
                "cuda_event_seconds": 0.0,
                "copy_count": 0,
                "copy_bytes": 0,
                "copy_count_by_direction": {
                    direction: 0 for direction in COPY_DIRECTIONS
                },
                "copy_bytes_by_direction": {
                    direction: 0 for direction in COPY_DIRECTIONS
                },
            }
            for name in REGION_NAMES
        },
        "external_transfers": {
            name: {
                "calls": 0,
                "host_wall_seconds": 0.0,
                "cuda_event_samples": 0,
                "cuda_event_seconds": 0.0,
                "copy_count": 0,
                "copy_bytes": 0,
                "copy_count_by_direction": {
                    direction: 0 for direction in COPY_DIRECTIONS
                },
                "copy_bytes_by_direction": {
                    direction: 0 for direction in COPY_DIRECTIONS
                },
            }
            for name in EXTERNAL_TRANSFER_NAMES
        },
        "window_reconciliation_pass": True,
    }


def _add_budget(target: dict[str, Any], budget: dict[str, Any]) -> None:
    target["records"] += 1
    target["generated_tokens"] += budget["generated_tokens"]
    target["model_elapsed_seconds"] += budget["model_elapsed_seconds"]
    target["host_unattributed_seconds"] += budget["host_unattributed_seconds"]
    target["window_reconciliation_pass"] &= budget["reconciliation_pass"]
    for name, source in budget["regions"].items():
        destination = target["regions"][name]
        for field in (
            "calls",
            "host_wall_seconds",
            "cuda_event_samples",
            "cuda_event_seconds",
            "copy_count",
            "copy_bytes",
        ):
            destination[field] += source[field]
        for direction in COPY_DIRECTIONS:
            destination["copy_count_by_direction"][direction] += source[
                "copy_count_by_direction"
            ][direction]
            destination["copy_bytes_by_direction"][direction] += source[
                "copy_bytes_by_direction"
            ][direction]
    for name, source in budget["external_transfers"].items():
        destination = target["external_transfers"][name]
        for field in (
            "calls",
            "host_wall_seconds",
            "copy_count",
            "copy_bytes",
        ):
            destination[field] += source[field]
        for direction in COPY_DIRECTIONS:
            destination["copy_count_by_direction"][direction] += source[
                "copy_count_by_direction"
            ][direction]
            destination["copy_bytes_by_direction"][direction] += source[
                "copy_bytes_by_direction"
            ][direction]


def _finish_aggregate(target: dict[str, Any]) -> dict[str, Any]:
    tokens = target["generated_tokens"]
    model_seconds = target["model_elapsed_seconds"]
    measured_host_seconds = sum(
        region["host_wall_seconds"] for region in target["regions"].values()
    )
    accounted_host_seconds = measured_host_seconds + target[
        "host_unattributed_seconds"
    ]
    error_seconds = abs(model_seconds - accounted_host_seconds)
    error_fraction = error_seconds / model_seconds if model_seconds > 0 else math.inf
    cuda_seconds = sum(
        region["cuda_event_seconds"] for region in target["regions"].values()
    )
    explicit_sync_wall_seconds = target["regions"]["sync"][
        "host_wall_seconds"
    ]
    cpu_region_wall_seconds = measured_host_seconds - explicit_sync_wall_seconds
    transition_count_by_direction = {
        direction: sum(
            region["copy_count_by_direction"][direction]
            for region in target["regions"].values()
        )
        for direction in COPY_DIRECTIONS
    }
    transition_bytes_by_direction = {
        direction: sum(
            region["copy_bytes_by_direction"][direction]
            for region in target["regions"].values()
        )
        for direction in COPY_DIRECTIONS
    }
    external_transfer_wall_seconds = sum(
        transfer["host_wall_seconds"]
        for transfer in target["external_transfers"].values()
    )
    target.update(
        {
            "measured_host_seconds": measured_host_seconds,
            "accounted_host_seconds": accounted_host_seconds,
            "cuda_event_measured_seconds": cuda_seconds,
            "cpu_region_wall_seconds": cpu_region_wall_seconds,
            "explicit_sync_wall_seconds": explicit_sync_wall_seconds,
            "explicit_sync_count": target["regions"]["sync"]["calls"],
            "transition_count_by_direction": transition_count_by_direction,
            "transition_bytes_by_direction": transition_bytes_by_direction,
            "external_transfer_wall_seconds": external_transfer_wall_seconds,
            "reconciliation_error_seconds": error_seconds,
            "reconciliation_error_fraction": error_fraction,
            "reconciliation_limit_fraction": RECONCILIATION_LIMIT_FRACTION,
            "reconciliation_pass": (
                target["window_reconciliation_pass"]
                and error_fraction <= RECONCILIATION_LIMIT_FRACTION
            ),
            "tokens_per_second": tokens / model_seconds,
            "per_token": {
                "model_elapsed_ms": model_seconds * 1000.0 / tokens,
                "measured_host_ms": measured_host_seconds * 1000.0 / tokens,
                "host_unattributed_ms": (
                    target["host_unattributed_seconds"] * 1000.0 / tokens
                ),
                "cuda_event_measured_ms": cuda_seconds * 1000.0 / tokens,
                "cpu_region_wall_ms": (
                    cpu_region_wall_seconds * 1000.0 / tokens
                ),
                "explicit_sync_wall_ms": (
                    explicit_sync_wall_seconds * 1000.0 / tokens
                ),
                "explicit_sync_count": (
                    target["regions"]["sync"]["calls"] / tokens
                ),
            },
        }
    )
    return target


def summarize(payload: Any) -> dict[str, Any]:
    root = _object(payload, name="profile")
    if root.get("schema_version") != 1:
        raise ValueError("profile.schema_version must be 1")
    metadata = _object(root.get("metadata"), name="profile.metadata")
    if metadata.get("profile_pass_kind") != "untraced_budget":
        raise ValueError("profile must use profile_pass_kind=untraced_budget")
    if metadata.get("profile_detail_ranges") is not False:
        raise ValueError("untraced budget profile must disable detail ranges")
    if metadata.get("profile_cuda_capture") is not False:
        raise ValueError("untraced budget profile must disable profiler capture")
    generation = root.get("generation")
    if not isinstance(generation, list) or not generation:
        raise ValueError("profile.generation must be a non-empty list")

    overall = _empty_aggregate()
    by_label: dict[str, dict[str, Any]] = {}
    windows = []
    for index, record_value in enumerate(generation):
        record = _object(record_value, name=f"profile.generation[{index}]")
        generated_tokens = _count(
            record.get("generated_tokens"),
            name=f"profile.generation[{index}].generated_tokens",
            positive=True,
        )
        budget = _parse_budget(
            record.get("untraced_budget"),
            name=f"profile.generation[{index}].untraced_budget",
            expected_generated_tokens=generated_tokens,
        )
        model_elapsed = _number(
            record.get("model_elapsed_seconds"),
            name=f"profile.generation[{index}].model_elapsed_seconds",
            positive=True,
        )
        if not _close(model_elapsed, budget["model_elapsed_seconds"]):
            raise ValueError(
                f"profile.generation[{index}] model time disagrees with budget"
            )
        label = str(record.get("profile_label") or "unknown")
        target = by_label.setdefault(label, _empty_aggregate())
        _add_budget(target, budget)
        _add_budget(overall, budget)
        windows.append(
            {
                "index": index,
                "profile_label": label,
                "context_type": record.get("context_type"),
                "generated_tokens": generated_tokens,
                "model_elapsed_seconds": model_elapsed,
                "reconciliation_error_fraction": budget[
                    "reconciliation_error_fraction"
                ],
                "reconciliation_pass": budget["reconciliation_pass"],
            }
        )

    result = {
        "schema_version": 1,
        "profile_pass_kind": "untraced_budget",
        "reconciliation_limit_fraction": RECONCILIATION_LIMIT_FRACTION,
        "overall": _finish_aggregate(overall),
        "by_label": {
            label: _finish_aggregate(target)
            for label, target in sorted(by_label.items())
        },
        "windows": windows,
    }
    result["pass"] = result["overall"]["reconciliation_pass"] and all(
        target["reconciliation_pass"] for target in result["by_label"].values()
    )
    return result


def add_control_comparison(
    report: dict[str, Any],
    control_payload: Any,
) -> dict[str, Any]:
    control = _object(control_payload, name="control_profile")
    metadata = _object(control.get("metadata"), name="control_profile.metadata")
    if metadata.get("profile_pass_kind") != "untraced_control":
        raise ValueError("control profile must use profile_pass_kind=untraced_control")
    summary = _object(control.get("summary"), name="control_profile.summary")
    by_label = _object(
        summary.get("generation_by_label"),
        name="control_profile.summary.generation_by_label",
    )
    if set(by_label) != set(report["by_label"]):
        raise ValueError("control and budget profile labels differ")

    comparisons: dict[str, Any] = {}
    control_total_seconds = 0.0
    control_total_tokens = 0
    control_total_records = 0
    for label, budget in report["by_label"].items():
        control_label = _object(
            by_label[label],
            name=f"control_profile.summary.generation_by_label.{label}",
        )
        control_seconds = _number(
            control_label.get("model_elapsed_seconds"),
            name=f"control_profile.summary.generation_by_label.{label}.model_elapsed_seconds",
            positive=True,
        )
        control_tokens = _count(
            control_label.get("generated_tokens"),
            name=f"control_profile.summary.generation_by_label.{label}.generated_tokens",
            positive=True,
        )
        control_records = _count(
            control_label.get("records"),
            name=f"control_profile.summary.generation_by_label.{label}.records",
            positive=True,
        )
        if control_tokens != budget["generated_tokens"]:
            raise ValueError(f"control and budget token counts differ for {label}")
        if control_records != budget["records"]:
            raise ValueError(f"control and budget record counts differ for {label}")
        delta = budget["model_elapsed_seconds"] - control_seconds
        comparisons[label] = {
            "records": control_records,
            "generated_tokens": control_tokens,
            "control_model_elapsed_seconds": control_seconds,
            "budget_model_elapsed_seconds": budget["model_elapsed_seconds"],
            "instrumentation_overhead_seconds": delta,
            "instrumentation_overhead_fraction": delta / control_seconds,
        }
        control_total_seconds += control_seconds
        control_total_tokens += control_tokens
        control_total_records += control_records

    budget_total = report["overall"]["model_elapsed_seconds"]
    report["paired_control"] = {
        "by_label": comparisons,
        "overall": {
            "records": control_total_records,
            "generated_tokens": control_total_tokens,
            "control_model_elapsed_seconds": control_total_seconds,
            "budget_model_elapsed_seconds": budget_total,
            "instrumentation_overhead_seconds": budget_total - control_total_seconds,
            "instrumentation_overhead_fraction": (
                (budget_total - control_total_seconds) / control_total_seconds
            ),
        },
    }
    return report


def _text_report(report: dict[str, Any]) -> str:
    overall = report["overall"]
    lines = [
        f"pass={str(report['pass']).lower()}",
        f"records={overall['records']}",
        f"generated_tokens={overall['generated_tokens']}",
        f"model_elapsed_seconds={overall['model_elapsed_seconds']:.9f}",
        f"tokens_per_second={overall['tokens_per_second']:.6f}",
        f"reconciliation_error_fraction={overall['reconciliation_error_fraction']:.9f}",
        f"host_unattributed_seconds={overall['host_unattributed_seconds']:.9f}",
        f"cuda_event_measured_seconds={overall['cuda_event_measured_seconds']:.9f}",
    ]
    for name in REGION_NAMES:
        region = overall["regions"][name]
        lines.append(
            f"region.{name}=calls:{region['calls']},"
            f"host_s:{region['host_wall_seconds']:.9f},"
            f"cuda_s:{region['cuda_event_seconds']:.9f},"
            f"copies:{region['copy_count']},bytes:{region['copy_bytes']}"
        )
    paired = report.get("paired_control")
    if isinstance(paired, dict) and isinstance(paired.get("overall"), dict):
        comparison = paired["overall"]
        lines.extend(
            (
                "control_model_elapsed_seconds="
                f"{comparison['control_model_elapsed_seconds']:.9f}",
                "instrumentation_overhead_seconds="
                f"{comparison['instrumentation_overhead_seconds']:.9f}",
                "instrumentation_overhead_fraction="
                f"{comparison['instrumentation_overhead_fraction']:.9f}",
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--control-profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path)
    args = parser.parse_args()

    report = summarize(json.loads(args.profile.read_text(encoding="utf-8")))
    if args.control_profile is not None:
        report = add_control_comparison(
            report,
            json.loads(args.control_profile.read_text(encoding="utf-8")),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(_text_report(report), encoding="utf-8")
    if not report["pass"]:
        print("STOP_UNTRACED_BUDGET_RECONCILIATION", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
