"""Validate and summarize the real-prefix decoder-prefill graph scout."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
EXPECTED_PREFILLS = 87
REQUIRED_SAVING_SECONDS = 1.503
FP16_VARIANTS = frozenset({"exact_fp16_graph", "bucket64_fp16_graph"})
FP32_VARIANTS = frozenset({"exact_fp32_graph", "bucket64_fp32_graph"})
VARIANTS = tuple(sorted(FP16_VARIANTS | FP32_VARIANTS))
EXPECTED_BUCKET_MODES = {
    "exact_fp16_graph": "exact",
    "exact_fp32_graph": "exact",
    "bucket64_fp16_graph": "bucket64",
    "bucket64_fp32_graph": "bucket64",
}


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _number(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0 or (positive and parsed <= 0):
        qualifier = "positive " if positive else "non-negative "
        raise ValueError(f"{name} must be a finite {qualifier}number")
    return parsed


def _parse_records(value: Any, *, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != EXPECTED_PREFILLS:
        raise ValueError(f"{name} must contain exactly {EXPECTED_PREFILLS} records")
    parsed: list[dict[str, Any]] = []
    indexes: set[int] = set()
    for offset, raw in enumerate(value):
        record = _object(raw, name=f"{name}[{offset}]")
        index = record.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"{name}[{offset}].index must be a non-negative integer")
        indexes.add(index)
        parsed.append(record)
    if indexes != set(range(EXPECTED_PREFILLS)):
        raise ValueError(f"{name} indexes must be exactly 0..{EXPECTED_PREFILLS - 1}")
    return sorted(parsed, key=lambda record: int(record["index"]))


def summarize(payload: Any) -> dict[str, Any]:
    root = _object(payload, name="input")
    if root.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")

    metadata = _object(root.get("metadata"), name="metadata")
    baseline_main = _number(
        metadata.get("baseline_main_seconds"),
        name="metadata.baseline_main_seconds",
        positive=True,
    )
    baseline_request = _number(
        metadata.get("baseline_request_seconds"),
        name="metadata.baseline_request_seconds",
        positive=True,
    )
    if metadata.get("captured_prefills") != EXPECTED_PREFILLS:
        raise ValueError(
            f"metadata.captured_prefills must equal {EXPECTED_PREFILLS}"
        )
    eager = _object(root.get("eager_fp32"), name="eager_fp32")
    eager_records = _parse_records(eager.get("records"), name="eager_fp32.records")
    baseline_prefill = sum(
        _number(
            record.get("execution_ms"),
            name=f"eager_fp32.records[{record['index']}].execution_ms",
            positive=True,
        )
        for record in eager_records
    ) / 1_000.0

    variants = _object(root.get("variants"), name="variants")
    if set(variants) != set(VARIANTS):
        raise ValueError(f"variants must be exactly {list(VARIANTS)}")

    reports: dict[str, Any] = {}
    any_promotion = False
    for name in VARIANTS:
        variant = _object(variants[name], name=f"variants.{name}")
        bucket_mode = variant.get("bucket_mode")
        if bucket_mode != EXPECTED_BUCKET_MODES[name]:
            raise ValueError(
                f"variants.{name}.bucket_mode must be "
                f"{EXPECTED_BUCKET_MODES[name]!r}"
            )
        graph_count = variant.get("graph_count")
        if (
            isinstance(graph_count, bool)
            or not isinstance(graph_count, int)
            or graph_count <= 0
        ):
            raise ValueError(f"variants.{name}.graph_count must be a positive integer")
        peak_allocated_bytes = variant.get("peak_allocated_bytes")
        retained_graph_bytes = variant.get("estimated_retained_graph_bytes")
        for field, value in (
            ("peak_allocated_bytes", peak_allocated_bytes),
            ("estimated_retained_graph_bytes", retained_graph_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"variants.{name}.{field} must be a non-negative integer"
                )
        records = _parse_records(
            variant.get("records"),
            name=f"variants.{name}.records",
        )
        setup_seconds = _number(
            variant.get("capture_setup_seconds"),
            name=f"variants.{name}.capture_setup_seconds",
        ) + _number(
            variant.get("dtype_setup_seconds", 0.0),
            name=f"variants.{name}.dtype_setup_seconds",
        )
        candidate_replay = sum(
            _number(
                record.get("replay_ms"),
                name=f"variants.{name}.records[{record['index']}].replay_ms",
                positive=True,
            )
            + _number(
                record.get("copy_ms"),
                name=f"variants.{name}.records[{record['index']}].copy_ms",
            )
            for record in records
        ) / 1_000.0
        deterministic = all(record.get("repeat_deterministic") is True for record in records)
        same_precision_graph_exact = all(
            record.get("same_precision_graph_exact") is True for record in records
        )
        fp32_reference_exact = all(
            record.get("fp32_reference_exact") is True for record in records
        )
        finite = all(record.get("finite") is True for record in records)
        cache_valid = all(record.get("cache_valid") is True for record in records)
        correctness_pass = (
            deterministic
            and same_precision_graph_exact
            and finite
            and cache_valid
            and (fp32_reference_exact if name in FP32_VARIANTS else True)
        )
        warm_saving = baseline_prefill - candidate_replay
        cold_saving = warm_saving - setup_seconds
        projected_main = baseline_main - cold_saving
        projected_request = baseline_request - cold_saving
        performance_pass = cold_saving >= REQUIRED_SAVING_SECONDS
        promotion_pass = correctness_pass and performance_pass
        any_promotion |= promotion_pass
        max_drift = max(
            _number(
                record.get("fp32_max_abs_drift"),
                name=f"variants.{name}.records[{record['index']}].fp32_max_abs_drift",
            )
            for record in records
        )
        reports[name] = {
            "scope": "main_generation_decoder_prefill",
            "bucket_mode": bucket_mode,
            "graph_count": graph_count,
            "baseline_prefill_seconds": baseline_prefill,
            "copy_plus_replay_seconds": candidate_replay,
            "capture_plus_dtype_setup_seconds": setup_seconds,
            "warm_saving_seconds": warm_saving,
            "cold_request_saving_seconds": cold_saving,
            "projected_main_seconds": projected_main,
            "projected_request_seconds": projected_request,
            "deterministic": deterministic,
            "same_precision_graph_exact": same_precision_graph_exact,
            "fp32_reference_exact": fp32_reference_exact,
            "finite": finite,
            "cache_valid": cache_valid,
            "fp32_max_abs_drift": max_drift,
            "correctness_pass": correctness_pass,
            "performance_pass": performance_pass,
            "promotion_pass": promotion_pass,
            "peak_allocated_bytes": peak_allocated_bytes,
            "estimated_retained_graph_bytes": retained_graph_bytes,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "PROMOTE_PREFILL_SCOUT" if any_promotion else "STOP_PREFILL_SCOUT",
        "promotion_pass": any_promotion,
        "constants": {
            "expected_prefills": EXPECTED_PREFILLS,
            "required_saving_seconds": REQUIRED_SAVING_SECONDS,
            "baseline_main_seconds": baseline_main,
            "baseline_request_seconds": baseline_request,
        },
        "baseline_prefill_seconds": baseline_prefill,
        "variants": reports,
    }


def _text(report: dict[str, Any]) -> str:
    lines = [
        report["decision"],
        f"baseline_prefill_seconds={report['baseline_prefill_seconds']:.6f}",
    ]
    for name, variant in report["variants"].items():
        lines.append(
            f"{name}: cold_saving={variant['cold_request_saving_seconds']:.6f}s "
            f"warm_saving={variant['warm_saving_seconds']:.6f}s "
            f"correctness={variant['correctness_pass']} "
            f"promotion={variant['promotion_pass']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-text", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(json.loads(args.input.read_text()))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_text.write_text(_text(report))
    print(_text(report), end="")
    if not report["promotion_pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
