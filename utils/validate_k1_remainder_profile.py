"""Validate eager-control versus captured-K1 remainder ownership."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LABELS = ("timing_context", "main_generation")


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def validate(payload: Any, *, role: str) -> dict[str, Any]:
    root = _object(payload, name="profile")
    if role not in {"control", "candidate"}:
        raise ValueError("role must be control or candidate")
    generation = root.get("generation")
    if not isinstance(generation, list) or not generation:
        raise ValueError("profile.generation must be a non-empty list")

    totals = {
        label: {
            "records": 0,
            "remainder_steps": 0,
            "remainder_graph_replays": 0,
            "eager_remainder_steps": 0,
        }
        for label in LABELS
    }
    expected_backend = "eager" if role == "control" else "cuda_graph"
    for index, raw_record in enumerate(generation):
        record = _object(raw_record, name=f"generation[{index}]")
        label = record.get("profile_label")
        if label not in totals:
            continue
        totals[label]["records"] += 1
        graphs = _object(
            record.get("optimized_cuda_graphs"),
            name=f"generation[{index}].optimized_cuda_graphs",
        )
        runtime = _object(
            graphs.get("k8_candidate"),
            name=f"generation[{index}].optimized_cuda_graphs.k8_candidate",
        )
        if runtime.get("block_size") != 4:
            raise ValueError(f"generation[{index}] must retain K4")
        if runtime.get("remainder_backend") != expected_backend:
            raise ValueError(
                f"generation[{index}] remainder backend must be {expected_backend}"
            )
        values = {
            key: _nonnegative_int(
                runtime.get(key),
                name=f"generation[{index}].k8_candidate.{key}",
            )
            for key in (
                "remainder_steps",
                "remainder_graph_replays",
                "eager_remainder_steps",
            )
        }
        if role == "control":
            if values["remainder_graph_replays"] != 0:
                raise ValueError(f"generation[{index}] control captured a remainder")
            if values["eager_remainder_steps"] != values["remainder_steps"]:
                raise ValueError(
                    f"generation[{index}] control remainder accounting diverged"
                )
        else:
            if values["eager_remainder_steps"] != 0:
                raise ValueError(f"generation[{index}] candidate used an eager remainder")
            if values["remainder_graph_replays"] != values["remainder_steps"]:
                raise ValueError(
                    f"generation[{index}] candidate remainder accounting diverged"
                )
        for key, value in values.items():
            totals[label][key] += value

    for label, values in totals.items():
        if values["records"] <= 0:
            raise ValueError(f"profile is missing {label} records")
        if values["remainder_steps"] <= 0:
            raise ValueError(f"{role} {label} executed no remainder steps")
        owned = (
            values["eager_remainder_steps"]
            if role == "control"
            else values["remainder_graph_replays"]
        )
        if owned != values["remainder_steps"]:
            raise ValueError(f"{role} {label} did not own every remainder step")

    return {
        "schema_version": 1,
        "role": role,
        "remainder_backend": expected_backend,
        "labels": totals,
        "pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--role", choices=("control", "candidate"), required=True)
    parsed = parser.parse_args()
    report = validate(
        json.loads(parsed.profile.read_text(encoding="utf-8")),
        role=parsed.role,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
