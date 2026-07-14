"""Validate K1 timing, K4+K1-remainder main, INT8 MLP, and stage precision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.validate_int8_mlp_full_song_profile import validate_profile as validate_int8
from utils.validate_k4_profile_contract import validate as validate_k4
from utils.validate_stage_precision_hybrid_profile import validate as validate_stage


LABELS = ("timing_context", "main_generation")


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_remainder_ownership(payload: dict[str, Any]) -> dict[str, Any]:
    generation = payload.get("generation")
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
    for index, raw in enumerate(generation):
        record = _object(raw, name=f"generation[{index}]")
        label = record.get("profile_label")
        if label not in totals:
            continue
        graphs = _object(
            record.get("optimized_cuda_graphs"),
            name=f"generation[{index}].optimized_cuda_graphs",
        )
        runtime = _object(
            graphs.get("k8_candidate"),
            name=f"generation[{index}].optimized_cuda_graphs.k8_candidate",
        )
        expected_block = 1 if label == "timing_context" else 4
        if runtime.get("block_size") != expected_block:
            raise ValueError(f"generation[{index}] has wrong stage block size")
        if runtime.get("remainder_backend") != "cuda_graph":
            raise ValueError(f"generation[{index}] did not enable K1 remainder graphs")
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
        if values["eager_remainder_steps"] != 0:
            raise ValueError(f"generation[{index}] used an eager remainder")
        if values["remainder_graph_replays"] != values["remainder_steps"]:
            raise ValueError(f"generation[{index}] K1 remainder accounting diverged")
        totals[label]["records"] += 1
        for key, value in values.items():
            totals[label][key] += value

    for label, values in totals.items():
        if values["records"] <= 0:
            raise ValueError(f"profile is missing {label} records")
    main = totals["main_generation"]
    if main["remainder_steps"] <= 0:
        raise ValueError("main generation executed no captured K1 remainder steps")
    return totals


def validate(payload: Any, *, role: str) -> dict[str, Any]:
    if role not in {"control", "candidate"}:
        raise ValueError("role must be control or candidate")
    root = _object(payload, name="profile")
    stage = validate_stage(root, role=role)
    k4 = validate_k4(
        root,
        role="candidate",
        block_size=4,
        timing_block_size=1,
    )
    int8 = validate_int8(root, role="candidate")
    remainders = _validate_remainder_ownership(root)
    return {
        "schema_version": 1,
        "role": role,
        "stage_precision": stage,
        "decode_blocks": k4,
        "int8_mlp": int8,
        "remainder_ownership": remainders,
        "pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--role", choices=("control", "candidate"), required=True)
    args = parser.parse_args()
    report = validate(
        json.loads(args.profile.read_text(encoding="utf-8")),
        role=args.role,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
