"""Fail-loud validation that a profile did or did not execute the K4 runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LABELS = ("timing_context", "main_generation")
RNG_POLICY = "counter_request_seed_window_prompt_v2"


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def validate(payload: Any, *, role: str, block_size: int) -> dict[str, Any]:
    root = _object(payload, name="profile")
    if role not in {"baseline", "candidate"}:
        raise ValueError("role must be baseline or candidate")
    if isinstance(block_size, bool) or block_size <= 1:
        raise ValueError("block_size must be an integer greater than one")
    generation = root.get("generation")
    if not isinstance(generation, list) or not generation:
        raise ValueError("profile.generation must be a non-empty list")

    totals = {
        label: {"records": 0, "block_replays": 0, "eligible_steps": 0}
        for label in LABELS
    }
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
        candidate = graphs.get("k8_candidate")
        if role == "baseline":
            if candidate is not None:
                raise ValueError(f"baseline generation[{index}] contains K4 metadata")
            continue

        candidate = _object(
            candidate,
            name=f"generation[{index}].optimized_cuda_graphs.k8_candidate",
        )
        if candidate.get("block_size") != block_size:
            raise ValueError(f"candidate generation[{index}] has wrong block_size")
        if candidate.get("rng_policy") != RNG_POLICY:
            raise ValueError(f"candidate generation[{index}] has wrong RNG policy")
        if candidate.get("rng_exact") is not False:
            raise ValueError(f"candidate generation[{index}] must declare RNG drift")
        if candidate.get("rng_early_eos_isolation") is not True:
            raise ValueError(
                f"candidate generation[{index}] lacks early-EOS RNG isolation"
            )
        if candidate.get("capture_state_restore_synchronized") is not True:
            raise ValueError(
                f"candidate generation[{index}] lacks synchronized capture restore"
            )

        values = {
            key: _nonnegative_int(
                candidate.get(key),
                name=f"generation[{index}].k8_candidate.{key}",
            )
            for key in (
                "prefill_steps",
                "eligible_steps",
                "block_replays",
                "remainder_steps",
                "physical_steps",
                "logical_steps",
                "wasted_steps",
            )
        }
        if values["eligible_steps"] != values["block_replays"] * block_size:
            raise ValueError(f"candidate generation[{index}] block accounting diverged")
        if values["physical_steps"] != (
            values["logical_steps"] + values["wasted_steps"]
        ):
            raise ValueError(
                f"candidate generation[{index}] physical/logical accounting diverged"
            )
        if values["physical_steps"] != (
            values["prefill_steps"]
            + values["eligible_steps"]
            + values["remainder_steps"]
        ):
            raise ValueError(f"candidate generation[{index}] work accounting diverged")
        totals[label]["block_replays"] += values["block_replays"]
        totals[label]["eligible_steps"] += values["eligible_steps"]

    for label, values in totals.items():
        if values["records"] <= 0:
            raise ValueError(f"profile is missing {label} records")
        if role == "candidate" and (
            values["block_replays"] <= 0 or values["eligible_steps"] <= 0
        ):
            raise ValueError(f"candidate {label} did not execute K4 blocks")

    return {
        "schema_version": 1,
        "role": role,
        "block_size": block_size,
        "rng_policy": RNG_POLICY if role == "candidate" else None,
        "labels": totals,
        "pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--role", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--block-size", type=int, default=4)
    parsed = parser.parse_args()
    report = validate(
        json.loads(parsed.profile.read_text(encoding="utf-8")),
        role=parsed.role,
        block_size=parsed.block_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
