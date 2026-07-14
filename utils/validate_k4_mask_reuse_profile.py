"""Validate request-local decoder-mask reuse evidence in a K4 profile."""

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
    if role not in {"control", "candidate"}:
        raise ValueError("role must be control or candidate")
    root = _object(payload, name="profile")
    generation = root.get("generation")
    if not isinstance(generation, list) or not generation:
        raise ValueError("profile.generation must be a non-empty list")
    totals = {
        label: {
            "records": 0,
            "block_replays": 0,
            "sync_calls": 0,
            "owned_reuse_calls": 0,
            "external_copy_calls": 0,
            "reuse_opportunities": 0,
            "records_with_reuse_opportunity": 0,
            "records_without_reuse_opportunity": 0,
            "allocation_count_max": 0,
            "buffer_bytes_max": 0,
        }
        for label in LABELS
    }
    for index, raw_record in enumerate(generation):
        record = _object(raw_record, name=f"generation[{index}]")
        label = record.get("profile_label")
        if label not in totals:
            continue
        candidate = _object(
            _object(
                record.get("optimized_cuda_graphs"),
                name=f"generation[{index}].optimized_cuda_graphs",
            ).get("k8_candidate"),
            name=f"generation[{index}].optimized_cuda_graphs.k8_candidate",
        )
        if candidate.get("block_size") != 4:
            raise ValueError(f"generation[{index}] is not a K4 profile")
        block_replays = _nonnegative_int(
            candidate.get("block_replays"),
            name=f"generation[{index}].block_replays",
        )
        requested = candidate.get("decoder_attention_mask_reuse_requested")
        reuse = _object(
            candidate.get("decoder_attention_mask_reuse"),
            name=f"generation[{index}].decoder_attention_mask_reuse",
        )
        enabled = reuse.get("enabled")
        expected = role == "candidate"
        if requested is not expected or enabled is not expected:
            raise ValueError(
                f"generation[{index}] decoder mask reuse role mismatch"
            )
        row = totals[label]
        row["records"] += 1
        row["block_replays"] += block_replays
        if not expected:
            continue
        values = {
            key: _nonnegative_int(
                reuse.get(key),
                name=f"generation[{index}].decoder_attention_mask_reuse.{key}",
            )
            for key in (
                "allocation_count",
                "buffer_capacity_elements",
                "buffer_bytes",
                "active_length",
                "reset_calls",
                "sync_calls",
                "owned_reuse_calls",
                "external_copy_calls",
                "external_copy_bytes",
                "tail_fill_calls",
                "activated_elements",
            )
        }
        if values["allocation_count"] != 1:
            raise ValueError(f"generation[{index}] allocated more than one mask buffer")
        if values["buffer_capacity_elements"] <= 0 or values["buffer_bytes"] <= 0:
            raise ValueError(f"generation[{index}] has no reusable mask capacity")
        if values["active_length"] > values["buffer_capacity_elements"]:
            raise ValueError(f"generation[{index}] mask active length exceeds capacity")
        if values["reset_calls"] < 1:
            raise ValueError(f"generation[{index}] mask storage was not request-reset")
        if values["tail_fill_calls"] != 0:
            raise ValueError(f"generation[{index}] used a per-block mask fill")
        if values["sync_calls"] != block_replays:
            raise ValueError(f"generation[{index}] mask sync/block accounting diverged")
        if values["external_copy_calls"] > block_replays:
            raise ValueError(
                f"generation[{index}] copied more external masks than block handoffs"
            )
        reuse_opportunities = block_replays - values["external_copy_calls"]
        if values["owned_reuse_calls"] != reuse_opportunities:
            raise ValueError(f"generation[{index}] mask ownership accounting diverged")
        row["sync_calls"] += values["sync_calls"]
        row["owned_reuse_calls"] += values["owned_reuse_calls"]
        row["external_copy_calls"] += values["external_copy_calls"]
        row["reuse_opportunities"] += reuse_opportunities
        opportunity_key = (
            "records_with_reuse_opportunity"
            if reuse_opportunities > 0
            else "records_without_reuse_opportunity"
        )
        row[opportunity_key] += 1
        row["allocation_count_max"] = max(
            row["allocation_count_max"], values["allocation_count"]
        )
        row["buffer_bytes_max"] = max(
            row["buffer_bytes_max"], values["buffer_bytes"]
        )
    for label, row in totals.items():
        if row["records"] <= 0 or row["block_replays"] <= 0:
            raise ValueError(f"profile lacks K4 work for {label}")
        if role == "candidate" and row["sync_calls"] != row["block_replays"]:
            raise ValueError(f"candidate aggregate mask accounting diverged for {label}")
        if role == "candidate" and (
            row["owned_reuse_calls"] != row["reuse_opportunities"]
        ):
            raise ValueError(
                f"candidate aggregate reuse opportunities diverged for {label}"
            )
    return {
        "schema_version": 1,
        "role": role,
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
