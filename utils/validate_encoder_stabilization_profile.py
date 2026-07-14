"""Validate profiling evidence for the encoder stabilization copy cleanup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "mapperatorinator.encoder-stabilization-profile.v1"
COUNTER_KEYS = frozenset(
    {
        "calls",
        "skipped_copies",
        "skipped_identity_copies",
        "skipped_shared_storage_copies",
        "executed_copies",
        "executed_bytes",
        "holder_replacements",
    }
)


class EncoderStabilizationProfileError(ValueError):
    pass


def _object(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EncoderStabilizationProfileError(f"{name} must be an object")
    return value


def _counters(value: Any, *, name: str) -> dict[str, int]:
    raw = _object(value, name=name)
    if set(raw) != COUNTER_KEYS:
        raise EncoderStabilizationProfileError(
            f"{name} counter keys differ: expected {sorted(COUNTER_KEYS)}, "
            f"got {sorted(raw)}"
        )
    parsed: dict[str, int] = {}
    for key in sorted(COUNTER_KEYS):
        count = raw[key]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise EncoderStabilizationProfileError(
                f"{name}.{key} must be a non-negative integer"
            )
        parsed[key] = count
    if parsed["calls"] != parsed["skipped_copies"] + parsed["executed_copies"]:
        raise EncoderStabilizationProfileError(
            f"{name} calls must equal skipped_copies + executed_copies"
        )
    if parsed["skipped_copies"] != (
        parsed["skipped_identity_copies"]
        + parsed["skipped_shared_storage_copies"]
    ):
        raise EncoderStabilizationProfileError(
            f"{name} skipped copies must be identity or exact is_set_to storage"
        )
    if parsed["holder_replacements"] > parsed["executed_copies"]:
        raise EncoderStabilizationProfileError(
            f"{name} holder replacements exceed executed copies"
        )
    if parsed["executed_copies"] > 0 and parsed["executed_bytes"] <= 0:
        raise EncoderStabilizationProfileError(
            f"{name} executed copies require a positive byte count"
        )
    return parsed


def validate_profile(payload: Any) -> dict[str, Any]:
    root = _object(payload, name="profile")
    if root.get("schema_version") != 1:
        raise EncoderStabilizationProfileError("profile schema_version must be 1")
    metadata = _object(root.get("metadata"), name="profile.metadata")
    if metadata.get("profile_pass_kind") != "untraced_control":
        raise EncoderStabilizationProfileError(
            "encoder stabilization evidence must be an untraced control"
        )
    if metadata.get("authoritative_performance") is not True:
        raise EncoderStabilizationProfileError(
            "encoder stabilization evidence must be authoritative"
        )
    if metadata.get("profile_detail_ranges") is not False:
        raise EncoderStabilizationProfileError(
            "encoder stabilization evidence must not enable detail ranges"
        )
    if metadata.get("profile_cuda_capture") is not False:
        raise EncoderStabilizationProfileError(
            "encoder stabilization evidence must not enable profiler capture"
        )

    generation = root.get("generation")
    if not isinstance(generation, list) or not generation:
        raise EncoderStabilizationProfileError("profile.generation must be non-empty")
    totals = {key: 0 for key in sorted(COUNTER_KEYS)}
    records = 0
    labels: dict[str, int] = {}
    for index, raw_record in enumerate(generation):
        record = _object(raw_record, name=f"profile.generation[{index}]")
        if record.get("decoder_loop_backend") != "active_prefix_cuda_graph":
            continue
        counters = _counters(
            record.get("encoder_stabilization"),
            name=f"profile.generation[{index}].encoder_stabilization",
        )
        records += 1
        label = str(record.get("profile_label"))
        labels[label] = labels.get(label, 0) + 1
        for key, value in counters.items():
            totals[key] += value

    if records == 0:
        raise EncoderStabilizationProfileError(
            "profile contains no active-prefix encoder stabilization records"
        )
    if set(labels) != {"timing_context", "main_generation"}:
        raise EncoderStabilizationProfileError(
            "encoder stabilization evidence must cover timing_context and main_generation"
        )
    if totals["skipped_copies"] <= 0:
        raise EncoderStabilizationProfileError(
            "candidate did not skip any encoder stabilization copies"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "pass": True,
        "records": records,
        "labels": dict(sorted(labels.items())),
        "totals": totals,
        "skip_policy": {
            "identity_only": True,
            "shared_storage_requires_is_set_to": True,
            "other_skip_reasons": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.profile.read_text(encoding="utf-8"))
    report = validate_profile(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
