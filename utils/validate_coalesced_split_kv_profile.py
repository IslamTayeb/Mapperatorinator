"""Validate the incremental full-song coalesced split-KV dispatch contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROFILE_LABELS = ("timing_context", "main_generation")
ROLES = ("baseline", "candidate")
GENERIC = "native_q1_rope_cache_self_attention"
SPLIT = f"{GENERIC}_split_kv_8"
COALESCED = f"{SPLIT}_coalesced"
SPLIT_PREFIX = f"{SPLIT}_prefix_"
PREFIX = f"{COALESCED}_prefix_"
SUPPORTED_PREFIXES = frozenset(range(192, 833, 64))


class CoalescedProfileError(ValueError):
    pass


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CoalescedProfileError(f"{name} must be an object")
    return value


def validate_profile(payload: Any, *, role: str) -> dict[str, Any]:
    if role not in ROLES:
        raise CoalescedProfileError(f"role must be one of {ROLES}")
    root = _object(payload, name="profile")
    if root.get("schema_version") != 1:
        raise CoalescedProfileError("profile.schema_version must be 1")
    generation = root.get("generation")
    if not isinstance(generation, list) or not generation:
        raise CoalescedProfileError("profile.generation must be non-empty")
    totals = {label: {GENERIC: 0, SPLIT: 0, COALESCED: 0} for label in PROFILE_LABELS}
    standard_prefix_totals: dict[int, int] = {}
    prefix_totals: dict[int, int] = {}
    seen = {label: False for label in PROFILE_LABELS}
    for index, raw in enumerate(generation):
        record = _object(raw, name=f"profile.generation[{index}]")
        label = record.get("profile_label")
        if label not in totals:
            continue
        seen[label] = True
        hits = _object(
            record.get("optimized_dispatch_capture_hits"),
            name=f"profile.generation[{index}].hits",
        )
        for key in (GENERIC, SPLIT, COALESCED):
            value = hits.get(key, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CoalescedProfileError(f"{label}.{key} must be non-negative int")
            totals[label][key] += value
        for key, value in hits.items():
            if key.startswith(SPLIT_PREFIX):
                suffix = key.removeprefix(SPLIT_PREFIX)
                try:
                    prefix = int(suffix)
                except ValueError as exc:
                    raise CoalescedProfileError(
                        f"malformed accepted split-KV prefix key {key}"
                    ) from exc
                if str(prefix) != suffix or prefix not in SUPPORTED_PREFIXES:
                    raise CoalescedProfileError(
                        f"unsupported accepted split-KV prefix {suffix}"
                    )
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise CoalescedProfileError(f"{key} must be non-negative int")
                standard_prefix_totals[prefix] = (
                    standard_prefix_totals.get(prefix, 0) + value
                )
                continue
            if not key.startswith(PREFIX):
                continue
            suffix = key.removeprefix(PREFIX)
            try:
                prefix = int(suffix)
            except ValueError as exc:
                raise CoalescedProfileError(f"malformed coalesced prefix key {key}") from exc
            if str(prefix) != suffix or prefix not in SUPPORTED_PREFIXES:
                raise CoalescedProfileError(f"unsupported coalesced prefix {suffix}")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CoalescedProfileError(f"{key} must be non-negative int")
            prefix_totals[prefix] = prefix_totals.get(prefix, 0) + value
    missing = [label for label, present in seen.items() if not present]
    if missing:
        raise CoalescedProfileError(f"profile is missing labels {missing}")
    if totals["timing_context"][COALESCED] != 0:
        raise CoalescedProfileError("timing generation must not execute coalesced split-KV")
    main = totals["main_generation"]
    if main[SPLIT] != sum(standard_prefix_totals.values()):
        raise CoalescedProfileError(
            "accepted split-KV aggregate does not equal per-prefix counts"
        )
    if role == "baseline":
        if main[COALESCED] != 0 or prefix_totals:
            raise CoalescedProfileError("baseline unexpectedly executed coalesced split-KV")
    else:
        if main[COALESCED] <= 0:
            raise CoalescedProfileError("candidate did not execute coalesced split-KV")
        if main[COALESCED] != sum(prefix_totals.values()):
            raise CoalescedProfileError(
                "coalesced split-KV aggregate does not equal per-prefix counts"
            )
        if not standard_prefix_totals or prefix_totals != standard_prefix_totals:
            raise CoalescedProfileError(
                "candidate did not execute every live eligible split-KV prefix"
            )
        if main[COALESCED] != main[SPLIT]:
            raise CoalescedProfileError(
                "candidate did not replace every eligible split-KV dispatch"
            )
        if main[GENERIC] - main[SPLIT] <= 0:
            raise CoalescedProfileError("candidate did not retain accepted short fallback")
    return {
        "role": role,
        "timing_counts": totals["timing_context"],
        "main_counts": main,
        "accepted_split_prefix_counts": dict(sorted(standard_prefix_totals.items())),
        "coalesced_prefix_counts": dict(sorted(prefix_totals.items())),
        "pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    cli = parser.parse_args()
    try:
        payload = json.loads(cli.profile.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read profile: {exc}") from exc
    print(json.dumps(validate_profile(payload, role=cli.role), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
