"""Validate that both reciprocal arms retain one identical shared-RoPE stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


ROLES = (
    "baseline_first",
    "candidate_first",
    "candidate_second",
    "baseline_second",
)
COMPOSITION_VERSION = "k4-split-kv-mixed-weight-shared-rope-v1"
SHARED_ROPE_VERSION = "shared-decoder-rope-v1"


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _string_list(value: Any, *, name: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{name} must be a non-empty string list")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicates")
    return value


def _shared_material(payload: Any, *, name: str) -> dict[str, Any]:
    root = _object(payload, name=name)
    if root.get("result_class") != "documented-drift":
        raise ValueError(f"{name} must retain documented-drift result class")
    if root.get("exactness_claim") is not False:
        raise ValueError(f"{name} must not relabel the overall stack exact")
    if root.get("combined_runtime") != COMPOSITION_VERSION:
        raise ValueError(f"{name} has the wrong combined runtime")
    shared = _object(root.get("shared_rope"), name=f"{name}.shared_rope")
    if shared.get("version") != SHARED_ROPE_VERSION:
        raise ValueError(f"{name} has the wrong shared-RoPE version")
    if shared.get("scope") != "main-model-only":
        raise ValueError(f"{name} has the wrong shared-RoPE scope")
    if shared.get("incremental_exactness_claim") is not True:
        raise ValueError(f"{name} lost the shared-RoPE incremental exactness claim")
    if shared.get("original_decoder_forward_required") is not True:
        raise ValueError(f"{name} no longer requires the original decoder forward")

    stats = _object(shared.get("stats"), name=f"{name}.shared_rope.stats")
    module_count = _positive_int(stats.get("module_count"), name="module_count")
    group_count = _positive_int(stats.get("group_count"), name="group_count")
    forwards = _positive_int(stats.get("forwards"), name="forwards")
    computes = _positive_int(stats.get("computes"), name="computes")
    reuses = _positive_int(stats.get("reuses"), name="reuses")
    eliminated = _positive_int(
        stats.get("eliminated_per_forward"),
        name="eliminated_per_forward",
    )
    if module_count <= group_count or eliminated != module_count - group_count:
        raise ValueError(f"{name} shared-RoPE topology cannot eliminate work")
    if computes != forwards * group_count or stats.get("expected_computes") != computes:
        raise ValueError(f"{name} shared-RoPE compute accounting diverged")
    if reuses != forwards * eliminated or stats.get("expected_reuses") != reuses:
        raise ValueError(f"{name} shared-RoPE reuse accounting diverged")

    members = _string_list(stats.get("member_names"), name="member_names")
    if len(members) != module_count:
        raise ValueError(f"{name} shared-RoPE member count diverged")
    group_members = _object(stats.get("group_members"), name="group_members")
    group_computes = _object(stats.get("group_computes"), name="group_computes")
    if len(group_members) != group_count or set(group_members) != set(group_computes):
        raise ValueError(f"{name} shared-RoPE group topology diverged")
    flattened: list[str] = []
    for group_id, raw_names in sorted(group_members.items()):
        names = _string_list(raw_names, name=f"group_members.{group_id}")
        flattened.extend(names)
        if group_computes[group_id] != forwards:
            raise ValueError(f"{name} shared-RoPE group compute count diverged")
    if sorted(flattened) != sorted(members) or len(flattened) != len(set(flattened)):
        raise ValueError(f"{name} shared-RoPE group membership diverged")
    return {
        "combined_runtime": root["combined_runtime"],
        "shared_rope": shared,
    }


def validate(payloads: Mapping[str, Any]) -> dict[str, Any]:
    if set(payloads) != set(ROLES):
        raise ValueError(f"payload roles must be exactly {list(ROLES)}")
    material = {
        role: _shared_material(payloads[role], name=role) for role in ROLES
    }
    reference = material["baseline_first"]
    divergent = [role for role in ROLES[1:] if material[role] != reference]
    if divergent:
        raise ValueError(
            "shared-RoPE topology/execution differs across reciprocal arms: "
            f"{divergent}"
        )
    return {
        "schema_version": 1,
        "roles": list(ROLES),
        "composition_version": COMPOSITION_VERSION,
        "shared_rope_version": SHARED_ROPE_VERSION,
        "material": reference,
        "pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for role in ROLES:
        parser.add_argument(
            f"--{role.replace('_', '-')}-initialization",
            type=Path,
            required=True,
        )
    parsed = parser.parse_args()
    payloads = {
        role: json.loads(
            getattr(parsed, f"{role}_initialization").read_text(encoding="utf-8")
        )
        for role in ROLES
    }
    print(json.dumps(validate(payloads), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
