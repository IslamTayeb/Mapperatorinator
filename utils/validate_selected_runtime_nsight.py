"""Validate that every selected-stack Nsight arm exercised the retained topology."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from osuT5.osuT5.inference.optimized.kernels.weight_only_runtime import (
    CROSS_FP16_PACKED,
)
from utils.validate_int8_mlp_full_song_profile import validate_profile as validate_int8
from utils.validate_k1_remainder_profile import validate as validate_k1
from utils.validate_k4_profile_contract import validate as validate_k4
from utils.validate_weight_only_full_song_profile import (
    validate_profile as validate_weight_only,
)


RUN_IDS = ("selected_control", "selected_graph", "selected_node")
PASS_KINDS = {
    "selected_control": "untraced_control",
    "selected_graph": "nsys_graph",
    "selected_node": "nsys_node",
}
RUNTIME_NAME = "k4-k1-int8-fp16-packed-cross"
RUNTIME_FACTORY = (
    "utils.final_confirmation_runtime:kblock_shared_rope_weight_plugin"
)
INITIALIZER = "initialize_approximate_int8_mlp_weight_only_cross"
EXPECTED_MAIN_STEPS = 8294


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _load(path: Path) -> dict[str, Any]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), name=str(path))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_equal(value: Any, expected: Any, *, name: str) -> None:
    if value != expected:
        raise ValueError(f"{name} must be {expected!r}, got {value!r}")


def _runtime_contract(evidence: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    _required_equal(evidence.get("candidate"), True, name=f"{run_id}.candidate")
    _required_equal(
        evidence.get("expected_main_steps"),
        EXPECTED_MAIN_STEPS,
        name=f"{run_id}.expected_main_steps",
    )
    main = _object(
        _object(evidence.get("labels"), name=f"{run_id}.labels").get(
            "main_generation"
        ),
        name=f"{run_id}.labels.main_generation",
    )
    _required_equal(
        main.get("logical_steps"),
        EXPECTED_MAIN_STEPS,
        name=f"{run_id}.main.logical_steps",
    )

    runtime = _object(evidence.get("runtime"), name=f"{run_id}.runtime")
    _required_equal(runtime.get("name"), RUNTIME_NAME, name=f"{run_id}.runtime.name")
    _required_equal(
        runtime.get("factory"), RUNTIME_FACTORY, name=f"{run_id}.runtime.factory"
    )
    _required_equal(
        runtime.get("binding_count"), 2, name=f"{run_id}.runtime.binding_count"
    )
    spec = _object(runtime.get("spec"), name=f"{run_id}.runtime.spec")
    kwargs = _object(spec.get("kwargs"), name=f"{run_id}.runtime.spec.kwargs")
    expected_kwargs = {
        "block_size": 4,
        "graph_remainders": True,
        "initializer_name": INITIALIZER,
        "initializer_kwargs": {"mode": CROSS_FP16_PACKED},
        "shared_rope_binding_index": 0,
        "initializer_binding_index": 0,
        "minimum_bindings": 2,
    }
    _required_equal(kwargs, expected_kwargs, name=f"{run_id}.runtime.spec.kwargs")

    initialization = _object(
        runtime.get("initialization"), name=f"{run_id}.runtime.initialization"
    )
    cross = _object(
        initialization.get("cross_candidate"),
        name=f"{run_id}.runtime.initialization.cross_candidate",
    )
    for key, expected in (
        ("mode", CROSS_FP16_PACKED),
        ("scope", "main-model-only"),
        ("attention_accumulation", "fp32"),
        ("production_selector_unchanged", True),
        ("projection_delta_only", True),
        ("accepted_q1_bmm", True),
        ("incremental_exactness_required", True),
    ):
        _required_equal(cross.get(key), expected, name=f"{run_id}.cross.{key}")
    overlay = _object(
        initialization.get("int8_mlp_overlay"),
        name=f"{run_id}.runtime.initialization.int8_mlp_overlay",
    )
    _required_equal(
        overlay.get("dispatch_counter"),
        "int8_weight_mlp_tail",
        name=f"{run_id}.int8.dispatch_counter",
    )

    hooks = runtime.get("temporary_hooks")
    if not isinstance(hooks, list) or len(hooks) != 3:
        raise ValueError(f"{run_id}.runtime.temporary_hooks must contain three hooks")
    by_kind = {
        _object(item, name=f"{run_id}.runtime.temporary_hooks[]").get("kind"): item
        for item in hooks
    }
    block = _object(by_kind.get("block_decode"), name=f"{run_id}.block_decode")
    _required_equal(block.get("block_size"), 4, name=f"{run_id}.block_size")
    _required_equal(
        block.get("graph_remainders"), True, name=f"{run_id}.graph_remainders"
    )
    initializer = _object(
        by_kind.get("runtime_initializer"), name=f"{run_id}.runtime_initializer"
    )
    for key, expected in (
        ("binding_index", 0),
        ("name", INITIALIZER),
        ("kwargs", {"mode": CROSS_FP16_PACKED}),
    ):
        _required_equal(initializer.get(key), expected, name=f"{run_id}.initializer.{key}")
    shared = _object(
        by_kind.get("shared_decoder_rope"), name=f"{run_id}.shared_decoder_rope"
    )
    _required_equal(shared.get("binding_index"), 0, name=f"{run_id}.rope.binding")
    stats = _object(shared.get("stats"), name=f"{run_id}.rope.stats")
    forwards = _positive_int(stats.get("forwards"), name=f"{run_id}.rope.forwards")
    reuses = _positive_int(stats.get("reuses"), name=f"{run_id}.rope.reuses")
    _required_equal(
        stats.get("computes"), stats.get("expected_computes"), name=f"{run_id}.rope.computes"
    )
    _required_equal(reuses, stats.get("expected_reuses"), name=f"{run_id}.rope.reuses")
    _positive_int(stats.get("module_count"), name=f"{run_id}.rope.module_count")
    _positive_int(stats.get("group_count"), name=f"{run_id}.rope.group_count")

    return {
        "name": runtime["name"],
        "factory": runtime["factory"],
        "binding_count": runtime["binding_count"],
        "spec_kwargs": kwargs,
        "cross_candidate": cross,
        "int8_overlay": {
            "version": overlay.get("version"),
            "scope": overlay.get("scope"),
            "dispatch_counter": overlay.get("dispatch_counter"),
        },
        "shared_rope": {
            "module_count": stats["module_count"],
            "group_count": stats["group_count"],
            "forwards": forwards,
            "computes": stats["computes"],
            "reuses": reuses,
        },
    }


def _validate_one(path: Path, *, run_id: str) -> dict[str, Any]:
    evidence = _load(path)
    profile_path = Path(evidence.get("profile_path", ""))
    if not profile_path.is_file():
        raise FileNotFoundError(profile_path)
    _required_equal(
        _sha256(profile_path), evidence.get("profile_sha256"), name=f"{run_id}.profile_sha256"
    )
    profile = _load(profile_path)
    metadata = _object(profile.get("metadata"), name=f"{run_id}.profile.metadata")
    _required_equal(
        metadata.get("profile_pass_kind"),
        PASS_KINDS[run_id],
        name=f"{run_id}.profile_pass_kind",
    )
    runtime_evidence_path = path.parent / "runtime-evidence.json"
    _required_equal(
        _load(runtime_evidence_path),
        evidence.get("runtime"),
        name=f"{run_id}.runtime_evidence",
    )
    return {
        "runtime": _runtime_contract(evidence, run_id=run_id),
        "weight_only": validate_weight_only(
            profile,
            role="candidate",
            cross_mode=CROSS_FP16_PACKED,
        ),
        "int8_mlp": validate_int8(profile, role="candidate"),
        "k1_remainder": validate_k1(profile, role="candidate"),
        "k4": validate_k4(profile, role="candidate", block_size=4),
    }


def validate(evidence_paths: dict[str, Path]) -> dict[str, Any]:
    if set(evidence_paths) != set(RUN_IDS):
        raise ValueError(f"evidence paths must be exactly {list(RUN_IDS)}")
    runs = {
        run_id: _validate_one(evidence_paths[run_id], run_id=run_id)
        for run_id in RUN_IDS
    }
    reference = runs["selected_control"]
    for run_id in RUN_IDS[1:]:
        _required_equal(runs[run_id], reference, name=f"{run_id}.topology")
    return {
        "schema_version": "mapperatorinator.selected-runtime-nsight-topology.v1",
        "pass": True,
        "runtime_name": RUNTIME_NAME,
        "expected_main_steps": EXPECTED_MAIN_STEPS,
        "common_topology": reference,
        "runs": {run_id: {"evidence_path": str(evidence_paths[run_id].resolve())} for run_id in RUN_IDS},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-control-evidence", type=Path, required=True)
    parser.add_argument("--selected-graph-evidence", type=Path, required=True)
    parser.add_argument("--selected-node-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(
        {
            "selected_control": args.selected_control_evidence,
            "selected_graph": args.selected_graph_evidence,
            "selected_node": args.selected_node_evidence,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
