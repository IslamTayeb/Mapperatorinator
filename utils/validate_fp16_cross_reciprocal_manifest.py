"""Validate the four-arm incremental FP16-packed cross composition manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROLES = (
    "baseline_first",
    "candidate_first",
    "candidate_second",
    "baseline_second",
)
BASELINE_ROLES = frozenset(("baseline_first", "baseline_second"))
CANDIDATE_ROLES = frozenset(("candidate_first", "candidate_second"))
CONTROL_RUNTIME = "k4-split-kv-mixed-weight-shared-rope-k1-remainder-int8-mlp-v1"
SHARED_ROPE_VERSION = "shared-decoder-rope-v1"
INT8_COUNTER = "int8_weight_mlp_tail"
FP16_CROSS_COUNTER = "fp16_packed_cross_projection_candidate"
RNG_POLICY = "counter_request_seed_window_prompt_v2"


class CrossReciprocalManifestError(ValueError):
    pass


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CrossReciprocalManifestError(f"{name} must be an object")
    return value


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CrossReciprocalManifestError(f"{name} must be a positive integer")
    return value


def _load(path: Path) -> dict[str, Any]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), name=str(path))
    except (OSError, json.JSONDecodeError) as exc:
        raise CrossReciprocalManifestError(f"cannot read {path}: {exc}") from exc


def _required_equal(
    value: Any,
    expected: Any,
    *,
    name: str,
) -> None:
    if value != expected:
        raise CrossReciprocalManifestError(
            f"{name} must be {expected!r}, got {value!r}"
        )


def _validate_initialization(
    payload: dict[str, Any],
    *,
    role: str,
    cross_mode: str,
) -> dict[str, Any]:
    shared = _object(payload.get("shared_rope"), name=f"{role}.shared_rope")
    _required_equal(
        shared.get("version"), SHARED_ROPE_VERSION, name=f"{role}.shared_rope.version"
    )
    _required_equal(
        shared.get("scope"), "main-model-only", name=f"{role}.shared_rope.scope"
    )
    _required_equal(
        shared.get("incremental_exactness_claim"),
        True,
        name=f"{role}.shared_rope.incremental_exactness_claim",
    )
    _required_equal(
        shared.get("original_decoder_forward_required"),
        True,
        name=f"{role}.shared_rope.original_decoder_forward_required",
    )
    shared_stats = _object(shared.get("stats"), name=f"{role}.shared_rope.stats")
    forwards = _positive_int(
        shared_stats.get("forwards"), name=f"{role}.shared_rope.stats.forwards"
    )
    reuses = _positive_int(
        shared_stats.get("reuses"), name=f"{role}.shared_rope.stats.reuses"
    )
    _required_equal(
        shared_stats.get("computes"),
        shared_stats.get("expected_computes"),
        name=f"{role}.shared_rope.stats.computes",
    )
    _required_equal(
        reuses,
        shared_stats.get("expected_reuses"),
        name=f"{role}.shared_rope.stats.reuses",
    )

    overlay = _object(
        payload.get("int8_mlp_overlay"), name=f"{role}.int8_mlp_overlay"
    )
    _required_equal(
        overlay.get("dispatch_counter"),
        INT8_COUNTER,
        name=f"{role}.int8_mlp_overlay.dispatch_counter",
    )

    cross = _object(payload.get("cross_candidate"), name=f"{role}.cross_candidate")
    expected_mode = "accepted" if role in BASELINE_ROLES else cross_mode
    for key, expected in (
        ("mode", expected_mode),
        ("accepted_q1_bmm", True),
        ("attention_accumulation", "fp32"),
        ("production_selector_unchanged", True),
    ):
        _required_equal(cross.get(key), expected, name=f"{role}.cross_candidate.{key}")

    expected_runtime = (
        CONTROL_RUNTIME
        if role in BASELINE_ROLES
        else f"{CONTROL_RUNTIME.removesuffix('-v1')}-{cross_mode}-v1"
    )
    _required_equal(
        payload.get("combined_runtime"),
        expected_runtime,
        name=f"{role}.combined_runtime",
    )

    cross_runtime = payload.get("cross_runtime")
    if role in BASELINE_ROLES:
        if cross_runtime is not None:
            raise CrossReciprocalManifestError(
                f"{role}.cross_runtime must be absent from the control"
            )
    else:
        cross_runtime = _object(cross_runtime, name=f"{role}.cross_runtime")
        for key, expected in (
            ("mode", cross_mode),
            ("incremental_control", CONTROL_RUNTIME),
            ("incremental_exactness_required", True),
            ("packed_projection_delta_only", True),
            ("accepted_q1_bmm_required", True),
            ("original_decoder_forward_required", True),
        ):
            _required_equal(
                cross_runtime.get(key), expected, name=f"{role}.cross_runtime.{key}"
            )
    return {"shared_rope_forwards": forwards, "shared_rope_reuses": reuses}


def _validate_reports(
    reports: dict[str, dict[str, Any]],
    *,
    role: str,
    cross_mode: str,
) -> dict[str, Any]:
    weight = reports["weight"]
    for key, expected in (
        ("role", "candidate"),
        ("candidate_enabled_for_main", True),
        ("candidate_disabled_for_timing", True),
    ):
        _required_equal(weight.get(key), expected, name=f"{role}.weight.{key}")
    expected_mode = "accepted" if role in BASELINE_ROLES else cross_mode
    _required_equal(
        weight.get("cross_mode"), expected_mode, name=f"{role}.weight.cross_mode"
    )
    mixed = _object(
        weight.get("main_weight_only_dispatch_counts"), name=f"{role}.weight.mixed"
    )
    self_count = _positive_int(
        mixed.get("weight_only_self_attention_block"),
        name=f"{role}.weight.mixed.self",
    )
    mlp_count = _positive_int(
        mixed.get("weight_only_mlp_tail"), name=f"{role}.weight.mixed.mlp"
    )
    _positive_int(
        mixed.get("weight_only_final_projection"),
        name=f"{role}.weight.mixed.final",
    )
    q1_bmm_count = _positive_int(
        weight.get("main_q1_bmm_cross_attention_count"),
        name=f"{role}.weight.q1_bmm",
    )
    _required_equal(q1_bmm_count, mlp_count, name=f"{role}.weight.q1_bmm")

    self_attention = _object(
        weight.get("main_effective_self_attention_counts"),
        name=f"{role}.weight.self_attention",
    )
    _required_equal(
        self_attention.get("native_q1_rope_cache_self_attention"),
        self_count,
        name=f"{role}.weight.self_attention.native_q1",
    )
    split_count = _positive_int(
        self_attention.get("native_q1_rope_cache_self_attention_split_kv_8"),
        name=f"{role}.weight.self_attention.split_kv",
    )
    fallback_count = _positive_int(
        self_attention.get("accepted_fallback"),
        name=f"{role}.weight.self_attention.accepted_fallback",
    )
    _required_equal(
        split_count + fallback_count,
        self_count,
        name=f"{role}.weight.self_attention.total",
    )
    cross_counts = _object(
        weight.get("main_cross_candidate_dispatch_counts"),
        name=f"{role}.weight.cross_counts",
    )
    expected_cross_count = 0 if role in BASELINE_ROLES else q1_bmm_count
    _required_equal(
        cross_counts.get(FP16_CROSS_COUNTER),
        expected_cross_count,
        name=f"{role}.weight.cross_counts.{FP16_CROSS_COUNTER}",
    )

    int8 = reports["int8"]
    for key, expected in (
        ("role", "candidate"),
        ("overlay_present", True),
        ("pass", True),
    ):
        _required_equal(int8.get(key), expected, name=f"{role}.int8.{key}")
    int8_dispatch = _object(
        int8.get("dispatch_totals"), name=f"{role}.int8.dispatch_totals"
    )
    int8_base = _object(
        int8.get("mixed_mlp_hook_totals"), name=f"{role}.int8.mixed_mlp"
    )
    _required_equal(int8_dispatch.get("timing_context"), 0, name=f"{role}.int8.timing")
    _required_equal(
        int8_dispatch.get("main_generation"), mlp_count, name=f"{role}.int8.main"
    )
    _required_equal(int8_dispatch, int8_base, name=f"{role}.int8.dispatch_ownership")

    k1 = reports["k1"]
    for key, expected in (
        ("role", "candidate"),
        ("remainder_backend", "cuda_graph"),
        ("pass", True),
    ):
        _required_equal(k1.get(key), expected, name=f"{role}.k1.{key}")
    k1_labels = _object(k1.get("labels"), name=f"{role}.k1.labels")
    for label in ("timing_context", "main_generation"):
        values = _object(k1_labels.get(label), name=f"{role}.k1.{label}")
        steps = _positive_int(
            values.get("remainder_steps"), name=f"{role}.k1.{label}.steps"
        )
        _required_equal(
            values.get("remainder_graph_replays"),
            steps,
            name=f"{role}.k1.{label}.graph_replays",
        )
        _required_equal(
            values.get("eager_remainder_steps"),
            0,
            name=f"{role}.k1.{label}.eager_steps",
        )

    k4 = reports["k4"]
    for key, expected in (
        ("role", "candidate"),
        ("block_size", 4),
        ("rng_policy", RNG_POLICY),
        ("pass", True),
    ):
        _required_equal(k4.get(key), expected, name=f"{role}.k4.{key}")
    k4_labels = _object(k4.get("labels"), name=f"{role}.k4.labels")
    for label in ("timing_context", "main_generation"):
        values = _object(k4_labels.get(label), name=f"{role}.k4.{label}")
        _positive_int(values.get("block_replays"), name=f"{role}.k4.{label}.replays")
        _positive_int(values.get("eligible_steps"), name=f"{role}.k4.{label}.steps")

    return {
        "mixed_dispatches": mixed,
        "q1_bmm_cross_attention": q1_bmm_count,
        "self_attention": self_attention,
        "int8_dispatches": int8_dispatch,
        "k1_labels": k1_labels,
        "k4_labels": k4_labels,
    }


def validate_run_root(root: Path, *, cross_mode: str) -> dict[str, Any]:
    if cross_mode != "fp16_packed_projections":
        raise CrossReciprocalManifestError(
            "cross_mode must be fp16_packed_projections"
        )
    role_reports: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        initialization = _load(root / "candidate-initialization" / f"{role}.json")
        init_signature = _validate_initialization(
            initialization, role=role, cross_mode=cross_mode
        )
        reports = {
            "weight": _load(root / f"{role}.weight-only-validation.json"),
            "int8": _load(root / f"{role}.int8-mlp-validation.json"),
            "k1": _load(root / f"{role}.k1-remainder-validation.json"),
            "k4": _load(root / f"{role}.k4-validation.json"),
        }
        role_reports[role] = {
            "initialization": init_signature,
            "topology": _validate_reports(
                reports, role=role, cross_mode=cross_mode
            ),
            "cross_mode": "accepted" if role in BASELINE_ROLES else cross_mode,
        }

    reference = role_reports["baseline_first"]["topology"]
    reference_initialization = role_reports["baseline_first"]["initialization"]
    common_fields = (
        "mixed_dispatches",
        "q1_bmm_cross_attention",
        "self_attention",
        "int8_dispatches",
        "k1_labels",
        "k4_labels",
    )
    for role in ROLES[1:]:
        _required_equal(
            role_reports[role]["initialization"],
            reference_initialization,
            name=f"{role}.initialization",
        )
        topology = role_reports[role]["topology"]
        for field in common_fields:
            _required_equal(
                topology[field], reference[field], name=f"{role}.topology.{field}"
            )

    return {
        "schema_version": 1,
        "pass": True,
        "cross_mode": cross_mode,
        "incremental_delta": "fp16-packed-cross-projections-only",
        "common_topology": reference,
        "roles": role_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--cross-mode", choices=("fp16_packed_projections",), required=True
    )
    args = parser.parse_args()
    print(
        json.dumps(
            validate_run_root(args.run_root, cross_mode=args.cross_mode),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
