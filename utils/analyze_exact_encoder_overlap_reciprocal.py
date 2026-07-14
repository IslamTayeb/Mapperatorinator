"""Analyze reciprocal full-song exact main-encoder overlap runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from osuT5.osuT5.inference.optimized.scout.encoder_overlap import (
    MATERIAL_AUDIT_VERSION,
    OVERLAP_VERSION,
)
from utils.run_k4_shared_rope_approximate_weight_only import COMPOSITION_VERSION
from utils.fixed_seed_inference import SEED_POLICY_VERSION, stage_seed
from utils.nsight_agent_profile import (
    _profile_graph_cache_signature,
    _profile_label_signature,
)
from utils.summarize_inference_profile import compare_reciprocal_profiles


LABELS = ("timing_context", "main_generation")


def _load(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _number(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result <= 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _stage(profile: dict[str, Any], name: str) -> float:
    rows = [
        row
        for row in profile.get("stages", [])
        if isinstance(row, dict) and row.get("name") == name
    ]
    if len(rows) != 1:
        raise ValueError(f"profile requires exactly one {name!r} stage")
    return _number(rows[0].get("wall_seconds"), name=f"stage.{name}", positive=True)


def _complete_wall(profile: dict[str, Any]) -> float:
    stages = [row for row in profile.get("stages", []) if isinstance(row, dict)]
    if not stages:
        raise ValueError("profile stages are empty")
    started = _number(
        stages[0].get("started_at_perf_counter_seconds"),
        name="first stage start",
    )
    finished = _number(
        stages[-1].get("finished_at_perf_counter_seconds"),
        name="last stage finish",
    )
    if finished <= started:
        raise ValueError("profile complete wall is not positive")
    return finished - started


def _records(profile: dict[str, Any], label: str) -> list[dict[str, Any]]:
    rows = [
        row
        for row in profile.get("generation", [])
        if isinstance(row, dict) and row.get("profile_label") == label
    ]
    if not rows:
        raise ValueError(f"profile lacks {label!r} generation records")
    return rows


def _label_metrics(profile: dict[str, Any], label: str) -> dict[str, Any]:
    rows = _records(profile, label)
    model_seconds = sum(
        _number(
            row.get("model_elapsed_seconds"),
            name=f"{label}.model_elapsed_seconds",
            positive=True,
        )
        for row in rows
    )
    outer_seconds = sum(
        _number(row.get("wall_seconds"), name=f"{label}.wall_seconds", positive=True)
        for row in rows
    )
    tokens = sum(int(row.get("generated_tokens", 0) or 0) for row in rows)
    return {
        "records": len(rows),
        "tokens": tokens,
        "model_seconds": model_seconds,
        "outer_seconds": outer_seconds,
        "tokens_per_second": tokens / model_seconds,
    }


def _profile_metrics(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "timing": _label_metrics(profile, "timing_context"),
        "main": _label_metrics(profile, "main_generation"),
        "timing_stage_seconds": _stage(profile, "timing_context_generation"),
        "main_stage_seconds": _stage(profile, "main_generation"),
        "complete_request_seconds": _complete_wall(profile),
    }


def _validate_seed(profile: dict[str, Any], *, base_seed: int) -> None:
    metadata = profile.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("profile metadata is missing")
    if metadata.get("reciprocal_seed_policy") != SEED_POLICY_VERSION:
        raise ValueError("profile reciprocal seed policy is missing")
    for label in LABELS:
        if metadata.get(f"reciprocal_seed_{label}") != stage_seed(base_seed, label):
            raise ValueError(f"profile reciprocal seed changed for {label}")


def _validate_candidate(
    profile: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    if evidence.get("mode") != "candidate":
        raise ValueError("candidate evidence mode changed")
    external = evidence.get("encoder_overlap")
    metadata = profile.get("metadata")
    internal = (
        metadata.get("optimized_exact_main_encoder_overlap")
        if isinstance(metadata, dict)
        else None
    )
    if not isinstance(external, dict) or not isinstance(internal, dict):
        raise ValueError("candidate overlap evidence is missing")
    expected = {
        "version": OVERLAP_VERSION,
        "production_default": False,
        "batch_size": 1,
        "separate_timing_main_models": True,
        "labels_completed": list(LABELS),
        "manifest_changed_after_launch": False,
    }
    failures = {
        key: {
            "expected": value,
            "profile": internal.get(key),
            "evidence": external.get(key),
        }
        for key, value in expected.items()
        if internal.get(key) != value or external.get(key) != value
    }
    if failures:
        raise ValueError(f"candidate overlap contract changed: {failures}")
    windows = external.get("live_window_count")
    if isinstance(windows, bool) or not isinstance(windows, int) or windows <= 0:
        raise ValueError("candidate overlap live window count is invalid")
    if external.get("main_model_generate_calls") != windows:
        raise ValueError("candidate did not consume every exact B1 encoder row")
    launch_window = external.get("launch_after_timing_window")
    if (
        isinstance(launch_window, bool)
        or not isinstance(launch_window, int)
        or launch_window < 2
        or launch_window >= windows
    ):
        raise ValueError("candidate overlap launch window is invalid")
    if not external.get("all_encoder_outputs_finite"):
        raise ValueError("candidate encoder output was not finite")
    manifest = external.get("launch_manifest")
    if not isinstance(manifest, dict) or manifest.get("graph_count", 0) <= 0:
        raise ValueError("candidate lacks a stable accepted graph manifest")
    for key in (
        "pinned_input_setup_seconds",
        "launch_host_seconds",
        "encoder_elapsed_seconds",
        "encoder_overlap_with_timing_seconds",
        "encoder_after_timing_seconds",
        "main_join_wait_seconds",
    ):
        _number(external.get(key), name=f"candidate.{key}")
    for key in (
        "output_store_bytes",
        "peak_allocated_vram_bytes",
        "peak_reserved_vram_bytes",
        "incremental_peak_allocated_vram_bytes",
        "incremental_peak_reserved_vram_bytes",
    ):
        value = external.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"candidate.{key} must be a non-negative integer")
    waits = external.get("per_row_main_join_wait_seconds")
    if not isinstance(waits, list) or len(waits) != windows:
        raise ValueError("candidate lacks per-row main join waits")
    if any(_number(value, name="candidate.row_join") < 0 for value in waits):
        raise ValueError("candidate row join wait is invalid")
    return external


def _validate_initialization(payload: dict[str, Any], *, role: str) -> dict[str, Any]:
    if payload.get("combined_runtime") != COMPOSITION_VERSION:
        raise ValueError(f"{role} did not initialize the current combined runtime")
    if payload.get("result_class") != "documented-drift":
        raise ValueError(f"{role} mixed-weight result class changed")
    if payload.get("exactness_claim") is not False:
        raise ValueError(f"{role} mixed-weight runtime claimed exactness")
    shared = payload.get("shared_rope")
    if not isinstance(shared, dict) or shared.get("incremental_exactness_claim") is not True:
        raise ValueError(f"{role} lacks exact shared-RoPE initialization evidence")
    if shared.get("original_decoder_forward_required") is not True:
        raise ValueError(f"{role} did not preserve the original decoder forward")
    stats = shared.get("stats")
    if not isinstance(stats, dict) or stats.get("reuses", 0) <= 0:
        raise ValueError(f"{role} shared-RoPE reuse evidence is invalid")
    return payload


def _validate_material_audit(
    profile: dict[str, Any],
    evidence: dict[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    audit = evidence.get("material_audit")
    metadata = profile.get("metadata")
    internal = (
        metadata.get("optimized_encoder_material_audit")
        if isinstance(metadata, dict)
        else None
    )
    if not isinstance(audit, dict) or audit != internal:
        raise ValueError(f"{role} material audit is missing or differs from profile")
    if audit.get("version") != MATERIAL_AUDIT_VERSION:
        raise ValueError(f"{role} material audit version changed")
    if audit.get("production_default") is not False:
        raise ValueError(f"{role} material audit leaked into production defaults")
    if audit.get("labels_completed") != list(LABELS):
        raise ValueError(f"{role} material audit labels changed")
    if audit.get("processor_count") != 2:
        raise ValueError(f"{role} material audit processor topology changed")
    copied = audit.get("total_copied_bytes")
    seconds = audit.get("total_copy_and_hash_seconds")
    if isinstance(copied, bool) or not isinstance(copied, int) or copied <= 0:
        raise ValueError(f"{role} material audit copied no bytes")
    _number(seconds, name=f"{role}.material_copy_seconds", positive=True)
    stages = audit.get("stages")
    if not isinstance(stages, dict) or set(stages) != set(LABELS):
        raise ValueError(f"{role} material audit stages changed")
    for label in LABELS:
        stage = stages[label]
        if not isinstance(stage, dict) or stage.get("profile_label") != label:
            raise ValueError(f"{role}.{label} material audit is malformed")
        if label == "timing_context":
            expected = {
                "profile_label": "timing_context",
                "captured": False,
                "reason": "avoid_synchronizing_main_encoder_overlap_stream",
                "copied_bytes": 0,
                "copy_and_hash_seconds": 0.0,
            }
            if stage != expected:
                raise ValueError(f"{role} timing material audit synchronized overlap")
            continue
        if stage.get("captured") is not True:
            raise ValueError(f"{role}.{label} raw material was not captured")
        if not isinstance(stage.get("aggregate_sha256"), str) or len(
            stage["aggregate_sha256"]
        ) != 64:
            raise ValueError(f"{role}.{label} lacks raw material digest")
        if stage.get("cross_layer_count", 0) <= 0:
            raise ValueError(f"{role}.{label} lacks cross-KV layers")
        rows = stage.get("cross_kv")
        if not isinstance(rows, list) or len(rows) != stage["cross_layer_count"]:
            raise ValueError(f"{role}.{label} cross-KV material is incomplete")
        rng = stage.get("rng_after_stage")
        if not isinstance(rng, dict) or any(
            not isinstance(rng.get(device), str) or len(rng[device]) != 64
            for device in ("cpu", "cuda")
        ):
            raise ValueError(f"{role}.{label} final RNG material is incomplete")
    return audit


def _material_exactness(audits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    labels: dict[str, Any] = {}
    for label in ("main_generation",):
        stages = {role: audit["stages"][label] for role, audit in audits.items()}
        raw = {role: stage["aggregate_sha256"] for role, stage in stages.items()}
        rng = {role: stage["rng_after_stage"] for role, stage in stages.items()}
        detailed = {
            role: {
                "encoder": stage["encoder"],
                "cross_kv": stage["cross_kv"],
            }
            for role, stage in stages.items()
        }
        labels[label] = {
            "raw_encoder_cross_kv_equal": len(set(raw.values())) == 1
            and len({json.dumps(value, sort_keys=True) for value in detailed.values()}) == 1,
            "rng_after_stage_equal": len(
                {json.dumps(value, sort_keys=True) for value in rng.values()}
            )
            == 1,
            "aggregate_sha256": raw,
            "rng_after_stage": rng,
        }
        labels[label]["pass"] = bool(
            labels[label]["raw_encoder_cross_kv_equal"]
            and labels[label]["rng_after_stage_equal"]
        )
    return {
        "scope": "main last-window raw encoder and every cross-KV layer; timing "
        "material copy intentionally omitted to preserve overlap",
        "labels": labels,
        "pass": all(row["pass"] for row in labels.values()),
    }


def _validate_baseline(profile: dict[str, Any], evidence: dict[str, Any]) -> None:
    if evidence.get("mode") != "baseline" or evidence.get("encoder_overlap") is not None:
        raise ValueError("baseline evidence unexpectedly contains overlap")
    metadata = profile.get("metadata")
    if isinstance(metadata, dict) and "optimized_exact_main_encoder_overlap" in metadata:
        raise ValueError("baseline profile unexpectedly contains overlap metadata")


def _exact_signatures(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    labels: dict[str, Any] = {}
    for label in LABELS:
        left = _profile_label_signature(baseline, label)
        right = _profile_label_signature(candidate, label)
        labels[label] = {
            "token_stream_equal": (
                left["token_stream_sha256"] is not None
                and left["token_stream_sha256"] == right["token_stream_sha256"]
            ),
            "stopping_equal": (
                left["stopping_sha256"] is not None
                and left["stopping_sha256"] == right["stopping_sha256"]
            ),
            "self_consistent": bool(left["self_consistent"] and right["self_consistent"]),
            "baseline": left,
            "candidate": right,
        }
        labels[label]["pass"] = all(
            labels[label][key]
            for key in ("token_stream_equal", "stopping_equal", "self_consistent")
        )
    left_graph = _profile_graph_cache_signature(baseline, LABELS)
    right_graph = _profile_graph_cache_signature(candidate, LABELS)
    graph_equal = (
        left_graph["status"] == "available"
        and right_graph["status"] == "available"
        and left_graph["sha256"] == right_graph["sha256"]
    )
    return {
        "labels": labels,
        "accepted_graph_cache": {
            "evidence_class": "accepted graph/cache behavior signature",
            "baseline": left_graph,
            "candidate": right_graph,
            "equal": graph_equal,
        },
        "pass": all(row["pass"] for row in labels.values()) and graph_equal,
    }


def analyze(
    profile_paths: dict[str, Path],
    evidence_paths: dict[str, Path],
    initialization_paths: dict[str, Path],
    *,
    base_seed: int,
    minimum_request_saving_seconds: float,
) -> dict[str, Any]:
    profiles = {
        role: _load(path, name=f"{role} profile")
        for role, path in profile_paths.items()
    }
    evidence = {
        role: _load(path, name=f"{role} evidence")
        for role, path in evidence_paths.items()
    }
    initialization = {
        role: _validate_initialization(
            _load(path, name=f"{role} initialization"),
            role=role,
        )
        for role, path in initialization_paths.items()
    }
    for profile in profiles.values():
        _validate_seed(profile, base_seed=base_seed)
    for role in ("baseline_first", "baseline_second"):
        _validate_baseline(profiles[role], evidence[role])
    candidate_evidence = {
        role: _validate_candidate(profiles[role], evidence[role])
        for role in ("candidate_second", "candidate_first")
    }
    material_audits = {
        role: _validate_material_audit(
            profiles[role], evidence[role], role=role
        )
        for role in profiles
    }
    material_exactness = _material_exactness(material_audits)
    reciprocal = compare_reciprocal_profiles(
        profile_paths["baseline_first"],
        profile_paths["candidate_second"],
        profile_paths["candidate_first"],
        profile_paths["baseline_second"],
        labels=list(LABELS),
        regression_tolerance_pct=100.0,
    )
    pairs = (
        ("baseline_first", "candidate_second"),
        ("baseline_second", "candidate_first"),
    )
    metrics = {role: _profile_metrics(profile) for role, profile in profiles.items()}
    orders: list[dict[str, Any]] = []
    exact_orders: dict[str, Any] = {}
    for baseline_role, candidate_role in pairs:
        baseline = metrics[baseline_role]
        candidate = metrics[candidate_role]
        request_saving = (
            baseline["complete_request_seconds"]
            - candidate["complete_request_seconds"]
        )
        timing_slowdown = (
            candidate["timing_stage_seconds"] - baseline["timing_stage_seconds"]
        )
        orders.append(
            {
                "baseline": baseline_role,
                "candidate": candidate_role,
                "complete_request_seconds_saved": request_saving,
                "complete_request_speedup_pct": (
                    request_saving / baseline["complete_request_seconds"] * 100.0
                ),
                "timing_stage_slowdown_seconds": timing_slowdown,
                "timing_stage_slowdown_pct": (
                    timing_slowdown / baseline["timing_stage_seconds"] * 100.0
                ),
                "main_stage_seconds_saved": (
                    baseline["main_stage_seconds"] - candidate["main_stage_seconds"]
                ),
            }
        )
        exact_orders[candidate_role] = _exact_signatures(
            profiles[baseline_role],
            profiles[candidate_role],
        )
    candidate_repeat = _exact_signatures(
        profiles["candidate_first"], profiles["candidate_second"]
    )
    mean_saving = sum(
        order["complete_request_seconds_saved"] for order in orders
    ) / len(orders)
    exactness_pass = (
        reciprocal["same_calculation_pass"]
        and reciprocal["token_equivalence_pass"]
        and reciprocal["output_artifact_pass"]
        and all(row["pass"] for row in exact_orders.values())
        and candidate_repeat["pass"]
        and material_exactness["pass"]
    )
    no_order_regression = all(
        order["complete_request_seconds_saved"] >= 0.0 for order in orders
    )
    performance_pass = (
        no_order_regression and mean_saving >= minimum_request_saving_seconds
    )
    decision = (
        "PASS_EXACT_ENCODER_OVERLAP"
        if exactness_pass and performance_pass
        else "STOP_EXACT_ENCODER_OVERLAP"
    )
    return {
        "schema_version": 1,
        "decision": decision,
        "minimum_request_saving_seconds": minimum_request_saving_seconds,
        "mean_complete_request_seconds_saved": mean_saving,
        "no_order_regression_pass": no_order_regression,
        "performance_pass": performance_pass,
        "exactness_pass": exactness_pass,
        "cache_evidence_scope": (
            "accepted graph/cache behavior signature plus byte-exact last-window "
            "raw encoder and every cross-KV layer"
        ),
        "profiles": {key: str(value) for key, value in profile_paths.items()},
        "evidence": {key: str(value) for key, value in evidence_paths.items()},
        "initialization": {
            key: str(value) for key, value in initialization_paths.items()
        },
        "initialization_evidence": initialization,
        "metrics": metrics,
        "orders": orders,
        "candidate_overlap": candidate_evidence,
        "exact_orders": exact_orders,
        "candidate_repeatability": candidate_repeat,
        "material_audits": material_audits,
        "raw_material_and_rng_exactness": material_exactness,
        "reciprocal_exactness": reciprocal,
    }


def _text(report: dict[str, Any]) -> str:
    lines = [
        f"decision={report['decision']}",
        f"mean_complete_request_seconds_saved={report['mean_complete_request_seconds_saved']:.6f}",
        f"minimum_request_saving_seconds={report['minimum_request_saving_seconds']:.6f}",
        f"exactness_pass={str(report['exactness_pass']).lower()}",
        f"performance_pass={str(report['performance_pass']).lower()}",
        f"no_order_regression_pass={str(report['no_order_regression_pass']).lower()}",
    ]
    for index, order in enumerate(report["orders"]):
        lines.extend(
            (
                f"order_{index}_request_seconds_saved={order['complete_request_seconds_saved']:.6f}",
                f"order_{index}_timing_slowdown_seconds={order['timing_stage_slowdown_seconds']:.6f}",
            )
        )
    for role, evidence in sorted(report["candidate_overlap"].items()):
        lines.extend(
            (
                f"{role}_encoder_elapsed_seconds={evidence['encoder_elapsed_seconds']:.6f}",
                f"{role}_encoder_overlap_seconds={evidence['encoder_overlap_with_timing_seconds']:.6f}",
                f"{role}_main_join_wait_seconds={evidence['main_join_wait_seconds']:.6f}",
                f"{role}_incremental_peak_vram_bytes={evidence['incremental_peak_allocated_vram_bytes']}",
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    for role in (
        "baseline-first",
        "candidate-second",
        "candidate-first",
        "baseline-second",
    ):
        parser.add_argument(f"--{role}-profile", type=Path, required=True)
        parser.add_argument(f"--{role}-evidence", type=Path, required=True)
        parser.add_argument(f"--{role}-initialization", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--minimum-request-saving-seconds", type=float, default=1.0)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    args = parser.parse_args()
    if args.minimum_request_saving_seconds <= 0:
        parser.error("minimum request saving must be positive")
    keys = (
        "baseline_first",
        "candidate_second",
        "candidate_first",
        "baseline_second",
    )
    report = analyze(
        {key: getattr(args, f"{key}_profile") for key in keys},
        {key: getattr(args, f"{key}_evidence") for key in keys},
        {key: getattr(args, f"{key}_initialization") for key in keys},
        base_seed=args.base_seed,
        minimum_request_saving_seconds=args.minimum_request_saving_seconds,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.text_output.write_text(_text(report), encoding="utf-8")
    print(_text(report), end="")
    if report["decision"] != "PASS_EXACT_ENCODER_OVERLAP":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
