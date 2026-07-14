"""Analyze reciprocal full-song encoder-store profiles and the B1/B16 drift audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from osuT5.osuT5.inference.optimized.scout.encoder_store import (
    STORE_VERSION,
    SUPPORTED_LABELS,
)
from utils.fixed_seed_inference import SEED_POLICY_VERSION, stage_seed
from utils.summarize_inference_profile import compare_reciprocal_profiles


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
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _records(profile: dict[str, Any], label: str) -> list[dict[str, Any]]:
    result = [
        row
        for row in profile.get("generation", [])
        if isinstance(row, dict) and row.get("profile_label") == label
    ]
    if not result:
        raise ValueError(f"profile is missing generation label {label!r}")
    return result


def _tokens(profile: dict[str, Any], label: str) -> list[int]:
    result: list[int] = []
    for index, row in enumerate(_records(profile, label)):
        value = row.get("generated_token_ids")
        if not isinstance(value, list):
            raise ValueError(f"{label}[{index}] lacks generated_token_ids")
        result.extend(int(token) for token in value)
    return result


def _token_comparison(reference: list[int], candidate: list[int]) -> dict[str, Any]:
    aligned = min(len(reference), len(candidate))
    mismatches = sum(
        left != right
        for left, right in zip(reference[:aligned], candidate[:aligned], strict=True)
    )
    first = next(
        (
            index
            for index, (left, right) in enumerate(
                zip(reference[:aligned], candidate[:aligned], strict=True)
            )
            if left != right
        ),
        None,
    )
    if first is None and len(reference) != len(candidate):
        first = aligned
    return {
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "count_delta": len(candidate) - len(reference),
        "aligned_count": aligned,
        "aligned_mismatch_count": mismatches,
        "aligned_mismatch_fraction": mismatches / aligned if aligned else 0.0,
        "first_mismatch": first,
        "exact": reference == candidate,
    }


def _stage(profile: dict[str, Any], name: str) -> float:
    matches = [
        row
        for row in profile.get("stages", [])
        if isinstance(row, dict) and row.get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError(f"profile requires exactly one {name!r} stage")
    return _number(matches[0].get("wall_seconds"), name=f"stage.{name}")


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


def _label_metrics(profile: dict[str, Any], label: str) -> dict[str, Any]:
    rows = _records(profile, label)
    tokens = sum(int(row.get("generated_tokens", 0) or 0) for row in rows)
    model = sum(
        _number(
            row.get("model_elapsed_seconds"),
            name=f"{label}.model_elapsed_seconds",
            positive=True,
        )
        for row in rows
    )
    outer = sum(
        _number(row.get("wall_seconds"), name=f"{label}.wall_seconds")
        for row in rows
    )
    return {
        "records": len(rows),
        "tokens": tokens,
        "model_seconds": model,
        "outer_seconds": outer,
        "tokens_per_second": tokens / model,
        "per_window_generated_tokens": [
            int(row.get("generated_tokens", 0) or 0) for row in rows
        ],
    }


def _profile_metrics(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "timing": _label_metrics(profile, "timing_context"),
        "main": _label_metrics(profile, "main_generation"),
        "timing_stage_seconds": _stage(profile, "timing_context_generation"),
        "main_stage_seconds": _stage(profile, "main_generation"),
        "complete_request_seconds": _complete_wall(profile),
    }


def _validate_seed_metadata(profile: dict[str, Any], *, base_seed: int) -> None:
    metadata = profile.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("profile metadata is missing")
    if metadata.get("reciprocal_seed_policy") != SEED_POLICY_VERSION:
        raise ValueError("profile reciprocal seed policy is missing")
    for label in SUPPORTED_LABELS:
        if metadata.get(f"reciprocal_seed_{label}") != stage_seed(base_seed, label):
            raise ValueError(f"profile reciprocal seed changed for {label}")


def _validate_candidate_store(
    profile: dict[str, Any],
    evidence: dict[str, Any],
    *,
    batch_size: int,
) -> dict[str, Any]:
    metadata = profile.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("candidate profile metadata is missing")
    store = metadata.get("optimized_batched_encoder_store")
    if not isinstance(store, dict):
        raise ValueError("candidate profile lacks encoder-store metadata")
    external = evidence.get("encoder_store")
    if not isinstance(external, dict):
        raise ValueError("candidate evidence lacks encoder_store")
    required = {
        "version": STORE_VERSION,
        "batch_size": batch_size,
        "labels_completed": list(SUPPORTED_LABELS),
        "production_default": False,
    }
    failures = {
        key: {"expected": expected, "profile": store.get(key), "evidence": external.get(key)}
        for key, expected in required.items()
        if store.get(key) != expected or external.get(key) != expected
    }
    for key in (
        "store_count",
        "shared_across_timing_main",
        "total_complete_precompute_seconds",
        "total_output_store_bytes",
    ):
        if store.get(key) != external.get(key):
            failures[key] = {
                "profile": store.get(key),
                "evidence": external.get(key),
            }
    if failures:
        raise ValueError(f"candidate encoder-store evidence mismatch: {failures}")
    profile_entries = store.get("entries")
    external_entries = external.get("entries")
    if not isinstance(profile_entries, list) or not isinstance(external_entries, list):
        raise ValueError("candidate encoder-store entries are missing")
    if len(profile_entries) != len(external_entries):
        raise ValueError("candidate encoder-store entry count changed")
    labels_seen: list[str] = []
    for index, (profile_entry, external_entry) in enumerate(
        zip(profile_entries, external_entries, strict=True)
    ):
        for key in (
            "entry_index",
            "labels",
            "live_window_count",
            "output_store_bytes",
            "conditioning_sha256",
            "complete_precompute_seconds",
        ):
            if profile_entry.get(key) != external_entry.get(key):
                raise ValueError(
                    f"candidate encoder-store entry {index} field {key} changed"
                )
        labels_seen.extend(external_entry.get("labels", []))
        uses = external_entry.get("uses")
        if not isinstance(uses, dict):
            raise ValueError(f"candidate encoder-store entry {index} uses are missing")
        for label in external_entry.get("labels", []):
            row = uses.get(label)
            if not isinstance(row, dict) or row.get("rows_reused", 0) <= 0:
                raise ValueError(f"candidate did not reuse encoder rows for {label}")
    if sorted(labels_seen) != sorted(SUPPORTED_LABELS):
        raise ValueError(f"candidate store labels changed: {labels_seen}")
    return external


def _baseline_has_no_store(profile: dict[str, Any]) -> None:
    metadata = profile.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("baseline profile metadata is missing")
    if "optimized_batched_encoder_store" in metadata:
        raise ValueError("baseline unexpectedly contains encoder-store metadata")


def _osu_structure(path: Path) -> dict[str, int]:
    section = None
    counts = {"hit_objects": 0, "timing_points": 0}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line
            continue
        if not line or line.startswith("//"):
            continue
        if section == "[HitObjects]":
            counts["hit_objects"] += 1
        elif section == "[TimingPoints]":
            counts["timing_points"] += 1
    return counts


def analyze(
    profile_paths: dict[str, Path],
    evidence_paths: dict[str, Path],
    audit_paths: dict[str, Path],
    *,
    batch_size: int,
    base_seed: int,
) -> dict[str, Any]:
    profiles = {
        role: _load(path, name=f"{role} profile")
        for role, path in profile_paths.items()
    }
    evidence = {
        role: _load(path, name=f"{role} evidence")
        for role, path in evidence_paths.items()
    }
    for profile in profiles.values():
        _validate_seed_metadata(profile, base_seed=base_seed)
    for role in ("baseline_first", "baseline_second"):
        _baseline_has_no_store(profiles[role])
        if evidence[role].get("mode") != "baseline":
            raise ValueError(f"{role} evidence mode changed")
    stores = {
        role: _validate_candidate_store(
            profiles[role],
            evidence[role],
            batch_size=batch_size,
        )
        for role in ("candidate_second", "candidate_first")
    }

    audits = {
        role: _load(path, name=f"{role} encoder drift audit")
        for role, path in audit_paths.items()
    }
    audit_rows: dict[str, dict[str, Any]] = {}
    audit_window_counts = set()
    for role, audit in audits.items():
        audit_variant = audit.get("variants", {}).get(str(batch_size))
        if not isinstance(audit_variant, dict):
            raise ValueError(f"{role} drift audit lacks B{batch_size}")
        drift = audit_variant.get("encoder_drift_vs_b1")
        if not isinstance(drift, dict):
            raise ValueError(f"{role} drift audit lacks B1 comparison")
        audit_window_counts.add(audit.get("metadata", {}).get("live_window_count"))
        audit_rows[role] = {
            "b1_complete_precompute_seconds": audit["variants"]["1"][
                "complete_precompute_seconds"
            ],
            "candidate_complete_precompute_seconds": audit_variant[
                "complete_precompute_seconds"
            ],
            "complete_seconds_saved_vs_b1": audit_variant[
                "complete_seconds_saved_vs_b1"
            ],
            "max_abs_encoder_drift": drift.get("max_abs"),
            "mean_window_max_abs_encoder_drift": drift.get(
                "mean_window_max_abs"
            ),
            "exact_window_count": drift.get("exact_window_count"),
            "window_count": drift.get("window_count"),
            "incremental_peak_vram_bytes": audit_variant.get(
                "incremental_peak_vram_bytes"
            ),
            "output_store_bytes": audit_variant.get("output_store_bytes"),
        }
    if len(audit_window_counts) != 1:
        raise ValueError("main and timing drift audits have different window counts")
    audit_windows = next(iter(audit_window_counts))
    for store in stores.values():
        if any(
            entry.get("live_window_count") != audit_windows
            for entry in store["entries"]
        ):
            raise ValueError("drift audit and candidate store window counts differ")

    reciprocal = compare_reciprocal_profiles(
        profile_paths["baseline_first"],
        profile_paths["candidate_second"],
        profile_paths["candidate_first"],
        profile_paths["baseline_second"],
        labels=["main_generation", "timing_context"],
        regression_tolerance_pct=100.0,
    )
    metrics = {role: _profile_metrics(profile) for role, profile in profiles.items()}
    order_pairs = (
        ("baseline_first", "candidate_second"),
        ("baseline_second", "candidate_first"),
    )
    orders: list[dict[str, Any]] = []
    for baseline_role, candidate_role in order_pairs:
        baseline = metrics[baseline_role]
        candidate = metrics[candidate_role]
        request_saved = (
            baseline["complete_request_seconds"]
            - candidate["complete_request_seconds"]
        )
        orders.append(
            {
                "baseline": baseline_role,
                "candidate": candidate_role,
                "complete_request_seconds_saved": request_saved,
                "complete_request_speedup_pct": (
                    request_saved / baseline["complete_request_seconds"] * 100.0
                ),
                "timing_stage_seconds_saved": (
                    baseline["timing_stage_seconds"]
                    - candidate["timing_stage_seconds"]
                ),
                "main_stage_seconds_saved": (
                    baseline["main_stage_seconds"]
                    - candidate["main_stage_seconds"]
                ),
                "timing_tokens": _token_comparison(
                    _tokens(profiles[baseline_role], "timing_context"),
                    _tokens(profiles[candidate_role], "timing_context"),
                ),
                "main_tokens": _token_comparison(
                    _tokens(profiles[baseline_role], "main_generation"),
                    _tokens(profiles[candidate_role], "main_generation"),
                ),
            }
        )

    repeatability = {}
    for label in SUPPORTED_LABELS:
        repeatability[f"baseline_{label}"] = _token_comparison(
            _tokens(profiles["baseline_first"], label),
            _tokens(profiles["baseline_second"], label),
        )
        repeatability[f"candidate_{label}"] = _token_comparison(
            _tokens(profiles["candidate_first"], label),
            _tokens(profiles["candidate_second"], label),
        )
    candidate_artifacts = [
        Path(profiles[role]["metadata"]["result_file_path"])
        for role in ("candidate_first", "candidate_second")
    ]
    baseline_artifacts = [
        Path(profiles[role]["metadata"]["result_file_path"])
        for role in ("baseline_first", "baseline_second")
    ]
    repeatability_pass = all(row["exact"] for row in repeatability.values())
    request_nonregression = all(
        order["complete_request_seconds_saved"] >= 0.0 for order in orders
    )
    report = {
        "schema_version": 1,
        "decision": (
            "PASS_FULL_SONG_GATE"
            if repeatability_pass and request_nonregression
            else "STOP_ENCODER_STORE"
        ),
        "batch_size": batch_size,
        "profiles": {key: str(value) for key, value in profile_paths.items()},
        "evidence": {key: str(value) for key, value in evidence_paths.items()},
        "audit_paths": {key: str(value) for key, value in audit_paths.items()},
        "audits": audit_rows,
        "metrics": metrics,
        "orders": orders,
        "mean_complete_request_seconds_saved": sum(
            order["complete_request_seconds_saved"] for order in orders
        )
        / len(orders),
        "repeatability": repeatability,
        "repeatability_pass": repeatability_pass,
        "request_nonregression_pass": request_nonregression,
        "baseline_output_repeat": {
            "sha256_equal": profiles["baseline_first"]["metadata"].get(
                "result_file_sha256"
            )
            == profiles["baseline_second"]["metadata"].get("result_file_sha256"),
            "structure": [_osu_structure(path) for path in baseline_artifacts],
        },
        "candidate_output_repeat": {
            "sha256_equal": profiles["candidate_first"]["metadata"].get(
                "result_file_sha256"
            )
            == profiles["candidate_second"]["metadata"].get("result_file_sha256"),
            "structure": [_osu_structure(path) for path in candidate_artifacts],
        },
        "reciprocal_exactness": reciprocal,
    }
    return report


def _text(report: dict[str, Any]) -> str:
    lines = [
        f"decision={report['decision']}",
        f"batch_size={report['batch_size']}",
        f"mean_complete_request_seconds_saved={report['mean_complete_request_seconds_saved']:.6f}",
        f"main_encoder_max_abs_drift={report['audits']['main']['max_abs_encoder_drift']}",
        f"timing_encoder_max_abs_drift={report['audits']['timing']['max_abs_encoder_drift']}",
        f"main_encoder_precompute_seconds_saved={report['audits']['main']['complete_seconds_saved_vs_b1']}",
        f"timing_encoder_precompute_seconds_saved={report['audits']['timing']['complete_seconds_saved_vs_b1']}",
        f"repeatability_pass={str(report['repeatability_pass']).lower()}",
        f"request_nonregression_pass={str(report['request_nonregression_pass']).lower()}",
    ]
    for index, order in enumerate(report["orders"]):
        lines.append(
            f"order_{index}_complete_request_seconds_saved="
            f"{order['complete_request_seconds_saved']:.6f}"
        )
        lines.append(
            f"order_{index}_main_token_count_delta={order['main_tokens']['count_delta']}"
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
    parser.add_argument("--main-audit", type=Path, required=True)
    parser.add_argument("--timing-audit", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    args = parser.parse_args()
    keys = (
        "baseline_first",
        "candidate_second",
        "candidate_first",
        "baseline_second",
    )
    report = analyze(
        {
            key: getattr(args, f"{key}_profile")
            for key in keys
        },
        {
            key: getattr(args, f"{key}_evidence")
            for key in keys
        },
        {"main": args.main_audit, "timing": args.timing_audit},
        batch_size=args.batch_size,
        base_seed=args.base_seed,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.text_output.write_text(_text(report), encoding="utf-8")
    print(_text(report), end="")
    if report["decision"] != "PASS_FULL_SONG_GATE":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
