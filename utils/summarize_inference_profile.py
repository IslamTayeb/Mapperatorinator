from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_LABELS = ("main_generation", "timing_context")
CONTRACT_METADATA_KEYS = (
    "model_path",
    "audio_path",
    "seed",
    "precision",
    "attn_implementation",
    "use_server",
    "parallel",
    "temperature",
    "timing_temperature",
    "mania_column_temperature",
    "taiko_hit_temperature",
    "timeshift_bias",
    "top_p",
    "top_k",
    "do_sample",
    "num_beams",
    "cfg_scale",
    "lookback",
    "lookahead",
    "start_time",
    "end_time",
    "in_context",
    "output_type",
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"profile must contain a JSON object: {path}")
    return payload


def _records(profile: dict[str, Any], label: str) -> list[dict[str, Any]]:
    return [
        record
        for record in profile.get("generation", [])
        if isinstance(record, dict) and record.get("profile_label") == label
    ]


def _record_key(record: dict[str, Any], index: int) -> tuple[Any, ...]:
    return (
        record.get("context_type"),
        record.get("mode"),
        record.get("sequence_index", record.get("batch_start_index", index)),
    )


def _aggregate(records: list[dict[str, Any]]) -> dict[str, float | int]:
    tokens = sum(int(record.get("generated_tokens", 0) or 0) for record in records)
    model_seconds = sum(float(record.get("model_elapsed_seconds", 0.0) or 0.0) for record in records)
    wall_seconds = sum(float(record.get("wall_seconds", 0.0) or 0.0) for record in records)
    return {
        "records": len(records),
        "generated_tokens": tokens,
        "model_elapsed_seconds": model_seconds,
        "wall_seconds": wall_seconds,
        "tokens_per_second": tokens / model_seconds if model_seconds > 0 else 0.0,
    }


def _stage_wall_seconds(profile: dict[str, Any]) -> float:
    return sum(
        float(stage.get("wall_seconds", 0.0) or 0.0)
        for stage in profile.get("stages", [])
        if isinstance(stage, dict)
    )


def _flatten_token_ids(profile: dict[str, Any], label: str) -> list[int] | None:
    flattened: list[int] = []
    for record in _records(profile, label):
        token_groups = record.get("generated_token_ids_per_sample")
        if token_groups is not None:
            if not isinstance(token_groups, list) or any(not isinstance(group, list) for group in token_groups):
                return None
            for group in token_groups:
                flattened.extend(int(token) for token in group)
            continue
        tokens = record.get("generated_token_ids")
        if not isinstance(tokens, list):
            return None
        flattened.extend(int(token) for token in tokens)
    return flattened


def _metric(
    baseline: float,
    candidate: float,
    *,
    higher_is_better: bool,
    tolerance_pct: float,
) -> dict[str, Any]:
    if baseline == 0:
        pct = 0.0 if candidate == 0 else None
        passed = candidate == 0 or (higher_is_better and candidate > 0)
    else:
        pct = (candidate - baseline) / baseline * 100.0
        epsilon = 1e-9
        passed = (
            pct >= -tolerance_pct - epsilon
            if higher_is_better
            else pct <= tolerance_pct + epsilon
        )
    return {
        "baseline": baseline,
        "candidate": candidate,
        "pct": pct,
        "higher_is_better": higher_is_better,
        "pass": passed,
    }


def _contract(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_metadata = baseline.get("metadata", {})
    candidate_metadata = candidate.get("metadata", {})
    missing = [
        key
        for key in CONTRACT_METADATA_KEYS
        if key not in baseline_metadata or key not in candidate_metadata
    ]
    mismatches = [
        {
            "key": key,
            "baseline": baseline_metadata[key],
            "candidate": candidate_metadata[key],
        }
        for key in CONTRACT_METADATA_KEYS
        if key not in missing and baseline_metadata[key] != candidate_metadata[key]
    ]
    return {"pass": not missing and not mismatches, "missing_keys": missing, "mismatches": mismatches}


def _artifact(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_metadata = baseline.get("metadata", {})
    candidate_metadata = candidate.get("metadata", {})
    values = {
        "baseline_sha256": baseline_metadata.get("result_file_sha256"),
        "candidate_sha256": candidate_metadata.get("result_file_sha256"),
        "baseline_size_bytes": baseline_metadata.get("result_file_size_bytes"),
        "candidate_size_bytes": candidate_metadata.get("result_file_size_bytes"),
    }
    missing = [key for key, value in values.items() if value is None]
    passed = (
        not missing
        and values["baseline_sha256"] == values["candidate_sha256"]
        and values["baseline_size_bytes"] == values["candidate_size_bytes"]
    )
    return {
        **values,
        "missing_keys": missing,
        "status": "PASS" if passed else ("not_checked" if missing else "FAIL"),
        "pass": passed,
    }


def _tokens(baseline: dict[str, Any], candidate: dict[str, Any], label: str) -> dict[str, Any]:
    baseline_tokens = _flatten_token_ids(baseline, label)
    candidate_tokens = _flatten_token_ids(candidate, label)
    if baseline_tokens is None or candidate_tokens is None:
        return {
            "status": "not_checked",
            "pass": False,
            "baseline_len": None if baseline_tokens is None else len(baseline_tokens),
            "candidate_len": None if candidate_tokens is None else len(candidate_tokens),
            "first_mismatch": None,
        }
    first_mismatch = next(
        (
            index
            for index, pair in enumerate(zip(baseline_tokens, candidate_tokens))
            if pair[0] != pair[1]
        ),
        min(len(baseline_tokens), len(candidate_tokens)),
    )
    passed = baseline_tokens == candidate_tokens
    return {
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "baseline_len": len(baseline_tokens),
        "candidate_len": len(candidate_tokens),
        "first_mismatch": None if passed else first_mismatch,
    }


def _performance(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    label: str,
    tolerance_pct: float,
) -> dict[str, Any]:
    baseline_records = _records(baseline, label)
    candidate_records = _records(candidate, label)
    baseline_total = _aggregate(baseline_records)
    candidate_total = _aggregate(candidate_records)
    metrics = {
        "tokens_per_second": _metric(
            float(baseline_total["tokens_per_second"]),
            float(candidate_total["tokens_per_second"]),
            higher_is_better=True,
            tolerance_pct=tolerance_pct,
        ),
        "model_elapsed_seconds": _metric(
            float(baseline_total["model_elapsed_seconds"]),
            float(candidate_total["model_elapsed_seconds"]),
            higher_is_better=False,
            tolerance_pct=tolerance_pct,
        ),
        "outer_wall_seconds": _metric(
            float(baseline_total["wall_seconds"]),
            float(candidate_total["wall_seconds"]),
            higher_is_better=False,
            tolerance_pct=tolerance_pct,
        ),
        "total_stage_wall_seconds": _metric(
            _stage_wall_seconds(baseline),
            _stage_wall_seconds(candidate),
            higher_is_better=False,
            tolerance_pct=tolerance_pct,
        ),
    }
    record_shape_match = (
        len(baseline_records) == len(candidate_records)
        and all(
            _record_key(base, index) == _record_key(cand, index)
            and int(base.get("generated_tokens", 0) or 0) == int(cand.get("generated_tokens", 0) or 0)
            for index, (base, cand) in enumerate(zip(baseline_records, candidate_records))
        )
    )
    return {
        "pass": bool(baseline_records) and record_shape_match and metrics["tokens_per_second"]["pass"] and metrics["model_elapsed_seconds"]["pass"],
        "metrics": metrics,
        "records_match": record_shape_match,
        "generated_tokens_match": baseline_total["generated_tokens"] == candidate_total["generated_tokens"],
        "baseline": baseline_total,
        "candidate": candidate_total,
    }


def compare_profiles(
    baseline_path: Path,
    candidate_path: Path,
    *,
    label: str,
    regression_tolerance_pct: float = 0.0,
) -> dict[str, Any]:
    baseline = _load(baseline_path)
    candidate = _load(candidate_path)
    report = {
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "label": label,
        "regression_tolerance_pct": regression_tolerance_pct,
        "same_calculation": _contract(baseline, candidate),
        "token_equivalence": _tokens(baseline, candidate, label),
        "output_artifact": _artifact(baseline, candidate),
        "performance": _performance(baseline, candidate, label, regression_tolerance_pct),
    }
    print(
        f"{label}: contract={'PASS' if report['same_calculation']['pass'] else 'FAIL'} "
        f"tokens={report['token_equivalence']['status']} "
        f"output={report['output_artifact']['status']} "
        f"performance={'PASS' if report['performance']['pass'] else 'FAIL'}"
    )
    return report


def compare_profiles_for_labels(
    baseline_path: Path,
    candidate_path: Path,
    *,
    labels: list[str],
    regression_tolerance_pct: float = 0.0,
) -> dict[str, Any]:
    reports = {
        label: compare_profiles(
            baseline_path,
            candidate_path,
            label=label,
            regression_tolerance_pct=regression_tolerance_pct,
        )
        for label in labels
    }
    return {
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "labels": labels,
        "reports": reports,
        "same_calculation_pass": all(report["same_calculation"]["pass"] for report in reports.values()),
        "token_equivalence_pass": all(report["token_equivalence"]["pass"] for report in reports.values()),
        "output_artifact_pass": all(report["output_artifact"]["pass"] for report in reports.values()),
        "performance_pass": all(report["performance"]["pass"] for report in reports.values()),
    }


def compare_reciprocal_profiles(
    baseline_first: Path,
    candidate_second: Path,
    candidate_first: Path,
    baseline_second: Path,
    *,
    labels: list[str],
    regression_tolerance_pct: float = 5.0,
) -> dict[str, Any]:
    orders = {
        "baseline_first": compare_profiles_for_labels(
            baseline_first,
            candidate_second,
            labels=labels,
            regression_tolerance_pct=regression_tolerance_pct,
        ),
        "candidate_first": compare_profiles_for_labels(
            baseline_second,
            candidate_first,
            labels=labels,
            regression_tolerance_pct=regression_tolerance_pct,
        ),
    }
    return {
        "schema_version": 1,
        "labels": labels,
        "orders": orders,
        "same_calculation_pass": all(order["same_calculation_pass"] for order in orders.values()),
        "token_equivalence_pass": all(order["token_equivalence_pass"] for order in orders.values()),
        "output_artifact_pass": all(order["output_artifact_pass"] for order in orders.values()),
        "performance_pass": all(order["performance_pass"] for order in orders.values()),
    }


def _labels(value: str) -> list[str]:
    labels = [label.strip() for label in value.split(",") if label.strip()]
    if not labels:
        raise argparse.ArgumentTypeError("at least one profile label is required")
    return labels


def _write_json(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        print(json.dumps(report, indent=2))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _passed(report: dict[str, Any]) -> bool:
    return all(
        report.get(key) is True
        for key in (
            "same_calculation_pass",
            "token_equivalence_pass",
            "output_artifact_pass",
            "performance_pass",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare exact-output inference profile JSON.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--compare", nargs=2, type=Path, metavar=("BASELINE", "CANDIDATE"))
    mode.add_argument(
        "--reciprocal",
        nargs=4,
        type=Path,
        metavar=("BASELINE_FIRST", "CANDIDATE_SECOND", "CANDIDATE_FIRST", "BASELINE_SECOND"),
    )
    parser.add_argument("--labels", type=_labels, default=list(DEFAULT_LABELS))
    parser.add_argument("--regression-tolerance-pct", type=float, default=5.0)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    if args.regression_tolerance_pct < 0:
        parser.error("--regression-tolerance-pct must be non-negative")
    if args.compare:
        report = compare_profiles_for_labels(
            *args.compare,
            labels=args.labels,
            regression_tolerance_pct=args.regression_tolerance_pct,
        )
    else:
        report = compare_reciprocal_profiles(
            *args.reciprocal,
            labels=args.labels,
            regression_tolerance_pct=args.regression_tolerance_pct,
        )
    _write_json(args.json_output, report)
    raise SystemExit(0 if _passed(report) else 1)


if __name__ == "__main__":
    main()
