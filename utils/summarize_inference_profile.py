from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONTRACT_METADATA_KEYS = [
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
    "profile_record_token_ids",
]


def _fmt_seconds(value: Any) -> str:
    try:
        return f"{float(value):8.3f}s"
    except (TypeError, ValueError):
        return "     n/a"


def _print_table(title: str, rows: list[tuple[str, Any]], limit: int | None = None) -> None:
    print(title)
    for name, value in rows[:limit]:
        print(f"  {_fmt_seconds(value)}  {name}")
    print()


def _generation_name(record: dict[str, Any]) -> str:
    context = record.get("context_type", "unknown")
    label = record.get("profile_label", "unknown")
    mode = record.get("mode", "unknown")
    if "sequence_index" in record:
        unit = f"seq={record['sequence_index']}"
    else:
        unit = f"batch_start={record.get('batch_start_index', 'n/a')}"
    tokens = record.get("generated_tokens", "n/a")
    tok_s = record.get("tokens_per_second")
    tok_s_text = f", {tok_s:.1f} tok/s" if isinstance(tok_s, (int, float)) else ""
    return f"{label}/{context}/{mode}/{unit}, generated={tokens}{tok_s_text}"


def _load_profile(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(path: Path, *, limit: int) -> None:
    profile = _load_profile(path)
    print(f"Profile: {path}")

    metadata = profile.get("metadata", {})
    if metadata:
        print("Run")
        for key in [
            "audio_path",
            "result_path",
            "device",
            "precision",
            "attn_implementation",
            "use_server",
            "parallel",
            "sequence_count",
            "song_length_ms",
        ]:
            if key in metadata:
                print(f"  {key}: {metadata[key]}")
        print()

    stages = profile.get("stages", [])
    stage_rows = sorted(
        ((stage.get("name", "unknown"), stage.get("wall_seconds")) for stage in stages),
        key=lambda row: float(row[1] or 0.0),
        reverse=True,
    )
    _print_table("Stages by wall time", stage_rows, limit)

    generation = profile.get("generation", [])
    generation_rows = sorted(
        ((_generation_name(record), record.get("wall_seconds")) for record in generation),
        key=lambda row: float(row[1] or 0.0),
        reverse=True,
    )
    _print_table("Slowest generation records by outer wall time", generation_rows, limit)

    summary = profile.get("summary", {})
    by_context = summary.get("generation_by_context", {})
    if by_context:
        print("Generation by context")
        for context, values in sorted(
                by_context.items(),
                key=lambda item: float(item[1].get("model_elapsed_seconds", 0.0) or 0.0),
                reverse=True,
        ):
            elapsed = values.get("model_elapsed_seconds", 0.0)
            wall = values.get("wall_seconds", 0.0)
            tokens = values.get("generated_tokens", 0)
            tok_s = values.get("tokens_per_second", 0.0)
            records = values.get("records", 0)
            print(
                f"  {context}: model={_fmt_seconds(elapsed)}, wall={_fmt_seconds(wall)}, "
                f"tokens={tokens}, tok/s={tok_s:.1f}, records={records}"
            )


def _summary_for_label(profile: dict[str, Any], label: str) -> dict[str, Any]:
    by_label = profile.get("summary", {}).get("generation_by_label", {})
    if label in by_label:
        return by_label[label]
    return {}


def _total_stage_wall_seconds(profile: dict[str, Any]) -> float:
    return sum(float(stage.get("wall_seconds", 0.0) or 0.0) for stage in profile.get("stages", []))


def _records_for_label(profile: dict[str, Any], label: str) -> list[dict[str, Any]]:
    return [record for record in profile.get("generation", []) if record.get("profile_label") == label]


def _record_key(record: dict[str, Any], index: int) -> str:
    context = record.get("context_type", "unknown")
    mode = record.get("mode", "unknown")
    if "sequence_index" in record:
        unit = f"seq{record['sequence_index']}"
    else:
        unit = f"batch{record.get('batch_start_index', index)}"
    return f"{context}/{mode}/{unit}"


def _record_tokens_per_second(record: dict[str, Any]) -> float:
    value = record.get("tokens_per_second")
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    generated_tokens = int(record.get("generated_tokens", 0) or 0)
    model_elapsed = float(record.get("model_elapsed_seconds", 0.0) or 0.0)
    return generated_tokens / model_elapsed if model_elapsed > 0 else 0.0


def _flatten_token_ids(profile: dict[str, Any], label: str) -> list[int] | None:
    tokens: list[int] = []
    saw_tokens = False
    for record in profile.get("generation", []):
        if record.get("profile_label") != label:
            continue
        if "generated_token_ids" in record:
            value = record.get("generated_token_ids")
            if value is None:
                continue
            saw_tokens = True
            tokens.extend(int(token) for token in value)
        elif "generated_token_ids_per_sample" in record:
            value = record.get("generated_token_ids_per_sample")
            if value is None:
                continue
            saw_tokens = True
            for sample in value:
                tokens.extend(int(token) for token in sample)
    return tokens if saw_tokens else None


def _compare_number(name: str, baseline: float, candidate: float, *, higher_is_better: bool) -> None:
    delta = candidate - baseline
    pct = (delta / baseline * 100.0) if baseline else 0.0
    direction = "better" if (delta >= 0) == higher_is_better else "worse"
    print(f"  {name}: baseline={baseline:.3f}, candidate={candidate:.3f}, delta={delta:+.3f} ({pct:+.1f}%, {direction})")


def _metric_comparison(
        baseline: float,
        candidate: float,
        *,
        higher_is_better: bool,
        tolerance_pct: float,
) -> dict[str, Any]:
    delta = candidate - baseline
    pct = (delta / baseline * 100.0) if baseline else 0.0
    tolerance = abs(baseline) * tolerance_pct / 100.0
    if higher_is_better:
        passed = candidate >= baseline - tolerance
    else:
        passed = candidate <= baseline + tolerance
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "pct": pct,
        "higher_is_better": higher_is_better,
        "pass": passed,
    }


def _compare_contract_metadata(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_metadata = baseline.get("metadata", {})
    candidate_metadata = candidate.get("metadata", {})
    mismatches = []
    missing = []
    for key in CONTRACT_METADATA_KEYS:
        if key not in baseline_metadata or key not in candidate_metadata:
            missing.append(key)
            continue
        if baseline_metadata[key] != candidate_metadata[key]:
            mismatches.append((key, baseline_metadata[key], candidate_metadata[key]))

    print("Same-calculation metadata contract")
    passed = not mismatches and not missing
    if mismatches or missing:
        print("  FAIL")
        for key, baseline_value, candidate_value in mismatches:
            print(f"  {key}: baseline={baseline_value!r}, candidate={candidate_value!r}")
    else:
        print("  PASS")
    if missing:
        print(f"  missing_keys: {', '.join(missing)}")
    print()
    return {
        "pass": passed,
        "mismatches": [
            {"key": key, "baseline": baseline_value, "candidate": candidate_value}
            for key, baseline_value, candidate_value in mismatches
        ],
        "missing_keys": missing,
    }


def _compare_label_records(
        baseline: dict[str, Any],
        candidate: dict[str, Any],
        *,
        label: str,
        regression_tolerance_pct: float,
) -> dict[str, Any]:
    baseline_records = _records_for_label(baseline, label)
    candidate_records = _records_for_label(candidate, label)
    window_results = []
    comparable_count = min(len(baseline_records), len(candidate_records))
    for index in range(comparable_count):
        baseline_record = baseline_records[index]
        candidate_record = candidate_records[index]
        metric_results = {
            "tokens_per_second": _metric_comparison(
                _record_tokens_per_second(baseline_record),
                _record_tokens_per_second(candidate_record),
                higher_is_better=True,
                tolerance_pct=regression_tolerance_pct,
            ),
            "model_elapsed_seconds": _metric_comparison(
                float(baseline_record.get("model_elapsed_seconds", 0.0) or 0.0),
                float(candidate_record.get("model_elapsed_seconds", 0.0) or 0.0),
                higher_is_better=False,
                tolerance_pct=regression_tolerance_pct,
            ),
            "outer_wall_seconds": _metric_comparison(
                float(baseline_record.get("wall_seconds", 0.0) or 0.0),
                float(candidate_record.get("wall_seconds", 0.0) or 0.0),
                higher_is_better=False,
                tolerance_pct=regression_tolerance_pct,
            ),
        }
        generated_tokens_match = (
            int(baseline_record.get("generated_tokens", 0) or 0)
            == int(candidate_record.get("generated_tokens", 0) or 0)
        )
        keys_match = _record_key(baseline_record, index) == _record_key(candidate_record, index)
        window_results.append({
            "index": index,
            "baseline_key": _record_key(baseline_record, index),
            "candidate_key": _record_key(candidate_record, index),
            "keys_match": keys_match,
            "generated_tokens_match": generated_tokens_match,
            "metrics": metric_results,
            "pass": (
                keys_match
                and generated_tokens_match
                and all(metric["pass"] for metric in metric_results.values())
            ),
        })

    failed = [record for record in window_results if not record["pass"]]
    pass_result = (
        len(baseline_records) == len(candidate_records)
        and all(record["pass"] for record in window_results)
    )
    print("Per-window no-regression")
    if pass_result:
        print(f"  PASS ({len(window_results)} records)")
    else:
        print(
            "  FAIL "
            f"(baseline_records={len(baseline_records)}, candidate_records={len(candidate_records)}, "
            f"failed_records={len(failed)})"
        )
        for record in failed[:5]:
            print(
                "  {index}: baseline={baseline}, candidate={candidate}".format(
                    index=record["index"],
                    baseline=record["baseline_key"],
                    candidate=record["candidate_key"],
                )
            )
    print()
    return {
        "pass": pass_result,
        "baseline_records": len(baseline_records),
        "candidate_records": len(candidate_records),
        "failed_records": len(failed),
        "windows": window_results,
    }


def compare_profiles(
        baseline_path: Path,
        candidate_path: Path,
        *,
        label: str,
        regression_tolerance_pct: float = 0.0,
) -> dict[str, Any]:
    baseline = _load_profile(baseline_path)
    candidate = _load_profile(candidate_path)
    baseline_summary = _summary_for_label(baseline, label)
    candidate_summary = _summary_for_label(candidate, label)
    report: dict[str, Any] = {
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "label": label,
        "regression_tolerance_pct": regression_tolerance_pct,
        "same_calculation": {},
        "performance": {
            "pass": False,
            "metrics": {},
            "generated_tokens_match": None,
            "records_match": None,
            "per_window": {},
        },
        "token_equivalence": {
            "status": "not_checked",
            "pass": False,
            "baseline_len": None,
            "candidate_len": None,
            "first_mismatch": None,
        },
    }

    print(f"Baseline:  {baseline_path}")
    print(f"Candidate: {candidate_path}")
    print(f"Label:     {label}")
    print()

    report["same_calculation"] = _compare_contract_metadata(baseline, candidate)

    if not baseline_summary or not candidate_summary:
        print("Missing generation summary for requested label.")
        report["performance"]["missing_summary"] = True
        return report

    metric_specs = {
        "tokens_per_second": True,
        "model_elapsed_seconds": False,
        "outer_wall_seconds": False,
        "total_stage_wall_seconds": False,
    }
    metric_values = {
        "tokens_per_second": (
            float(baseline_summary.get("tokens_per_second", 0.0) or 0.0),
            float(candidate_summary.get("tokens_per_second", 0.0) or 0.0),
        ),
        "model_elapsed_seconds": (
            float(baseline_summary.get("model_elapsed_seconds", 0.0) or 0.0),
            float(candidate_summary.get("model_elapsed_seconds", 0.0) or 0.0),
        ),
        "outer_wall_seconds": (
            float(baseline_summary.get("wall_seconds", 0.0) or 0.0),
            float(candidate_summary.get("wall_seconds", 0.0) or 0.0),
        ),
        "total_stage_wall_seconds": (
            _total_stage_wall_seconds(baseline),
            _total_stage_wall_seconds(candidate),
        ),
    }
    for name, higher_is_better in metric_specs.items():
        baseline_value, candidate_value = metric_values[name]
        _compare_number(name, baseline_value, candidate_value, higher_is_better=higher_is_better)
        report["performance"]["metrics"][name] = _metric_comparison(
            baseline_value,
            candidate_value,
            higher_is_better=higher_is_better,
            tolerance_pct=regression_tolerance_pct,
        )

    baseline_tokens_count = baseline_summary.get("generated_tokens")
    candidate_tokens_count = candidate_summary.get("generated_tokens")
    baseline_records = baseline_summary.get("records")
    candidate_records = candidate_summary.get("records")
    generated_tokens_match = baseline_tokens_count == candidate_tokens_count
    records_match = baseline_records == candidate_records
    report["performance"]["generated_tokens_match"] = generated_tokens_match
    report["performance"]["records_match"] = records_match
    report["performance"]["per_window"] = _compare_label_records(
        baseline,
        candidate,
        label=label,
        regression_tolerance_pct=regression_tolerance_pct,
    )
    report["performance"]["pass"] = (
        all(metric["pass"] for metric in report["performance"]["metrics"].values())
        and generated_tokens_match
        and records_match
        and report["performance"]["per_window"]["pass"]
    )
    print(f"  generated_tokens: baseline={baseline_tokens_count}, candidate={candidate_tokens_count}")
    print(f"  records: baseline={baseline_records}, candidate={candidate_records}")
    print(f"  no_regression_gate: {'PASS' if report['performance']['pass'] else 'FAIL'}")
    print()

    baseline_tokens = _flatten_token_ids(baseline, label)
    candidate_tokens = _flatten_token_ids(candidate, label)
    if baseline_tokens is None or candidate_tokens is None:
        print("Token equivalence: not checked; rerun with profile_record_token_ids=true.")
        return report
    if baseline_tokens == candidate_tokens:
        print(f"Token equivalence: PASS ({len(baseline_tokens)} generated token IDs match).")
        report["token_equivalence"] = {
            "status": "PASS",
            "pass": True,
            "baseline_len": len(baseline_tokens),
            "candidate_len": len(candidate_tokens),
            "first_mismatch": None,
        }
        return report

    mismatch = next(
        (idx for idx, pair in enumerate(zip(baseline_tokens, candidate_tokens)) if pair[0] != pair[1]),
        min(len(baseline_tokens), len(candidate_tokens)),
    )
    print(
        "Token equivalence: FAIL "
        f"(baseline_len={len(baseline_tokens)}, candidate_len={len(candidate_tokens)}, first_mismatch={mismatch})."
    )
    report["token_equivalence"] = {
        "status": "FAIL",
        "pass": False,
        "baseline_len": len(baseline_tokens),
        "candidate_len": len(candidate_tokens),
        "first_mismatch": mismatch,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Mapperatorinator inference profile JSON.")
    parser.add_argument("profile", type=Path, nargs="?", help="Path to a .profile.json file.")
    parser.add_argument("--limit", type=int, default=12, help="Number of stage/window rows to print.")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "CANDIDATE"),
        type=Path,
        help="Compare two profile JSON files.",
    )
    parser.add_argument("--label", default="main_generation", help="Generation label to compare.")
    parser.add_argument(
        "--regression-tolerance-pct",
        type=float,
        default=0.0,
        help="Tolerance used by --require-no-regression. Default 0 means no tolerated degradation.",
    )
    parser.add_argument(
        "--require-contract-match",
        action="store_true",
        help="Exit nonzero if same-calculation metadata differs.",
    )
    parser.add_argument(
        "--require-token-equivalence",
        action="store_true",
        help="Exit nonzero unless generated token IDs are present and exactly match.",
    )
    parser.add_argument(
        "--require-no-regression",
        action="store_true",
        help="Exit nonzero if candidate throughput, model time, wall time, token count, or record count regresses.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Shortcut for all --require-* comparison gates.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Write a machine-readable comparison report.",
    )
    args = parser.parse_args()
    if args.compare:
        report = compare_profiles(
            args.compare[0],
            args.compare[1],
            label=args.label,
            regression_tolerance_pct=args.regression_tolerance_pct,
        )
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

        require_contract_match = args.require_contract_match or args.strict
        require_token_equivalence = args.require_token_equivalence or args.strict
        require_no_regression = args.require_no_regression or args.strict
        failed = (
            (require_contract_match and not report["same_calculation"].get("pass", False))
            or (require_token_equivalence and not report["token_equivalence"].get("pass", False))
            or (require_no_regression and not report["performance"].get("pass", False))
        )
        raise SystemExit(1 if failed else 0)
    elif args.profile:
        summarize(args.profile, limit=args.limit)
    else:
        parser.error("provide a profile path or --compare BASELINE CANDIDATE")


if __name__ == "__main__":
    main()
