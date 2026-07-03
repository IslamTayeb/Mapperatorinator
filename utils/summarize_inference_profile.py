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


def _load_json(path: Path) -> dict[str, Any]:
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


def _suite_aggregate(manifest: dict[str, Any], scope: str) -> dict[str, Any]:
    aggregate = manifest.get("aggregate", {})
    if scope == "warmed_runs":
        warmed = aggregate.get("warmed_runs")
        if warmed is not None:
            return warmed
        return aggregate.get("all_runs", {})
    return aggregate.get("all_runs", {})


def _suite_runs_for_scope(manifest: dict[str, Any], scope: str) -> list[dict[str, Any]]:
    runs = manifest.get("runs", [])
    if scope == "warmed_runs":
        return [
            run
            for run in runs
            if int(run.get("repeat_index", run.get("run_index", 0)) or 0) > 0
        ]
    return runs


def _compare_suite_scope_availability(
        baseline: dict[str, Any],
        candidate: dict[str, Any],
        *,
        scope: str,
) -> dict[str, Any]:
    baseline_runs = _suite_runs_for_scope(baseline, scope)
    candidate_runs = _suite_runs_for_scope(candidate, scope)
    missing = []
    if not baseline_runs:
        missing.append("baseline")
    if not candidate_runs:
        missing.append("candidate")
    passed = not missing
    print(f"Suite scope availability ({scope})")
    print(f"  {'PASS' if passed else 'FAIL'}")
    if missing:
        print(f"  missing_{scope}: {', '.join(missing)}")
    print()
    return {
        "pass": passed,
        "baseline_runs": len(baseline_runs),
        "candidate_runs": len(candidate_runs),
        "missing": missing,
    }


def _suite_run_key(run: dict[str, Any], index: int) -> str:
    return "run{index}/song{song}/repeat{repeat}".format(
        index=run.get("run_index", index),
        song=run.get("song_index", "n/a"),
        repeat=run.get("repeat_index", "n/a"),
    )


def _compare_suite_shape(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    keys = ["run_kind", "song_count", "seed_step"]
    mismatches = []
    missing = []
    baseline_schema = int(baseline.get("schema_version", 0) or 0)
    candidate_schema = int(candidate.get("schema_version", 0) or 0)
    if baseline_schema < 3 or candidate_schema < 3:
        mismatches.append({
            "key": "schema_version",
            "baseline": baseline.get("schema_version"),
            "candidate": candidate.get("schema_version"),
            "expected": "both >= 3",
        })
    for key in keys:
        if baseline.get(key) != candidate.get(key):
            mismatches.append({"key": key, "baseline": baseline.get(key), "candidate": candidate.get(key)})
    baseline_runs = baseline.get("runs", [])
    candidate_runs = candidate.get("runs", [])
    if len(baseline_runs) != len(candidate_runs):
        mismatches.append({"key": "run_count", "baseline": len(baseline_runs), "candidate": len(candidate_runs)})

    run_contract_keys = [
        "run_index",
        "repeat_index",
        "song_index",
        "song_id",
        "audio_path",
        "start_time",
        "end_time",
        "seed",
        "sequence_count",
        "song_length_ms",
        "mode",
        "batch_size",
        "batch_start_index",
    ]
    for index in range(min(len(baseline_runs), len(candidate_runs))):
        baseline_run = baseline_runs[index]
        candidate_run = candidate_runs[index]
        for key in run_contract_keys:
            if key not in baseline_run and key not in candidate_run:
                continue
            if key not in baseline_run or key not in candidate_run:
                missing.append({
                    "index": index,
                    "key": key,
                    "baseline_key": _suite_run_key(baseline_run, index),
                    "candidate_key": _suite_run_key(candidate_run, index),
                })
                continue
            if baseline_run.get(key) != candidate_run.get(key):
                mismatches.append({
                    "index": index,
                    "key": key,
                    "baseline": baseline_run.get(key),
                    "candidate": candidate_run.get(key),
                    "baseline_key": _suite_run_key(baseline_run, index),
                    "candidate_key": _suite_run_key(candidate_run, index),
                })

    if baseline.get("run_kind") == "serial_multi_song" or candidate.get("run_kind") == "serial_multi_song":
        if int(baseline.get("song_count", 0) or 0) < 5 or int(candidate.get("song_count", 0) or 0) < 5:
            mismatches.append({
                "key": "song_count",
                "baseline": baseline.get("song_count"),
                "candidate": candidate.get("song_count"),
                "expected": "serial_multi_song performance evidence should use at least 5 songs",
            })

    passed = len(mismatches) == 0 and len(missing) == 0
    print("Suite shape/contract")
    print(f"  {'PASS' if passed else 'FAIL'}")
    for mismatch in mismatches[:8]:
        print(f"  {mismatch['key']}: baseline={mismatch.get('baseline')!r}, candidate={mismatch.get('candidate')!r}")
    if missing:
        print(f"  missing_contract_fields: {len(missing)}")
    print()
    return {"pass": passed, "mismatches": mismatches, "missing": missing}


def _compare_suite_token_hashes(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_runs = baseline.get("runs", [])
    candidate_runs = candidate.get("runs", [])
    paired = min(len(baseline_runs), len(candidate_runs))
    mismatches = []
    missing = []
    for index in range(paired):
        baseline_run = baseline_runs[index]
        candidate_run = candidate_runs[index]
        baseline_hash = baseline_run.get("main_token_sha256")
        candidate_hash = candidate_run.get("main_token_sha256")
        if baseline_hash is None or candidate_hash is None:
            missing.append(_suite_run_key(baseline_run, index))
            continue
        if (
                baseline_hash != candidate_hash
                or baseline_run.get("main_token_count") != candidate_run.get("main_token_count")
                or baseline_run.get("main_generated_tokens") != candidate_run.get("main_generated_tokens")
        ):
            mismatches.append({
                "index": index,
                "baseline_key": _suite_run_key(baseline_run, index),
                "candidate_key": _suite_run_key(candidate_run, index),
                "baseline_hash": baseline_hash,
                "candidate_hash": candidate_hash,
                "baseline_tokens": baseline_run.get("main_token_count"),
                "candidate_tokens": candidate_run.get("main_token_count"),
            })
    passed = len(mismatches) == 0 and len(missing) == 0 and len(baseline_runs) == len(candidate_runs)
    print("Suite token equivalence")
    if passed:
        print(f"  PASS ({paired} paired runs)")
    else:
        print(f"  FAIL (paired={paired}, mismatches={len(mismatches)}, missing={len(missing)})")
        for mismatch in mismatches[:5]:
            print(
                f"  {mismatch['baseline_key']} -> {mismatch['candidate_key']}: "
                f"baseline_hash={mismatch['baseline_hash']}, candidate_hash={mismatch['candidate_hash']}"
            )
        if missing:
            print(f"  missing_hashes: {', '.join(missing[:5])}")
    print()
    return {
        "pass": passed,
        "paired_runs": paired,
        "mismatches": mismatches,
        "missing": missing,
    }


def _compare_suite_metric_block(
        baseline_block: dict[str, Any],
        candidate_block: dict[str, Any],
        *,
        regression_tolerance_pct: float,
) -> dict[str, Any]:
    metric_specs = {
        "tokens_per_second": True,
        "model_elapsed_seconds": False,
        "wall_seconds": False,
    }
    metrics = {}
    for key, higher_is_better in metric_specs.items():
        metrics[key] = _metric_comparison(
            float(baseline_block.get(key, 0.0) or 0.0),
            float(candidate_block.get(key, 0.0) or 0.0),
            higher_is_better=higher_is_better,
            tolerance_pct=regression_tolerance_pct,
        )
    generated_tokens_match = baseline_block.get("generated_tokens") == candidate_block.get("generated_tokens")
    records_match = baseline_block.get("records") == candidate_block.get("records")
    if "runs" in baseline_block or "runs" in candidate_block:
        records_match = records_match and baseline_block.get("runs") == candidate_block.get("runs")
    return {
        "pass": all(metric["pass"] for metric in metrics.values()) and generated_tokens_match and records_match,
        "metrics": metrics,
        "generated_tokens_match": generated_tokens_match,
        "records_match": records_match,
    }


def _aggregate_suite_timing_runs(selected: list[dict[str, Any]]) -> dict[str, Any]:
    generated_tokens = sum(int(run.get("timing_generated_tokens") or 0) for run in selected)
    model_elapsed_seconds = sum(float(run.get("timing_model_elapsed_seconds") or 0.0) for run in selected)
    wall_seconds = sum(float(run.get("timing_wall_seconds") or 0.0) for run in selected)
    return {
        "runs": len(selected),
        "records": len(selected),
        "generated_tokens": generated_tokens,
        "model_elapsed_seconds": model_elapsed_seconds,
        "wall_seconds": wall_seconds,
        "tokens_per_second": generated_tokens / model_elapsed_seconds if model_elapsed_seconds > 0 else 0.0,
    }


def _aggregate_suite_main_runs(selected: list[dict[str, Any]]) -> dict[str, Any]:
    generated_tokens = sum(int(run.get("main_generated_tokens") or 0) for run in selected)
    model_elapsed_seconds = sum(float(run.get("main_model_elapsed_seconds") or 0.0) for run in selected)
    wall_seconds = sum(float(run.get("main_wall_seconds") or 0.0) for run in selected)
    return {
        "runs": len(selected),
        "records": len(selected),
        "generated_tokens": generated_tokens,
        "model_elapsed_seconds": model_elapsed_seconds,
        "wall_seconds": wall_seconds,
        "tokens_per_second": generated_tokens / model_elapsed_seconds if model_elapsed_seconds > 0 else 0.0,
    }


def _compare_suite_per_song(
        baseline: dict[str, Any],
        candidate: dict[str, Any],
        *,
        scope: str,
        regression_tolerance_pct: float,
) -> dict[str, Any]:
    baseline_by_song: dict[int, list[dict[str, Any]]] = {}
    candidate_by_song: dict[int, list[dict[str, Any]]] = {}
    for run in _suite_runs_for_scope(baseline, scope):
        baseline_by_song.setdefault(int(run.get("song_index", -1)), []).append(run)
    for run in _suite_runs_for_scope(candidate, scope):
        candidate_by_song.setdefault(int(run.get("song_index", -1)), []).append(run)

    baseline_keys = set(baseline_by_song)
    candidate_keys = set(candidate_by_song)
    missing = sorted(baseline_keys.symmetric_difference(candidate_keys))
    songs = []
    for song_index in sorted(baseline_keys & candidate_keys):
        baseline_runs = baseline_by_song[song_index]
        candidate_runs = candidate_by_song[song_index]
        report = _compare_suite_metric_block(
            _aggregate_suite_main_runs(baseline_runs),
            _aggregate_suite_main_runs(candidate_runs),
            regression_tolerance_pct=regression_tolerance_pct,
        )
        report.update({
            "song_index": song_index,
            "song_id": baseline_runs[0].get("song_id"),
            "runs": len(baseline_runs),
        })
        songs.append(report)

    failed = [song for song in songs if not song.get("pass", False)]
    passed = not missing and not failed
    print(f"Suite per-song main-generation no-regression ({scope})")
    print(f"  {'PASS' if passed else 'FAIL'} (songs={len(songs)}, failed={len(failed)}, missing={len(missing)})")
    for song in failed[:5]:
        metric = song["metrics"]["tokens_per_second"]
        print(
            "  song_index={song_index}, song_id={song_id}: "
            "baseline={baseline:.3f}, candidate={candidate:.3f}".format(
                song_index=song["song_index"],
                song_id=song.get("song_id"),
                baseline=float(metric["baseline"]),
                candidate=float(metric["candidate"]),
            )
        )
    if missing:
        print(f"  missing_song_indices: {missing[:8]}")
    print()
    return {
        "pass": passed,
        "songs": songs,
        "failed": failed,
        "missing_song_indices": missing,
    }


def _compare_suite_optional_block(
        name: str,
        baseline_block: Any,
        candidate_block: Any,
        *,
        regression_tolerance_pct: float,
) -> dict[str, Any]:
    if not isinstance(baseline_block, dict) or not isinstance(candidate_block, dict):
        report = {
            "available": False,
            "pass": False,
            "missing": True,
            "metrics": {},
            "generated_tokens_match": False,
            "records_match": False,
        }
        print(name)
        print("  not available")
        print()
        return report

    report = _compare_suite_metric_block(
        baseline_block,
        candidate_block,
        regression_tolerance_pct=regression_tolerance_pct,
    )
    report["available"] = True
    _print_suite_metric_block(name, report)
    return report


def _print_suite_metric_block(name: str, report: dict[str, Any]) -> None:
    print(name)
    for metric_name, metric in report["metrics"].items():
        _compare_number(
            metric_name,
            float(metric["baseline"]),
            float(metric["candidate"]),
            higher_is_better=bool(metric["higher_is_better"]),
        )
    print(f"  generated_tokens_match: {report['generated_tokens_match']}")
    print(f"  records_match: {report['records_match']}")
    print(f"  no_regression_gate: {'PASS' if report['pass'] else 'FAIL'}")
    print()


def compare_suite_manifests(
        baseline_path: Path,
        candidate_path: Path,
        *,
        scope: str,
        regression_tolerance_pct: float = 0.0,
) -> dict[str, Any]:
    baseline = _load_json(baseline_path)
    candidate = _load_json(candidate_path)
    baseline_block = _suite_aggregate(baseline, scope)
    candidate_block = _suite_aggregate(candidate, scope)
    report: dict[str, Any] = {
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "scope": scope,
        "regression_tolerance_pct": regression_tolerance_pct,
        "shape": {},
        "scope_availability": {},
        "token_equivalence": {},
        "performance": {},
        "timing_context": {},
        "segments": {},
        "per_song": {},
        "cold_run0": {},
    }

    print(f"Baseline suite:  {baseline_path}")
    print(f"Candidate suite: {candidate_path}")
    print(f"Scope:           {scope}")
    print()

    report["shape"] = _compare_suite_shape(baseline, candidate)
    report["scope_availability"] = _compare_suite_scope_availability(baseline, candidate, scope=scope)
    report["token_equivalence"] = _compare_suite_token_hashes(baseline, candidate)
    report["performance"] = _compare_suite_metric_block(
        baseline_block,
        candidate_block,
        regression_tolerance_pct=regression_tolerance_pct,
    )
    _print_suite_metric_block(f"Suite no-regression ({scope})", report["performance"])

    report["segments"] = {
        "first_records": _compare_suite_optional_block(
            f"Suite first-record no-regression ({scope})",
            baseline_block.get("first_records"),
            candidate_block.get("first_records"),
            regression_tolerance_pct=regression_tolerance_pct,
        ),
        "remaining_records": _compare_suite_optional_block(
            f"Suite remaining-record no-regression ({scope})",
            baseline_block.get("remaining_records"),
            candidate_block.get("remaining_records"),
            regression_tolerance_pct=regression_tolerance_pct,
        ),
    }

    baseline_timing_block = _aggregate_suite_timing_runs(_suite_runs_for_scope(baseline, scope))
    candidate_timing_block = _aggregate_suite_timing_runs(_suite_runs_for_scope(candidate, scope))
    report["timing_context"] = _compare_suite_optional_block(
        f"Suite timing-context no-regression ({scope})",
        baseline_timing_block,
        candidate_timing_block,
        regression_tolerance_pct=regression_tolerance_pct,
    )
    report["per_song"] = _compare_suite_per_song(
        baseline,
        candidate,
        scope=scope,
        regression_tolerance_pct=regression_tolerance_pct,
    )

    baseline_runs = baseline.get("runs", [])
    candidate_runs = candidate.get("runs", [])
    if baseline_runs and candidate_runs:
        report["cold_run0"] = _compare_suite_metric_block(
            {
                "records": 1,
                "generated_tokens": baseline_runs[0].get("main_generated_tokens"),
                "tokens_per_second": baseline_runs[0].get("main_tokens_per_second"),
                "model_elapsed_seconds": baseline_runs[0].get("main_model_elapsed_seconds"),
                "wall_seconds": baseline_runs[0].get("main_wall_seconds"),
            },
            {
                "records": 1,
                "generated_tokens": candidate_runs[0].get("main_generated_tokens"),
                "tokens_per_second": candidate_runs[0].get("main_tokens_per_second"),
                "model_elapsed_seconds": candidate_runs[0].get("main_model_elapsed_seconds"),
                "wall_seconds": candidate_runs[0].get("main_wall_seconds"),
            },
            regression_tolerance_pct=regression_tolerance_pct,
        )
        _print_suite_metric_block("Cold run0 diagnostic (not scoped acceptance unless selected)", report["cold_run0"])
    return report


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


def _parse_compare_labels(*, label: str, labels: str | None, strict_full_song: bool) -> list[str]:
    if strict_full_song:
        return ["main_generation", "timing_context"]
    if labels is None:
        return [label]
    parsed = [item.strip() for item in labels.split(",") if item.strip()]
    if not parsed:
        raise ValueError("--labels must include at least one generation label")
    return parsed


def compare_profiles_for_labels(
        baseline_path: Path,
        candidate_path: Path,
        *,
        labels: list[str],
        regression_tolerance_pct: float = 0.0,
) -> dict[str, Any]:
    reports = {}
    for index, label in enumerate(labels):
        if index:
            print()
            print("=" * 80)
            print()
        reports[label] = compare_profiles(
            baseline_path,
            candidate_path,
            label=label,
            regression_tolerance_pct=regression_tolerance_pct,
        )
    return {
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "labels": labels,
        "reports": reports,
        "same_calculation_pass": all(
            report.get("same_calculation", {}).get("pass", False)
            for report in reports.values()
        ),
        "token_equivalence_pass": all(
            report.get("token_equivalence", {}).get("pass", False)
            for report in reports.values()
        ),
        "performance_pass": all(
            report.get("performance", {}).get("pass", False)
            for report in reports.values()
        ),
    }


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
    parser.add_argument(
        "--compare-suite",
        nargs=2,
        metavar=("BASE_MANIFEST", "CANDIDATE_MANIFEST"),
        type=Path,
        help="Compare two profile_inference_suite suite_manifest.json files.",
    )
    parser.add_argument(
        "--suite-scope",
        choices=["all_runs", "warmed_runs"],
        default="warmed_runs",
        help="Aggregate scope for --compare-suite no-regression gating. Default: warmed_runs.",
    )
    parser.add_argument("--label", default="main_generation", help="Generation label to compare.")
    parser.add_argument(
        "--labels",
        default=None,
        help=(
            "Comma-separated generation labels to compare in one command, e.g. "
            "main_generation,timing_context."
        ),
    )
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
        "--require-suite-segment-no-regression",
        "--require-suite-segments",
        dest="require_suite_segment_no_regression",
        action="store_true",
        help=(
            "For --compare-suite, exit nonzero if selected-scope first-record or remaining-record "
            "main-generation segment metrics regress."
        ),
    )
    parser.add_argument(
        "--require-suite-timing-no-regression",
        "--require-suite-timing",
        dest="require_suite_timing_no_regression",
        action="store_true",
        help="For --compare-suite, exit nonzero if selected-scope timing-context metrics regress.",
    )
    parser.add_argument(
        "--require-per-song-no-regression",
        "--require-per-song-non-regression",
        dest="require_per_song_no_regression",
        action="store_true",
        help="For --compare-suite, exit nonzero if any selected-scope song-level main-generation metrics regress.",
    )
    parser.add_argument(
        "--require-mode-contract",
        action="store_true",
        help=(
            "For --compare-suite, exit nonzero if suite schema, run order, song/window/seed, "
            "or available mode/batch contract fields differ."
        ),
    )
    parser.add_argument(
        "--gate-cold-run0",
        action="store_true",
        help=(
            "For --compare-suite, also fail if run0 regresses. This is not included in --strict "
            "because warmed/batch claims report cold cost separately."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Shortcut for default strict comparison gates. In suite mode this includes contract, "
            "token, aggregate, segment, timing, per-song, and scope checks; cold run0 still "
            "requires --gate-cold-run0."
        ),
    )
    parser.add_argument(
        "--strict-full-song",
        action="store_true",
        help=(
            "For --compare, shortcut for --strict --labels main_generation,timing_context. "
            "Use for full-song promotion gates."
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Write a machine-readable comparison report.",
    )
    args = parser.parse_args()
    if args.compare and args.compare_suite:
        parser.error("use either --compare or --compare-suite, not both")
    if args.strict_full_song and not args.compare:
        parser.error("--strict-full-song requires --compare")
    if args.compare:
        try:
            labels = _parse_compare_labels(
                label=args.label,
                labels=args.labels,
                strict_full_song=args.strict_full_song,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if len(labels) == 1 and not args.labels and not args.strict_full_song:
            report = compare_profiles(
                args.compare[0],
                args.compare[1],
                label=labels[0],
                regression_tolerance_pct=args.regression_tolerance_pct,
            )
        else:
            report = compare_profiles_for_labels(
                args.compare[0],
                args.compare[1],
                labels=labels,
                regression_tolerance_pct=args.regression_tolerance_pct,
            )
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

        strict_compare = args.strict or args.strict_full_song
        require_contract_match = args.require_contract_match or strict_compare
        require_token_equivalence = args.require_token_equivalence or strict_compare
        require_no_regression = args.require_no_regression or strict_compare
        if "reports" in report:
            failed = (
                (require_contract_match and not report.get("same_calculation_pass", False))
                or (require_token_equivalence and not report.get("token_equivalence_pass", False))
                or (require_no_regression and not report.get("performance_pass", False))
            )
        else:
            failed = (
                (require_contract_match and not report["same_calculation"].get("pass", False))
                or (require_token_equivalence and not report["token_equivalence"].get("pass", False))
                or (require_no_regression and not report["performance"].get("pass", False))
            )
        raise SystemExit(1 if failed else 0)
    elif args.compare_suite:
        report = compare_suite_manifests(
            args.compare_suite[0],
            args.compare_suite[1],
            scope=args.suite_scope,
            regression_tolerance_pct=args.regression_tolerance_pct,
        )
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

        require_contract_match = args.require_contract_match or args.strict
        require_token_equivalence = args.require_token_equivalence or args.strict
        require_no_regression = args.require_no_regression or args.strict
        require_segment_no_regression = args.require_suite_segment_no_regression or args.strict
        require_timing_no_regression = args.require_suite_timing_no_regression or args.strict
        failed = (
            ((require_contract_match or args.require_mode_contract) and not report["shape"].get("pass", False))
            or (args.strict and not report["scope_availability"].get("pass", False))
            or (require_token_equivalence and not report["token_equivalence"].get("pass", False))
            or (require_no_regression and not report["performance"].get("pass", False))
            or (
                require_segment_no_regression
                and not all(block.get("pass", False) for block in report["segments"].values())
            )
            or (
                require_timing_no_regression
                and not report["timing_context"].get("pass", False)
            )
            or (
                (args.require_per_song_no_regression or args.strict)
                and not report["per_song"].get("pass", False)
            )
            or (args.gate_cold_run0 and not report["cold_run0"].get("pass", False))
        )
        raise SystemExit(1 if failed else 0)
    elif args.profile:
        summarize(args.profile, limit=args.limit)
    else:
        parser.error("provide a profile path, --compare BASELINE CANDIDATE, or --compare-suite BASE CANDIDATE")


if __name__ == "__main__":
    main()
