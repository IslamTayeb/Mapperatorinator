from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
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
            "result_file_sha256",
            "result_file_size_bytes",
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
            cuda_event = values.get("model_generate_cuda_event_seconds")
            host_gap = values.get("model_generate_host_gap_seconds")
            ledger_text = ""
            if isinstance(cuda_event, (int, float)) and isinstance(host_gap, (int, float)):
                ledger_text = f", generate_cuda={_fmt_seconds(cuda_event)}, host_gap={_fmt_seconds(host_gap)}"
            print(
                f"  {context}: model={_fmt_seconds(elapsed)}, wall={_fmt_seconds(wall)}, "
                f"tokens={tokens}, tok/s={tok_s:.1f}, records={records}{ledger_text}"
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


def _compare_suite_output_hashes(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_runs = baseline.get("runs", [])
    candidate_runs = candidate.get("runs", [])
    paired = min(len(baseline_runs), len(candidate_runs))
    mismatches = []
    missing = []
    for index in range(paired):
        baseline_run = baseline_runs[index]
        candidate_run = candidate_runs[index]
        baseline_hash = baseline_run.get("result_file_sha256")
        candidate_hash = candidate_run.get("result_file_sha256")
        if baseline_hash is None or candidate_hash is None:
            missing.append(_suite_run_key(baseline_run, index))
            continue
        if (
                baseline_hash != candidate_hash
                or baseline_run.get("result_file_size_bytes") != candidate_run.get("result_file_size_bytes")
        ):
            mismatches.append({
                "index": index,
                "baseline_key": _suite_run_key(baseline_run, index),
                "candidate_key": _suite_run_key(candidate_run, index),
                "baseline_hash": baseline_hash,
                "candidate_hash": candidate_hash,
                "baseline_size_bytes": baseline_run.get("result_file_size_bytes"),
                "candidate_size_bytes": candidate_run.get("result_file_size_bytes"),
            })
    passed = len(mismatches) == 0 and len(missing) == 0 and len(baseline_runs) == len(candidate_runs)
    print("Suite output artifact equivalence")
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
            print(f"  missing_output_hashes: {', '.join(missing[:5])}")
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


STATIC_SERVER_TOKEN_STATUS = "not_checked_shared_server_rng"
CONTINUOUS_SCHEDULER_TOKEN_STATUS = "scheduler_only_scripted_tokens"
CONTINUOUS_SCHEDULER_STATE_HASH_FIELDS = [
    "initial_rng_state_hash",
    "final_rng_state_hash",
    "logits_processor_state_hash",
    "cache_state_hash",
]


def _continuous_token_sha256(tokens: list[int]) -> str:
    payload = json.dumps([int(token) for token in tokens], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compare_static_server_contract(
        baseline: dict[str, Any],
        candidate: dict[str, Any],
        *,
        allow_server_batch_timeout_change: bool,
        allow_server_max_batch_size_change: bool,
) -> dict[str, Any]:
    mismatches = []
    for key in ["run_kind", "song_count", "repeats", "max_workers"]:
        if baseline.get(key) != candidate.get(key):
            mismatches.append({"key": key, "baseline": baseline.get(key), "candidate": candidate.get(key)})

    baseline_fingerprint = dict(baseline.get("server_config_fingerprint") or {})
    candidate_fingerprint = dict(candidate.get("server_config_fingerprint") or {})
    if allow_server_batch_timeout_change:
        baseline_fingerprint.pop("server_batch_timeout", None)
        candidate_fingerprint.pop("server_batch_timeout", None)
    if allow_server_max_batch_size_change:
        baseline_fingerprint.pop("max_batch_size", None)
        candidate_fingerprint.pop("max_batch_size", None)
    if baseline_fingerprint != candidate_fingerprint:
        mismatches.append({
            "key": "server_config_fingerprint",
            "baseline": baseline_fingerprint,
            "candidate": candidate_fingerprint,
        })

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
        "beatmap_path",
        "start_time",
        "end_time",
        "seed",
        "requested_seed",
        "sequence_count",
        "song_length_ms",
    ]
    for index in range(min(len(baseline_runs), len(candidate_runs))):
        baseline_run = baseline_runs[index]
        candidate_run = candidate_runs[index]
        for key in run_contract_keys:
            if key not in baseline_run and key not in candidate_run:
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

    passed = not mismatches
    print("Static-server contract")
    print(f"  {'PASS' if passed else 'FAIL'}")
    for mismatch in mismatches[:8]:
        print(f"  {mismatch['key']}: baseline={mismatch.get('baseline')!r}, candidate={mismatch.get('candidate')!r}")
    print()
    return {
        "pass": passed,
        "mismatches": mismatches,
        "allow_server_batch_timeout_change": allow_server_batch_timeout_change,
        "allow_server_max_batch_size_change": allow_server_max_batch_size_change,
    }


def _compare_static_server_result_class(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_aggregate = baseline.get("aggregate", {})
    candidate_aggregate = candidate.get("aggregate", {})
    checks = {
        "baseline_result_class": baseline_aggregate.get("result_class"),
        "candidate_result_class": candidate_aggregate.get("result_class"),
        "baseline_server_batch_observed": bool(baseline_aggregate.get("server_batch_observed", False)),
        "candidate_server_batch_observed": bool(candidate_aggregate.get("server_batch_observed", False)),
        "baseline_same_calculation": bool(baseline_aggregate.get("same_calculation", baseline.get("same_calculation", True))),
        "candidate_same_calculation": bool(candidate_aggregate.get("same_calculation", candidate.get("same_calculation", True))),
        "baseline_throughput_claim_scope": baseline_aggregate.get(
            "throughput_claim_scope",
            baseline.get("throughput_claim_scope"),
        ),
        "candidate_throughput_claim_scope": candidate_aggregate.get(
            "throughput_claim_scope",
            candidate.get("throughput_claim_scope"),
        ),
    }
    passed = (
        checks["baseline_result_class"] == "static_server_batch"
        and checks["candidate_result_class"] == "static_server_batch"
        and checks["baseline_server_batch_observed"]
        and checks["candidate_server_batch_observed"]
        and not checks["baseline_same_calculation"]
        and not checks["candidate_same_calculation"]
        and checks["baseline_throughput_claim_scope"] == checks["candidate_throughput_claim_scope"]
    )
    print("Static-server result class")
    print(f"  {'PASS' if passed else 'FAIL'}")
    for key, value in checks.items():
        print(f"  {key}: {value}")
    print()
    return {"pass": passed, **checks}


def _compare_static_server_token_status(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    statuses = []
    mismatches = []
    for side, manifest in [("baseline", baseline), ("candidate", candidate)]:
        for index, run in enumerate(manifest.get("runs", [])):
            status = run.get("token_equivalence_status")
            statuses.append({"side": side, "index": index, "status": status})
            if status != STATIC_SERVER_TOKEN_STATUS:
                mismatches.append({
                    "side": side,
                    "index": index,
                    "key": _suite_run_key(run, index),
                    "status": status,
                    "expected": STATIC_SERVER_TOKEN_STATUS,
                })
    passed = not mismatches
    print("Static-server token-equivalence status")
    print(f"  {'PASS' if passed else 'FAIL'}")
    if passed:
        print(f"  all runs are labelled {STATIC_SERVER_TOKEN_STATUS}")
    else:
        for mismatch in mismatches[:8]:
            print(
                "  {side} {key}: status={status!r}, expected={expected!r}".format(
                    **mismatch
                )
            )
    print()
    return {"pass": passed, "statuses": statuses, "mismatches": mismatches}


def _float_close(left: Any, right: Any, *, rel_tol: float = 1e-6, abs_tol: float = 1e-6) -> bool:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return left == right
    return abs(float(left) - float(right)) <= max(abs_tol, rel_tol * max(abs(float(left)), abs(float(right)), 1.0))


def _static_server_percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    index = max(0, min(len(sorted_values) - 1, int(len(sorted_values) * percentile + 0.999999) - 1))
    return sorted_values[index]


def _static_server_batch_summary_from_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, dict[str, Any]] = {}
    for run in runs:
        batch_summary = run.get("generation_batch_summary")
        if not isinstance(batch_summary, dict):
            continue
        by_label = batch_summary.get("by_label")
        if not isinstance(by_label, dict):
            continue
        for label, label_summary in by_label.items():
            if not isinstance(label_summary, dict):
                continue
            target = aggregate.setdefault(
                str(label),
                {
                    "records": 0,
                    "modes": {},
                    "batch_size_histogram": {},
                    "server_batch_size_histogram": {},
                    "server_batch_count": 0,
                    "server_request_record_count": 0,
                    "server_total_queue_wait_seconds": 0.0,
                    "server_max_queue_wait_seconds": 0.0,
                    "server_total_first_queue_wait_seconds": 0.0,
                    "server_max_first_queue_wait_seconds": 0.0,
                    "server_batching_modes": {},
                    "server_elapsed_seconds_attributions": {},
                    "server_batch_count_attributed": 0,
                    "server_unique_batch_size_histogram": {},
                    "server_unique_batch_elapsed_seconds_sum": 0.0,
                    "server_unique_batch_elapsed_seconds_max": 0.0,
                    "_seen_server_batch_ids": set(),
                },
            )
            target["records"] += int(label_summary.get("records", 0) or 0)
            target["server_batch_count_attributed"] += int(label_summary.get("server_batch_count", 0) or 0)
            target["server_request_record_count"] += int(
                label_summary.get("server_request_record_count", 0) or 0
            )
            target["server_total_queue_wait_seconds"] += float(
                label_summary.get("server_total_queue_wait_seconds", 0.0) or 0.0
            )
            target["server_max_queue_wait_seconds"] = max(
                float(target["server_max_queue_wait_seconds"]),
                float(label_summary.get("server_max_queue_wait_seconds", 0.0) or 0.0),
            )
            target["server_total_first_queue_wait_seconds"] += float(
                label_summary.get("server_total_first_queue_wait_seconds", 0.0) or 0.0
            )
            target["server_max_first_queue_wait_seconds"] = max(
                float(target["server_max_first_queue_wait_seconds"]),
                float(label_summary.get("server_max_first_queue_wait_seconds", 0.0) or 0.0),
            )
            for field in (
                "modes",
                "batch_size_histogram",
                "server_batch_size_histogram",
                "server_batching_modes",
                "server_elapsed_seconds_attributions",
            ):
                values = label_summary.get(field)
                if not isinstance(values, dict):
                    continue
                for key, value in values.items():
                    target[field][str(key)] = int(target[field].get(str(key), 0)) + int(value)
            server_batches = label_summary.get("server_batches")
            if isinstance(server_batches, list):
                for batch in server_batches:
                    if not isinstance(batch, dict):
                        continue
                    batch_id = batch.get("batch_id")
                    batch_size = batch.get("batch_size")
                    if batch_id is None or batch_size is None:
                        continue
                    seen_key = str(batch_id)
                    if seen_key in target["_seen_server_batch_ids"]:
                        continue
                    target["_seen_server_batch_ids"].add(seen_key)
                    target["server_batch_count"] += 1
                    size_key = str(int(batch_size))
                    target["server_unique_batch_size_histogram"][size_key] = int(
                        target["server_unique_batch_size_histogram"].get(size_key, 0)
                    ) + 1
                    elapsed = batch.get("elapsed_seconds")
                    if isinstance(elapsed, (int, float)):
                        target["server_unique_batch_elapsed_seconds_sum"] += float(elapsed)
                        target["server_unique_batch_elapsed_seconds_max"] = max(
                            float(target["server_unique_batch_elapsed_seconds_max"]),
                            float(elapsed),
                        )
            elif int(label_summary.get("server_batch_count", 0) or 0) > 0:
                target["server_batch_count"] += int(label_summary.get("server_batch_count", 0) or 0)
    for label_summary in aggregate.values():
        label_summary.pop("_seen_server_batch_ids", None)
    return {"by_label": aggregate}


def _static_server_batch_observed(batching: dict[str, Any]) -> bool:
    for label_summary in (batching.get("by_label") or {}).values():
        hist = label_summary.get("server_unique_batch_size_histogram") or label_summary.get(
            "server_batch_size_histogram"
        )
        if isinstance(hist, dict) and any(int(size) > 1 and int(count) > 0 for size, count in hist.items()):
            return True
    return False


def _static_server_batch_ledger_failures(manifest: dict[str, Any], side: str) -> list[dict[str, Any]]:
    failures = []
    for run_index, run in enumerate(manifest.get("runs", [])):
        batch_summary = run.get("generation_batch_summary")
        if not isinstance(batch_summary, dict):
            continue
        for label, label_summary in (batch_summary.get("by_label") or {}).items():
            if not isinstance(label_summary, dict):
                failures.append({
                    "side": side,
                    "path": f"runs[{run_index}].generation_batch_summary.by_label.{label}",
                    "reason": "label summary is not a dict",
                })
                continue
            batches = label_summary.get("server_batches")
            batch_count = int(label_summary.get("server_batch_count", 0) or 0)
            if isinstance(batches, list):
                if len(batches) != batch_count:
                    failures.append({
                        "side": side,
                        "path": f"runs[{run_index}].generation_batch_summary.by_label.{label}.server_batches",
                        "reason": f"len(server_batches)={len(batches)} != server_batch_count={batch_count}",
                    })
                for batch_index, batch in enumerate(batches):
                    if not isinstance(batch, dict):
                        failures.append({
                            "side": side,
                            "path": (
                                f"runs[{run_index}].generation_batch_summary.by_label."
                                f"{label}.server_batches[{batch_index}]"
                            ),
                            "reason": "batch entry is not a dict",
                        })
                        continue
                    for key in ("batch_id", "batch_size", "request_count", "work_items"):
                        if not isinstance(batch.get(key), int):
                            failures.append({
                                "side": side,
                                "path": (
                                    f"runs[{run_index}].generation_batch_summary.by_label."
                                    f"{label}.server_batches[{batch_index}].{key}"
                                ),
                                "reason": f"expected int, got {batch.get(key)!r}",
                            })
                    for key in ("elapsed_seconds", "queue_wait_seconds"):
                        value = batch.get(key)
                        if value is not None and not isinstance(value, (int, float)):
                            failures.append({
                                "side": side,
                                "path": (
                                    f"runs[{run_index}].generation_batch_summary.by_label."
                                    f"{label}.server_batches[{batch_index}].{key}"
                                ),
                                "reason": f"expected number or null, got {value!r}",
                            })
            elif batch_count:
                failures.append({
                    "side": side,
                    "path": f"runs[{run_index}].generation_batch_summary.by_label.{label}.server_batches",
                    "reason": "missing server_batches for nonzero server_batch_count",
                })
    return failures


def _append_static_server_aggregate_failure(
        failures: list[dict[str, Any]],
        side: str,
        key: str,
        expected: Any,
        actual: Any,
) -> None:
    if _float_close(expected, actual):
        return
    failures.append({
        "side": side,
        "path": f"aggregate.{key}",
        "expected": expected,
        "actual": actual,
    })


def _validate_static_server_manifest(manifest: dict[str, Any], *, side: str) -> dict[str, Any]:
    failures = []
    aggregate = manifest.get("aggregate")
    runs = manifest.get("runs")
    if not isinstance(aggregate, dict):
        failures.append({"side": side, "path": "aggregate", "reason": "missing aggregate"})
        aggregate = {}
    if not isinstance(runs, list):
        failures.append({"side": side, "path": "runs", "reason": "missing runs list"})
        runs = []

    if manifest.get("schema_version") != 1:
        failures.append({"side": side, "path": "schema_version", "expected": 1, "actual": manifest.get("schema_version")})
    if manifest.get("run_kind") != "static_server_batch":
        failures.append({
            "side": side,
            "path": "run_kind",
            "expected": "static_server_batch",
            "actual": manifest.get("run_kind"),
        })
    if bool(manifest.get("same_calculation", True)):
        failures.append({"side": side, "path": "same_calculation", "expected": False, "actual": manifest.get("same_calculation")})
    if manifest.get("throughput_claim_scope") != "static_ipc_concurrent_full_song_requests":
        failures.append({
            "side": side,
            "path": "throughput_claim_scope",
            "expected": "static_ipc_concurrent_full_song_requests",
            "actual": manifest.get("throughput_claim_scope"),
        })
    if manifest.get("token_equivalence_status") != STATIC_SERVER_TOKEN_STATUS:
        failures.append({
            "side": side,
            "path": "token_equivalence_status",
            "expected": STATIC_SERVER_TOKEN_STATUS,
            "actual": manifest.get("token_equivalence_status"),
        })

    request_walls = sorted(float(run.get("request_wall_seconds") or 0.0) for run in runs)
    main_tokens = sum(int(run.get("main_generated_tokens") or 0) for run in runs)
    timing_tokens = sum(int(run.get("timing_generated_tokens") or 0) for run in runs)
    main_model_elapsed = sum(float(run.get("main_model_elapsed_seconds") or 0.0) for run in runs)
    timing_model_elapsed = sum(float(run.get("timing_model_elapsed_seconds") or 0.0) for run in runs)
    scheduler_wall = float(aggregate.get("scheduler_wall_seconds", 0.0) or 0.0)
    expected_batched = _static_server_batch_summary_from_runs(runs)
    server_batch_observed = _static_server_batch_observed(expected_batched)
    result_class = "static_server_batch" if server_batch_observed else "static_server_no_batch_observed"

    expected_values = {
        "runs": len(runs),
        "result_class": result_class,
        "server_batch_observed": server_batch_observed,
        "same_calculation": False,
        "throughput_claim_scope": "static_ipc_concurrent_full_song_requests",
        "token_equivalence_status": STATIC_SERVER_TOKEN_STATUS,
        "main_generated_tokens": main_tokens,
        "timing_generated_tokens": timing_tokens,
        "request_wall_seconds_sum": sum(request_walls),
        "request_wall_seconds_max": max(request_walls) if request_walls else 0.0,
        "request_wall_seconds_p95": _static_server_percentile(request_walls, 0.95),
        "main_model_elapsed_seconds_sum": main_model_elapsed,
        "timing_model_elapsed_seconds_sum": timing_model_elapsed,
        "main_tokens_per_scheduler_second": main_tokens / scheduler_wall if scheduler_wall > 0 else 0.0,
        "timing_tokens_per_scheduler_second": timing_tokens / scheduler_wall if scheduler_wall > 0 else 0.0,
        "main_tokens_per_request_model_second_attributed": (
            main_tokens / main_model_elapsed if main_model_elapsed > 0 else 0.0
        ),
        "timing_tokens_per_request_model_second_attributed": (
            timing_tokens / timing_model_elapsed if timing_model_elapsed > 0 else 0.0
        ),
    }
    for key, expected in expected_values.items():
        _append_static_server_aggregate_failure(failures, side, key, expected, aggregate.get(key))

    if (aggregate.get("batching") or {}) != expected_batched:
        failures.append({
            "side": side,
            "path": "aggregate.batching",
            "expected": expected_batched,
            "actual": aggregate.get("batching"),
        })
    failures.extend(_static_server_batch_ledger_failures(manifest, side))

    passed = not failures
    print(f"Static-server manifest self-validation ({side})")
    print(f"  {'PASS' if passed else 'FAIL'}")
    if not passed:
        for failure in failures[:8]:
            detail = failure.get("reason")
            if detail is None:
                detail = f"expected={failure.get('expected')!r}, actual={failure.get('actual')!r}"
            print(f"  {failure.get('path')}: {detail}")
    print()
    return {
        "pass": passed,
        "failures": failures,
        "expected_aggregate": expected_values,
        "expected_batching": expected_batched,
    }


def _compare_static_server_performance(
        baseline: dict[str, Any],
        candidate: dict[str, Any],
        *,
        regression_tolerance_pct: float,
) -> dict[str, Any]:
    baseline_aggregate = baseline.get("aggregate", {})
    candidate_aggregate = candidate.get("aggregate", {})
    metric_specs = {
        "main_tokens_per_scheduler_second": True,
        "scheduler_wall_seconds": False,
        "request_wall_seconds_p95": False,
        "request_wall_seconds_max": False,
    }
    if int(baseline_aggregate.get("timing_generated_tokens", 0) or 0) > 0:
        metric_specs["timing_tokens_per_scheduler_second"] = True
    metrics = {}
    print("Static-server scheduler throughput")
    for key, higher_is_better in metric_specs.items():
        baseline_value = float(baseline_aggregate.get(key, 0.0) or 0.0)
        candidate_value = float(candidate_aggregate.get(key, 0.0) or 0.0)
        _compare_number(key, baseline_value, candidate_value, higher_is_better=higher_is_better)
        metrics[key] = _metric_comparison(
            baseline_value,
            candidate_value,
            higher_is_better=higher_is_better,
            tolerance_pct=regression_tolerance_pct,
        )

    attributed_metric = _metric_comparison(
        float(baseline_aggregate.get("main_tokens_per_request_model_second_attributed", 0.0) or 0.0),
        float(candidate_aggregate.get("main_tokens_per_request_model_second_attributed", 0.0) or 0.0),
        higher_is_better=True,
        tolerance_pct=regression_tolerance_pct,
    )
    baseline_tokens = int(baseline_aggregate.get("main_generated_tokens", 0) or 0)
    candidate_tokens = int(candidate_aggregate.get("main_generated_tokens", 0) or 0)
    generated_tokens_non_decreasing = candidate_tokens >= baseline_tokens
    baseline_timing_tokens = int(baseline_aggregate.get("timing_generated_tokens", 0) or 0)
    candidate_timing_tokens = int(candidate_aggregate.get("timing_generated_tokens", 0) or 0)
    timing_tokens_non_decreasing = candidate_timing_tokens >= baseline_timing_tokens
    passed = all(metric["pass"] for metric in metrics.values()) and generated_tokens_non_decreasing
    if baseline_timing_tokens > 0:
        passed = passed and timing_tokens_non_decreasing
    print(f"  main_generated_tokens: baseline={baseline_tokens}, candidate={candidate_tokens}")
    print(f"  generated_tokens_non_decreasing: {generated_tokens_non_decreasing}")
    print(f"  timing_generated_tokens: baseline={baseline_timing_tokens}, candidate={candidate_timing_tokens}")
    print(f"  timing_tokens_non_decreasing: {timing_tokens_non_decreasing}")
    print(
        "  attributed_request_model_tok/s: baseline={baseline:.3f}, candidate={candidate:.3f}".format(
            baseline=attributed_metric["baseline"],
            candidate=attributed_metric["candidate"],
        )
    )
    print(f"  no_regression_gate: {'PASS' if passed else 'FAIL'}")
    print()
    return {
        "pass": passed,
        "metrics": metrics,
        "attributed_request_model_tokens_per_second": attributed_metric,
        "generated_tokens_non_decreasing": generated_tokens_non_decreasing,
        "timing_tokens_non_decreasing": timing_tokens_non_decreasing,
        "baseline_main_generated_tokens": baseline_tokens,
        "candidate_main_generated_tokens": candidate_tokens,
        "baseline_timing_generated_tokens": baseline_timing_tokens,
        "candidate_timing_generated_tokens": candidate_timing_tokens,
    }


def compare_static_server_manifests(
        baseline_path: Path,
        candidate_path: Path,
        *,
        regression_tolerance_pct: float = 0.0,
        allow_server_batch_timeout_change: bool = False,
        allow_server_max_batch_size_change: bool = False,
) -> dict[str, Any]:
    baseline = _load_json(baseline_path)
    candidate = _load_json(candidate_path)
    report: dict[str, Any] = {
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "regression_tolerance_pct": regression_tolerance_pct,
        "allow_server_batch_timeout_change": allow_server_batch_timeout_change,
        "allow_server_max_batch_size_change": allow_server_max_batch_size_change,
        "contract": {},
        "result_class": {},
        "token_status": {},
        "performance": {},
        "self_validation": {},
    }

    print(f"Baseline static server manifest:  {baseline_path}")
    print(f"Candidate static server manifest: {candidate_path}")
    print()

    baseline_validation = _validate_static_server_manifest(baseline, side="baseline")
    candidate_validation = _validate_static_server_manifest(candidate, side="candidate")
    report["self_validation"] = {
        "pass": bool(baseline_validation["pass"] and candidate_validation["pass"]),
        "baseline": baseline_validation,
        "candidate": candidate_validation,
    }
    report["contract"] = _compare_static_server_contract(
        baseline,
        candidate,
        allow_server_batch_timeout_change=allow_server_batch_timeout_change,
        allow_server_max_batch_size_change=allow_server_max_batch_size_change,
    )
    report["result_class"] = _compare_static_server_result_class(baseline, candidate)
    report["token_status"] = _compare_static_server_token_status(baseline, candidate)
    report["performance"] = _compare_static_server_performance(
        baseline,
        candidate,
        regression_tolerance_pct=regression_tolerance_pct,
    )
    return report


def _continuous_request_key(request: dict[str, Any], index: int) -> str:
    return str(request.get("request_id", f"request{index}"))


def _compare_continuous_scheduler_contract(
        baseline: dict[str, Any],
        candidate: dict[str, Any],
) -> dict[str, Any]:
    mismatches = []
    for key in [
        "schema_version",
        "run_kind",
        "request_count",
        "config",
        "compatibility_key",
    ]:
        if baseline.get(key) != candidate.get(key):
            mismatches.append({"key": key, "baseline": baseline.get(key), "candidate": candidate.get(key)})

    baseline_requests = baseline.get("requests", [])
    candidate_requests = candidate.get("requests", [])
    if len(baseline_requests) != len(candidate_requests):
        mismatches.append({"key": "request_count", "baseline": len(baseline_requests), "candidate": len(candidate_requests)})

    request_contract_keys = [
        "request_id",
        "prompt_tokens",
        "max_new_tokens",
        "eos_token_ids",
        "planned_arrival_step",
        "metadata",
    ]
    for index in range(min(len(baseline_requests), len(candidate_requests))):
        baseline_request = baseline_requests[index]
        candidate_request = candidate_requests[index]
        for key in request_contract_keys:
            if baseline_request.get(key) != candidate_request.get(key):
                mismatches.append({
                    "index": index,
                    "key": key,
                    "baseline": baseline_request.get(key),
                    "candidate": candidate_request.get(key),
                    "baseline_request": _continuous_request_key(baseline_request, index),
                    "candidate_request": _continuous_request_key(candidate_request, index),
                })

    passed = not mismatches
    print("Continuous-scheduler contract")
    print(f"  {'PASS' if passed else 'FAIL'}")
    for mismatch in mismatches[:8]:
        print(f"  {mismatch['key']}: baseline={mismatch.get('baseline')!r}, candidate={mismatch.get('candidate')!r}")
    print()
    return {"pass": passed, "mismatches": mismatches}


def _compare_continuous_scheduler_result_class(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "baseline_result_class": baseline.get("result_class"),
        "candidate_result_class": candidate.get("result_class"),
        "baseline_model_generation_executed": bool(baseline.get("model_generation_executed", True)),
        "candidate_model_generation_executed": bool(candidate.get("model_generation_executed", True)),
        "baseline_state_hash_policy": baseline.get("state_hash_policy"),
        "candidate_state_hash_policy": candidate.get("state_hash_policy"),
    }
    passed = (
        checks["baseline_result_class"] == "continuous_scheduler_dry_run"
        and checks["candidate_result_class"] == "continuous_scheduler_dry_run"
        and not checks["baseline_model_generation_executed"]
        and not checks["candidate_model_generation_executed"]
        and checks["baseline_state_hash_policy"] == "required"
        and checks["candidate_state_hash_policy"] == "required"
    )
    print("Continuous-scheduler result class")
    print(f"  {'PASS' if passed else 'FAIL'}")
    for key, value in checks.items():
        print(f"  {key}: {value}")
    print()
    return {"pass": passed, **checks}


def _compare_continuous_scheduler_tokens(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    mismatches = []
    baseline_requests = baseline.get("requests", [])
    candidate_requests = candidate.get("requests", [])
    for side, manifest in [("baseline", baseline), ("candidate", candidate)]:
        if manifest.get("token_equivalence_status") != CONTINUOUS_SCHEDULER_TOKEN_STATUS:
            mismatches.append({
                "side": side,
                "key": "token_equivalence_status",
                "status": manifest.get("token_equivalence_status"),
                "expected": CONTINUOUS_SCHEDULER_TOKEN_STATUS,
            })
        for index, request in enumerate(manifest.get("requests", [])):
            status = request.get("token_equivalence_status")
            if status != CONTINUOUS_SCHEDULER_TOKEN_STATUS:
                mismatches.append({
                    "side": side,
                    "index": index,
                    "request_id": _continuous_request_key(request, index),
                    "key": "request.token_equivalence_status",
                    "status": status,
                    "expected": CONTINUOUS_SCHEDULER_TOKEN_STATUS,
                })

    semantic_keys = ["generated_token_count", "generated_token_sha256", "stop_reason"]
    for index in range(min(len(baseline_requests), len(candidate_requests))):
        baseline_request = baseline_requests[index]
        candidate_request = candidate_requests[index]
        for key in semantic_keys:
            if baseline_request.get(key) != candidate_request.get(key):
                mismatches.append({
                    "index": index,
                    "request_id": _continuous_request_key(baseline_request, index),
                    "key": key,
                    "baseline": baseline_request.get(key),
                    "candidate": candidate_request.get(key),
                })

    baseline_tokens = int((baseline.get("aggregate") or {}).get("total_generated_tokens", 0) or 0)
    candidate_tokens = int((candidate.get("aggregate") or {}).get("total_generated_tokens", 0) or 0)
    total_tokens_match = baseline_tokens == candidate_tokens
    if not total_tokens_match:
        mismatches.append({
            "key": "aggregate.total_generated_tokens",
            "baseline": baseline_tokens,
            "candidate": candidate_tokens,
        })

    passed = not mismatches
    print("Continuous-scheduler scripted-token equivalence")
    print(f"  {'PASS' if passed else 'FAIL'}")
    print(f"  total_generated_tokens: baseline={baseline_tokens}, candidate={candidate_tokens}")
    for mismatch in mismatches[:8]:
        print(f"  {mismatch['key']}: baseline={mismatch.get('baseline')!r}, candidate={mismatch.get('candidate')!r}")
    print()
    return {
        "pass": passed,
        "mismatches": mismatches,
        "baseline_total_generated_tokens": baseline_tokens,
        "candidate_total_generated_tokens": candidate_tokens,
    }


def _compare_continuous_scheduler_state_ledger(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    mismatches = []
    baseline_requests = baseline.get("requests", [])
    candidate_requests = candidate.get("requests", [])
    state_keys = [
        "initial_rng_state_hash",
        "final_rng_state_hash",
        "logits_processor_state_hash",
        "cache_state_hash",
        "missing_state_hash_fields",
    ]
    lifecycle_keys = [
        "enqueue_step",
        "activation_step",
        "finish_step",
        "queue_wait_steps",
        "decode_steps",
        "latency_steps",
        "cache_slot_id",
        "slot_generation",
    ]
    for index in range(min(len(baseline_requests), len(candidate_requests))):
        baseline_request = baseline_requests[index]
        candidate_request = candidate_requests[index]
        for key in state_keys:
            baseline_value = baseline_request.get(key)
            candidate_value = candidate_request.get(key)
            if (
                    baseline_value is None
                    or candidate_value is None
                    or baseline_value != candidate_value
                    or (key == "missing_state_hash_fields" and baseline_value)
            ):
                mismatches.append({
                    "index": index,
                    "request_id": _continuous_request_key(baseline_request, index),
                    "key": key,
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                })
        for key in lifecycle_keys:
            if baseline_request.get(key) != candidate_request.get(key):
                mismatches.append({
                    "index": index,
                    "request_id": _continuous_request_key(baseline_request, index),
                    "key": key,
                    "baseline": baseline_request.get(key),
                    "candidate": candidate_request.get(key),
                })

    passed = not mismatches
    print("Continuous-scheduler state/lifecycle ledger")
    print(f"  {'PASS' if passed else 'FAIL'}")
    if passed:
        print(f"  requests_checked: {min(len(baseline_requests), len(candidate_requests))}")
    else:
        for mismatch in mismatches[:8]:
            print(f"  {mismatch['key']}: baseline={mismatch.get('baseline')!r}, candidate={mismatch.get('candidate')!r}")
    print()
    return {"pass": passed, "mismatches": mismatches}


def _compare_continuous_scheduler_shape(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    mismatches = []
    shape_keys = [
        "active_batch_size_histogram",
        "steps",
        "cache_slot_events",
    ]
    for key in shape_keys:
        if baseline.get(key) != candidate.get(key):
            mismatches.append({"key": key, "baseline": baseline.get(key), "candidate": candidate.get(key)})
    baseline_stop_reasons = (baseline.get("aggregate") or {}).get("stop_reason_counts")
    candidate_stop_reasons = (candidate.get("aggregate") or {}).get("stop_reason_counts")
    if baseline_stop_reasons != candidate_stop_reasons:
        mismatches.append({
            "key": "aggregate.stop_reason_counts",
            "baseline": baseline_stop_reasons,
            "candidate": candidate_stop_reasons,
        })

    passed = not mismatches
    print("Continuous-scheduler scheduling shape")
    print(f"  {'PASS' if passed else 'FAIL'}")
    if passed:
        print(f"  active_batch_size_histogram: {baseline.get('active_batch_size_histogram')}")
    else:
        for mismatch in mismatches[:8]:
            print(f"  {mismatch['key']}: baseline={mismatch.get('baseline')!r}, candidate={mismatch.get('candidate')!r}")
    print()
    return {"pass": passed, "mismatches": mismatches}


def _compare_continuous_scheduler_performance(
        baseline: dict[str, Any],
        candidate: dict[str, Any],
        *,
        regression_tolerance_pct: float,
) -> dict[str, Any]:
    baseline_aggregate = baseline.get("aggregate") or {}
    candidate_aggregate = candidate.get("aggregate") or {}
    metric = _metric_comparison(
        float(baseline_aggregate.get("scheduler_cpu_wall_seconds", 0.0) or 0.0),
        float(candidate_aggregate.get("scheduler_cpu_wall_seconds", 0.0) or 0.0),
        higher_is_better=False,
        tolerance_pct=regression_tolerance_pct,
    )
    baseline_tokens = int(baseline_aggregate.get("total_generated_tokens", 0) or 0)
    candidate_tokens = int(candidate_aggregate.get("total_generated_tokens", 0) or 0)
    generated_tokens_non_decreasing = candidate_tokens >= baseline_tokens
    passed = metric["pass"] and generated_tokens_non_decreasing
    print("Continuous-scheduler CPU timing diagnostic")
    _compare_number(
        "scheduler_cpu_wall_seconds",
        metric["baseline"],
        metric["candidate"],
        higher_is_better=False,
    )
    print(f"  total_generated_tokens: baseline={baseline_tokens}, candidate={candidate_tokens}")
    print(f"  generated_tokens_non_decreasing: {generated_tokens_non_decreasing}")
    print(f"  no_regression_gate: {'PASS' if passed else 'FAIL'}")
    print()
    return {
        "pass": passed,
        "scheduler_cpu_wall_seconds": metric,
        "generated_tokens_non_decreasing": generated_tokens_non_decreasing,
        "baseline_total_generated_tokens": baseline_tokens,
        "candidate_total_generated_tokens": candidate_tokens,
    }


def _counter_to_manifest_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_continuous_scheduler_manifest(manifest: dict[str, Any], *, side: str) -> dict[str, Any]:
    requests = manifest.get("requests", [])
    steps = manifest.get("steps", [])
    cache_slot_events = manifest.get("cache_slot_events", [])
    aggregate = manifest.get("aggregate") or {}
    mismatches: list[dict[str, Any]] = []

    if _int_or_none(manifest.get("request_count")) != len(requests):
        mismatches.append({
            "side": side,
            "key": "request_count",
            "manifest": manifest.get("request_count"),
            "computed": len(requests),
        })

    completed_requests = 0
    total_generated_tokens = 0
    stop_reason_counts: Counter = Counter()
    planned_arrival_counts: Counter = Counter()
    request_by_id: dict[str, dict[str, Any]] = {}
    for index, request in enumerate(requests):
        request_id = _continuous_request_key(request, index)
        request_by_id[request_id] = request
        generated_tokens = [int(token) for token in request.get("generated_tokens", [])]
        total_generated_tokens += len(generated_tokens)
        if request.get("generated_token_count") != len(generated_tokens):
            mismatches.append({
                "side": side,
                "request_id": request_id,
                "key": "generated_token_count",
                "manifest": request.get("generated_token_count"),
                "computed": len(generated_tokens),
            })
        computed_hash = _continuous_token_sha256(generated_tokens)
        if request.get("generated_token_sha256") != computed_hash:
            mismatches.append({
                "side": side,
                "request_id": request_id,
                "key": "generated_token_sha256",
                "manifest": request.get("generated_token_sha256"),
                "computed": computed_hash,
            })

        missing_hashes = [
            field
            for field in CONTINUOUS_SCHEDULER_STATE_HASH_FIELDS
            if request.get(field) in (None, "")
        ]
        if request.get("missing_state_hash_fields") != missing_hashes:
            mismatches.append({
                "side": side,
                "request_id": request_id,
                "key": "missing_state_hash_fields",
                "manifest": request.get("missing_state_hash_fields"),
                "computed": missing_hashes,
            })

        enqueue_step = request.get("enqueue_step")
        activation_step = request.get("activation_step")
        finish_step = request.get("finish_step")
        planned_arrival_step = request.get("planned_arrival_step")
        if planned_arrival_step is not None:
            planned_arrival_counts[planned_arrival_step] += 1
        if activation_step is not None and planned_arrival_step is not None and activation_step < planned_arrival_step:
            mismatches.append({
                "side": side,
                "request_id": request_id,
                "key": "activation_before_planned_arrival",
                "planned_arrival_step": planned_arrival_step,
                "activation_step": activation_step,
            })
        if enqueue_step is not None and activation_step is not None:
            computed_queue_wait = activation_step - enqueue_step
            if request.get("queue_wait_steps") != computed_queue_wait:
                mismatches.append({
                    "side": side,
                    "request_id": request_id,
                    "key": "queue_wait_steps",
                    "manifest": request.get("queue_wait_steps"),
                    "computed": computed_queue_wait,
                })
        if activation_step is not None and finish_step is not None:
            computed_decode_steps = finish_step - activation_step + 1
            if request.get("decode_steps") != computed_decode_steps:
                mismatches.append({
                    "side": side,
                    "request_id": request_id,
                    "key": "decode_steps",
                    "manifest": request.get("decode_steps"),
                    "computed": computed_decode_steps,
                })
        if enqueue_step is not None and finish_step is not None:
            computed_latency_steps = finish_step - enqueue_step + 1
            if request.get("latency_steps") != computed_latency_steps:
                mismatches.append({
                    "side": side,
                    "request_id": request_id,
                    "key": "latency_steps",
                    "manifest": request.get("latency_steps"),
                    "computed": computed_latency_steps,
                })
        if request.get("stop_reason") is not None:
            completed_requests += 1
        stop_reason_counts[request.get("stop_reason")] += 1

    active_histogram: Counter = Counter()
    for step in steps:
        step_index = step.get("step_index")
        decoded = step.get("decoded") or []
        active_batch_size = int(step.get("active_batch_size", 0) or 0)
        if active_batch_size != len(decoded):
            mismatches.append({
                "side": side,
                "step_index": step_index,
                "key": "active_batch_size",
                "manifest": active_batch_size,
                "computed": len(decoded),
            })
        if active_batch_size:
            active_histogram[active_batch_size] += 1

        for activation in step.get("activated") or []:
            request_id = activation.get("request_id")
            request = request_by_id.get(request_id)
            if request is None:
                mismatches.append({
                    "side": side,
                    "step_index": step_index,
                    "key": "activation_unknown_request",
                    "request_id": request_id,
                })
                continue
            planned_arrival_step = request.get("planned_arrival_step")
            if planned_arrival_step is not None and step_index is not None and step_index < planned_arrival_step:
                mismatches.append({
                    "side": side,
                    "request_id": request_id,
                    "key": "step_activation_before_planned_arrival",
                    "planned_arrival_step": planned_arrival_step,
                    "step_index": step_index,
                })

        for decoded in decoded:
            request_id = decoded.get("request_id")
            request = request_by_id.get(request_id)
            if request is None:
                mismatches.append({
                    "side": side,
                    "step_index": step_index,
                    "key": "decode_unknown_request",
                    "request_id": request_id,
                })
                continue
            finish_step = request.get("finish_step")
            activation_step = request.get("activation_step")
            if activation_step is not None and step_index is not None and step_index < activation_step:
                mismatches.append({
                    "side": side,
                    "request_id": request_id,
                    "key": "decode_before_activation",
                    "activation_step": activation_step,
                    "step_index": step_index,
                })
            if finish_step is not None and step_index is not None and step_index > finish_step:
                mismatches.append({
                    "side": side,
                    "request_id": request_id,
                    "key": "decode_after_finish",
                    "finish_step": finish_step,
                    "step_index": step_index,
                })

    expected_active_histogram = _counter_to_manifest_dict(active_histogram)
    if manifest.get("active_batch_size_histogram") != expected_active_histogram:
        mismatches.append({
            "side": side,
            "key": "active_batch_size_histogram",
            "manifest": manifest.get("active_batch_size_histogram"),
            "computed": expected_active_histogram,
        })
    if aggregate.get("active_batch_size_histogram") != expected_active_histogram:
        mismatches.append({
            "side": side,
            "key": "aggregate.active_batch_size_histogram",
            "manifest": aggregate.get("active_batch_size_histogram"),
            "computed": expected_active_histogram,
        })

    acquire_events = [event for event in cache_slot_events if event.get("event") == "acquire"]
    release_events = [event for event in cache_slot_events if event.get("event") == "release"]
    if _int_or_none(aggregate.get("cache_slot_acquire_count")) != len(acquire_events):
        mismatches.append({
            "side": side,
            "key": "aggregate.cache_slot_acquire_count",
            "manifest": aggregate.get("cache_slot_acquire_count"),
            "computed": len(acquire_events),
        })
    if _int_or_none(aggregate.get("cache_slot_release_count")) != len(release_events):
        mismatches.append({
            "side": side,
            "key": "aggregate.cache_slot_release_count",
            "manifest": aggregate.get("cache_slot_release_count"),
            "computed": len(release_events),
        })

    acquire_by_request = Counter(event.get("request_id") for event in acquire_events)
    release_by_request = Counter(event.get("request_id") for event in release_events)
    for request in requests:
        request_id = request.get("request_id")
        if acquire_by_request[request_id] != 1:
            mismatches.append({
                "side": side,
                "request_id": request_id,
                "key": "cache_slot_acquire_events",
                "computed": acquire_by_request[request_id],
            })
        expected_releases = 1 if request.get("stop_reason") is not None else 0
        if release_by_request[request_id] != expected_releases:
            mismatches.append({
                "side": side,
                "request_id": request_id,
                "key": "cache_slot_release_events",
                "computed": release_by_request[request_id],
                "expected": expected_releases,
            })

    last_generation_by_slot: dict[Any, int] = {}
    for event in acquire_events:
        slot_id = event.get("cache_slot_id")
        generation = int(event.get("slot_generation", 0) or 0)
        previous_generation = last_generation_by_slot.get(slot_id, 0)
        if generation <= previous_generation:
            mismatches.append({
                "side": side,
                "key": "slot_generation_monotonic",
                "cache_slot_id": slot_id,
                "previous_generation": previous_generation,
                "slot_generation": generation,
            })
        last_generation_by_slot[slot_id] = generation

    if _int_or_none(aggregate.get("request_count")) != len(requests):
        mismatches.append({
            "side": side,
            "key": "aggregate.request_count",
            "manifest": aggregate.get("request_count"),
            "computed": len(requests),
        })
    if _int_or_none(aggregate.get("completed_request_count")) != completed_requests:
        mismatches.append({
            "side": side,
            "key": "aggregate.completed_request_count",
            "manifest": aggregate.get("completed_request_count"),
            "computed": completed_requests,
        })
    if _int_or_none(aggregate.get("total_generated_tokens")) != total_generated_tokens:
        mismatches.append({
            "side": side,
            "key": "aggregate.total_generated_tokens",
            "manifest": aggregate.get("total_generated_tokens"),
            "computed": total_generated_tokens,
        })
    expected_stop_reason_counts = _counter_to_manifest_dict(stop_reason_counts)
    if aggregate.get("stop_reason_counts") != expected_stop_reason_counts:
        mismatches.append({
            "side": side,
            "key": "aggregate.stop_reason_counts",
            "manifest": aggregate.get("stop_reason_counts"),
            "computed": expected_stop_reason_counts,
        })
    expected_arrival_counts = _counter_to_manifest_dict(planned_arrival_counts)
    if aggregate.get("planned_arrival_step_histogram") != expected_arrival_counts:
        mismatches.append({
            "side": side,
            "key": "aggregate.planned_arrival_step_histogram",
            "manifest": aggregate.get("planned_arrival_step_histogram"),
            "computed": expected_arrival_counts,
        })
    missing_state_hash_request_count = sum(
        1
        for request in requests
        if any(request.get(field) in (None, "") for field in CONTINUOUS_SCHEDULER_STATE_HASH_FIELDS)
    )
    if _int_or_none(aggregate.get("missing_state_hash_request_count")) != missing_state_hash_request_count:
        mismatches.append({
            "side": side,
            "key": "aggregate.missing_state_hash_request_count",
            "manifest": aggregate.get("missing_state_hash_request_count"),
            "computed": missing_state_hash_request_count,
        })

    passed = not mismatches
    print(f"Continuous-scheduler manifest self-validation ({side})")
    print(f"  {'PASS' if passed else 'FAIL'}")
    if passed:
        print(f"  requests={len(requests)}, steps={len(steps)}, cache_events={len(cache_slot_events)}")
    else:
        for mismatch in mismatches[:8]:
            print(f"  {mismatch['key']}: manifest={mismatch.get('manifest')!r}, computed={mismatch.get('computed')!r}")
    print()
    return {"pass": passed, "mismatches": mismatches}


def compare_continuous_scheduler_manifests(
        baseline_path: Path,
        candidate_path: Path,
        *,
        regression_tolerance_pct: float = 0.0,
) -> dict[str, Any]:
    baseline = _load_json(baseline_path)
    candidate = _load_json(candidate_path)
    report: dict[str, Any] = {
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "regression_tolerance_pct": regression_tolerance_pct,
        "contract": {},
        "result_class": {},
        "scripted_token_equivalence": {},
        "state_ledger": {},
        "scheduling_shape": {},
        "manifest_self_validation": {},
        "cpu_timing": {},
    }

    print(f"Baseline continuous scheduler manifest:  {baseline_path}")
    print(f"Candidate continuous scheduler manifest: {candidate_path}")
    print()

    report["contract"] = _compare_continuous_scheduler_contract(baseline, candidate)
    report["result_class"] = _compare_continuous_scheduler_result_class(baseline, candidate)
    report["scripted_token_equivalence"] = _compare_continuous_scheduler_tokens(baseline, candidate)
    report["state_ledger"] = _compare_continuous_scheduler_state_ledger(baseline, candidate)
    report["scheduling_shape"] = _compare_continuous_scheduler_shape(baseline, candidate)
    report["manifest_self_validation"] = {
        "baseline": _validate_continuous_scheduler_manifest(baseline, side="baseline"),
        "candidate": _validate_continuous_scheduler_manifest(candidate, side="candidate"),
    }
    report["cpu_timing"] = _compare_continuous_scheduler_performance(
        baseline,
        candidate,
        regression_tolerance_pct=regression_tolerance_pct,
    )
    return report


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
        "output_artifact_equivalence": {},
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
    report["output_artifact_equivalence"] = _compare_suite_output_hashes(baseline, candidate)
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


def _compare_output_artifacts(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_metadata = baseline.get("metadata", {})
    candidate_metadata = candidate.get("metadata", {})
    baseline_hash = baseline_metadata.get("result_file_sha256")
    candidate_hash = candidate_metadata.get("result_file_sha256")
    baseline_size = baseline_metadata.get("result_file_size_bytes")
    candidate_size = candidate_metadata.get("result_file_size_bytes")
    missing = [
        key
        for key, value in {
            "baseline.result_file_sha256": baseline_hash,
            "candidate.result_file_sha256": candidate_hash,
            "baseline.result_file_size_bytes": baseline_size,
            "candidate.result_file_size_bytes": candidate_size,
        }.items()
        if value is None
    ]
    passed = not missing and baseline_hash == candidate_hash and baseline_size == candidate_size
    print("Output artifact equivalence")
    if passed:
        print(f"  PASS sha256={baseline_hash}, size_bytes={baseline_size}")
    elif missing:
        print("  NOT CHECKED")
        print(f"  missing_keys: {', '.join(missing)}")
    else:
        print("  FAIL")
        print(f"  baseline_sha256={baseline_hash}, candidate_sha256={candidate_hash}")
        print(f"  baseline_size_bytes={baseline_size}, candidate_size_bytes={candidate_size}")
    print()
    return {
        "pass": passed,
        "status": "PASS" if passed else ("not_checked" if missing else "FAIL"),
        "missing_keys": missing,
        "baseline_sha256": baseline_hash,
        "candidate_sha256": candidate_hash,
        "baseline_size_bytes": baseline_size,
        "candidate_size_bytes": candidate_size,
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
        "output_artifact": {
            "status": "not_checked",
            "pass": False,
        },
    }

    print(f"Baseline:  {baseline_path}")
    print(f"Candidate: {candidate_path}")
    print(f"Label:     {label}")
    print()

    report["same_calculation"] = _compare_contract_metadata(baseline, candidate)
    report["output_artifact"] = _compare_output_artifacts(baseline, candidate)

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
        "output_artifact_pass": all(
            report.get("output_artifact", {}).get("pass", False)
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
        "--compare-static-server",
        nargs=2,
        metavar=("BASE_MANIFEST", "CANDIDATE_MANIFEST"),
        type=Path,
        help="Compare two profile_static_server_batch static_server_batch_manifest.json files.",
    )
    parser.add_argument(
        "--compare-continuous-scheduler",
        nargs=2,
        metavar=("BASE_MANIFEST", "CANDIDATE_MANIFEST"),
        type=Path,
        help="Compare two CPU-only continuous_scheduler_manifest.json dry-run files.",
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
        "--require-output-equivalence",
        action="store_true",
        help="Exit nonzero unless generated output artifact hashes are present and exactly match.",
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
        "--allow-server-batch-timeout-change",
        action="store_true",
        help=(
            "For --compare-static-server, allow server_batch_timeout to differ in the server config "
            "fingerprint. Use only when evaluating that scheduler knob explicitly."
        ),
    )
    parser.add_argument(
        "--allow-server-max-batch-size-change",
        action="store_true",
        help=(
            "For --compare-static-server, allow max_batch_size to differ in the server config "
            "fingerprint. Use only when evaluating static server batch-cap scaling explicitly."
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
            "Also requires generated output artifact hash equivalence. Use for full-song promotion gates."
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Write a machine-readable comparison report.",
    )
    args = parser.parse_args()
    compare_modes = [
        bool(args.compare),
        bool(args.compare_suite),
        bool(args.compare_static_server),
        bool(args.compare_continuous_scheduler),
    ]
    if sum(compare_modes) > 1:
        parser.error(
            "use only one of --compare, --compare-suite, --compare-static-server, "
            "or --compare-continuous-scheduler"
        )
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
        require_output_equivalence = args.require_output_equivalence or args.strict_full_song
        require_no_regression = args.require_no_regression or strict_compare
        if "reports" in report:
            failed = (
                (require_contract_match and not report.get("same_calculation_pass", False))
                or (require_token_equivalence and not report.get("token_equivalence_pass", False))
                or (require_output_equivalence and not report.get("output_artifact_pass", False))
                or (require_no_regression and not report.get("performance_pass", False))
            )
        else:
            failed = (
                (require_contract_match and not report["same_calculation"].get("pass", False))
                or (require_token_equivalence and not report["token_equivalence"].get("pass", False))
                or (require_output_equivalence and not report["output_artifact"].get("pass", False))
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
        require_output_equivalence = args.require_output_equivalence
        require_no_regression = args.require_no_regression or args.strict
        require_segment_no_regression = args.require_suite_segment_no_regression or args.strict
        require_timing_no_regression = args.require_suite_timing_no_regression or args.strict
        failed = (
            ((require_contract_match or args.require_mode_contract) and not report["shape"].get("pass", False))
            or (args.strict and not report["scope_availability"].get("pass", False))
            or (require_token_equivalence and not report["token_equivalence"].get("pass", False))
            or (
                require_output_equivalence
                and not report["output_artifact_equivalence"].get("pass", False)
            )
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
    elif args.compare_static_server:
        report = compare_static_server_manifests(
            args.compare_static_server[0],
            args.compare_static_server[1],
            regression_tolerance_pct=args.regression_tolerance_pct,
            allow_server_batch_timeout_change=args.allow_server_batch_timeout_change,
            allow_server_max_batch_size_change=args.allow_server_max_batch_size_change,
        )
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

        failed = (
            ((args.require_mode_contract or args.strict)
             and not report["self_validation"].get("pass", False))
            or ((args.require_contract_match or args.require_mode_contract or args.strict)
             and not report["contract"].get("pass", False))
            or ((args.require_mode_contract or args.strict)
                and not report["result_class"].get("pass", False))
            or ((args.require_token_equivalence or args.strict)
                and not report["token_status"].get("pass", False))
            or ((args.require_no_regression or args.strict)
                and not report["performance"].get("pass", False))
        )
        raise SystemExit(1 if failed else 0)
    elif args.compare_continuous_scheduler:
        report = compare_continuous_scheduler_manifests(
            args.compare_continuous_scheduler[0],
            args.compare_continuous_scheduler[1],
            regression_tolerance_pct=args.regression_tolerance_pct,
        )
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

        failed = (
            ((args.require_contract_match or args.require_mode_contract or args.strict)
             and not report["contract"].get("pass", False))
            or ((args.require_mode_contract or args.strict)
                and not report["result_class"].get("pass", False))
            or ((args.require_token_equivalence or args.strict)
                and not report["scripted_token_equivalence"].get("pass", False))
            or ((args.require_mode_contract or args.strict)
                and not report["state_ledger"].get("pass", False))
            or ((args.require_mode_contract or args.strict)
                and not report["scheduling_shape"].get("pass", False))
            or ((args.require_mode_contract or args.strict)
                and not (
                    report["manifest_self_validation"].get("baseline", {}).get("pass", False)
                    and report["manifest_self_validation"].get("candidate", {}).get("pass", False)
                ))
            or (args.require_no_regression and not report["cpu_timing"].get("pass", False))
        )
        raise SystemExit(1 if failed else 0)
    elif args.profile:
        summarize(args.profile, limit=args.limit)
    else:
        parser.error(
            "provide a profile path, --compare BASELINE CANDIDATE, "
            "--compare-suite BASE CANDIDATE, --compare-static-server BASE CANDIDATE, "
            "or --compare-continuous-scheduler BASE CANDIDATE"
        )


if __name__ == "__main__":
    main()
