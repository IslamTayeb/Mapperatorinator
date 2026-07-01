from __future__ import annotations

from typing import Any


def generation_records_for_label(profile: dict[str, Any], label: str) -> list[dict[str, Any]]:
    return [record for record in profile.get("generation", []) if record.get("profile_label") == label]


def generation_record_key(record: dict[str, Any], index: int) -> str:
    context = record.get("context_type", "unknown")
    mode = record.get("mode", "unknown")
    if "sequence_index" in record:
        unit = f"seq{record['sequence_index']}"
    else:
        unit = f"batch{record.get('batch_start_index', index)}"
    return f"{context}/{mode}/{unit}"


def _generated_tokens(record: dict[str, Any]) -> int:
    return int(record.get("generated_tokens", 0) or 0)


def _model_elapsed_seconds(record: dict[str, Any]) -> float:
    return float(record.get("model_elapsed_seconds", 0.0) or 0.0)


def _wall_seconds(record: dict[str, Any]) -> float:
    return float(record.get("wall_seconds", 0.0) or 0.0)


def _tokens_per_second(tokens: int, seconds: float) -> float:
    return tokens / seconds if seconds > 0 else 0.0


def summarize_generation_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    generated_tokens = _generated_tokens(record)
    model_elapsed_seconds = _model_elapsed_seconds(record)
    wall_seconds = _wall_seconds(record)
    return {
        "index": index,
        "key": generation_record_key(record, index),
        "profile_label": record.get("profile_label"),
        "mode": record.get("mode"),
        "context_type": record.get("context_type"),
        "sequence_index": record.get("sequence_index"),
        "batch_start_index": record.get("batch_start_index"),
        "generated_tokens": generated_tokens,
        "model_elapsed_seconds": model_elapsed_seconds,
        "wall_seconds": wall_seconds,
        "tokens_per_second": _tokens_per_second(generated_tokens, model_elapsed_seconds),
        "wall_tokens_per_second": _tokens_per_second(generated_tokens, wall_seconds),
        "torch_profiled": bool(record.get("torch_profiled", False)),
    }


def aggregate_generation_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    generated_tokens = sum(_generated_tokens(record) for record in records)
    model_elapsed_seconds = sum(_model_elapsed_seconds(record) for record in records)
    wall_seconds = sum(_wall_seconds(record) for record in records)
    return {
        "records": len(records),
        "generated_tokens": generated_tokens,
        "model_elapsed_seconds": model_elapsed_seconds,
        "wall_seconds": wall_seconds,
        "tokens_per_second": _tokens_per_second(generated_tokens, model_elapsed_seconds),
        "wall_tokens_per_second": _tokens_per_second(generated_tokens, wall_seconds),
    }


def first_record_breakdown(profile: dict[str, Any], label: str) -> dict[str, Any]:
    records = generation_records_for_label(profile, label)
    first = summarize_generation_record(records[0], 0) if records else None
    remaining = records[1:] if records else []
    return {
        "label": label,
        "records": len(records),
        "first_record": first,
        "remaining_records": aggregate_generation_records(remaining),
        "all_records": aggregate_generation_records(records),
    }
