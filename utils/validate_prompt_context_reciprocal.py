from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any


ROLES = ("baseline_first", "candidate_first", "candidate_second", "baseline_second")
BASELINE_ROLES = ("baseline_first", "baseline_second")
CANDIDATE_ROLES = ("candidate_first", "candidate_second")
MIN_PROMPT_SAVING_SECONDS = 0.20
MIN_MAIN_STAGE_SAVING_SECONDS = 0.15
MAX_MODEL_REGRESSION_FRACTION = 0.01
MAX_RECONCILIATION_ERROR_SECONDS = 0.20


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _list(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _load_profile(root: Path, role: str) -> dict[str, Any]:
    path_file = root / f"{role}.profile-path.txt"
    if not path_file.is_file():
        raise ValueError(f"missing profile path for {role}")
    profile_path = Path(path_file.read_text(encoding="utf-8").strip())
    if not profile_path.is_file():
        raise ValueError(f"missing profile for {role}: {profile_path}")
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    return _object(payload, name=f"{role} profile")


def _stage(profile: dict[str, Any], name: str, *, role: str) -> dict[str, Any]:
    rows = [
        _object(row, name=f"{role} stage")
        for row in _list(profile.get("stages"), name=f"{role}.stages")
        if isinstance(row, dict) and row.get("name") == name
    ]
    if len(rows) != 1:
        raise ValueError(f"{role} must contain exactly one {name} stage")
    return rows[0]


def _main_rows(profile: dict[str, Any], *, role: str) -> list[dict[str, Any]]:
    rows = [
        _object(row, name=f"{role} generation row")
        for row in _list(profile.get("generation"), name=f"{role}.generation")
        if isinstance(row, dict) and row.get("profile_label") == "main_generation"
    ]
    if not rows:
        raise ValueError(f"{role} contains no main-generation rows")
    return rows


def _signature(rows: list[dict[str, Any]], *, role: str) -> list[dict[str, Any]]:
    result = []
    for index, row in enumerate(rows):
        digest = row.get("prompt_sha256_per_sample")
        if (
            not isinstance(digest, list)
            or len(digest) != 1
            or not isinstance(digest[0], str)
            or len(digest[0]) != 64
        ):
            raise ValueError(f"{role} row {index} has no prompt SHA-256")
        generated = row.get("generated_token_ids")
        if not isinstance(generated, list):
            raise ValueError(f"{role} row {index} has no generated token IDs")
        result.append(
            {
                "sequence_index": row.get("sequence_index"),
                "frame_time_ms": row.get("frame_time_ms"),
                "prompt_width": row.get("prompt_width"),
                "prompt_tokens_per_sample": row.get("prompt_tokens_per_sample"),
                "prompt_sha256": digest[0],
                "generated_token_ids": generated,
                "trim_lookback": row.get("trim_lookback"),
                "trim_lookahead": row.get("trim_lookahead"),
            }
        )
    return result


def _metrics(profile: dict[str, Any], *, role: str) -> dict[str, float]:
    rows = _main_rows(profile, role=role)
    prompt = sum(
        _number(row.get("prompt_wall_seconds"), name=f"{role}.prompt wall")
        for row in rows
    )
    model = sum(
        _number(row.get("model_elapsed_seconds"), name=f"{role}.model wall")
        for row in rows
    )
    tokens = sum(int(row.get("generated_tokens", 0)) for row in rows)
    if tokens <= 0:
        raise ValueError(f"{role} generated no main tokens")
    main_stage = _stage(profile, "main_generation", role=role)
    validate_stage = _stage(profile, "validate_inputs", role=role)
    write_stages = [
        _stage(profile, name, role=role)
        for name in ("write_osu", "write_osz")
        if any(
            isinstance(row, dict) and row.get("name") == name
            for row in profile["stages"]
        )
    ]
    if len(write_stages) != 1:
        raise ValueError(f"{role} must contain one final write stage")
    complete = _number(
        write_stages[0].get("finished_at_perf_counter_seconds"),
        name=f"{role}.write finished",
    ) - _number(
        validate_stage.get("started_at_perf_counter_seconds"),
        name=f"{role}.validate started",
    )
    if complete <= 0.0:
        raise ValueError(f"{role} complete request wall must be positive")
    return {
        "prompt_wall_seconds": prompt,
        "main_model_seconds": model,
        "main_tokens": float(tokens),
        "fixed_8294_main_seconds": model * 8294.0 / tokens,
        "main_stage_wall_seconds": _number(
            main_stage.get("wall_seconds"),
            name=f"{role}.main stage",
        ),
        "complete_request_wall_seconds": complete,
    }


def _median(runs: dict[str, dict[str, float]], roles: tuple[str, str], key: str) -> float:
    return statistics.median(runs[role][key] for role in roles)


def validate(run_root: Path) -> dict[str, Any]:
    profiles = {role: _load_profile(run_root, role) for role in ROLES}
    signatures = {
        role: _signature(_main_rows(profile, role=role), role=role)
        for role, profile in profiles.items()
    }
    baseline_repeat = signatures["baseline_first"] == signatures["baseline_second"]
    candidate_repeat = signatures["candidate_first"] == signatures["candidate_second"]
    cross_exact = signatures["baseline_first"] == signatures["candidate_first"]
    if not baseline_repeat or not candidate_repeat or not cross_exact:
        raise ValueError("prompt/token signatures are not exact and repeat-stable")

    for role in BASELINE_ROLES:
        metadata = _object(profiles[role].get("metadata"), name=f"{role}.metadata")
        if "prompt_context_preparation" in metadata:
            raise ValueError(f"{role} unexpectedly enabled prompt-context preparation")
    candidate_metadata = []
    for role in CANDIDATE_ROLES:
        metadata = _object(profiles[role].get("metadata"), name=f"{role}.metadata")
        prompt = _object(
            metadata.get("prompt_context_preparation"),
            name=f"{role}.prompt_context_preparation",
        )
        if prompt.get("scope") != "main_generation_only":
            raise ValueError(f"{role} prompt-context scope is wrong")
        if prompt.get("exactness_required") is not True:
            raise ValueError(f"{role} prompt-context exactness is not required")
        if prompt.get("prompt_calls") != len(signatures[role]):
            raise ValueError(f"{role} prompt call count does not match windows")
        if prompt.get("range_queries", 0) <= prompt.get("prompt_calls", 0):
            raise ValueError(f"{role} did not use binary-search ranges")
        candidate_metadata.append(prompt)
    if candidate_metadata[0] != candidate_metadata[1]:
        raise ValueError("candidate prompt-context accounting is not repeat-stable")

    metrics = {
        role: _metrics(profile, role=role) for role, profile in profiles.items()
    }
    baseline = {
        key: _median(metrics, BASELINE_ROLES, key)
        for key in metrics["baseline_first"]
    }
    candidate = {
        key: _median(metrics, CANDIDATE_ROLES, key)
        for key in metrics["candidate_first"]
    }
    savings = {
        key: baseline[key] - candidate[key]
        for key in (
            "prompt_wall_seconds",
            "fixed_8294_main_seconds",
            "main_stage_wall_seconds",
            "complete_request_wall_seconds",
        )
    }
    pairs = {
        "first": {
            key: metrics["baseline_first"][key] - metrics["candidate_first"][key]
            for key in savings
        },
        "second": {
            key: metrics["baseline_second"][key] - metrics["candidate_second"][key]
            for key in savings
        },
    }
    prompt_pass = all(
        pair["prompt_wall_seconds"] >= MIN_PROMPT_SAVING_SECONDS
        for pair in pairs.values()
    )
    main_stage_pass = (
        savings["main_stage_wall_seconds"] >= MIN_MAIN_STAGE_SAVING_SECONDS
        and all(pair["main_stage_wall_seconds"] > 0.0 for pair in pairs.values())
    )
    complete_pass = savings["complete_request_wall_seconds"] > 0.0
    model_regression_fraction = max(
        0.0,
        -savings["fixed_8294_main_seconds"] / baseline["fixed_8294_main_seconds"],
    )
    model_pass = model_regression_fraction <= MAX_MODEL_REGRESSION_FRACTION
    reconciliation_error = abs(
        savings["prompt_wall_seconds"] - savings["main_stage_wall_seconds"]
    )
    reconciliation_pass = reconciliation_error <= MAX_RECONCILIATION_ERROR_SECONDS
    promotion_pass = all(
        (
            prompt_pass,
            main_stage_pass,
            complete_pass,
            model_pass,
            reconciliation_pass,
        )
    )
    report = {
        "schema_version": 1,
        "result_class": "exact-incremental-cpu-prompt-context",
        "parity": {
            "baseline_repeat_stable": baseline_repeat,
            "candidate_repeat_stable": candidate_repeat,
            "prompt_and_tokens_exact": cross_exact,
        },
        "candidate_metadata": candidate_metadata[0],
        "metrics": {
            "runs": metrics,
            "baseline_median": baseline,
            "candidate_median": candidate,
            "median_savings": savings,
            "reciprocal_pair_savings": pairs,
            "model_regression_fraction": model_regression_fraction,
            "prompt_to_main_stage_reconciliation_error_seconds": reconciliation_error,
        },
        "gates": {
            "minimum_prompt_saving_seconds": MIN_PROMPT_SAVING_SECONDS,
            "minimum_main_stage_saving_seconds": MIN_MAIN_STAGE_SAVING_SECONDS,
            "maximum_model_regression_fraction": MAX_MODEL_REGRESSION_FRACTION,
            "maximum_reconciliation_error_seconds": MAX_RECONCILIATION_ERROR_SECONDS,
            "prompt_pass": prompt_pass,
            "main_stage_pass": main_stage_pass,
            "complete_request_pass": complete_pass,
            "model_pass": model_pass,
            "reconciliation_pass": reconciliation_pass,
            "promotion_pass": promotion_pass,
        },
    }
    if not promotion_pass:
        raise ValueError(json.dumps(report, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.run_root)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
