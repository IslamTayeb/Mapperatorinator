"""Compare an FP16 candidate gate with the accepted same-precision baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .analyze_fp16_fresh_baseline import (
        EXPECTED_PRESET_VERSION,
        SCHEMA_VERSION as FP16_RUN_SCHEMA_VERSION,
    )
except ImportError:  # Direct ``python utils/...py`` execution.
    from analyze_fp16_fresh_baseline import (  # type: ignore[no-redef]
        EXPECTED_PRESET_VERSION,
        SCHEMA_VERSION as FP16_RUN_SCHEMA_VERSION,
    )


SCHEMA_VERSION = "mapperatorinator.fp16-reciprocal-candidate.v1"
CANDIDATE_PRESET_VERSION = "candidate-shared-fp16-device-temperature-v1"


class FP16ReciprocalError(RuntimeError):
    pass


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FP16ReciprocalError(f"{name} must be an object")
    return value


def _load(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FP16ReciprocalError(f"{name} is missing or empty: {path}")
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), name=name)
    except json.JSONDecodeError as exc:
        raise FP16ReciprocalError(f"{name} is invalid JSON: {exc}") from exc


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FP16ReciprocalError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise FP16ReciprocalError(f"{name} must be finite and positive")
    return result


def _validate_analysis(
    value: Mapping[str, Any],
    *,
    name: str,
    expected_runs: int,
) -> None:
    expected = {
        "schema_version": FP16_RUN_SCHEMA_VERSION,
        "status": "PASS",
        "precision": "fp16",
        "fresh_processes": True,
        "authoritative_performance": True,
        "run_count": expected_runs,
    }
    failures = {
        key: {"expected": wanted, "actual": value.get(key)}
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    if failures:
        raise FP16ReciprocalError(f"{name} analysis contract changed: {failures}")
    checks = _object(value.get("checks"), name=f"{name}.checks")
    failed = sorted(key for key, passed in checks.items() if passed is not True)
    if failed:
        raise FP16ReciprocalError(f"{name} contains failed checks: {failed}")
    runs = value.get("runs")
    audits = value.get("exactness_audits")
    if not isinstance(runs, list) or len(runs) != expected_runs:
        raise FP16ReciprocalError(f"{name} run inventory changed")
    if not isinstance(audits, list) or len(audits) != 2:
        raise FP16ReciprocalError(f"{name} requires two exactness audits")


def _run_signature(run: Mapping[str, Any]) -> dict[str, Any]:
    generation = _object(run.get("generation"), name="run.generation")
    signatures = _object(run.get("token_signatures"), name="run.token_signatures")
    labels: dict[str, Any] = {}
    for label in ("timing_context", "main_generation"):
        metrics = _object(generation.get(label), name=f"run.generation.{label}")
        signature = _object(
            signatures.get(label), name=f"run.token_signatures.{label}"
        )
        labels[label] = {
            "generated_tokens": signature.get("generated_tokens"),
            "token_stream_sha256": signature.get("token_stream_sha256"),
            "stopping_sha256": signature.get("stopping_sha256"),
            "metric_generated_tokens": metrics.get("generated_tokens"),
        }
    return {
        "labels": labels,
        "output_sha256": run.get("output_sha256"),
        "output_structure": run.get("output_structure"),
    }


def _audit_signature(audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "output_sha256": audit.get("output_sha256"),
        "strict_exactness": audit.get("strict_exactness"),
    }


def _median_metrics(value: Mapping[str, Any]) -> dict[str, float]:
    raw_runs = value.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise FP16ReciprocalError("analysis has no run metrics")
    rows: dict[str, list[float]] = {
        "process_wall_seconds": [],
        "request_to_final_osu_wall_seconds": [],
        "timing_synchronized_model_seconds": [],
        "main_synchronized_model_seconds": [],
        "combined_synchronized_model_seconds": [],
    }
    for index, raw in enumerate(raw_runs):
        run = _object(raw, name=f"runs[{index}]")
        generation = _object(run.get("generation"), name=f"runs[{index}].generation")
        timing = _object(
            generation.get("timing_context"),
            name=f"runs[{index}].generation.timing_context",
        )
        main = _object(
            generation.get("main_generation"),
            name=f"runs[{index}].generation.main_generation",
        )
        timing_seconds = _finite(
            timing.get("synchronized_model_seconds"), name="timing model seconds"
        )
        main_seconds = _finite(
            main.get("synchronized_model_seconds"), name="main model seconds"
        )
        rows["process_wall_seconds"].append(
            _finite(run.get("process_wall_seconds"), name="process wall")
        )
        rows["request_to_final_osu_wall_seconds"].append(
            _finite(
                run.get("request_to_final_osu_wall_seconds"), name="request wall"
            )
        )
        rows["timing_synchronized_model_seconds"].append(timing_seconds)
        rows["main_synchronized_model_seconds"].append(main_seconds)
        rows["combined_synchronized_model_seconds"].append(
            timing_seconds + main_seconds
        )
    return {key: float(statistics.median(values)) for key, values in rows.items()}


def _delta(baseline: float, candidate: float) -> dict[str, float]:
    saved = baseline - candidate
    return {
        "baseline_seconds": baseline,
        "candidate_seconds": candidate,
        "saved_seconds": saved,
        "improvement_pct": saved / baseline * 100.0,
    }


def analyze(
    baseline_path: Path,
    candidate_path: Path,
    *,
    mode: str,
) -> dict[str, Any]:
    expected_candidate_runs = 1 if mode == "initial" else 5 if mode == "confirm" else 0
    if expected_candidate_runs == 0:
        raise FP16ReciprocalError("mode must be initial or confirm")
    baseline = _load(baseline_path.resolve(), name="baseline analysis")
    candidate = _load(candidate_path.resolve(), name="candidate analysis")
    _validate_analysis(baseline, name="baseline", expected_runs=5)
    _validate_analysis(
        candidate, name="candidate", expected_runs=expected_candidate_runs
    )
    baseline_preset = baseline.get(
        "preset_version", baseline.get("accepted_preset_version")
    )
    if baseline_preset != EXPECTED_PRESET_VERSION:
        raise FP16ReciprocalError("accepted FP16 baseline preset changed")
    if candidate.get("preset_version") != CANDIDATE_PRESET_VERSION:
        raise FP16ReciprocalError("candidate FP16 preset changed")
    if candidate.get("device_conditional_temperature") is not True:
        raise FP16ReciprocalError("candidate device specialization was not verified")

    baseline_runs = baseline["runs"]
    candidate_runs = candidate["runs"]
    reference_run = _run_signature(
        _object(baseline_runs[0], name="baseline.runs[0]")
    )
    candidate_signatures = [
        _run_signature(_object(value, name=f"candidate.runs[{index}]"))
        for index, value in enumerate(candidate_runs)
    ]
    reference_audit = _audit_signature(
        _object(baseline["exactness_audits"][0], name="baseline audit")
    )
    candidate_audits = [
        _audit_signature(_object(value, name=f"candidate audit {index}"))
        for index, value in enumerate(candidate["exactness_audits"])
    ]
    checks = {
        "fixed_work": baseline.get("fixed_work") == candidate.get("fixed_work"),
        "tokens_stopping_and_output": all(
            value == reference_run for value in candidate_signatures
        ),
        "rng_cache_and_audit_output": all(
            value == reference_audit for value in candidate_audits
        ),
        "candidate_repeatability": len(
            {_hash(value) for value in candidate_signatures}
        )
        == 1,
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise FP16ReciprocalError(
            "FP16 reciprocal parity failed: " + ", ".join(failed)
        )

    baseline_metrics = _median_metrics(baseline)
    candidate_metrics = _median_metrics(candidate)
    deltas = {
        key: _delta(baseline_metrics[key], candidate_metrics[key])
        for key in baseline_metrics
    }
    promotion_eligible = (
        deltas["combined_synchronized_model_seconds"]["saved_seconds"] > 0
        or deltas["request_to_final_osu_wall_seconds"]["saved_seconds"] > 0
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            ("PASS_INITIAL" if mode == "initial" else "PASS_CONFIRMED")
            if promotion_eligible
            else "STOP_NO_GAIN"
        ),
        "mode": mode,
        "promotion_eligible": promotion_eligible,
        "exact_same_precision_parity": True,
        "baseline_analysis": str(baseline_path.resolve()),
        "candidate_analysis": str(candidate_path.resolve()),
        "candidate_runtime_identity": candidate.get("runtime_identity"),
        "checks": checks,
        "deltas": deltas,
    }
    result["analysis_sha256"] = _hash(result)
    return result


def _text(value: Mapping[str, Any]) -> str:
    lines = [
        f"status={value['status']}",
        f"mode={value['mode']}",
        f"promotion_eligible={str(value['promotion_eligible']).lower()}",
        "precision=fp16",
        "exact_same_precision_parity=true",
    ]
    for key, delta in value["deltas"].items():
        lines.extend(
            (
                f"{key}_baseline={delta['baseline_seconds']:.6f}",
                f"{key}_candidate={delta['candidate_seconds']:.6f}",
                f"{key}_saved={delta['saved_seconds']:.6f}",
                f"{key}_improvement_pct={delta['improvement_pct']:.6f}",
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-analysis", type=Path, required=True)
    parser.add_argument("--candidate-analysis", type=Path, required=True)
    parser.add_argument("--mode", choices=("initial", "confirm"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        value = analyze(
            args.baseline_analysis,
            args.candidate_analysis,
            mode=args.mode,
        )
        exit_code = 0
    except Exception as exc:
        value = {
            "schema_version": SCHEMA_VERSION,
            "status": "ERROR",
            "mode": args.mode,
            "promotion_eligible": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        value["analysis_sha256"] = _hash(value)
        exit_code = 2
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = _text(value) if exit_code == 0 else f"status=ERROR\nerror={value['error']}\n"
    args.text_output.write_text(report, encoding="utf-8")
    print(report, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
