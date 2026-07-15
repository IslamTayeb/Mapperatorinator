"""Fail-closed FP16 split-KV full-song reciprocal analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils import analyze_fp32_fresh_baseline as common
from utils import nsight_agent_profile as nsight
from utils.analyze_fp16_fresh_baseline import (
    EXPECTED_PRESET_VERSION,
    LABELS,
    SCHEMA_VERSION as FP16_RUN_SCHEMA_VERSION,
)


SCHEMA_VERSION = "mapperatorinator.fp16-split-kv-reciprocal.v1"
CANDIDATE_PRESET_VERSION = "candidate-fp16-all-fused-split-kv-v3"


class FP16SplitKVReciprocalError(RuntimeError):
    pass


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FP16SplitKVReciprocalError(f"{name} must be an object")
    return value


def _load(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FP16SplitKVReciprocalError(f"{name} is missing or empty: {path}")
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), name=name)
    except json.JSONDecodeError as exc:
        raise FP16SplitKVReciprocalError(f"{name} is invalid JSON: {exc}") from exc


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


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
        raise FP16SplitKVReciprocalError(
            f"{name} analysis contract changed: {failures}"
        )
    checks = _object(value.get("checks"), name=f"{name}.checks")
    failed = sorted(key for key, passed in checks.items() if passed is not True)
    if failed:
        raise FP16SplitKVReciprocalError(
            f"{name} contains failed checks: {failed}"
        )
    if not isinstance(value.get("runs"), list) or len(value["runs"]) != expected_runs:
        raise FP16SplitKVReciprocalError(f"{name} run inventory changed")
    if not isinstance(value.get("exactness_audits"), list) or len(
        value["exactness_audits"]
    ) != 2:
        raise FP16SplitKVReciprocalError(
            f"{name} requires two exactness audits"
        )


def _load_profile(run_dir: Path) -> dict[str, Any]:
    try:
        path = common._path_from_pointer(run_dir, "profile-path.txt")
        return nsight._load_inference_profile(path)
    except (common.BaselineError, ValueError) as exc:
        raise FP16SplitKVReciprocalError(str(exc)) from exc


def _profile_tokens(profile: Mapping[str, Any], label: str) -> list[list[list[int]]]:
    material: list[list[list[int]]] = []
    for index, record in enumerate(profile.get("generation", [])):
        if not isinstance(record, dict) or record.get("profile_label") != label:
            continue
        groups = nsight._record_token_groups(record)
        if groups is None:
            raise FP16SplitKVReciprocalError(
                f"baseline {label} record {index} lacks token IDs"
            )
        material.append(groups)
    if not material:
        raise FP16SplitKVReciprocalError(f"baseline {label} token stream is empty")
    return material


def _flatten(value: Any) -> list[int]:
    result: list[int] = []

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for child in node:
                visit(child)
            return
        if isinstance(node, bool) or not isinstance(node, int):
            raise FP16SplitKVReciprocalError("token material contains a non-integer")
        result.append(node)

    visit(value)
    return result


def _token_drift(baseline: Any, candidate: Any) -> dict[str, Any]:
    baseline_tokens = _flatten(baseline)
    candidate_tokens = _flatten(candidate)
    aligned = min(len(baseline_tokens), len(candidate_tokens))
    mismatches = sum(
        left != right
        for left, right in zip(
            baseline_tokens[:aligned],
            candidate_tokens[:aligned],
            strict=True,
        )
    )
    return {
        "exact": baseline_tokens == candidate_tokens,
        "baseline_tokens": len(baseline_tokens),
        "candidate_tokens": len(candidate_tokens),
        "length_delta": len(candidate_tokens) - len(baseline_tokens),
        "aligned_positions": aligned,
        "aligned_mismatches": mismatches,
        "aligned_mismatch_fraction": mismatches / aligned if aligned else 0.0,
        "baseline_final_token": baseline_tokens[-1] if baseline_tokens else None,
        "candidate_final_token": candidate_tokens[-1] if candidate_tokens else None,
    }


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FP16SplitKVReciprocalError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise FP16SplitKVReciprocalError(f"{name} must be finite and positive")
    return result


def _median_metrics(value: Mapping[str, Any]) -> dict[str, float]:
    raw_runs = value.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise FP16SplitKVReciprocalError("analysis has no run metrics")
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
        timing = _object(generation.get("timing_context"), name="timing generation")
        main = _object(generation.get("main_generation"), name="main generation")
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


def _candidate_signature(run: Mapping[str, Any]) -> dict[str, Any]:
    signatures = _object(run.get("token_signatures"), name="token_signatures")
    return {
        label: {
            "generated_tokens": _object(
                signatures.get(label), name=f"token_signatures.{label}"
            ).get("generated_tokens"),
            "token_stream_sha256": signatures[label].get("token_stream_sha256"),
            "stopping_sha256": signatures[label].get("stopping_sha256"),
            "token_groups_by_record": signatures[label].get(
                "token_groups_by_record"
            ),
        }
        for label in LABELS
    }


def analyze(
    baseline_run_root: Path,
    baseline_analysis_path: Path,
    candidate_analysis_path: Path,
    *,
    mode: str,
) -> dict[str, Any]:
    candidate_count = 1 if mode == "initial" else 5 if mode == "confirm" else 0
    if candidate_count == 0:
        raise FP16SplitKVReciprocalError("mode must be initial or confirm")
    baseline = _load(baseline_analysis_path.resolve(), name="baseline analysis")
    candidate = _load(candidate_analysis_path.resolve(), name="candidate analysis")
    _validate_analysis(baseline, name="baseline", expected_runs=5)
    _validate_analysis(candidate, name="candidate", expected_runs=candidate_count)
    baseline_preset = baseline.get(
        "preset_version", baseline.get("accepted_preset_version")
    )
    if baseline_preset != EXPECTED_PRESET_VERSION:
        raise FP16SplitKVReciprocalError("accepted FP16 baseline preset changed")
    if candidate.get("preset_version") != CANDIDATE_PRESET_VERSION:
        raise FP16SplitKVReciprocalError("candidate FP16 split-KV preset changed")
    if candidate.get("split_kv") is not True:
        raise FP16SplitKVReciprocalError("candidate split-KV hits were not verified")

    baseline_profile = _load_profile(baseline_run_root.resolve() / "run-01")
    baseline_tokens = {
        label: _profile_tokens(baseline_profile, label) for label in LABELS
    }
    baseline_run = _object(baseline["runs"][0], name="baseline.runs[0]")
    baseline_signatures = _object(
        baseline_run.get("token_signatures"), name="baseline token signatures"
    )
    candidate_runs = [
        _object(value, name=f"candidate.runs[{index}]")
        for index, value in enumerate(candidate["runs"])
    ]
    candidate_signatures = [
        _candidate_signature(value) for value in candidate_runs
    ]
    candidate_reference = candidate_signatures[0]
    candidate_audits = [
        _object(value, name=f"candidate audit {index}")
        for index, value in enumerate(candidate["exactness_audits"])
    ]
    audit_signatures = [
        {
            label: {
                key: _object(
                    _object(audit.get("token_signatures"), name="audit signatures").get(
                        label
                    ),
                    name=f"audit signatures.{label}",
                ).get(key)
                for key in (
                    "generated_tokens",
                    "token_stream_sha256",
                    "stopping_sha256",
                    "token_groups_by_record",
                )
            }
            for label in LABELS
        }
        for audit in candidate_audits
    ]
    exactness_values = [audit.get("strict_exactness") for audit in candidate_audits]
    output_values = [audit.get("output_sha256") for audit in candidate_audits]
    checks = {
        "candidate_run_repeatability": all(
            value == candidate_reference for value in candidate_signatures
        ),
        "candidate_audit_token_repeatability": all(
            value == candidate_reference for value in audit_signatures
        ),
        "candidate_rng_cache_repeatability": len(
            {_hash(value) for value in exactness_values}
        )
        == 1,
        "candidate_output_repeatability": all(
            value == candidate_runs[0].get("output_sha256")
            for value in output_values
        ),
        "candidate_fixed_work": candidate.get("fixed_work")
        == {
            "timing_tokens": candidate_reference["timing_context"][
                "generated_tokens"
            ],
            "main_tokens": candidate_reference["main_generation"][
                "generated_tokens"
            ],
        },
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise FP16SplitKVReciprocalError(
            "FP16 split-KV determinism/cache gate failed: " + ", ".join(failed)
        )

    drift = {
        label: {
            "tokens": _token_drift(
                baseline_tokens[label],
                candidate_reference[label]["token_groups_by_record"],
            ),
            "stopping_exact": _object(
                baseline_signatures.get(label), name=f"baseline signature {label}"
            ).get("stopping_sha256")
            == candidate_reference[label]["stopping_sha256"],
        }
        for label in LABELS
    }
    drift["final_map"] = {
        "bytes_exact": baseline_run.get("output_sha256")
        == candidate_runs[0].get("output_sha256"),
        "baseline_sha256": baseline_run.get("output_sha256"),
        "candidate_sha256": candidate_runs[0].get("output_sha256"),
        "baseline_structure": baseline_run.get("output_structure"),
        "candidate_structure": candidate_runs[0].get("output_structure"),
    }

    baseline_metrics = _median_metrics(baseline)
    candidate_metrics = _median_metrics(candidate)
    deltas = {
        key: _delta(baseline_metrics[key], candidate_metrics[key])
        for key in baseline_metrics
    }
    promotion_eligible = (
        deltas["main_synchronized_model_seconds"]["saved_seconds"] > 0
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
        "candidate_deterministic_and_cache_safe": True,
        "same_precision_drift": drift,
        "baseline_run_root": str(baseline_run_root.resolve()),
        "baseline_analysis": str(baseline_analysis_path.resolve()),
        "candidate_analysis": str(candidate_analysis_path.resolve()),
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
        "candidate_deterministic_and_cache_safe=true",
    ]
    for label in LABELS:
        token_drift = value["same_precision_drift"][label]["tokens"]
        lines.extend(
            (
                f"{label}_tokens_exact={str(token_drift['exact']).lower()}",
                f"{label}_aligned_mismatch_fraction={token_drift['aligned_mismatch_fraction']:.9f}",
                f"{label}_length_delta={token_drift['length_delta']}",
                f"{label}_stopping_exact={str(value['same_precision_drift'][label]['stopping_exact']).lower()}",
            )
        )
    lines.append(
        "final_map_bytes_exact="
        + str(value["same_precision_drift"]["final_map"]["bytes_exact"]).lower()
    )
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
    parser.add_argument("--baseline-run-root", type=Path, required=True)
    parser.add_argument("--baseline-analysis", type=Path, required=True)
    parser.add_argument("--candidate-analysis", type=Path, required=True)
    parser.add_argument("--mode", choices=("initial", "confirm"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        value = analyze(
            args.baseline_run_root,
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
