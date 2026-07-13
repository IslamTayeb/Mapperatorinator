"""Summarize the FP16 native-prefix component gate.

Input JSON schema (schema_version 1)::

    {
      "schema_version": 1,
      "variants": {
        "fp32_accepted": {
          "buckets": {
            "128": {
              "full_model_replay_ms_per_call": 2.1,
              "capture_setup_seconds": 0.04,
              "prefix_replay_ms_per_layer": 0.08,
              "checks": {
                "finite_outputs": true,
                "cache_shapes_valid": true,
                "active_slot_writes_valid": true,
                "future_slots_untouched": true,
                "cross_cache_unchanged": true,
                "storage_ownership_valid": true,
                "candidate_repeat_deterministic": true,
                "graph_repeat_deterministic": true,
                "memory_stable": true
              }
            }
          }
        },
        "fp32_shared_specialized": {"buckets": {"...": "same bucket schema"}},
        "fp16_framework_self_bmm_cross": {"buckets": {"...": "same bucket schema"}},
        "fp16_native_self_bmm_cross": {"buckets": {"...": "same bucket schema"}}
      }
    }

Each variant must contain either exactly the sentinel buckets 128, 576, and
640, or all current accepted production buckets.  Sentinel evidence is useful
for bounded sizing but can never promote a candidate.  All variants must cover
the same buckets.

``full_model_replay_ms_per_call`` is the worst reciprocal CUDA-graph replay
time for one
complete one-token model forward; it already includes all decoder layers.
``capture_setup_seconds`` is the aggregate once-per-production-signature setup
and capture wall for that prefix, not a per-replay average.  The projection
charges its baseline-to-candidate delta separately so flat setup is not hidden.
``prefix_replay_ms_per_layer`` is the matched self+cross-prefix boundary used
to compare the two FP16 self-attention policies while both retain accepted-style
q1 BMM cross-attention.  The native-self variant is retained only when it is at
least five percent faster than framework self-attention.  The recaptured FP32
shared-specialized graph is a parity control: it must be exact and no more than
one percent slower than the cached accepted graph before any FP16 candidate can
advance.

The full-bucket projection is::

    replay_v = sum(count[b] * replay_ms[v, b]) / 1000
    setup_v = sum(capture_setup_seconds[v, b])
    projected_main_v = 28.243
        + (replay_v - replay_fp32)
        + max(0, setup_v - setup_fp32)

For sentinel-only sizing, unmeasured replay and setup deltas are conservatively
set to zero.  That result can authorize collection of the remaining buckets,
but it is not promotable evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


BUCKET_COUNTS = {
    128: 30,
    192: 64,
    256: 64,
    320: 71,
    384: 161,
    448: 448,
    512: 1_166,
    576: 1_647,
    640: 2_470,
    704: 1_142,
    768: 242,
    832: 92,
}
SENTINEL_BUCKETS = frozenset({128, 576, 640})
ALL_BUCKETS = frozenset(BUCKET_COUNTS)
VARIANTS = (
    "fp32_accepted",
    "fp32_shared_specialized",
    "fp16_framework_self_bmm_cross",
    "fp16_native_self_bmm_cross",
)
CHECKS = (
    "finite_outputs",
    "cache_shapes_valid",
    "active_slot_writes_valid",
    "future_slots_untouched",
    "cross_cache_unchanged",
    "storage_ownership_valid",
    "candidate_repeat_deterministic",
    "graph_repeat_deterministic",
    "memory_stable",
)
DRIFT_FIELDS = (
    "layer_output_max_abs",
    "cache_key_slot_max_abs",
    "cache_value_slot_max_abs",
    "logits_max_abs",
)
BASELINE_MAIN_SECONDS = 28.243
BASELINE_MAIN_TOKENS = 7_684
BASELINE_MAIN_TPS = BASELINE_MAIN_TOKENS / BASELINE_MAIN_SECONDS
TARGET_MAIN_SECONDS = 25.419
TARGET_MAIN_TPS = BASELINE_MAIN_TOKENS / TARGET_MAIN_SECONDS
TARGET_TPS_DELTA = TARGET_MAIN_TPS - BASELINE_MAIN_TPS
DECODER_LAYERS = 12
NATIVE_PREFIX_MIN_ADVANTAGE_PCT = 5.0
FP32_PARITY_MAX_REGRESSION_PCT = 1.0


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _number(value: Any, *, name: str, strictly_positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if not strictly_positive and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _bucket_key(value: Any, *, name: str) -> int:
    try:
        bucket = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} has a non-integer bucket key: {value!r}") from exc
    if str(bucket) != str(value):
        raise ValueError(f"{name} has a non-canonical bucket key: {value!r}")
    if bucket not in BUCKET_COUNTS:
        raise ValueError(f"{name} contains unsupported bucket {bucket}")
    return bucket


def _parse_bucket(
    payload: Any,
    *,
    variant: str,
    bucket: int,
) -> dict[str, Any]:
    name = f"variants.{variant}.buckets.{bucket}"
    entry = _object(payload, name=name)
    parsed = {
        "full_model_replay_ms_per_call": _number(
            entry.get("full_model_replay_ms_per_call"),
            name=f"{name}.full_model_replay_ms_per_call",
            strictly_positive=True,
        ),
        "capture_setup_seconds": _number(
            entry.get("capture_setup_seconds"),
            name=f"{name}.capture_setup_seconds",
            strictly_positive=False,
        ),
        "prefix_replay_ms_per_layer": _number(
            entry.get("prefix_replay_ms_per_layer"),
            name=f"{name}.prefix_replay_ms_per_layer",
            strictly_positive=True,
        ),
    }
    checks = _object(entry.get("checks"), name=f"{name}.checks")
    missing = [check for check in CHECKS if check not in checks]
    if missing:
        raise ValueError(f"{name}.checks is missing required checks: {missing}")
    invalid = [check for check in CHECKS if type(checks[check]) is not bool]
    if invalid:
        raise ValueError(f"{name}.checks must contain JSON booleans for: {invalid}")
    parsed["checks"] = {check: checks[check] for check in CHECKS}
    drift = _object(entry.get("drift"), name=f"{name}.drift")
    missing_drift = [field for field in DRIFT_FIELDS if field not in drift]
    if missing_drift:
        raise ValueError(f"{name}.drift is missing required fields: {missing_drift}")
    parsed["drift"] = {
        field: _number(
            drift[field],
            name=f"{name}.drift.{field}",
            strictly_positive=False,
        )
        for field in DRIFT_FIELDS
    }
    return parsed


def _parse(payload: Any) -> tuple[str, dict[str, dict[int, dict[str, Any]]]]:
    root = _object(payload, name="input")
    if root.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    variants = _object(root.get("variants"), name="variants")
    missing_variants = [variant for variant in VARIANTS if variant not in variants]
    if missing_variants:
        raise ValueError(f"variants is missing required entries: {missing_variants}")

    parsed: dict[str, dict[int, dict[str, Any]]] = {}
    bucket_sets: dict[str, frozenset[int]] = {}
    for variant in VARIANTS:
        variant_payload = _object(variants[variant], name=f"variants.{variant}")
        buckets_payload = _object(
            variant_payload.get("buckets"),
            name=f"variants.{variant}.buckets",
        )
        buckets: dict[int, dict[str, Any]] = {}
        for raw_bucket, entry in buckets_payload.items():
            bucket = _bucket_key(raw_bucket, name=f"variants.{variant}.buckets")
            if bucket in buckets:
                raise ValueError(f"variants.{variant}.buckets repeats bucket {bucket}")
            buckets[bucket] = _parse_bucket(entry, variant=variant, bucket=bucket)
        parsed[variant] = buckets
        bucket_sets[variant] = frozenset(buckets)

    reference_buckets = bucket_sets[VARIANTS[0]]
    mismatched = {
        variant: sorted(buckets)
        for variant, buckets in bucket_sets.items()
        if buckets != reference_buckets
    }
    if mismatched:
        raise ValueError(f"all variants must cover identical buckets: {mismatched}")
    if reference_buckets == SENTINEL_BUCKETS:
        evidence_mode = "sentinel_only"
    elif reference_buckets == ALL_BUCKETS:
        evidence_mode = "all_production_buckets"
    else:
        raise ValueError(
            "bucket coverage must be exactly sentinel buckets "
            f"{sorted(SENTINEL_BUCKETS)} or all production buckets {sorted(ALL_BUCKETS)}; "
            f"got {sorted(reference_buckets)}"
        )
    return evidence_mode, parsed


def _correctness(
    variants: dict[str, dict[int, dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    reports: dict[str, Any] = {}
    passed: dict[str, bool] = {}
    for variant, buckets in variants.items():
        failures = {}
        for bucket, entry in sorted(buckets.items()):
            bucket_failures = [
                check for check, result in entry["checks"].items() if not result
            ]
            failures[str(bucket)] = bucket_failures
        failures = {bucket: checks for bucket, checks in failures.items() if checks}
        passed[variant] = not failures
        reports[variant] = {"pass": passed[variant], "failures": failures}
    return reports, passed


def _weighted(
    buckets: dict[int, dict[str, Any]],
) -> dict[str, float]:
    replay_seconds = sum(
        BUCKET_COUNTS[bucket] * entry["full_model_replay_ms_per_call"]
        for bucket, entry in buckets.items()
    ) / 1000.0
    capture_setup_seconds = sum(
        entry["capture_setup_seconds"] for entry in buckets.values()
    )
    prefix_seconds = DECODER_LAYERS * sum(
        BUCKET_COUNTS[bucket] * entry["prefix_replay_ms_per_layer"]
        for bucket, entry in buckets.items()
    ) / 1000.0
    return {
        "full_model_replay_seconds": replay_seconds,
        "capture_setup_seconds": capture_setup_seconds,
        "prefix_replay_seconds_12_layers": prefix_seconds,
    }


def summarize(payload: Any) -> dict[str, Any]:
    """Validate and summarize one component report payload."""

    evidence_mode, variants = _parse(payload)
    measured_buckets = sorted(next(iter(variants.values())))
    measured_replays = sum(BUCKET_COUNTS[bucket] for bucket in measured_buckets)
    total_replays = sum(BUCKET_COUNTS.values())
    correctness_reports, correctness_pass = _correctness(variants)
    weighted = {variant: _weighted(buckets) for variant, buckets in variants.items()}

    baseline = weighted["fp32_accepted"]
    projections: dict[str, Any] = {}
    for variant in VARIANTS[1:]:
        values = weighted[variant]
        replay_delta = (
            values["full_model_replay_seconds"]
            - baseline["full_model_replay_seconds"]
        )
        capture_delta = max(
            0.0,
            values["capture_setup_seconds"] - baseline["capture_setup_seconds"],
        )
        projected_main = BASELINE_MAIN_SECONDS + replay_delta + capture_delta
        projections[variant] = {
            "replay_delta_seconds": replay_delta,
            "capture_setup_delta_seconds": capture_delta,
            "projected_main_seconds": projected_main,
            "projected_tokens_per_second_at_current_7684_tokens": (
                BASELINE_MAIN_TOKENS / projected_main
            ),
            "target_main_seconds": TARGET_MAIN_SECONDS,
            "target_pass": projected_main <= TARGET_MAIN_SECONDS,
        }

    baseline_valid = correctness_pass["fp32_accepted"]
    shared = weighted["fp32_shared_specialized"]
    fp32_replay_regression_pct = (
        (shared["full_model_replay_seconds"] - baseline["full_model_replay_seconds"])
        / baseline["full_model_replay_seconds"]
        * 100.0
    )
    fp32_exact = all(
        value == 0.0
        for entry in variants["fp32_shared_specialized"].values()
        for value in entry["drift"].values()
    )
    fp32_parity_pass = bool(
        baseline_valid
        and correctness_pass["fp32_shared_specialized"]
        and fp32_exact
        and fp32_replay_regression_pct <= FP32_PARITY_MAX_REGRESSION_PCT
    )
    fp32_parity = {
        "exact_drift_pass": fp32_exact,
        "correctness_pass": correctness_pass["fp32_shared_specialized"],
        "replay_regression_pct": fp32_replay_regression_pct,
        "maximum_regression_pct": FP32_PARITY_MAX_REGRESSION_PCT,
        "pass": fp32_parity_pass,
    }

    framework_variant = "fp16_framework_self_bmm_cross"
    native_variant = "fp16_native_self_bmm_cross"
    framework_prefix = weighted[framework_variant]["prefix_replay_seconds_12_layers"]
    native_prefix = weighted[native_variant]["prefix_replay_seconds_12_layers"]
    native_advantage_pct = (
        (framework_prefix - native_prefix) / framework_prefix * 100.0
    )
    native_timing_pass = native_advantage_pct >= NATIVE_PREFIX_MIN_ADVANTAGE_PCT
    native_correctness_pair_pass = bool(
        correctness_pass[framework_variant] and correctness_pass[native_variant]
    )
    native_comparison = {
        "framework_variant": framework_variant,
        "native_variant": native_variant,
        "framework_prefix_seconds": framework_prefix,
        "native_prefix_seconds": native_prefix,
        "native_advantage_pct": native_advantage_pct,
        "minimum_advantage_pct": NATIVE_PREFIX_MIN_ADVANTAGE_PCT,
        "timing_pass": native_timing_pass,
        "correctness_pair_pass": native_correctness_pair_pass,
        "retained": bool(native_timing_pass and native_correctness_pair_pass),
    }

    decisions = {
        framework_variant: {
            "correctness_pass": correctness_pass[framework_variant],
            "target_pass": projections[framework_variant]["target_pass"],
            "native_retention_required": False,
        },
        native_variant: {
            "correctness_pass": correctness_pass[native_variant],
            "target_pass": projections[native_variant]["target_pass"],
            "native_retention_required": True,
            "native_retention_pass": native_comparison["retained"],
        },
    }
    for variant, decision in decisions.items():
        decision["sizing_pass"] = bool(
            fp32_parity_pass
            and decision["correctness_pass"]
            and decision["target_pass"]
            and (
                not decision["native_retention_required"]
                or decision["native_retention_pass"]
            )
        )
        decision["promotion_pass"] = bool(
            evidence_mode == "all_production_buckets" and decision["sizing_pass"]
        )

    sizing_pass = any(decision["sizing_pass"] for decision in decisions.values())
    promotion_pass = any(decision["promotion_pass"] for decision in decisions.values())
    if not fp32_parity_pass:
        disposition = "STOP_FP32_PARITY"
    elif promotion_pass:
        disposition = "PROMOTE_TO_FIXED_WORK_LOOP"
    elif evidence_mode == "sentinel_only" and sizing_pass:
        disposition = "COLLECT_ALL_PRODUCTION_BUCKETS"
    else:
        disposition = "STOP_COMPONENT_SCOUT"

    return {
        "schema_version": 1,
        "result_class": "documented-drift-component-sizing",
        "constants": {
            "bucket_counts": {str(key): value for key, value in BUCKET_COUNTS.items()},
            "baseline_main_seconds": BASELINE_MAIN_SECONDS,
            "baseline_main_tokens": BASELINE_MAIN_TOKENS,
            "baseline_main_tps": BASELINE_MAIN_TPS,
            "target_tps_delta": TARGET_TPS_DELTA,
            "target_main_tps": TARGET_MAIN_TPS,
            "target_main_seconds": TARGET_MAIN_SECONDS,
            "decoder_layers": DECODER_LAYERS,
            "native_prefix_min_advantage_pct": NATIVE_PREFIX_MIN_ADVANTAGE_PCT,
            "fp32_parity_max_regression_pct": FP32_PARITY_MAX_REGRESSION_PCT,
        },
        "evidence": {
            "mode": evidence_mode,
            "measured_buckets": measured_buckets,
            "missing_buckets": sorted(ALL_BUCKETS - set(measured_buckets)),
            "measured_replays": measured_replays,
            "total_replays": total_replays,
            "coverage_fraction": measured_replays / total_replays,
            "sentinel_coverage_pass": SENTINEL_BUCKETS.issubset(measured_buckets),
            "promotion_coverage_pass": evidence_mode == "all_production_buckets",
            "unmeasured_delta_policy": (
                "none; all accepted production buckets measured"
                if evidence_mode == "all_production_buckets"
                else "zero replay and capture/setup delta outside measured sentinel buckets"
            ),
        },
        "correctness": {
            "baseline_pass": baseline_valid,
            "variants": correctness_reports,
        },
        "weighted": weighted,
        "projections": projections,
        "fp32_parity": fp32_parity,
        "native_prefix_comparison": native_comparison,
        "candidate_decisions": decisions,
        "sizing_pass": sizing_pass,
        "promotion_pass": promotion_pass,
        "disposition": disposition,
    }


def _write_json(path: Path | None, report: dict[str, Any]) -> None:
    output = json.dumps(report, indent=2) + "\n"
    if path is None:
        print(output, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize FP32/FP16 native-prefix component timing and correctness. "
            "See the module docstring for the strict schema and formulas."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", type=Path, help="component profiler JSON")
    parser.add_argument("--json-output", type=Path, help="write the summary JSON here")
    args = parser.parse_args()

    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        report = summarize(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    _write_json(args.json_output, report)
    raise SystemExit(0 if report["sizing_pass"] else 1)


if __name__ == "__main__":
    main()
