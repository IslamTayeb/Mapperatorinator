"""Decide mask reuse atop the relaxed K4+split+mixed+shared-RoPE stack."""

from __future__ import annotations

import argparse
from importlib import import_module
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

analyze = import_module("utils.analyze_reciprocal_full_song_candidate").analyze


SCHEMA_VERSION = "mapperatorinator.k4-shared-rope-mask-reuse-reciprocal.v3"
INCREMENTAL_EXACTNESS_SCOPE = (
    "mask-reuse-vs-relaxed-k4-split-weight-shared-rope-control"
)
WALL_METRICS = (
    "main_outer_stage_wall_seconds",
    "complete_request_wall_seconds",
)


def decide(
    analysis: Mapping[str, Any],
    *,
    minimum_wall_saving_seconds: float = 0.05,
    maximum_model_regression_fraction: float = 0.01,
    maximum_timing_stage_regression_fraction: float = 0.01,
) -> dict[str, Any]:
    for name, value in (
        ("minimum_wall_saving_seconds", minimum_wall_saving_seconds),
        ("maximum_model_regression_fraction", maximum_model_regression_fraction),
        (
            "maximum_timing_stage_regression_fraction",
            maximum_timing_stage_regression_fraction,
        ),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{name} must be finite and non-negative")
    if analysis.get("mode") != "relaxed":
        raise ValueError(
            "mask reuse must use relaxed analysis with incremental exactness gates"
        )
    parity = analysis.get("parity")
    if not isinstance(parity, Mapping):
        raise ValueError("analysis parity is missing")
    if parity.get("claim") != "relaxed-nonexact":
        raise ValueError("mask reuse analysis must retain the relaxed base claim")
    parity_pass = bool(
        parity.get("cross_candidate_exact")
        and parity.get("baseline_repeat_stable")
        and parity.get("candidate_repeat_stable")
        and parity.get("required_exact_labels_pass")
        and parity.get("dispatch_cache_topology", {}).get("pass")
    )
    metrics = analysis.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("analysis metrics are missing")
    walls = {}
    failures = []
    for metric in WALL_METRICS:
        values = metrics.get(metric)
        if not isinstance(values, Mapping):
            raise ValueError(f"analysis is missing {metric}")
        pairs = values.get("reciprocal_pairs")
        if not isinstance(pairs, list) or len(pairs) != 2:
            raise ValueError(f"{metric} must contain two reciprocal pairs")
        pair_improvements = [float(pair["improvement"]) for pair in pairs]
        median = float(values["reciprocal_improvement_median"])
        pairwise_nonregression = all(value > 0.0 for value in pair_improvements)
        saving_pass = median >= minimum_wall_saving_seconds
        walls[metric] = {
            "baseline_median": float(values["baseline_median"]),
            "candidate_median": float(values["candidate_median"]),
            "reciprocal_saving_seconds": median,
            "pair_savings_seconds": pair_improvements,
            "pairwise_nonregression_pass": pairwise_nonregression,
            "minimum_saving_pass": saving_pass,
        }
        if not pairwise_nonregression:
            failures.append(f"{metric} regressed in a reciprocal order")
        if not saving_pass:
            failures.append(
                f"{metric} saving {median:.6f}s is below "
                f"{minimum_wall_saving_seconds:.6f}s"
            )
    model = metrics.get("main_model_seconds")
    if not isinstance(model, Mapping):
        raise ValueError("analysis is missing main_model_seconds")
    model_delta_fraction = float(model["candidate_minus_baseline_fraction"])
    model_pass = model_delta_fraction <= maximum_model_regression_fraction
    if not model_pass:
        failures.append(
            "main synchronized model time regressed by "
            f"{model_delta_fraction * 100.0:.3f}%"
        )
    timing_stage = metrics.get("timing_outer_stage_wall_seconds")
    if not isinstance(timing_stage, Mapping):
        raise ValueError("analysis is missing timing_outer_stage_wall_seconds")
    timing_stage_delta_fraction = float(
        timing_stage["candidate_minus_baseline_fraction"]
    )
    timing_stage_pairs = timing_stage.get("reciprocal_pairs")
    if not isinstance(timing_stage_pairs, list) or len(timing_stage_pairs) != 2:
        raise ValueError(
            "timing_outer_stage_wall_seconds must contain two reciprocal pairs"
        )
    timing_stage_pair_regression_fractions = [
        max(0.0, -float(pair["improvement"])) / float(pair["baseline"])
        for pair in timing_stage_pairs
    ]
    timing_stage_pairwise_pass = all(
        value <= maximum_timing_stage_regression_fraction
        for value in timing_stage_pair_regression_fractions
    )
    timing_stage_pass = (
        timing_stage_delta_fraction <= maximum_timing_stage_regression_fraction
        and timing_stage_pairwise_pass
    )
    if timing_stage_delta_fraction > maximum_timing_stage_regression_fraction:
        failures.append(
            "median timing outer stage regressed by "
            f"{timing_stage_delta_fraction * 100.0:.3f}%"
        )
    if not timing_stage_pairwise_pass:
        failures.append(
            "timing outer stage exceeded the regression tolerance in a "
            "reciprocal order"
        )
    if not parity_pass:
        failures.append("exact token/stopping/output/dispatch parity failed")
    return {
        "schema_version": SCHEMA_VERSION,
        "incremental_exactness_scope": INCREMENTAL_EXACTNESS_SCOPE,
        "result_class": "promotion" if not failures else "valid_negative",
        "promotion_pass": not failures,
        "parity_pass": parity_pass,
        "thresholds": {
            "minimum_wall_saving_seconds": minimum_wall_saving_seconds,
            "maximum_model_regression_fraction": maximum_model_regression_fraction,
            "maximum_timing_stage_regression_fraction": (
                maximum_timing_stage_regression_fraction
            ),
        },
        "walls": walls,
        "main_model": {
            "baseline_median": float(model["baseline_median"]),
            "candidate_median": float(model["candidate_median"]),
            "candidate_minus_baseline_fraction": model_delta_fraction,
            "nonregression_pass": model_pass,
        },
        "timing_stage": {
            "baseline_median": float(timing_stage["baseline_median"]),
            "candidate_median": float(timing_stage["candidate_median"]),
            "candidate_minus_baseline_fraction": timing_stage_delta_fraction,
            "pair_regression_fractions": timing_stage_pair_regression_fractions,
            "pairwise_nonregression_pass": timing_stage_pairwise_pass,
            "nonregression_pass": timing_stage_pass,
        },
        "failures": failures,
        "analysis": analysis,
    }


def text_report(report: Mapping[str, Any]) -> str:
    lines = [
        f"result_class={report['result_class']}",
        f"promotion_pass={str(report['promotion_pass']).lower()}",
        f"parity_pass={str(report['parity_pass']).lower()}",
        f"incremental_exactness_scope={report['incremental_exactness_scope']}",
    ]
    for metric in WALL_METRICS:
        wall = report["walls"][metric]
        lines.append(
            f"{metric}=baseline:{wall['baseline_median']:.9f},"
            f"candidate:{wall['candidate_median']:.9f},"
            f"reciprocal_saving:{wall['reciprocal_saving_seconds']:.9f},"
            f"pairwise_nonregression:{str(wall['pairwise_nonregression_pass']).lower()},"
            f"minimum_saving:{str(wall['minimum_saving_pass']).lower()}"
        )
    lines.append(
        "main_model_seconds="
        f"baseline:{report['main_model']['baseline_median']:.9f},"
        f"candidate:{report['main_model']['candidate_median']:.9f},"
        "candidate_minus_baseline_fraction:"
        f"{report['main_model']['candidate_minus_baseline_fraction']:.9f},"
        f"nonregression:{str(report['main_model']['nonregression_pass']).lower()}"
    )
    lines.append(
        "timing_outer_stage_wall_seconds="
        f"baseline:{report['timing_stage']['baseline_median']:.9f},"
        f"candidate:{report['timing_stage']['candidate_median']:.9f},"
        "candidate_minus_baseline_fraction:"
        f"{report['timing_stage']['candidate_minus_baseline_fraction']:.9f},"
        "pairwise_nonregression:"
        f"{str(report['timing_stage']['pairwise_nonregression_pass']).lower()},"
        f"nonregression:{str(report['timing_stage']['nonregression_pass']).lower()}"
    )
    lines.extend(f"failure={failure}" for failure in report["failures"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    for role in (
        "baseline_first",
        "candidate_first",
        "candidate_second",
        "baseline_second",
    ):
        parser.add_argument(f"--{role.replace('_', '-')}-profile", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    parser.add_argument("--minimum-wall-saving-seconds", type=float, default=0.05)
    parsed = parser.parse_args()
    paths = {
        role: getattr(parsed, f"{role}_profile")
        for role in (
            "baseline_first",
            "candidate_first",
            "candidate_second",
            "baseline_second",
        )
    }
    dispatch_patterns = tuple(
        f"records.{label}[[]*].optimized_cuda_graphs.k8_candidate."
        "decoder_attention_mask_reuse*"
        for label in ("timing_context", "main_generation")
    )
    analysis = analyze(
        paths,
        mode="relaxed",
        optional_allowed_dispatch_deltas=dispatch_patterns,
        required_exact_labels=("timing_context", "main_generation"),
        require_dispatch_declaration=True,
    )
    report = decide(
        analysis,
        minimum_wall_saving_seconds=parsed.minimum_wall_saving_seconds,
    )
    parsed.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    parsed.text_output.write_text(text_report(report), encoding="utf-8")
    raise SystemExit(0 if report["promotion_pass"] else 3)


if __name__ == "__main__":
    main()
