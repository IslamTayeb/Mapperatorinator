from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ARMS = ("exact_fp32_native_self", "full_fp16", "fp16_weights_fp32_state")
METRICS = (
    "timing_model_seconds",
    "timing_tokens",
    "timing_tps",
    "fixed_timing_context_model_seconds_at_821_tokens",
    "main_model_seconds",
    "main_tokens",
    "main_tps",
    "fixed_8294_main_seconds",
    "complete_request_wall_seconds",
    "setup_plus_capture_seconds",
    "peak_cuda_memory_allocated_mb",
)


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def summarize(payloads: dict[str, Any]) -> dict[str, Any]:
    if set(payloads) != set(ARMS):
        raise ValueError(f"payload arms must be exactly {list(ARMS)}")
    rows = {}
    baseline_reference = None
    for arm in ARMS:
        root = _object(payloads[arm], name=arm)
        metrics = _object(root.get("metrics"), name=f"{arm}.metrics")
        parity = _object(root.get("parity"), name=f"{arm}.parity")
        values = {}
        baseline = {}
        for metric in METRICS:
            row = _object(metrics.get(metric), name=f"{arm}.metrics.{metric}")
            values[metric] = {
                key: float(row[key])
                for key in (
                    "baseline_median",
                    "candidate_median",
                    "improvement",
                    "reciprocal_improvement_median",
                    "reciprocal_improvement_range",
                )
            }
            if not all(math.isfinite(value) for value in values[metric].values()):
                raise ValueError(f"{arm}.{metric} contains non-finite values")
            baseline[metric] = values[metric]["baseline_median"]
        if baseline_reference is None:
            baseline_reference = baseline
        elif baseline != baseline_reference:
            raise ValueError("matrix analyses do not share identical control runs")
        rows[arm] = {
            "metrics": values,
            "parity": {
                "claim": parity.get("claim"),
                "candidate_repeat_stable": parity.get("candidate_repeat_stable"),
                "token_and_stopping_divergence": parity.get(
                    "token_and_stopping_divergence"
                ),
                "output_divergence": parity.get("output_divergence"),
            },
        }
    request_ranking = sorted(
        ARMS,
        key=lambda arm: rows[arm]["metrics"]["complete_request_wall_seconds"][
            "improvement"
        ],
        reverse=True,
    )
    timing_ranking = sorted(
        ARMS,
        key=lambda arm: rows[arm]["metrics"]["timing_model_seconds"]["improvement"],
        reverse=True,
    )
    return {
        "schema_version": 1,
        "measurement": {
            "reciprocal": True,
            "full_song": True,
            "fixed_main_tokens": 8_294,
            "fixed_timing_tokens": 821,
            "authoritative_tps": "untraced synchronized model time",
            "product_metric": "complete request wall",
        },
        "control": baseline_reference,
        "arms": rows,
        "rankings": {
            "complete_request_wall": request_ranking,
            "timing_model": timing_ranking,
        },
    }


def text_report(report: dict[str, Any]) -> str:
    lines = [
        "schema_version=1",
        "measurement.reciprocal=true",
        "measurement.full_song=true",
        "measurement.fixed_main_tokens=8294",
        "measurement.fixed_timing_tokens=821",
    ]
    for arm in ARMS:
        row = report["arms"][arm]
        lines.append(f"arm={arm}")
        for metric in METRICS:
            value = row["metrics"][metric]
            lines.append(
                f"arm.{arm}.{metric}=baseline:{value['baseline_median']:.9f},"
                f"candidate:{value['candidate_median']:.9f},"
                f"improvement:{value['improvement']:.9f},"
                f"reciprocal_median:{value['reciprocal_improvement_median']:.9f},"
                f"reciprocal_range:{value['reciprocal_improvement_range']:.9f}"
            )
        divergence = row["parity"]["token_and_stopping_divergence"]
        for label in ("timing_context", "main_generation"):
            values = divergence[label]
            lines.append(
                f"arm.{arm}.divergence.{label}=tokens_equal:"
                f"{str(values['token_stream_equal']).lower()},stopping_equal:"
                f"{str(values['stopping_equal']).lower()},aligned_mismatch_fraction:"
                f"{values['aligned_mismatch_fraction']:.9f}"
            )
        output = row["parity"]["output_divergence"]
        lines.append(
            f"arm.{arm}.divergence.final_map_equal="
            f"{str(output['final_map_equal']).lower()}"
        )
    lines.append(
        "ranking.complete_request_wall="
        + ",".join(report["rankings"]["complete_request_wall"])
    )
    lines.append(
        "ranking.timing_model=" + ",".join(report["rankings"]["timing_model"])
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    for arm in ARMS:
        parser.add_argument(f"--{arm.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    args = parser.parse_args()
    payloads = {
        arm: json.loads(
            getattr(args, arm).read_text(encoding="utf-8")
        )
        for arm in ARMS
    }
    report = summarize(payloads)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.text_output.write_text(text_report(report), encoding="utf-8")


if __name__ == "__main__":
    main()
