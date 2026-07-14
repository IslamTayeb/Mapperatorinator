"""Summarize paired in-process request-to-output inference walls."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PROFILE_SCHEMA_VERSION = 1
EXPECTED_PASS_KINDS = {
    "control": "untraced_control",
    "budget": "untraced_budget",
}
FINAL_WRITE_STAGES = frozenset({"write_osu", "write_osz"})
GROUP_STAGE_NAMES = {
    "pre_request_setup_load": frozenset(
        {
            "compile_args",
            "setup_inference_environment",
            "load_main_model",
            "load_timing_model",
            "load_diffusion_model",
            "build_generation_config",
        }
    ),
    "request_setup": frozenset({"validate_inputs", "setup_processors"}),
    "audio_preparation": frozenset({"audio_load", "audio_segment"}),
    "timing_generation": frozenset(
        {
            "super_timing_generation",
            "timing_context_generation",
            "load_reference_timing",
        }
    ),
    "timing_postprocess": frozenset(
        {"super_timing_postprocess", "timing_context_postprocess"}
    ),
    "main_generation": frozenset({"main_generation"}),
    "merge_resnap_postprocess_write": frozenset(
        {
            "merge_generated_events",
            "derive_timing_from_generated_events",
            "resnap_events",
            "diffusion_position_generation",
            "postprocess_generate_osu",
            "merge_with_reference_beatmap",
            "write_osu",
            "write_osz",
        }
    ),
}
REQUEST_GROUPS = tuple(name for name in GROUP_STAGE_NAMES if name != "pre_request_setup_load")
ALL_SUPPORTED_STAGE_NAMES = frozenset().union(*GROUP_STAGE_NAMES.values())


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _number(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result <= 0.0):
        qualifier = "positive finite" if positive else "finite non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _parse_profile(value: Any, *, role: str) -> dict[str, Any]:
    root = _object(value, name=f"{role}_profile")
    if root.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"{role}_profile.schema_version must be {PROFILE_SCHEMA_VERSION}"
        )
    metadata = _object(root.get("metadata"), name=f"{role}_profile.metadata")
    expected_pass = EXPECTED_PASS_KINDS[role]
    if metadata.get("profile_pass_kind") != expected_pass:
        raise ValueError(
            f"{role} profile must use profile_pass_kind={expected_pass}"
        )
    raw_stages = root.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError(f"{role}_profile.stages must be a non-empty list")

    stages: list[dict[str, Any]] = []
    for index, value in enumerate(raw_stages):
        raw = _object(value, name=f"{role}_profile.stages[{index}]")
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{role}_profile.stages[{index}].name must be non-empty")
        if name not in ALL_SUPPORTED_STAGE_NAMES:
            raise ValueError(f"{role} profile contains unsupported stage {name!r}")
        started = _number(
            raw.get("started_at_perf_counter_seconds"),
            name=f"{role}_profile.stages[{index}].started_at_perf_counter_seconds",
        )
        finished = _number(
            raw.get("finished_at_perf_counter_seconds"),
            name=f"{role}_profile.stages[{index}].finished_at_perf_counter_seconds",
        )
        if finished < started:
            raise ValueError(f"{role} stage {name!r} finishes before it starts")
        wall = _number(
            raw.get("wall_seconds"),
            name=f"{role}_profile.stages[{index}].wall_seconds",
        )
        if not math.isclose(wall, finished - started, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{role} stage {name!r} wall disagrees with timestamps")
        stages.append(
            {
                "index": index,
                "name": name,
                "wall_seconds": wall,
                "started_at_perf_counter_seconds": started,
                "finished_at_perf_counter_seconds": finished,
                "metadata": {
                    key: child
                    for key, child in raw.items()
                    if key
                    not in {
                        "name",
                        "wall_seconds",
                        "started_at_perf_counter_seconds",
                        "finished_at_perf_counter_seconds",
                    }
                },
            }
        )

    stages.sort(key=lambda stage: (stage["started_at_perf_counter_seconds"], stage["index"]))
    if [stage["index"] for stage in stages] != list(range(len(stages))):
        raise ValueError(f"{role} stages are not recorded in chronological order")
    for previous, current in zip(stages, stages[1:], strict=False):
        if current["started_at_perf_counter_seconds"] < (
            previous["finished_at_perf_counter_seconds"] - 1e-9
        ):
            raise ValueError(
                f"{role} stages overlap: {previous['name']!r} and {current['name']!r}"
            )

    by_name: dict[str, list[dict[str, Any]]] = {}
    for stage in stages:
        by_name.setdefault(stage["name"], []).append(stage)
    for required in ("compile_args", "validate_inputs", "main_generation"):
        if len(by_name.get(required, [])) != 1:
            raise ValueError(f"{role} profile must contain exactly one {required} stage")
    final_writes = [stage for stage in stages if stage["name"] in FINAL_WRITE_STAGES]
    if len(final_writes) != 1:
        raise ValueError(f"{role} profile must contain exactly one final write stage")

    compile_stage = by_name["compile_args"][0]
    request_start = by_name["validate_inputs"][0]
    final_write = final_writes[0]
    if compile_stage is not stages[0]:
        raise ValueError(f"{role} compile_args must be the first measured stage")
    if final_write is not stages[-1]:
        raise ValueError(f"{role} final write must be the last measured stage")
    if request_start["started_at_perf_counter_seconds"] < (
        compile_stage["started_at_perf_counter_seconds"]
    ):
        raise ValueError(f"{role} request starts before compile_args")
    if final_write["finished_at_perf_counter_seconds"] < (
        request_start["started_at_perf_counter_seconds"]
    ):
        raise ValueError(f"{role} final write finishes before request start")

    groups = {
        group_name: _group_summary(
            [stage for stage in stages if stage["name"] in stage_names]
        )
        for group_name, stage_names in GROUP_STAGE_NAMES.items()
    }
    request_stages = [
        stage
        for stage in stages
        if stage["started_at_perf_counter_seconds"]
        >= request_start["started_at_perf_counter_seconds"]
    ]
    cold_span = _outer_span(stages, compile_stage, final_write)
    request_span = _outer_span(request_stages, request_start, final_write)
    return {
        "profile_pass_kind": expected_pass,
        "final_write_stage": final_write["name"],
        "stage_sequence": [stage["name"] for stage in stages],
        "stage_totals_seconds": _stage_totals(stages),
        "stages": stages,
        "adjacent_interstage_gaps": _adjacent_gaps(stages),
        "groups": groups,
        "request_to_output": request_span,
        "cold_in_process": cold_span,
    }


def _stage_totals(stages: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for stage in stages:
        name = stage["name"]
        totals[name] = totals.get(name, 0.0) + stage["wall_seconds"]
    return totals


def _adjacent_gaps(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "after_stage": previous["name"],
            "before_stage": current["name"],
            "seconds": max(
                0.0,
                current["started_at_perf_counter_seconds"]
                - previous["finished_at_perf_counter_seconds"],
            ),
        }
        for previous, current in zip(stages, stages[1:], strict=False)
    ]


def _group_summary(stages: list[dict[str, Any]]) -> dict[str, Any]:
    if not stages:
        return {
            "stage_names": [],
            "stage_wall_seconds": 0.0,
            "timeline_span_seconds": 0.0,
            "residual_gap_seconds": 0.0,
        }
    stage_wall = sum(stage["wall_seconds"] for stage in stages)
    span = (
        stages[-1]["finished_at_perf_counter_seconds"]
        - stages[0]["started_at_perf_counter_seconds"]
    )
    residual = span - stage_wall
    if residual < -1e-9:
        raise ValueError("group stage wall exceeds its timeline span")
    return {
        "stage_names": [stage["name"] for stage in stages],
        "stage_wall_seconds": stage_wall,
        "timeline_span_seconds": span,
        "residual_gap_seconds": max(0.0, residual),
    }


def _outer_span(
    stages: list[dict[str, Any]],
    first: dict[str, Any],
    last: dict[str, Any],
) -> dict[str, Any]:
    wall = (
        last["finished_at_perf_counter_seconds"]
        - first["started_at_perf_counter_seconds"]
    )
    stage_wall = sum(stage["wall_seconds"] for stage in stages)
    residual = wall - stage_wall
    if residual < -1e-9:
        raise ValueError("stage wall exceeds outer timeline span")
    return {
        "start_stage": first["name"],
        "finish_stage": last["name"],
        "wall_seconds": wall,
        "stage_wall_seconds": stage_wall,
        "residual_gap_seconds": max(0.0, residual),
        "residual_gap_fraction": max(0.0, residual) / wall if wall > 0 else 0.0,
    }


def _comparison(control: float, budget: float) -> dict[str, Any]:
    delta = budget - control
    return {
        "control_seconds": control,
        "budget_seconds": budget,
        "delta_seconds": delta,
        "delta_fraction_of_control": delta / control if control > 0 else None,
    }


def summarize(control_payload: Any, budget_payload: Any) -> dict[str, Any]:
    control = _parse_profile(control_payload, role="control")
    budget = _parse_profile(budget_payload, role="budget")
    if control["stage_sequence"] != budget["stage_sequence"]:
        raise ValueError("control and budget stage sequences differ")
    if control["final_write_stage"] != budget["final_write_stage"]:
        raise ValueError("control and budget final write stages differ")

    comparisons = {
        "request_to_output": _comparison(
            control["request_to_output"]["wall_seconds"],
            budget["request_to_output"]["wall_seconds"],
        ),
        "cold_in_process": _comparison(
            control["cold_in_process"]["wall_seconds"],
            budget["cold_in_process"]["wall_seconds"],
        ),
        "groups": {
            group: _comparison(
                control["groups"][group]["stage_wall_seconds"],
                budget["groups"][group]["stage_wall_seconds"],
            )
            for group in GROUP_STAGE_NAMES
        },
        "residual_gaps": {
            span: _comparison(
                control[span]["residual_gap_seconds"],
                budget[span]["residual_gap_seconds"],
            )
            for span in ("request_to_output", "cold_in_process")
        },
    }
    return {
        "schema_version": 1,
        "measurement_scope": {
            "request_to_output": "validate_inputs start through final output write finish",
            "cold_in_process": "compile_args start through final output write finish",
            "authoritative_performance_role": "control",
            "budget_role": "instrumentation diagnostic",
            "stage_boundary_cuda_synchronization": True,
            "excludes": [
                "Python process launch and imports before compile_args",
                "profile JSON serialization after final output write",
            ],
            "cold_cache_caveat": (
                "cold_in_process is process-local; shared filesystem, model, and native "
                "build caches may already be warm"
            ),
        },
        "group_order": list(GROUP_STAGE_NAMES),
        "request_group_order": list(REQUEST_GROUPS),
        "control": control,
        "budget": budget,
        "paired_overhead": comparisons,
    }


def _text_report(report: dict[str, Any]) -> str:
    paired = report["paired_overhead"]
    lines = [
        "scope.request_to_output=validate_inputs_start_to_final_write_finish",
        "scope.cold_in_process=compile_args_start_to_final_write_finish",
    ]
    for span in ("request_to_output", "cold_in_process"):
        comparison = paired[span]
        fraction = comparison["delta_fraction_of_control"]
        fraction_text = "null" if fraction is None else f"{fraction:.9f}"
        lines.append(
            f"{span}=control_s:{comparison['control_seconds']:.9f},"
            f"budget_s:{comparison['budget_seconds']:.9f},"
            f"delta_s:{comparison['delta_seconds']:.9f},"
            f"delta_fraction:{fraction_text}"
        )
    for group in report["group_order"]:
        comparison = paired["groups"][group]
        fraction = comparison["delta_fraction_of_control"]
        fraction_text = "null" if fraction is None else f"{fraction:.9f}"
        lines.append(
            f"group.{group}=control_s:{comparison['control_seconds']:.9f},"
            f"budget_s:{comparison['budget_seconds']:.9f},"
            f"delta_s:{comparison['delta_seconds']:.9f},"
            f"delta_fraction:{fraction_text}"
        )
    for span in ("request_to_output", "cold_in_process"):
        comparison = paired["residual_gaps"][span]
        lines.append(
            f"residual.{span}=control_s:{comparison['control_seconds']:.9f},"
            f"budget_s:{comparison['budget_seconds']:.9f},"
            f"delta_s:{comparison['delta_seconds']:.9f}"
        )
    for role in ("control", "budget"):
        for stage, seconds in report[role]["stage_totals_seconds"].items():
            lines.append(f"stage.{role}.{stage}_seconds={seconds:.9f}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-profile", type=Path, required=True)
    parser.add_argument("--budget-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    args = parser.parse_args()

    report = summarize(
        json.loads(args.control_profile.read_text(encoding="utf-8")),
        json.loads(args.budget_profile.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.text_output.parent.mkdir(parents=True, exist_ok=True)
    args.text_output.write_text(_text_report(report), encoding="utf-8")


if __name__ == "__main__":
    main()
