"""Gate the selected-stack shared static-input arena reciprocal run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any


FIXED_MAIN_STEPS = 8_294
MIN_FIXED_MAIN_SAVING_SECONDS = 0.300
LABELS = ("timing_context", "main_generation")
ROLES = (
    "baseline_first",
    "candidate_first",
    "candidate_second",
    "baseline_second",
)


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _runtime_records(profile: dict[str, Any], *, role: str) -> dict[str, list[dict[str, Any]]]:
    generation = profile.get("generation")
    if not isinstance(generation, list) or not generation:
        raise ValueError(f"{role}.generation must be a non-empty list")
    records = {label: [] for label in LABELS}
    for index, raw in enumerate(generation):
        record = _object(raw, name=f"{role}.generation[{index}]")
        label = record.get("profile_label")
        if label not in records:
            continue
        graphs = _object(
            record.get("optimized_cuda_graphs"),
            name=f"{role}.generation[{index}].optimized_cuda_graphs",
        )
        runtime = _object(
            graphs.get("k8_candidate"),
            name=f"{role}.generation[{index}].k8_candidate",
        )
        records[str(label)].append(runtime)
    if any(not values for values in records.values()):
        raise ValueError(f"{role} is missing timing or main K4 records")
    return records


def _cache_topology(profile: dict[str, Any], *, role: str) -> dict[str, list[dict[str, Any]]]:
    generation = profile.get("generation")
    if not isinstance(generation, list):
        raise ValueError(f"{role}.generation must be a list")
    result = {label: [] for label in LABELS}
    for index, raw in enumerate(generation):
        record = _object(raw, name=f"{role}.generation[{index}]")
        label = record.get("profile_label")
        if label not in result:
            continue
        graphs = _object(record.get("optimized_cuda_graphs"), name=f"{role}.graphs")
        runtime = _object(graphs.get("k8_candidate"), name=f"{role}.runtime")
        buckets = _object(graphs.get("buckets"), name=f"{role}.buckets")
        result[str(label)].append({
            "graph_count": graphs.get("graph_count"),
            "decode_replays": graphs.get("decode_replays"),
            "buckets": {
                prefix: {
                    "graph_count": _object(value, name=f"{role}.bucket.{prefix}").get(
                        "graph_count"
                    ),
                    "decode_replays": _object(
                        value, name=f"{role}.bucket.{prefix}"
                    ).get("decode_replays"),
                }
                for prefix, value in sorted(buckets.items(), key=lambda item: int(item[0]))
            },
            "runtime": {
                key: runtime.get(key)
                for key in (
                    "block_size",
                    "prefill_steps",
                    "eligible_steps",
                    "block_replays",
                    "remainder_steps",
                    "remainder_backend",
                    "remainder_graph_replays",
                    "eager_remainder_steps",
                    "physical_steps",
                    "logical_steps",
                    "wasted_steps",
                    "rng_policy",
                    "rng_window_identity",
                )
            },
        })
    return result


def _transition_summary(
    records: dict[str, list[dict[str, Any]]],
    *,
    role: str,
    candidate: bool,
) -> dict[str, Any]:
    labels: dict[str, Any] = {}
    for label, values in records.items():
        stage_totals: dict[str, dict[str, float | int]] = {}
        device_totals: dict[str, dict[str, float | int]] = {}
        model_wall = 0.0
        accounted = 0.0
        unattributed = 0.0
        measured_records = 0
        refreshes = 0
        refresh_copy_calls = 0
        refresh_copy_bytes = 0
        ordinary_copy_calls = 0
        for index, runtime in enumerate(values):
            if runtime.get("shared_static_input_arena") is not candidate:
                raise ValueError(
                    f"{role}.{label}[{index}] has wrong shared-arena declaration"
                )
            refresh = runtime.get("static_input_arena_refreshes", 0)
            refresh_calls = runtime.get("static_input_arena_refresh_copy_calls", 0)
            refresh_bytes = runtime.get("static_input_arena_refresh_copy_bytes", 0)
            if any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in (refresh, refresh_calls, refresh_bytes)
            ):
                raise ValueError(f"{role}.{label}[{index}] has invalid arena counters")
            if candidate and (refresh != 1 or refresh_calls != 4 or refresh_bytes <= 0):
                raise ValueError(
                    f"{role}.{label}[{index}] must refresh exactly four tensors once"
                )
            if not candidate and any((refresh, refresh_calls, refresh_bytes)):
                raise ValueError(f"{role}.{label}[{index}] baseline used an arena")
            refreshes += refresh
            refresh_copy_calls += refresh_calls
            refresh_copy_bytes += refresh_bytes
            copy_calls = runtime.get("copy_calls")
            if isinstance(copy_calls, bool) or not isinstance(copy_calls, int):
                raise ValueError(f"{role}.{label}[{index}] copy_calls is invalid")
            ordinary_copy_calls += copy_calls

            timing = _object(
                runtime.get("transition_timing"),
                name=f"{role}.{label}[{index}].transition_timing",
            )
            if timing.get("schema_version") != 1 or timing.get("enabled") is not True:
                raise ValueError(f"{role}.{label}[{index}] timing metadata is invalid")
            current_wall = _number(
                timing.get("model_wall_seconds"),
                name=f"{role}.{label}[{index}].model_wall_seconds",
            )
            current_accounted = _number(
                timing.get("accounted_wall_seconds"),
                name=f"{role}.{label}[{index}].accounted_wall_seconds",
            )
            current_unattributed = timing.get("unattributed_wall_seconds")
            if isinstance(current_unattributed, bool) or not isinstance(
                current_unattributed, (int, float)
            ) or not math.isfinite(float(current_unattributed)):
                raise ValueError(f"{role}.{label}[{index}] unattributed wall is invalid")
            error = _number(
                timing.get("reconciliation_error_fraction"),
                name=f"{role}.{label}[{index}].reconciliation_error_fraction",
            )
            if current_wall > 0.0:
                measured_records += 1
                computed_error = abs(float(current_unattributed)) / current_wall
                if abs(
                    current_wall
                    - current_accounted
                    - float(current_unattributed)
                ) > max(1e-9, current_wall * 1e-6):
                    raise ValueError(
                        f"{role}.{label}[{index}] transition decomposition does not sum"
                    )
                if abs(error - computed_error) > 1e-9:
                    raise ValueError(
                        f"{role}.{label}[{index}] reconciliation error is inconsistent"
                    )
                if computed_error > 0.02:
                    raise ValueError(
                        f"{role}.{label}[{index}] transition wall misses 2% reconciliation"
                    )
            model_wall += current_wall
            accounted += current_accounted
            unattributed += float(current_unattributed)

            stages = _object(
                timing.get("stages"),
                name=f"{role}.{label}[{index}].stages",
            )
            expected_stages = {
                "prepare_inputs_for_generation",
                "graph_key_lookup",
                "static_input_refresh",
                "static_input_copies",
                "graph_replay",
                "status_d2h",
                "sync_model_kwargs",
            }
            if set(stages) != expected_stages:
                raise ValueError(f"{role}.{label}[{index}] stage schema changed")
            record_stage_sum = 0.0
            for name, raw_stage in stages.items():
                stage = _object(raw_stage, name=f"{role}.{label}.{name}")
                calls = stage.get("calls")
                if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
                    raise ValueError(f"{role}.{label}.{name}.calls is invalid")
                wall = _number(stage.get("wall_seconds"), name=f"{role}.{label}.{name}")
                record_stage_sum += wall
                total = stage_totals.setdefault(name, {"calls": 0, "wall_seconds": 0.0})
                total["calls"] = int(total["calls"]) + calls
                total["wall_seconds"] = float(total["wall_seconds"]) + wall
            if abs(record_stage_sum - current_accounted) > max(
                1e-9, current_wall * 1e-6
            ):
                raise ValueError(
                    f"{role}.{label}[{index}] accounted wall differs from stage sum"
                )

            device = _object(
                timing.get("device"),
                name=f"{role}.{label}[{index}].device",
            )
            if set(device) != {"static_input_copies", "graph_replay"}:
                raise ValueError(f"{role}.{label}[{index}] device schema changed")
            for name, raw_device in device.items():
                value = _object(raw_device, name=f"{role}.{label}.device.{name}")
                calls = value.get("calls")
                if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
                    raise ValueError(f"{role}.{label}.device.{name}.calls is invalid")
                seconds = _number(
                    value.get("seconds"), name=f"{role}.{label}.device.{name}.seconds"
                )
                total = device_totals.setdefault(name, {"calls": 0, "seconds": 0.0})
                total["calls"] = int(total["calls"]) + calls
                total["seconds"] = float(total["seconds"]) + seconds
        if measured_records <= 0 or model_wall <= 0.0:
            raise ValueError(f"{role}.{label} contains no steady-state transition timing")
        labels[label] = {
            "records": len(values),
            "measured_records": measured_records,
            "model_wall_seconds": model_wall,
            "accounted_wall_seconds": accounted,
            "unattributed_wall_seconds": unattributed,
            "reconciliation_error_fraction": abs(unattributed) / model_wall,
            "static_input_arena_refreshes": refreshes,
            "static_input_arena_refresh_copy_calls": refresh_copy_calls,
            "static_input_arena_refresh_copy_bytes": refresh_copy_bytes,
            "copy_calls": ordinary_copy_calls,
            "stages": stage_totals,
            "device": device_totals,
        }
    return labels


def analyze(
    reciprocal: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if reciprocal.get("schema_version") != "mapperatorinator.reciprocal-full-song-candidate.v1":
        raise ValueError("reciprocal analysis schema is wrong")
    parity = _object(reciprocal.get("parity"), name="reciprocal.parity")
    if parity.get("cross_candidate_exact") is not True:
        raise ValueError("shared arena changed tokens, stopping, or final output")
    if parity.get("required_exact_labels") != list(LABELS):
        raise ValueError("shared arena did not require exact timing and main labels")
    if parity.get("required_exact_labels_pass") is not True:
        raise ValueError("shared arena exact label gate failed")
    output = _object(parity.get("output_divergence"), name="output_divergence")
    if output.get("final_map_equal") is not True:
        raise ValueError("shared arena final osu is not byte-identical")

    metric = _object(
        _object(reciprocal.get("metrics"), name="metrics").get(
            "fixed_8294_main_seconds"
        ),
        name="metrics.fixed_8294_main_seconds",
    )
    saving = _finite(metric.get("improvement"), name="fixed main improvement")
    baseline = _number(metric.get("baseline_median"), name="fixed main baseline")
    candidate = _number(metric.get("candidate_median"), name="fixed main candidate")

    transitions = {
        role: _transition_summary(
            _runtime_records(profiles[role], role=role),
            role=role,
            candidate=role.startswith("candidate"),
        )
        for role in ROLES
    }
    cache_topologies = {
        role: _cache_topology(profiles[role], role=role) for role in ROLES
    }
    if cache_topologies["baseline_first"] != cache_topologies["baseline_second"]:
        raise ValueError("baseline cache topology is not reciprocal-repeat stable")
    if cache_topologies["candidate_first"] != cache_topologies["candidate_second"]:
        raise ValueError("candidate cache topology is not reciprocal-repeat stable")
    if cache_topologies["baseline_first"] != cache_topologies["candidate_first"]:
        raise ValueError("shared arena changed graph/cache topology")
    baseline_copies = statistics.median(
        transitions[role]["main_generation"]["copy_calls"]
        for role in ("baseline_first", "baseline_second")
    )
    candidate_copies = statistics.median(
        transitions[role]["main_generation"]["copy_calls"]
        for role in ("candidate_first", "candidate_second")
    )
    if candidate_copies >= baseline_copies:
        raise ValueError("shared arena did not reduce main static-input copies")

    pass_gate = saving >= MIN_FIXED_MAIN_SAVING_SECONDS
    return {
        "schema_version": 1,
        "candidate": "shared-per-request-static-input-arena",
        "fixed_main_steps": FIXED_MAIN_STEPS,
        "fixed_main": {
            "baseline_seconds": baseline,
            "candidate_seconds": candidate,
            "saving_seconds": saving,
            "minimum_saving_seconds": MIN_FIXED_MAIN_SAVING_SECONDS,
        },
        "exact_incremental_parity": True,
        "cache_topology_exact": True,
        "final_osu_byte_identical": True,
        "main_copy_calls": {
            "baseline_median": baseline_copies,
            "candidate_median": candidate_copies,
        },
        "transitions": transitions,
        "pass": pass_gate,
        "decision": "PROMOTE" if pass_gate else "STOP_SHARED_STATIC_INPUT_ARENA",
    }


def _text(report: dict[str, Any]) -> str:
    fixed = report["fixed_main"]
    copies = report["main_copy_calls"]
    return "\n".join((
        f"decision={report['decision']}",
        f"fixed_main_steps={report['fixed_main_steps']}",
        f"fixed_main_baseline_seconds={fixed['baseline_seconds']:.9f}",
        f"fixed_main_candidate_seconds={fixed['candidate_seconds']:.9f}",
        f"fixed_main_saving_seconds={fixed['saving_seconds']:.9f}",
        f"minimum_saving_seconds={fixed['minimum_saving_seconds']:.9f}",
        f"main_copy_calls_baseline_median={copies['baseline_median']:.3f}",
        f"main_copy_calls_candidate_median={copies['candidate_median']:.3f}",
        "exact_incremental_parity=true",
        "final_osu_byte_identical=true",
    )) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reciprocal-analysis", type=Path, required=True)
    for role in ROLES:
        parser.add_argument(f"--{role.replace('_', '-')}-profile", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    args = parser.parse_args()
    reciprocal = json.loads(args.reciprocal_analysis.read_text(encoding="utf-8"))
    profiles = {
        role: json.loads(
            getattr(args, f"{role}_profile").read_text(encoding="utf-8")
        )
        for role in ROLES
    }
    report = analyze(reciprocal, profiles)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.text_output.write_text(_text(report), encoding="utf-8")
    if not report["pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
