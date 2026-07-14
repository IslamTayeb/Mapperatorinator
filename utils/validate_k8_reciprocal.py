"""Validate reciprocal accepted/K=8 inference profiles without hiding changed work."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import nsight_agent_profile as nsight
from utils import summarize_inference_profile as exact


LABELS = ("timing_context", "main_generation")
RUN_NAMES = (
    "baseline_first",
    "candidate_second",
    "candidate_first",
    "baseline_second",
)
K8_POLICY = "counter_request_seed_window_prompt_v2"


class GateError(ValueError):
    pass


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GateError(f"profile must be a JSON object: {path}")
    if not isinstance(payload.get("metadata"), dict):
        raise GateError(f"profile metadata must be a JSON object: {path}")
    if not isinstance(payload.get("generation"), list):
        raise GateError(f"profile generation must be a JSON array: {path}")
    return payload


def _records(profile: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    return [
        record
        for record in profile["generation"]
        if isinstance(record, dict) and record.get("profile_label") == label
    ]


def _profile_contract(profile: Mapping[str, Any], *, name: str) -> list[str]:
    metadata = profile["metadata"]
    failures = []
    expected = {
        "inference_engine": "optimized",
        "precision": "fp32",
        "profile_pass_kind": "untraced_control",
        "authoritative_performance": True,
        "profile_detail_ranges": False,
        "profile_cuda_capture": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            failures.append(f"{name}.metadata.{key}!={value!r}")
    for label in LABELS:
        if not _records(profile, label):
            failures.append(f"{name} has no {label} records")
    return failures


def _artifact(profile: Mapping[str, Any]) -> dict[str, Any]:
    metadata = profile["metadata"]
    path_value = metadata.get("result_file_path")
    if not isinstance(path_value, str) or not path_value:
        raise GateError("profile is missing metadata.result_file_path")
    path = Path(path_value)
    if not path.is_file():
        raise GateError(f"profile result file does not exist: {path}")
    return {
        "path": str(path),
        "sha256": metadata.get("result_file_sha256"),
        "size_bytes": metadata.get("result_file_size_bytes"),
        "structure": nsight.summarize_osu_structure(path),
    }


def _same_signature(left: Mapping[str, Any], right: Mapping[str, Any], label: str) -> dict[str, Any]:
    left_signature = nsight._profile_label_signature(left, label)
    right_signature = nsight._profile_label_signature(right, label)
    available = (
        left_signature.get("status") == "available"
        and right_signature.get("status") == "available"
        and bool(left_signature.get("self_consistent"))
        and bool(right_signature.get("self_consistent"))
    )
    token_equal = (
        available
        and left_signature["token_stream_sha256"]
        == right_signature["token_stream_sha256"]
    )
    stopping_equal = (
        available
        and left_signature["stopping_sha256"] == right_signature["stopping_sha256"]
    )
    return {
        "available_and_self_consistent": available,
        "token_equal": token_equal,
        "stopping_equal": stopping_equal,
        "left": left_signature,
        "right": right_signature,
    }


def _same_artifact(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_metadata = left["metadata"]
    right_metadata = right["metadata"]
    return (
        left_metadata.get("result_file_sha256") is not None
        and left_metadata.get("result_file_sha256")
        == right_metadata.get("result_file_sha256")
        and left_metadata.get("result_file_size_bytes")
        == right_metadata.get("result_file_size_bytes")
    )


def _structure_comparison(
    baseline: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if baseline is None or candidate is None:
        return {
            "available": False,
            "equal": False,
            "numeric_deltas": {},
            "baseline": baseline,
            "candidate": candidate,
        }
    numeric_keys = (
        "timing_points",
        "uninherited_timing_points",
        "inherited_timing_points",
        "hit_objects",
        "malformed_lines",
        "nonfinite_values",
    )
    deltas = {
        key: candidate[key] - baseline[key]
        for key in numeric_keys
        if isinstance(baseline.get(key), (int, float))
        and not isinstance(baseline.get(key), bool)
        and isinstance(candidate.get(key), (int, float))
        and not isinstance(candidate.get(key), bool)
    }
    return {
        "available": True,
        "equal": baseline == candidate,
        "numeric_deltas": deltas,
        "baseline": baseline,
        "candidate": candidate,
    }


def _k8_stats(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    graphs = record.get("optimized_cuda_graphs")
    if not isinstance(graphs, Mapping):
        return None
    candidate = graphs.get("k8_candidate")
    return candidate if isinstance(candidate, Mapping) else None


def _k8_fallback_reason(record: Mapping[str, Any]) -> str | None:
    graphs = record.get("optimized_cuda_graphs")
    values = (
        record.get("k8_fallback_reason"),
        graphs.get("k8_fallback_reason") if isinstance(graphs, Mapping) else None,
    )
    return next((value for value in values if isinstance(value, str) and value.strip()), None)


def _integer(stats: Mapping[str, Any], key: str) -> int:
    value = stats.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise GateError(f"K8 {key} must be an integer")
    return value


def _finite_nonnegative(stats: Mapping[str, Any], key: str) -> float:
    value = stats.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateError(f"K8 {key} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise GateError(f"K8 {key} must be finite and non-negative")
    return number


def _validate_k8_profile(
    profile: Mapping[str, Any],
    *,
    name: str,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    failures: list[str] = []

    def new_totals() -> dict[str, Any]:
        return {
            "records": 0,
            "fallback_records": 0,
            "fallback_reasons": [],
            "generated_tokens": 0,
            "prefill_steps": 0,
            "eligible_steps": 0,
            "block_replays": 0,
            "remainder_steps": 0,
            "physical_steps": 0,
            "logical_steps": 0,
            "wasted_steps": 0,
            "status_reads": 0,
            "copy_calls": 0,
            "copy_bytes": 0,
            "prompt_seed_d2h_copy_calls": 0,
            "prompt_seed_d2h_copy_bytes": 0,
            "processor_signature_d2h_copy_calls": 0,
            "processor_signature_d2h_copy_bytes": 0,
            "prompt_seed_setup_seconds": 0.0,
            "processor_signature_setup_seconds": 0.0,
            "capture_seconds": 0.0,
        }

    totals_by_label = {label: new_totals() for label in LABELS}
    for record_index, record in enumerate(profile["generation"]):
        if not isinstance(record, Mapping):
            failures.append(f"{name}.generation[{record_index}] is not an object")
            continue
        label = record.get("profile_label")
        if label not in LABELS:
            continue
        totals = totals_by_label[str(label)]
        stats = _k8_stats(record)
        if stats is None:
            fallback_reason = _k8_fallback_reason(record)
            if label == "timing_context" and fallback_reason is None:
                failures.append(
                    f"{name}.timing_context[{record_index}] silently omitted K8; "
                    "require supported metadata or k8_fallback_reason"
                )
            elif fallback_reason is not None:
                totals["fallback_records"] += 1
                totals["fallback_reasons"].append(fallback_reason)
            continue
        totals["records"] += 1
        totals["generated_tokens"] += int(record.get("generated_tokens", 0) or 0)
        try:
            block_size = _integer(stats, "block_size")
            integer_keys = (
                "prefill_steps",
                "eligible_steps",
                "block_replays",
                "remainder_steps",
                "physical_steps",
                "logical_steps",
                "wasted_steps",
                "status_reads",
                "copy_calls",
                "copy_bytes",
                "rng_request_seed",
                "rng_window_identity",
                "prompt_seed_d2h_copy_calls",
                "prompt_seed_d2h_copy_bytes",
                "processor_signature_d2h_copy_calls",
                "processor_signature_d2h_copy_bytes",
            )
            values = {key: _integer(stats, key) for key in integer_keys}
            setup_values = {
                key: _finite_nonnegative(stats, key)
                for key in (
                    "prompt_seed_setup_seconds",
                    "processor_signature_setup_seconds",
                    "capture_seconds",
                )
            }
            if block_size != 8:
                failures.append(f"{name}.{label}[{record_index}] block_size={block_size}")
            if any(values[key] < 0 for key in integer_keys if key != "rng_request_seed"):
                failures.append(f"{name}.{label}[{record_index}] has negative K8 counters")
            if values["rng_window_identity"] <= 0:
                failures.append(f"{name}.{label}[{record_index}] has invalid window identity")
            if stats.get("rng_policy") != K8_POLICY:
                failures.append(f"{name}.{label}[{record_index}] has wrong RNG policy")
            if stats.get("rng_exact") is not False:
                failures.append(f"{name}.{label}[{record_index}] must declare RNG drift")
            if stats.get("rng_early_eos_isolation") is not True:
                failures.append(f"{name}.{label}[{record_index}] lacks EOS RNG isolation")
            if stats.get("sampling_mode") not in {"sample", "greedy"}:
                failures.append(f"{name}.{label}[{record_index}] has invalid sampling mode")
            if stats.get("parent_backend") != "cuda_python_child_graphs":
                failures.append(f"{name}.{label}[{record_index}] has wrong parent backend")
            if stats.get("capture_state_restore_synchronized") is not True:
                failures.append(
                    f"{name}.{label}[{record_index}] lacks synchronized capture restore"
                )
            if values["eligible_steps"] != block_size * values["block_replays"]:
                failures.append(f"{name}.{label}[{record_index}] block accounting diverged")
            if (
                values["prefill_steps"]
                + values["eligible_steps"]
                + values["remainder_steps"]
                != values["physical_steps"]
            ):
                failures.append(f"{name}.{label}[{record_index}] physical work diverged")
            if values["physical_steps"] - values["logical_steps"] != values["wasted_steps"]:
                failures.append(f"{name}.{label}[{record_index}] wasted work diverged")
            for key in totals:
                if key in {"records", "fallback_records", "fallback_reasons", "generated_tokens"}:
                    continue
                if key in values:
                    totals[key] += values[key]
                elif key in setup_values:
                    totals[key] += setup_values[key]
        except GateError as exc:
            failures.append(f"{name}.{label}[{record_index}]: {exc}")
    main_totals = totals_by_label["main_generation"]
    if main_totals["records"] == 0:
        failures.append(f"{name} has no main K8 metadata")
    if main_totals["block_replays"] <= 0:
        failures.append(f"{name} has no main K8 block replay")
    for label, totals in totals_by_label.items():
        if totals["records"] and totals["logical_steps"] != totals["generated_tokens"]:
            failures.append(
                f"{name}.{label} logical_steps={totals['logical_steps']} "
                f"does not equal generated_tokens={totals['generated_tokens']}"
            )
    return failures, totals_by_label


def _validate_baseline(profile: Mapping[str, Any], *, name: str) -> list[str]:
    failures = []
    for index, record in enumerate(profile["generation"]):
        if isinstance(record, Mapping) and _k8_stats(record) is not None:
            failures.append(f"{name}.generation[{index}] unexpectedly contains K8 metadata")
    return failures


def _performance(profile: Mapping[str, Any], label: str) -> dict[str, Any]:
    records = _records(profile, label)
    tokens = sum(int(record.get("generated_tokens", 0) or 0) for record in records)
    model_seconds = sum(float(record.get("model_elapsed_seconds", 0.0) or 0.0) for record in records)
    outer_seconds = sum(float(record.get("wall_seconds", 0.0) or 0.0) for record in records)
    stage_seconds = exact._stage_wall_seconds(dict(profile))
    return {
        "records": len(records),
        "generated_tokens": tokens,
        "model_elapsed_seconds": model_seconds,
        "outer_generation_wall_seconds": outer_seconds,
        "outer_minus_model_seconds": outer_seconds - model_seconds,
        "complete_request_stage_wall_seconds": stage_seconds,
        "generated_tokens_per_model_second": tokens / model_seconds if model_seconds > 0 else 0.0,
    }


def _pair_performance(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    candidate_k8: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    baseline_perf = _performance(baseline, label)
    candidate_perf = _performance(candidate, label)
    model_seconds = candidate_perf["model_elapsed_seconds"]
    logical = int(candidate_k8["logical_steps"])
    physical = int(candidate_k8["physical_steps"])
    wasted = int(candidate_k8["wasted_steps"])
    signature = _same_signature(baseline, candidate, label)
    fixed_work = bool(signature["token_equal"] and signature["stopping_equal"])
    saving = baseline_perf["model_elapsed_seconds"] - candidate_perf["model_elapsed_seconds"]
    return {
        "work_comparability": "fixed_work" if fixed_work else "changed_work",
        "fixed_work": fixed_work,
        "token_and_stopping": signature,
        "baseline": baseline_perf,
        "candidate": {
            **candidate_perf,
            "logical_steps": logical,
            "physical_steps": physical,
            "wasted_steps": wasted,
            "logical_steps_per_model_second": logical / model_seconds if model_seconds > 0 else 0.0,
            "physical_steps_per_model_second": physical / model_seconds if model_seconds > 0 else 0.0,
            "wasted_step_fraction": wasted / physical if physical > 0 else 0.0,
        },
        "observed_model_seconds_difference": saving,
        "fixed_work_model_seconds_saving": saving if fixed_work else None,
    }


def _repeatability(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    first_artifact: Mapping[str, Any],
    second_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    labels = {label: _same_signature(first, second, label) for label in LABELS}
    artifact_equal = _same_artifact(first, second)
    structure_equal = first_artifact["structure"] == second_artifact["structure"]
    passed = (
        artifact_equal
        and structure_equal
        and all(
            value["available_and_self_consistent"]
            and value["token_equal"]
            and value["stopping_equal"]
            for value in labels.values()
        )
    )
    return {
        "pass": passed,
        "labels": labels,
        "artifact_equal": artifact_equal,
        "structure_equal": structure_equal,
    }


def summarize(
    paths: Mapping[str, Path],
    *,
    scope: str,
    minimum_smoke_speedup_pct: float,
    full_fixed_work_saving_seconds: float,
) -> dict[str, Any]:
    if set(paths) != set(RUN_NAMES):
        raise GateError(f"profile paths must be exactly {RUN_NAMES}")
    profiles = {name: _load(paths[name]) for name in RUN_NAMES}
    failures: list[str] = []
    for name, profile in profiles.items():
        failures.extend(_profile_contract(profile, name=name))
    contract_pairs = (
        ("baseline_first", "candidate_second"),
        ("baseline_second", "candidate_first"),
        ("baseline_first", "baseline_second"),
        ("candidate_first", "candidate_second"),
    )
    contracts = {}
    for left, right in contract_pairs:
        contract = exact._contract(profiles[left], profiles[right])
        contracts[f"{left}_vs_{right}"] = contract
        if not contract["pass"]:
            failures.append(f"workload contract differs: {left} vs {right}")

    artifacts: dict[str, dict[str, Any]] = {}
    for name, profile in profiles.items():
        try:
            artifacts[name] = _artifact(profile)
        except GateError as exc:
            failures.append(f"{name}: {exc}")
            artifacts[name] = {"path": None, "sha256": None, "size_bytes": None, "structure": None}
    for name in ("baseline_first", "baseline_second"):
        failures.extend(_validate_baseline(profiles[name], name=name))
    candidate_stats = {}
    for name in ("candidate_first", "candidate_second"):
        candidate_failures, totals = _validate_k8_profile(profiles[name], name=name)
        failures.extend(candidate_failures)
        candidate_stats[name] = totals
        structure = artifacts[name]["structure"]
        if not isinstance(structure, Mapping) or structure.get("finite_and_well_formed") is not True:
            failures.append(f"{name} output is not finite and well formed")

    baseline_repeatability = _repeatability(
        profiles["baseline_first"],
        profiles["baseline_second"],
        artifacts["baseline_first"],
        artifacts["baseline_second"],
    )
    candidate_repeatability = _repeatability(
        profiles["candidate_first"],
        profiles["candidate_second"],
        artifacts["candidate_first"],
        artifacts["candidate_second"],
    )
    if not baseline_repeatability["pass"]:
        failures.append("baseline reciprocal repeats are not exact")
    if not candidate_repeatability["pass"]:
        failures.append("candidate reciprocal repeats are not deterministic")

    order_specs = {
        "baseline_first": ("baseline_first", "candidate_second"),
        "candidate_first": ("baseline_second", "candidate_first"),
    }
    orders = {}
    for order, (baseline_name, candidate_name) in order_specs.items():
        orders[order] = {
            label: _pair_performance(
                profiles[baseline_name],
                profiles[candidate_name],
                candidate_stats[candidate_name][label],
                label=label,
            )
            for label in LABELS
        }
        orders[order]["output"] = {
            "artifact_equal": _same_artifact(
                profiles[baseline_name],
                profiles[candidate_name],
            ),
            "structure": _structure_comparison(
                artifacts[baseline_name]["structure"],
                artifacts[candidate_name]["structure"],
            ),
        }
    main_speedups = []
    for order in orders.values():
        main = order["main_generation"]
        baseline_tps = main["baseline"]["generated_tokens_per_model_second"]
        candidate_tps = main["candidate"]["logical_steps_per_model_second"]
        speedup = (
            (candidate_tps - baseline_tps) / baseline_tps * 100.0
            if baseline_tps > 0
            else -math.inf
        )
        main["logical_tps_speedup_pct"] = speedup
        main_speedups.append(speedup)
    reciprocal_speedup = mean(main_speedups)
    if scope == "smoke" and reciprocal_speedup < minimum_smoke_speedup_pct:
        failures.append(
            "reciprocal main logical TPS speedup "
            f"{reciprocal_speedup:.3f}% is below {minimum_smoke_speedup_pct:.3f}%"
        )

    fixed_work_savings = [
        order["main_generation"]["fixed_work_model_seconds_saving"]
        for order in orders.values()
    ]
    full_fixed_work_available = all(value is not None for value in fixed_work_savings)
    full_fixed_work_saving = (
        mean(float(value) for value in fixed_work_savings)
        if full_fixed_work_available
        else None
    )
    promotion_pass = bool(
        scope == "full"
        and full_fixed_work_saving is not None
        and full_fixed_work_saving >= full_fixed_work_saving_seconds
        and not failures
    )
    graph_signatures = {
        name: nsight._profile_graph_cache_signature(profile, LABELS)
        for name, profile in profiles.items()
    }
    baseline_graph_repeat = (
        graph_signatures["baseline_first"].get("status") == "available"
        and graph_signatures["baseline_second"].get("status") == "available"
        and graph_signatures["baseline_first"].get("sha256")
        == graph_signatures["baseline_second"].get("sha256")
    )
    candidate_graph_repeat = (
        graph_signatures["candidate_first"].get("status") == "available"
        and graph_signatures["candidate_second"].get("status") == "available"
        and graph_signatures["candidate_first"].get("sha256")
        == graph_signatures["candidate_second"].get("sha256")
    )
    if not baseline_graph_repeat:
        failures.append("baseline reciprocal graph-cache signatures differ")
    if not candidate_graph_repeat:
        failures.append("candidate reciprocal graph-cache signatures differ")
    if failures:
        promotion_pass = False
    for order, (baseline_name, candidate_name) in order_specs.items():
        baseline_signature = graph_signatures[baseline_name]
        candidate_signature = graph_signatures[candidate_name]
        orders[order]["graph_cache"] = {
            "baseline": baseline_signature,
            "candidate": candidate_signature,
            "equal": (
                baseline_signature.get("status") == "available"
                and candidate_signature.get("status") == "available"
                and baseline_signature.get("sha256") == candidate_signature.get("sha256")
            ),
        }
    return {
        "schema_version": 1,
        "scope": scope,
        "profile_paths": {name: str(path) for name, path in paths.items()},
        "contract_checks": contracts,
        "artifacts": artifacts,
        "graph_cache_signatures": graph_signatures,
        "graph_cache_repeatability": {
            "baseline_pass": baseline_graph_repeat,
            "candidate_pass": candidate_graph_repeat,
        },
        "baseline_repeatability": baseline_repeatability,
        "candidate_repeatability": candidate_repeatability,
        "candidate_k8_totals": candidate_stats,
        "orders": orders,
        "reciprocal_main_logical_tps_speedup_pct": reciprocal_speedup,
        "fixed_work": {
            "available": full_fixed_work_available,
            "mean_main_model_seconds_saving": full_fixed_work_saving,
            "required_saving_seconds": full_fixed_work_saving_seconds,
            "promotion_pass": promotion_pass,
            "changed_tokens_never_count_as_fixed_work": True,
        },
        "feasibility_pass": not failures,
        "failures": failures,
    }


def _write_text(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        f"scope={report['scope']}",
        f"feasibility_pass={report['feasibility_pass']}",
        f"reciprocal_main_logical_tps_speedup_pct={report['reciprocal_main_logical_tps_speedup_pct']:.6f}",
        f"fixed_work_available={report['fixed_work']['available']}",
        f"fixed_work_main_saving_seconds={report['fixed_work']['mean_main_model_seconds_saving']}",
        f"promotion_pass={report['fixed_work']['promotion_pass']}",
    ]
    for order_name, order in report["orders"].items():
        candidate_name = (
            "candidate_second" if order_name == "baseline_first" else "candidate_first"
        )
        for label in LABELS:
            comparison = order[label]
            candidate = comparison["candidate"]
            k8 = report["candidate_k8_totals"][candidate_name][label]
            prefix = f"{order_name}.{label}"
            lines.extend(
                (
                    f"{prefix}.work_comparability={comparison['work_comparability']}",
                    f"{prefix}.baseline_tps={comparison['baseline']['generated_tokens_per_model_second']:.6f}",
                    f"{prefix}.candidate_logical_tps={candidate['logical_steps_per_model_second']:.6f}",
                    f"{prefix}.candidate_physical_tps={candidate['physical_steps_per_model_second']:.6f}",
                    f"{prefix}.candidate_wasted_steps={candidate['wasted_steps']}",
                    f"{prefix}.candidate_outer_minus_model_seconds={candidate['outer_minus_model_seconds']:.6f}",
                    f"{prefix}.complete_request_wall_seconds={candidate['complete_request_stage_wall_seconds']:.6f}",
                    f"{prefix}.status_reads={k8['status_reads']}",
                    f"{prefix}.graph_input_copy_calls={k8['copy_calls']}",
                    f"{prefix}.graph_input_copy_bytes={k8['copy_bytes']}",
                    f"{prefix}.prompt_seed_d2h_copy_calls={k8['prompt_seed_d2h_copy_calls']}",
                    f"{prefix}.processor_signature_d2h_copy_calls={k8['processor_signature_d2h_copy_calls']}",
                    f"{prefix}.explicit_setup_seconds={k8['prompt_seed_setup_seconds'] + k8['processor_signature_setup_seconds']:.6f}",
                    f"{prefix}.capture_seconds={k8['capture_seconds']:.6f}",
                )
            )
        lines.append(f"{order_name}.output_artifact_equal={order['output']['artifact_equal']}")
        lines.append(f"{order_name}.output_structure_equal={order['output']['structure']['equal']}")
        for key, value in order["output"]["structure"]["numeric_deltas"].items():
            lines.append(f"{order_name}.output_structure_delta.{key}={value}")
        lines.append(f"{order_name}.graph_cache_signature_equal={order['graph_cache']['equal']}")
    lines.extend(f"failure={failure}" for failure in report["failures"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("smoke", "full"), required=True)
    parser.add_argument("--baseline-first", type=Path, required=True)
    parser.add_argument("--candidate-second", type=Path, required=True)
    parser.add_argument("--candidate-first", type=Path, required=True)
    parser.add_argument("--baseline-second", type=Path, required=True)
    parser.add_argument("--minimum-smoke-speedup-pct", type=float, default=0.0)
    parser.add_argument("--full-fixed-work-saving-seconds", type=float, default=5.0)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    args = parser.parse_args()
    if args.full_fixed_work_saving_seconds < 0:
        parser.error("--full-fixed-work-saving-seconds must be non-negative")
    paths = {name: getattr(args, name) for name in RUN_NAMES}
    report = summarize(
        paths,
        scope=args.scope,
        minimum_smoke_speedup_pct=args.minimum_smoke_speedup_pct,
        full_fixed_work_saving_seconds=args.full_fixed_work_saving_seconds,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_text(args.text_output, report)
    raise SystemExit(0 if report["feasibility_pass"] else 1)


if __name__ == "__main__":
    main()
