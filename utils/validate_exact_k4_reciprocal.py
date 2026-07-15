"""Validate strict-FP32 accepted-K1 versus exact-K4 reciprocal profiles."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping

from utils import analyze_fp32_fresh_baseline as strict
from utils import nsight_agent_profile as nsight
from utils import summarize_inference_profile as profiles


LABELS = ("timing_context", "main_generation")
RUN_NAMES = (
    "baseline_first",
    "candidate_second",
    "candidate_first",
    "baseline_second",
)
RNG_POLICY = "default_cuda_generator_torch_multinomial_exact_offset_rollback"
PARENT_BACKEND = "ordinary_pytorch_cuda_graph_four_distinct_steps"
BLOCK_SIZE = 4
OFFSET_INCREMENT = 4


class GateError(RuntimeError):
    """Evidence does not satisfy the strict exact-K4 contract."""


def _load(path: Path) -> dict[str, Any]:
    payload = nsight._load_inference_profile(path)
    if not isinstance(payload.get("generation"), list):
        raise GateError(f"profile has no generation array: {path}")
    return payload


def _records(profile: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    return [
        record
        for record in profile["generation"]
        if isinstance(record, dict) and record.get("profile_label") == label
    ]


def _performance(profile: Mapping[str, Any], label: str) -> dict[str, Any]:
    records = _records(profile, label)
    tokens = sum(int(record.get("generated_tokens", 0) or 0) for record in records)
    model_seconds = sum(
        float(record.get("model_elapsed_seconds", 0.0) or 0.0)
        for record in records
    )
    outer_seconds = sum(
        float(record.get("wall_seconds", 0.0) or 0.0) for record in records
    )
    return {
        "records": len(records),
        "tokens": tokens,
        "model_seconds": model_seconds,
        "outer_seconds": outer_seconds,
        "complete_request_stage_wall_seconds": profiles._stage_wall_seconds(
            dict(profile)
        ),
        "tps": tokens / model_seconds if model_seconds > 0 else 0.0,
    }


def _label_exact(left: Mapping[str, Any], right: Mapping[str, Any], label: str) -> dict[str, Any]:
    left_signature = nsight._profile_label_signature(left, label)
    right_signature = nsight._profile_label_signature(right, label)
    passed = bool(
        left_signature.get("status") == "available"
        and right_signature.get("status") == "available"
        and left_signature.get("self_consistent")
        and right_signature.get("self_consistent")
        and left_signature.get("token_stream_sha256")
        == right_signature.get("token_stream_sha256")
        and left_signature.get("stopping_sha256")
        == right_signature.get("stopping_sha256")
        and left_signature.get("generated_tokens")
        == right_signature.get("generated_tokens")
    )
    return {
        "pass": passed,
        "left": left_signature,
        "right": right_signature,
    }


def _artifact_exact(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_metadata = left["metadata"]
    right_metadata = right["metadata"]
    passed = bool(
        left_metadata.get("result_file_sha256")
        and left_metadata.get("result_file_sha256")
        == right_metadata.get("result_file_sha256")
        and left_metadata.get("result_file_size_bytes")
        == right_metadata.get("result_file_size_bytes")
    )
    return {
        "pass": passed,
        "left_sha256": left_metadata.get("result_file_sha256"),
        "right_sha256": right_metadata.get("result_file_sha256"),
        "left_size_bytes": left_metadata.get("result_file_size_bytes"),
        "right_size_bytes": right_metadata.get("result_file_size_bytes"),
    }


def _dispatch_policy(profile: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}
    for index, record in enumerate(profile["generation"]):
        if not isinstance(record, Mapping) or record.get("profile_label") not in LABELS:
            continue
        hits = record.get("optimized_dispatch_capture_hits")
        if not isinstance(hits, Mapping):
            raise GateError(f"generation[{index}] lacks optimized dispatch metadata")
        by_label[str(record["profile_label"])].append(
            {
                "record_key": nsight._profile_record_key(record, index),
                "decoder_loop_backend": record.get("decoder_loop_backend"),
                "torch_compile_enabled": record.get("torch_compile_enabled"),
                "generation_compile_enabled": record.get("generation_compile_enabled"),
                "optimized_effective_config_version": record.get(
                    "optimized_effective_config_version"
                ),
                "native_cross_mlp_tail_requested": record.get(
                    "native_cross_mlp_tail_requested"
                ),
                "native_cross_mlp_tail_enabled": record.get(
                    "native_cross_mlp_tail_enabled"
                ),
                "native_cross_mlp_tail_disabled_reason": record.get(
                    "native_cross_mlp_tail_disabled_reason"
                ),
                # Capture multiplicity differs between K1 and K4. Eligibility and
                # actual use must not: compare positive-hit topology, not raw calls.
                "positive_dispatches": {
                    str(key): int(value) > 0 for key, value in sorted(hits.items())
                },
            }
        )
    if any(not records for records in by_label.values()):
        raise GateError("dispatch policy is missing timing or main records")
    return by_label


def _candidate_stats(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    graphs = record.get("optimized_cuda_graphs")
    if not isinstance(graphs, Mapping):
        return None
    value = graphs.get("exact_k4_candidate")
    return value if isinstance(value, Mapping) else None


def _integer(stats: Mapping[str, Any], key: str) -> int:
    value = stats.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise GateError(f"exact-K4 {key} must be an integer")
    return value


def _candidate_contract(profile: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    totals = {
        label: {
            "records": 0,
            "generated_tokens": 0,
            "prefill_steps": 0,
            "eligible_steps": 0,
            "block_replays": 0,
            "remainder_steps": 0,
            "physical_steps": 0,
            "logical_steps": 0,
            "wasted_steps": 0,
            "status_reads": 0,
            "terminal_rollbacks": 0,
            "generator_offset_corrections": 0,
        }
        for label in LABELS
    }
    failures: list[str] = []
    for index, record in enumerate(profile["generation"]):
        if not isinstance(record, Mapping) or record.get("profile_label") not in LABELS:
            continue
        label = str(record["profile_label"])
        stats = _candidate_stats(record)
        if stats is None:
            failures.append(f"{name}.generation[{index}] lacks exact-K4 metadata")
            continue
        totals[label]["records"] += 1
        totals[label]["generated_tokens"] += int(record.get("generated_tokens", 0) or 0)
        try:
            values = {
                key: _integer(stats, key)
                for key in (
                    "block_size",
                    "prefill_steps",
                    "eligible_steps",
                    "block_replays",
                    "remainder_steps",
                    "physical_steps",
                    "logical_steps",
                    "wasted_steps",
                    "status_reads",
                    "terminal_rollbacks",
                    "generator_offset_corrections",
                    "generator_offset_increment",
                )
            }
        except GateError as exc:
            failures.append(f"{name}.generation[{index}]: {exc}")
            continue
        if values["block_size"] != BLOCK_SIZE:
            failures.append(f"{name}.generation[{index}] block size is not four")
        if values["generator_offset_increment"] != OFFSET_INCREMENT:
            failures.append(f"{name}.generation[{index}] RNG offset increment is not four")
        if stats.get("rng_policy") != RNG_POLICY:
            failures.append(f"{name}.generation[{index}] RNG policy changed")
        if stats.get("rng_exact") is not True or stats.get("rng_drift") is not None:
            failures.append(f"{name}.generation[{index}] does not claim strict RNG exactness")
        if stats.get("parent_backend") != PARENT_BACKEND:
            failures.append(f"{name}.generation[{index}] parent graph backend changed")
        if stats.get("capture_state_restore_synchronized") is not True:
            failures.append(f"{name}.generation[{index}] capture restore is not synchronized")
        if values["eligible_steps"] != BLOCK_SIZE * values["block_replays"]:
            failures.append(f"{name}.generation[{index}] block accounting diverged")
        if (
            values["prefill_steps"]
            + values["eligible_steps"]
            + values["remainder_steps"]
            != values["physical_steps"]
        ):
            failures.append(f"{name}.generation[{index}] physical work diverged")
        if values["physical_steps"] - values["logical_steps"] != values["wasted_steps"]:
            failures.append(f"{name}.generation[{index}] rollback accounting diverged")
        if values["terminal_rollbacks"] != values["generator_offset_corrections"]:
            failures.append(f"{name}.generation[{index}] terminal/RNG rollback counts diverged")
        for key in totals[label]:
            if key in {"records", "generated_tokens"}:
                continue
            totals[label][key] += values[key]
    for label, values in totals.items():
        if values["records"] <= 0:
            failures.append(f"{name} has no {label} exact-K4 records")
        if values["logical_steps"] != values["generated_tokens"]:
            failures.append(f"{name}.{label} logical steps do not match tokens")
    if totals["main_generation"]["block_replays"] <= 0:
        failures.append(f"{name} did not execute an exact-K4 main block")
    return {"pass": not failures, "failures": failures, "totals": totals}


def _baseline_contract(profile: Mapping[str, Any], *, name: str) -> list[str]:
    failures: list[str] = []
    for index, record in enumerate(profile["generation"]):
        if isinstance(record, Mapping) and _candidate_stats(record) is not None:
            failures.append(f"{name}.generation[{index}] unexpectedly contains exact-K4")
    return failures


def _strict_exactness_pair(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_material = strict._strict_exactness_contract(baseline)
    candidate_material = strict._strict_exactness_contract(candidate)
    labels = {
        label: {
            "pass": baseline_material[label] == candidate_material[label],
            "baseline": baseline_material[label],
            "candidate": candidate_material[label],
        }
        for label in LABELS
    }
    token_stopping = {
        label: _label_exact(baseline, candidate, label) for label in LABELS
    }
    artifact = _artifact_exact(baseline, candidate)
    dispatch = {
        "pass": _dispatch_policy(baseline) == _dispatch_policy(candidate),
        "baseline": _dispatch_policy(baseline),
        "candidate": _dispatch_policy(candidate),
    }
    return {
        "pass": bool(
            all(value["pass"] for value in labels.values())
            and all(value["pass"] for value in token_stopping.values())
            and artifact["pass"]
            and dispatch["pass"]
        ),
        "strict_cache_and_rng": labels,
        "token_and_stopping": token_stopping,
        "artifact": artifact,
        "dispatch_policy": dispatch,
    }


def summarize(
    performance_paths: Mapping[str, Path],
    *,
    exactness_baseline: Path,
    exactness_candidate: Path,
    expected_commit: str,
    expected_branch: str,
    scope: str,
    minimum_speedup_pct: float,
    full_saving_seconds: float,
) -> dict[str, Any]:
    if set(performance_paths) != set(RUN_NAMES):
        raise GateError(f"performance paths must be exactly {RUN_NAMES}")
    if scope not in {"smoke", "full"}:
        raise GateError("scope must be smoke or full")
    performance = {name: _load(path) for name, path in performance_paths.items()}
    exactness = {
        "baseline": _load(exactness_baseline),
        "candidate": _load(exactness_candidate),
    }
    failures: list[str] = []
    for name, profile in performance.items():
        try:
            strict._runtime_contract(
                profile,
                expected_commit=expected_commit,
                expected_branch=expected_branch,
            )
        except strict.BaselineError as exc:
            failures.append(f"{name}: {exc}")
    for name, profile in exactness.items():
        try:
            strict._runtime_contract(
                profile,
                expected_commit=expected_commit,
                expected_branch=expected_branch,
                exactness_audit=True,
            )
        except strict.BaselineError as exc:
            failures.append(f"exactness_{name}: {exc}")

    for name in ("baseline_first", "baseline_second"):
        failures.extend(_baseline_contract(performance[name], name=name))
    candidate_contracts = {
        name: _candidate_contract(performance[name], name=name)
        for name in ("candidate_first", "candidate_second")
    }
    for contract in candidate_contracts.values():
        failures.extend(contract["failures"])
    exactness_candidate_contract = _candidate_contract(
        exactness["candidate"], name="exactness_candidate"
    )
    failures.extend(exactness_candidate_contract["failures"])
    failures.extend(_baseline_contract(exactness["baseline"], name="exactness_baseline"))

    for left, right in (
        ("baseline_first", "candidate_second"),
        ("baseline_second", "candidate_first"),
        ("baseline_first", "baseline_second"),
        ("candidate_first", "candidate_second"),
    ):
        contract = profiles._contract(performance[left], performance[right])
        if not contract["pass"]:
            failures.append(f"workload contract differs: {left} vs {right}")

    order_specs = {
        "baseline_first": ("baseline_first", "candidate_second"),
        "candidate_first": ("baseline_second", "candidate_first"),
    }
    orders: dict[str, Any] = {}
    speedups: list[float] = []
    savings: list[float] = []
    complete_request_wall_speedups: list[float] = []
    complete_request_wall_savings: list[float] = []
    for order, (baseline_name, candidate_name) in order_specs.items():
        labels: dict[str, Any] = {}
        for label in LABELS:
            baseline_perf = _performance(performance[baseline_name], label)
            candidate_perf = _performance(performance[candidate_name], label)
            exact = _label_exact(
                performance[baseline_name], performance[candidate_name], label
            )
            dispatch_equal = (
                _dispatch_policy(performance[baseline_name])[label]
                == _dispatch_policy(performance[candidate_name])[label]
            )
            saving = baseline_perf["model_seconds"] - candidate_perf["model_seconds"]
            speedup = (
                (candidate_perf["tps"] - baseline_perf["tps"])
                / baseline_perf["tps"]
                * 100.0
                if baseline_perf["tps"] > 0
                else -math.inf
            )
            labels[label] = {
                "pass": exact["pass"] and dispatch_equal,
                "exact": exact,
                "dispatch_policy_equal": dispatch_equal,
                "baseline": baseline_perf,
                "candidate": candidate_perf,
                "model_seconds_saving": saving,
                "tps_speedup_pct": speedup,
            }
            if not labels[label]["pass"]:
                failures.append(f"{order}.{label} changed exact work or dispatch policy")
            if label == "main_generation":
                speedups.append(speedup)
                savings.append(saving)
                baseline_wall = baseline_perf["complete_request_stage_wall_seconds"]
                candidate_wall = candidate_perf["complete_request_stage_wall_seconds"]
                complete_request_wall_savings.append(baseline_wall - candidate_wall)
                complete_request_wall_speedups.append(
                    (baseline_wall - candidate_wall) / baseline_wall * 100.0
                    if baseline_wall > 0
                    else -math.inf
                )
        artifact = _artifact_exact(
            performance[baseline_name], performance[candidate_name]
        )
        if not artifact["pass"]:
            failures.append(f"{order} final osu bytes differ")
        orders[order] = {"labels": labels, "artifact": artifact}

    baseline_repeat = _artifact_exact(
        performance["baseline_first"], performance["baseline_second"]
    )["pass"] and all(
        _label_exact(performance["baseline_first"], performance["baseline_second"], label)[
            "pass"
        ]
        for label in LABELS
    )
    candidate_repeat = _artifact_exact(
        performance["candidate_first"], performance["candidate_second"]
    )["pass"] and all(
        _label_exact(performance["candidate_first"], performance["candidate_second"], label)[
            "pass"
        ]
        for label in LABELS
    )
    if not baseline_repeat:
        failures.append("baseline reciprocal repeats are not exact")
    if not candidate_repeat:
        failures.append("candidate reciprocal repeats are not exact")

    try:
        exactness_pair = _strict_exactness_pair(
            exactness["baseline"], exactness["candidate"]
        )
    except (strict.BaselineError, GateError) as exc:
        failures.append(f"strict exactness evidence invalid: {exc}")
        exactness_pair = {"pass": False, "error": str(exc)}
    if not exactness_pair["pass"]:
        failures.append("strict cache/RNG/token/stopping/output/dispatch exactness failed")

    reciprocal_speedup = statistics.mean(speedups)
    reciprocal_saving = statistics.mean(savings)
    reciprocal_complete_wall_speedup = statistics.mean(
        complete_request_wall_speedups
    )
    reciprocal_complete_wall_saving = statistics.mean(
        complete_request_wall_savings
    )
    if reciprocal_speedup < minimum_speedup_pct:
        failures.append(
            f"reciprocal main TPS speedup {reciprocal_speedup:.3f}% is below "
            f"{minimum_speedup_pct:.3f}%"
        )
    if scope == "full" and reciprocal_saving < full_saving_seconds:
        failures.append(
            f"reciprocal main saving {reciprocal_saving:.3f}s is below "
            f"{full_saving_seconds:.3f}s"
        )
    return {
        "schema_version": "mapperatorinator.strict-fp32-exact-k4-reciprocal.v1",
        "scope": scope,
        "expected_commit": expected_commit,
        "expected_branch": expected_branch,
        "performance_paths": {name: str(path) for name, path in performance_paths.items()},
        "exactness_paths": {
            "baseline": str(exactness_baseline),
            "candidate": str(exactness_candidate),
        },
        "candidate_contracts": candidate_contracts,
        "exactness_candidate_contract": exactness_candidate_contract,
        "orders": orders,
        "baseline_repeatability_pass": baseline_repeat,
        "candidate_repeatability_pass": candidate_repeat,
        "strict_exactness": exactness_pair,
        "reciprocal_main_tps_speedup_pct": reciprocal_speedup,
        "reciprocal_main_model_seconds_saving": reciprocal_saving,
        "reciprocal_complete_request_wall_speedup_pct": (
            reciprocal_complete_wall_speedup
        ),
        "reciprocal_complete_request_wall_saving_seconds": (
            reciprocal_complete_wall_saving
        ),
        "minimum_speedup_pct": minimum_speedup_pct,
        "required_full_saving_seconds": full_saving_seconds,
        "pass": not failures,
        "failures": failures,
    }


def _write_text(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        f"scope={report['scope']}",
        f"pass={report['pass']}",
        f"reciprocal_main_tps_speedup_pct={report['reciprocal_main_tps_speedup_pct']}",
        f"reciprocal_main_model_seconds_saving={report['reciprocal_main_model_seconds_saving']}",
        "reciprocal_complete_request_wall_speedup_pct="
        f"{report['reciprocal_complete_request_wall_speedup_pct']}",
        "reciprocal_complete_request_wall_saving_seconds="
        f"{report['reciprocal_complete_request_wall_saving_seconds']}",
        f"strict_exactness_pass={report['strict_exactness']['pass']}",
        f"baseline_repeatability_pass={report['baseline_repeatability_pass']}",
        f"candidate_repeatability_pass={report['candidate_repeatability_pass']}",
    ]
    for failure in report["failures"]:
        lines.append(f"failure={failure}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("smoke", "full"), required=True)
    for name in RUN_NAMES:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--exactness-baseline", type=Path, required=True)
    parser.add_argument("--exactness-candidate", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--minimum-speedup-pct", type=float, default=0.0)
    parser.add_argument("--full-saving-seconds", type=float, default=1.412)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.expected_commit) != 40:
        parser.error("--expected-commit must be a full 40-character SHA")
    if args.full_saving_seconds < 0:
        parser.error("--full-saving-seconds must be non-negative")
    report = summarize(
        {name: getattr(args, name) for name in RUN_NAMES},
        exactness_baseline=args.exactness_baseline,
        exactness_candidate=args.exactness_candidate,
        expected_commit=args.expected_commit,
        expected_branch=args.expected_branch,
        scope=args.scope,
        minimum_speedup_pct=args.minimum_speedup_pct,
        full_saving_seconds=args.full_saving_seconds,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_text(args.text_output, report)
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
