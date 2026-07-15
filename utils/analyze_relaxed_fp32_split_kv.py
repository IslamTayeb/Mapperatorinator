"""Fail-loud diagnostic analysis for FP32 split-KV with self-cache hash drift."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import analyze_strict_fp32_candidate as strict
except ImportError:  # Direct ``python utils/...py`` execution.
    import analyze_strict_fp32_candidate as strict


SCHEMA_VERSION = "mapperatorinator.relaxed-fp32-split-kv-diagnostic.v1"


class RelaxedSplitKVDiagnosticError(RuntimeError):
    """The evidence is incomplete or drift escaped the declared surface."""


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RelaxedSplitKVDiagnosticError(f"{name} must be an object")
    return value


def _same(name: str, expected: Any, actual: Any) -> dict[str, Any]:
    result = strict._comparison(name, expected, actual)
    if not result["pass"]:
        raise RelaxedSplitKVDiagnosticError(f"{name} changed")
    return result


def _cache_drift(
    baseline_exactness: Mapping[str, Any],
    candidate_exactness: Mapping[str, Any],
) -> dict[str, Any]:
    allowed: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    if set(baseline_exactness) != set(strict.LABELS) or set(candidate_exactness) != set(
        strict.LABELS
    ):
        raise RelaxedSplitKVDiagnosticError("exactness labels changed")
    for label in strict.LABELS:
        baseline_records = baseline_exactness[label]
        candidate_records = candidate_exactness[label]
        if not isinstance(baseline_records, list) or not isinstance(
            candidate_records, list
        ):
            raise RelaxedSplitKVDiagnosticError(f"{label} exactness must be lists")
        if len(baseline_records) != len(candidate_records):
            raise RelaxedSplitKVDiagnosticError(f"{label} exactness record count changed")
        for index, (baseline_record, candidate_record) in enumerate(
            zip(baseline_records, candidate_records, strict=True)
        ):
            baseline_record = _object(
                baseline_record, name=f"baseline.{label}[{index}]"
            )
            candidate_record = _object(
                candidate_record, name=f"candidate.{label}[{index}]"
            )
            for field in ("record_key", "rng_before", "rng_after"):
                checks[f"{label}[{index}].{field}"] = _same(
                    f"{label}[{index}].{field}",
                    baseline_record.get(field),
                    candidate_record.get(field),
                )
            baseline_cache = _object(
                baseline_record.get("cache_writes"),
                name=f"baseline.{label}[{index}].cache_writes",
            )
            candidate_cache = _object(
                candidate_record.get("cache_writes"),
                name=f"candidate.{label}[{index}].cache_writes",
            )
            for field in (
                "self_sequence_length",
                "cross_sequence_length",
                "layer_count",
            ):
                checks[f"{label}[{index}].cache.{field}"] = _same(
                    f"{label}[{index}].cache.{field}",
                    baseline_cache.get(field),
                    candidate_cache.get(field),
                )
            baseline_layers = baseline_cache.get("layers")
            candidate_layers = candidate_cache.get("layers")
            if not isinstance(baseline_layers, list) or not isinstance(
                candidate_layers, list
            ):
                raise RelaxedSplitKVDiagnosticError("cache layers must be lists")
            if len(baseline_layers) != len(candidate_layers):
                raise RelaxedSplitKVDiagnosticError("cache layer count changed")
            record_self_drift = False
            for layer_index, (baseline_layer, candidate_layer) in enumerate(
                zip(baseline_layers, candidate_layers, strict=True)
            ):
                baseline_layer = _object(
                    baseline_layer,
                    name=f"baseline.{label}[{index}].layers[{layer_index}]",
                )
                candidate_layer = _object(
                    candidate_layer,
                    name=f"candidate.{label}[{index}].layers[{layer_index}]",
                )
                for field in ("layer_idx", "cross_keys", "cross_values"):
                    checks[f"{label}[{index}].layer[{layer_index}].{field}"] = _same(
                        f"{label}[{index}].layer[{layer_index}].{field}",
                        baseline_layer.get(field),
                        candidate_layer.get(field),
                    )
                for field in ("self_keys", "self_values"):
                    baseline_hash = baseline_layer.get(field)
                    candidate_hash = candidate_layer.get(field)
                    if baseline_hash != candidate_hash:
                        if label != "main_generation":
                            raise RelaxedSplitKVDiagnosticError(
                                f"{label} {field} drift is outside the allowed surface"
                            )
                        record_self_drift = True
                        allowed.append(
                            {
                                "label": label,
                                "record_index": index,
                                "record_key": baseline_record["record_key"],
                                "layer_idx": layer_index,
                                "field": field,
                                "baseline_sha256": baseline_hash,
                                "candidate_sha256": candidate_hash,
                            }
                        )
            aggregate_exact = (
                baseline_cache.get("aggregate_sha256")
                == candidate_cache.get("aggregate_sha256")
            )
            if label != "main_generation" and not aggregate_exact:
                raise RelaxedSplitKVDiagnosticError(
                    f"{label} aggregate cache hash changed"
                )
            if label == "main_generation" and aggregate_exact == record_self_drift:
                raise RelaxedSplitKVDiagnosticError(
                    "main cache aggregate/self-hash drift is inconsistent"
                )
    if not allowed:
        raise RelaxedSplitKVDiagnosticError(
            "no main self-cache hash drift was observed; use the strict analyzer"
        )
    return {
        "allowed_surface": "main_generation.self_cache_sha256_only",
        "drift_count": len(allowed),
        "drift": allowed,
        "rng_exact": True,
        "timing_cache_exact": True,
        "main_cross_cache_exact": True,
        "main_self_cache_exact": False,
        "structural_cache_metadata_exact": True,
        "checks": checks,
    }


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RelaxedSplitKVDiagnosticError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise RelaxedSplitKVDiagnosticError(f"{name} must be finite and nonnegative")
    return result


def _capture_seconds(record: Mapping[str, Any]) -> float:
    graphs = record.get("optimized_cuda_graphs")
    if not isinstance(graphs, dict):
        return 0.0
    return _finite(graphs.get("capture_seconds", 0.0), name="capture_seconds")


def _cold_diagnostics(profile: Mapping[str, Any]) -> dict[str, Any]:
    by_label: dict[str, Any] = {}
    for label in strict.LABELS:
        records = [
            _object(value, name=f"generation[{index}]")
            for index, value in enumerate(profile.get("generation", []))
            if isinstance(value, dict) and value.get("profile_label") == label
        ]
        if not records:
            raise RelaxedSplitKVDiagnosticError(f"{label} records are missing")
        rows = []
        for index, record in enumerate(records):
            wall = _finite(record.get("wall_seconds"), name=f"{label}[{index}].wall")
            model = _finite(
                record.get("model_elapsed_seconds"),
                name=f"{label}[{index}].model",
            )
            capture = _capture_seconds(record)
            non_model = wall - model
            if non_model < -1e-6:
                raise RelaxedSplitKVDiagnosticError(
                    f"{label}[{index}] model time exceeds outer wall"
                )
            rows.append(
                {
                    "record_index": index,
                    "wall_seconds": wall,
                    "synchronized_model_seconds": model,
                    "cuda_graph_capture_seconds": capture,
                    "non_model_seconds": max(0.0, non_model),
                    "non_model_excluding_graph_capture_seconds": max(
                        0.0, non_model - capture
                    ),
                }
            )
        by_label[label] = {
            "first_record": rows[0],
            "maximum_non_model_record": max(
                rows, key=lambda value: value["non_model_seconds"]
            ),
            "total_outer_wall_seconds": sum(value["wall_seconds"] for value in rows),
            "total_synchronized_model_seconds": sum(
                value["synchronized_model_seconds"] for value in rows
            ),
            "total_cuda_graph_capture_seconds": sum(
                value["cuda_graph_capture_seconds"] for value in rows
            ),
        }
    first_main = by_label["main_generation"]["first_record"]
    likely_cold_jit = (
        first_main["non_model_excluding_graph_capture_seconds"] >= 10.0
        and first_main["cuda_graph_capture_seconds"] < 1.0
    )
    return {
        "labels": by_label,
        "likely_cold_native_jit_or_build": likely_cold_jit,
        "classification": (
            "first-main cold native JIT/build dominates non-model wall"
            if likely_cold_jit
            else "no large first-main cold JIT/build signature"
        ),
    }


def _gpu_overlap(preflight_path: Path) -> dict[str, Any]:
    if not preflight_path.is_file():
        raise RelaxedSplitKVDiagnosticError(f"missing preflight: {preflight_path}")
    text = preflight_path.read_text(encoding="utf-8")
    match = re.search(r"^job_id=(\d+)$", text, flags=re.MULTILINE)
    if match is None:
        raise RelaxedSplitKVDiagnosticError("preflight job_id is missing")
    current = match.group(1)
    observed = sorted(
        {
            value
            for value in re.findall(
                r"^\s*(\d+)\s+.*gres/gpu.*$", text, flags=re.MULTILINE
            )
            if value != current
        }
    )
    return {
        "current_job_id": current,
        "other_gpu_job_ids_observed_in_preflight": observed,
        "pass": not observed,
    }


def analyze(
    baseline_run_root: Path,
    baseline_analysis_path: Path,
    candidate_run_root: Path,
    *,
    candidate_commit: str,
    candidate_branch: str,
    mode: str,
    dispatch_delta_spec_path: Path,
) -> dict[str, Any]:
    if mode != "initial":
        raise RelaxedSplitKVDiagnosticError(
            "documented-drift diagnostics currently require one initial run"
        )
    if len(candidate_commit) != 40 or not candidate_branch:
        raise RelaxedSplitKVDiagnosticError("candidate identity is invalid")
    baseline_evidence = strict._load_baseline(
        baseline_run_root.resolve(), baseline_analysis_path.resolve()
    )
    candidate_root = candidate_run_root.resolve()
    run_dir = candidate_root / "candidate-01"
    profile, profile_path, result_path = strict._load_run(run_dir)
    strict._runtime_contract(
        profile,
        commit=candidate_commit,
        branch=candidate_branch,
        exactness=False,
    )
    contract = strict._profile_contract(profile)
    result_contract = strict._result_contract(profile, result_path)
    metrics = strict._metrics(profile, run_dir)
    signatures = {
        label: strict._label_signature(profile, label) for label in strict.LABELS
    }

    audit_dir = candidate_root / "exactness-audit"
    audit_profile, audit_profile_path, audit_result_path = strict._load_run(audit_dir)
    strict._runtime_contract(
        audit_profile,
        commit=candidate_commit,
        branch=candidate_branch,
        exactness=True,
    )
    audit_contract = strict._profile_contract(audit_profile)
    audit_result = strict._result_contract(audit_profile, audit_result_path)
    audit_signatures = {
        label: strict._label_signature(audit_profile, label)
        for label in strict.LABELS
    }
    candidate_exactness = strict._strict_exactness(audit_profile)
    delta_spec = strict._load_dispatch_delta_spec(dispatch_delta_spec_path)
    dispatch_delta = strict._declared_policy_delta(
        baseline_evidence["control_contract"], contract, delta_spec
    )

    checks: dict[str, Any] = {
        "baseline_workload": _same(
            "baseline_workload",
            baseline_evidence["control_contract"]["workload"],
            contract["workload"],
        ),
        "declared_dispatch_metadata_delta": {
            "name": "declared_dispatch_metadata_delta",
            "pass": dispatch_delta["pass"],
            "observed_paths": dispatch_delta["observed_paths"],
        },
        "control_final_osu": _same(
            "control_final_osu", baseline_evidence["result_contract"], result_contract
        ),
        "audit_workload": _same(
            "audit_workload",
            baseline_evidence["audit_contract"]["workload"],
            audit_contract["workload"],
        ),
        "audit_candidate_policy": _same(
            "audit_candidate_policy",
            {
                "preset": contract["preset"],
                "dispatch_and_graph": contract["dispatch_and_graph"],
            },
            {
                "preset": audit_contract["preset"],
                "dispatch_and_graph": audit_contract["dispatch_and_graph"],
            },
        ),
        "audit_graph_signature": _same(
            "audit_graph_signature",
            contract["graph_signature_sha256"],
            audit_contract["graph_signature_sha256"],
        ),
        "audit_final_osu": _same(
            "audit_final_osu", baseline_evidence["result_contract"], audit_result
        ),
    }
    for label in strict.LABELS:
        checks[f"{label}_tokens"] = _same(
            f"{label}_tokens",
            baseline_evidence["label_signatures"][label]["token_stream_sha256"],
            signatures[label]["token_stream_sha256"],
        )
        checks[f"{label}_stopping"] = _same(
            f"{label}_stopping",
            baseline_evidence["label_signatures"][label]["stopping_sha256"],
            signatures[label]["stopping_sha256"],
        )
        checks[f"audit_{label}_tokens"] = _same(
            f"audit_{label}_tokens",
            baseline_evidence["audit_label_signatures"][label][
                "token_stream_sha256"
            ],
            audit_signatures[label]["token_stream_sha256"],
        )
        checks[f"audit_{label}_stopping"] = _same(
            f"audit_{label}_stopping",
            baseline_evidence["audit_label_signatures"][label]["stopping_sha256"],
            audit_signatures[label]["stopping_sha256"],
        )

    cache_drift = _cache_drift(
        baseline_evidence["strict_exactness"], candidate_exactness
    )
    candidate_values = {
        "process_wall_seconds": metrics["process_wall_seconds"],
        "request_to_final_osu_wall_seconds": metrics[
            "request_to_final_osu_wall_seconds"
        ],
        "combined_synchronized_model_seconds": metrics[
            "combined_synchronized_model_seconds"
        ],
        "timing_synchronized_model_seconds": metrics["generation"][
            "timing_context"
        ]["synchronized_model_seconds"],
        "main_synchronized_model_seconds": metrics["generation"]["main_generation"][
            "synchronized_model_seconds"
        ],
    }
    deltas = {
        name: strict._delta(baseline_evidence["medians"][name], value)
        for name, value in candidate_values.items()
    }
    main_tokens = signatures["main_generation"]["generated_tokens"]
    timing_tokens = signatures["timing_context"]["generated_tokens"]
    throughput = {
        "main": {
            "tokens": main_tokens,
            "baseline_tps": main_tokens
            / baseline_evidence["medians"]["main_synchronized_model_seconds"],
            "candidate_tps": main_tokens
            / candidate_values["main_synchronized_model_seconds"],
        },
        "timing": {
            "tokens": timing_tokens,
            "baseline_tps": timing_tokens
            / baseline_evidence["medians"]["timing_synchronized_model_seconds"],
            "candidate_tps": timing_tokens
            / candidate_values["timing_synchronized_model_seconds"],
        },
    }
    gpu_overlap = _gpu_overlap(candidate_root / "preflight.txt")
    cold = _cold_diagnostics(profile)
    reasons = ["single candidate process", "documented main self-cache hash drift"]
    if cold["likely_cold_native_jit_or_build"]:
        reasons.append("first-main cold native JIT/build")
    if not gpu_overlap["pass"]:
        reasons.append("another GPU job was observed in preflight")
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_DOCUMENTED_DRIFT_DIAGNOSTIC",
        "mode": mode,
        "promotion_eligible": False,
        "authoritative_performance": False,
        "non_authoritative_reasons": reasons,
        "exact_fixed_seed_tokens_rng_stopping_and_final_osu": True,
        "candidate": {
            "commit": candidate_commit,
            "branch": candidate_branch,
            "run_root": str(candidate_root),
            "profile_path": str(profile_path),
            "result_path": str(result_path),
            "audit_profile_path": str(audit_profile_path),
            "audit_result_path": str(audit_result_path),
        },
        "baseline": {
            "commit": baseline_evidence["commit"],
            "branch": baseline_evidence["branch"],
            "run_root": str(baseline_run_root.resolve()),
            "analysis_path": str(baseline_analysis_path.resolve()),
            "medians": baseline_evidence["medians"],
        },
        "checks": checks,
        "documented_cache_hash_drift": cache_drift,
        "declared_dispatch_delta": dispatch_delta,
        "gpu_overlap": gpu_overlap,
        "cold_jit_diagnostics": cold,
        "candidate_metrics": metrics,
        "deltas": deltas,
        "throughput": throughput,
    }
    result["analysis_sha256"] = strict._sha256(result)
    return result


def _text(result: Mapping[str, Any]) -> str:
    main = result["throughput"]["main"]
    timing = result["throughput"]["timing"]
    first_main = result["cold_jit_diagnostics"]["labels"]["main_generation"][
        "first_record"
    ]
    return "\n".join(
        (
            f"status={result['status']}",
            "promotion_eligible=false",
            "authoritative_performance=false",
            "precision=fp32",
            "tokens_rng_stopping_final_osu_exact=true",
            "main_self_cache_hash_exact=false",
            f"main_cache_hash_drift_count={result['documented_cache_hash_drift']['drift_count']}",
            f"main_baseline_seconds={result['deltas']['main_synchronized_model_seconds']['baseline_seconds']:.6f}",
            f"main_candidate_seconds={result['deltas']['main_synchronized_model_seconds']['candidate_seconds']:.6f}",
            f"main_saved_seconds={result['deltas']['main_synchronized_model_seconds']['saved_seconds']:.6f}",
            f"main_baseline_tps={main['baseline_tps']:.6f}",
            f"main_candidate_tps={main['candidate_tps']:.6f}",
            f"timing_baseline_tps={timing['baseline_tps']:.6f}",
            f"timing_candidate_tps={timing['candidate_tps']:.6f}",
            f"first_main_non_model_seconds={first_main['non_model_seconds']:.6f}",
            "likely_cold_native_jit_or_build="
            + str(
                result["cold_jit_diagnostics"]["likely_cold_native_jit_or_build"]
            ).lower(),
            "other_gpu_jobs_observed="
            + ",".join(result["gpu_overlap"]["other_gpu_job_ids_observed_in_preflight"]),
        )
    ) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run-root", type=Path, required=True)
    parser.add_argument("--baseline-analysis", type=Path, required=True)
    parser.add_argument("--candidate-run-root", type=Path, required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--candidate-branch", required=True)
    parser.add_argument("--mode", choices=("initial",), required=True)
    parser.add_argument("--dispatch-delta-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = analyze(
            args.baseline_run_root,
            args.baseline_analysis,
            args.candidate_run_root,
            candidate_commit=args.candidate_commit,
            candidate_branch=args.candidate_branch,
            mode=args.mode,
            dispatch_delta_spec_path=args.dispatch_delta_spec,
        )
        exit_code = 0
    except Exception as exc:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "ERROR",
            "promotion_eligible": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        result["analysis_sha256"] = strict._sha256(result)
        exit_code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.text_output.parent.mkdir(parents=True, exist_ok=True)
    report = _text(result) if exit_code == 0 else f"status=ERROR\nerror={result['error']}\n"
    args.text_output.write_text(report, encoding="utf-8")
    print(report, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
