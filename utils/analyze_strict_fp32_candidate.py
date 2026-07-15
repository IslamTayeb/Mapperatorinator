"""Fail-closed promotion analysis for strict-FP32 inference candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import analyze_fp32_fresh_baseline as baseline
    from . import nsight_agent_profile as nsight
except ImportError:  # Direct ``python utils/...py`` execution.
    import analyze_fp32_fresh_baseline as baseline
    import nsight_agent_profile as nsight


SCHEMA_VERSION = "mapperatorinator.strict-fp32-candidate-gate.v1"
MODES = ("initial", "confirm")
LABELS = baseline.LABELS
DISPATCH_DELTA_SCHEMA_VERSION = "mapperatorinator.dispatch-delta-allowlist.v1"
_MISSING = {"__mapperatorinator_missing__": True}


class CandidateGateError(RuntimeError):
    """Candidate evidence is incomplete, invalid, or inexact."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CandidateGateError(f"{name} must be an object")
    return value


def _required_file(path: Path, *, allow_empty: bool = False) -> Path:
    if not path.is_file():
        raise CandidateGateError(f"required artifact is missing: {path}")
    if not allow_empty and path.stat().st_size <= 0:
        raise CandidateGateError(f"required artifact is empty: {path}")
    return path


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    _required_file(path)
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), name=name)
    except json.JSONDecodeError as exc:
        raise CandidateGateError(f"invalid JSON in {path}: {exc}") from exc


def _verify_analysis_hash(payload: Mapping[str, Any], *, name: str) -> None:
    material = dict(payload)
    claimed = material.pop("analysis_sha256", None)
    if not isinstance(claimed, str) or claimed != _sha256(material):
        raise CandidateGateError(f"{name} analysis_sha256 is missing or invalid")


def _candidate_run_names(mode: str) -> tuple[str, ...]:
    if mode == "initial":
        return ("candidate-01",)
    if mode == "confirm":
        return tuple(f"candidate-{index:02d}" for index in range(1, 6))
    raise CandidateGateError(f"unsupported candidate mode: {mode!r}")


def _load_dispatch_delta_spec(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "schema_version": DISPATCH_DELTA_SCHEMA_VERSION,
            "allowed_paths": [],
            "required_paths": [],
            "source_path": None,
            "source_sha256": None,
        }
    payload = _load_json(path.resolve(), name="dispatch delta spec")
    expected_keys = {"schema_version", "allowed_paths", "required_paths"}
    if set(payload) != expected_keys:
        raise CandidateGateError(
            "dispatch delta spec must contain exactly " + str(sorted(expected_keys))
        )
    if payload.get("schema_version") != DISPATCH_DELTA_SCHEMA_VERSION:
        raise CandidateGateError("dispatch delta spec schema changed")
    parsed: dict[str, list[str]] = {}
    for key in ("allowed_paths", "required_paths"):
        raw_paths = payload.get(key)
        if not isinstance(raw_paths, list) or any(
            not isinstance(value, str) or not value for value in raw_paths
        ):
            raise CandidateGateError(f"dispatch delta spec {key} must be strings")
        if len(raw_paths) != len(set(raw_paths)):
            raise CandidateGateError(f"dispatch delta spec {key} contains duplicates")
        invalid = [
            value
            for value in raw_paths
            if not (value.startswith("preset.") or value.startswith("records."))
        ]
        if invalid:
            raise CandidateGateError(
                f"dispatch delta spec {key} leaves the policy surface: {invalid}"
            )
        parsed[key] = sorted(raw_paths)
    if not set(parsed["required_paths"]).issubset(parsed["allowed_paths"]):
        raise CandidateGateError("required dispatch paths must also be allowed")
    return {
        "schema_version": DISPATCH_DELTA_SCHEMA_VERSION,
        **parsed,
        "source_path": str(path.resolve()),
        "source_sha256": baseline._sha256_file(path.resolve()),
    }


def _contract_deltas(
    expected: Any,
    actual: Any,
    *,
    display_path: tuple[str, ...] = (),
    allowlist_path: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        deltas: list[dict[str, Any]] = []
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                deltas.append(
                    {
                        "path": ".".join((*display_path, str(key))),
                        "allowlist_path": ".".join((*allowlist_path, str(key))),
                        "baseline": expected.get(key, _MISSING),
                        "candidate": actual.get(key, _MISSING),
                    }
                )
                continue
            deltas.extend(
                _contract_deltas(
                    expected[key],
                    actual[key],
                    display_path=(*display_path, str(key)),
                    allowlist_path=(*allowlist_path, str(key)),
                )
            )
        return deltas
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise CandidateGateError(
                f"candidate policy changes list length at {'.'.join(display_path)}"
            )
        deltas = []
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual, strict=True)
        ):
            deltas.extend(
                _contract_deltas(
                    expected_item,
                    actual_item,
                    display_path=(*display_path, str(index)),
                    allowlist_path=(*allowlist_path, "*"),
                )
            )
        return deltas
    if type(expected) is not type(actual) or expected != actual:
        return [
            {
                "path": ".".join(display_path),
                "allowlist_path": ".".join(allowlist_path),
                "baseline": expected,
                "candidate": actual,
            }
        ]
    return []


def _declared_policy_delta(
    baseline_contract: Mapping[str, Any],
    candidate_contract: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_policy = {
        "preset": baseline_contract["preset"],
        "records": baseline_contract["dispatch_and_graph"],
    }
    candidate_policy = {
        "preset": candidate_contract["preset"],
        "records": candidate_contract["dispatch_and_graph"],
    }
    deltas = _contract_deltas(baseline_policy, candidate_policy)
    observed = sorted({delta["allowlist_path"] for delta in deltas})
    allowed = set(spec["allowed_paths"])
    required = set(spec["required_paths"])
    undeclared = sorted(set(observed) - allowed)
    missing_required = sorted(required - set(observed))
    if undeclared or missing_required:
        raise CandidateGateError(
            "candidate dispatch metadata delta is not declared: "
            f"undeclared={undeclared}, missing_required={missing_required}"
        )
    return {
        "pass": True,
        "spec": dict(spec),
        "observed_paths": observed,
        "deltas": deltas,
        "baseline_policy_sha256": _sha256(baseline_policy),
        "candidate_policy_sha256": _sha256(candidate_policy),
    }


def _pointer_path(run_dir: Path, name: str) -> Path:
    try:
        return baseline._path_from_pointer(run_dir, name)
    except baseline.BaselineError as exc:
        raise CandidateGateError(str(exc)) from exc


def _load_run(run_dir: Path) -> tuple[dict[str, Any], Path, Path]:
    if not run_dir.is_dir():
        raise CandidateGateError(f"missing run directory: {run_dir}")
    for artifact, allow_empty in (
        ("command.txt", False),
        ("stdout.txt", False),
        ("stderr.txt", True),
        ("profile-path.txt", False),
        ("result-path.txt", False),
        ("process-wall.json", False),
    ):
        _required_file(run_dir / artifact, allow_empty=allow_empty)
    profile_path = _pointer_path(run_dir, "profile-path.txt")
    result_path = _pointer_path(run_dir, "result-path.txt")
    try:
        profile = nsight._load_inference_profile(profile_path)
    except nsight.NsightProfileError as exc:
        raise CandidateGateError(str(exc)) from exc
    return profile, profile_path, result_path


def _runtime_contract(
    profile: Mapping[str, Any],
    *,
    commit: str,
    branch: str,
    exactness: bool,
) -> None:
    try:
        baseline._runtime_contract(
            profile,
            expected_commit=commit,
            expected_branch=branch,
            exactness_audit=exactness,
        )
    except baseline.BaselineError as exc:
        raise CandidateGateError(str(exc)) from exc


def _label_signature(profile: Mapping[str, Any], label: str) -> dict[str, Any]:
    signature = nsight._profile_label_signature(profile, label)
    if signature.get("status") != "available" or not signature.get(
        "self_consistent"
    ):
        raise CandidateGateError(
            f"{label} token/stopping signature is unavailable or inconsistent"
        )
    return signature


def _graph_signature(profile: Mapping[str, Any]) -> dict[str, Any]:
    signature = nsight._profile_graph_cache_signature(profile, LABELS)
    if signature.get("status") != "available":
        raise CandidateGateError("graph/cache signature is unavailable")
    return signature


def _profile_contract(profile: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return {
            "workload": baseline._workload_contract(profile),
            "preset": baseline._preset_contract(profile),
            "dispatch_and_graph": baseline._record_contract(profile),
            "graph_signature_sha256": _graph_signature(profile)["sha256"],
        }
    except baseline.BaselineError as exc:
        raise CandidateGateError(str(exc)) from exc


def _result_contract(profile: Mapping[str, Any], result_path: Path) -> dict[str, Any]:
    output_sha = baseline._sha256_file(result_path)
    metadata = _object(profile.get("metadata"), name="profile.metadata")
    if metadata.get("result_file_sha256") != output_sha:
        raise CandidateGateError("profile/output SHA-256 mismatch")
    if metadata.get("result_file_size_bytes") != result_path.stat().st_size:
        raise CandidateGateError("profile/output size mismatch")
    structure = nsight.summarize_osu_structure(result_path)
    if structure.get("malformed_by_section") != {
        "TimingPoints": 0,
        "HitObjects": 0,
    }:
        raise CandidateGateError("candidate output contains malformed rows")
    if structure.get("nonfinite_values") != 0:
        raise CandidateGateError("candidate output contains non-finite values")
    return {
        "sha256": output_sha,
        "size_bytes": result_path.stat().st_size,
        "structure": structure,
    }


def _metrics(profile: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    try:
        labels = {
            label: baseline._label_metrics(profile, label) for label in LABELS
        }
        return {
            "generation": labels,
            "combined_synchronized_model_seconds": sum(
                values["synchronized_model_seconds"] for values in labels.values()
            ),
            "request_to_final_osu_wall_seconds": baseline._request_wall_seconds(
                profile
            ),
            "process_wall_seconds": baseline._wall_artifact(run_dir)[
                "elapsed_seconds"
            ],
            "peak_cuda_memory_mb": baseline._peak_memory_mb(profile),
        }
    except baseline.BaselineError as exc:
        raise CandidateGateError(str(exc)) from exc


def _strict_exactness(profile: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return baseline._strict_exactness_contract(profile)
    except baseline.BaselineError as exc:
        raise CandidateGateError(str(exc)) from exc


def _load_baseline(
    run_root: Path,
    analysis_path: Path,
) -> dict[str, Any]:
    analysis = _load_json(analysis_path, name="baseline analysis")
    _verify_analysis_hash(analysis, name="baseline")
    if analysis.get("schema_version") != baseline.SCHEMA_VERSION:
        raise CandidateGateError("baseline analysis schema is not authoritative v1")
    if analysis.get("status") != "PASS" or not analysis.get(
        "authoritative_performance"
    ):
        raise CandidateGateError("baseline analysis is not an authoritative PASS")
    if analysis.get("run_count") != 5 or not analysis.get("fresh_processes"):
        raise CandidateGateError("baseline must contain five fresh processes")
    baseline_checks = _object(analysis.get("checks"), name="baseline.checks")
    failed_baseline_checks = sorted(
        name
        for name, raw in baseline_checks.items()
        if not isinstance(raw, dict) or raw.get("pass") is not True
    )
    if failed_baseline_checks:
        raise CandidateGateError(
            "baseline analysis contains failed checks: "
            + ", ".join(failed_baseline_checks)
        )

    precision = _object(
        analysis.get("precision_contract"), name="baseline.precision_contract"
    )
    baseline_commit = precision.get("expected_commit")
    baseline_branch = precision.get("expected_branch")
    if not isinstance(baseline_commit, str) or len(baseline_commit) != 40:
        raise CandidateGateError("baseline expected_commit must be a full SHA")
    if not isinstance(baseline_branch, str) or not baseline_branch:
        raise CandidateGateError("baseline expected_branch is missing")

    control_profile, control_profile_path, control_result = _load_run(
        run_root / "run-01"
    )
    audit_profile, audit_profile_path, audit_result = _load_run(
        run_root / "exactness-audit"
    )
    analysis_runs = analysis.get("runs")
    if not isinstance(analysis_runs, list) or len(analysis_runs) != 5:
        raise CandidateGateError("baseline analysis must expose all five runs")
    first_analysis_run = _object(analysis_runs[0], name="baseline.runs[0]")
    if Path(str(first_analysis_run.get("profile_path"))).resolve() != control_profile_path:
        raise CandidateGateError("baseline analysis/run-root profile lineage mismatch")
    if Path(str(first_analysis_run.get("result_path"))).resolve() != control_result:
        raise CandidateGateError("baseline analysis/run-root result lineage mismatch")
    analysis_audit = _object(
        analysis.get("exactness_audit"), name="baseline.exactness_audit"
    )
    if Path(str(analysis_audit.get("profile_path"))).resolve() != audit_profile_path:
        raise CandidateGateError("baseline exactness profile lineage mismatch")
    if Path(str(analysis_audit.get("result_path"))).resolve() != audit_result:
        raise CandidateGateError("baseline exactness result lineage mismatch")
    _runtime_contract(
        control_profile,
        commit=baseline_commit,
        branch=baseline_branch,
        exactness=False,
    )
    _runtime_contract(
        audit_profile,
        commit=baseline_commit,
        branch=baseline_branch,
        exactness=True,
    )
    control_result_contract = _result_contract(control_profile, control_result)
    audit_result_contract = _result_contract(audit_profile, audit_result)
    if control_result_contract != audit_result_contract:
        raise CandidateGateError("baseline control and exactness output differ")

    aggregates = _object(analysis.get("aggregates"), name="baseline.aggregates")
    required_aggregates = (
        "process_wall_seconds",
        "request_to_final_osu_wall_seconds",
        "timing_synchronized_model_seconds",
        "main_synchronized_model_seconds",
    )
    medians: dict[str, float] = {}
    for key in required_aggregates:
        values = _object(aggregates.get(key), name=f"baseline.aggregates.{key}")
        median = values.get("median")
        if (
            isinstance(median, bool)
            or not isinstance(median, (int, float))
            or not math.isfinite(float(median))
            or float(median) <= 0
        ):
            raise CandidateGateError(f"baseline median is invalid for {key}")
        medians[key] = float(median)
    combined_model_values: list[float] = []
    for index, raw_run in enumerate(analysis_runs):
        analysis_run = _object(raw_run, name=f"baseline.runs[{index}]")
        generation = _object(
            analysis_run.get("generation"),
            name=f"baseline.runs[{index}].generation",
        )
        combined = 0.0
        for label in LABELS:
            label_values = _object(
                generation.get(label),
                name=f"baseline.runs[{index}].generation.{label}",
            )
            value = label_values.get("synchronized_model_seconds")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise CandidateGateError(
                    f"baseline run {index} has invalid {label} synchronized time"
                )
            combined += float(value)
        combined_model_values.append(combined)
    medians["combined_synchronized_model_seconds"] = float(
        statistics.median(combined_model_values)
    )

    exactness = _strict_exactness(audit_profile)
    return {
        "analysis": analysis,
        "commit": baseline_commit,
        "branch": baseline_branch,
        "control_contract": _profile_contract(control_profile),
        "result_contract": control_result_contract,
        "label_signatures": {
            label: _label_signature(control_profile, label) for label in LABELS
        },
        "audit_label_signatures": {
            label: _label_signature(audit_profile, label) for label in LABELS
        },
        "audit_contract": _profile_contract(audit_profile),
        "strict_exactness": exactness,
        "medians": medians,
    }


def _comparison(name: str, expected: Any, actual: Any) -> dict[str, Any]:
    expected_sha = _sha256(expected)
    actual_sha = _sha256(actual)
    return {
        "name": name,
        "pass": expected_sha == actual_sha,
        "baseline_sha256": expected_sha,
        "candidate_sha256": actual_sha,
    }


def _metric_aggregate(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise CandidateGateError("cannot aggregate an empty metric")
    return {
        "median": float(statistics.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
        "range": float(max(values) - min(values)),
    }


def _delta(baseline_seconds: float, candidate_seconds: float) -> dict[str, float]:
    saved = baseline_seconds - candidate_seconds
    return {
        "baseline_seconds": baseline_seconds,
        "candidate_seconds": candidate_seconds,
        "saved_seconds": saved,
        "improvement_pct": saved / baseline_seconds * 100.0,
    }


def _validate_initial_analysis(
    path: Path,
    *,
    candidate_commit: str,
    candidate_branch: str,
    dispatch_delta_spec: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _load_json(path, name="initial candidate analysis")
    _verify_analysis_hash(payload, name="initial candidate")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise CandidateGateError("initial candidate analysis schema changed")
    if payload.get("mode") != "initial" or payload.get("status") != "PASS_INITIAL":
        raise CandidateGateError("confirmation requires a PASS_INITIAL analysis")
    if not payload.get("promotion_eligible"):
        raise CandidateGateError("initial candidate was not promotion eligible")
    identity = _object(payload.get("candidate"), name="initial.candidate")
    if identity.get("commit") != candidate_commit or identity.get("branch") != candidate_branch:
        raise CandidateGateError("confirmation candidate differs from initial candidate")
    initial_delta = _object(
        payload.get("declared_dispatch_delta"),
        name="initial.declared_dispatch_delta",
    )
    initial_spec = _object(initial_delta.get("spec"), name="initial dispatch spec")
    if initial_spec.get("source_sha256") != dispatch_delta_spec.get("source_sha256"):
        raise CandidateGateError("confirmation dispatch delta spec differs from initial")
    for key in ("allowed_paths", "required_paths"):
        if initial_spec.get(key) != dispatch_delta_spec.get(key):
            raise CandidateGateError("confirmation dispatch allowlist differs from initial")
    return payload


def analyze(
    baseline_run_root: Path,
    baseline_analysis_path: Path,
    candidate_run_root: Path,
    *,
    candidate_commit: str,
    candidate_branch: str,
    mode: str,
    initial_analysis_path: Path | None = None,
    dispatch_delta_spec_path: Path | None = None,
) -> dict[str, Any]:
    if len(candidate_commit) != 40:
        raise CandidateGateError("candidate commit must be a full 40-character SHA")
    if not candidate_branch:
        raise CandidateGateError("candidate branch must be nonempty")
    run_names = _candidate_run_names(mode)
    baseline_evidence = _load_baseline(
        baseline_run_root.resolve(), baseline_analysis_path.resolve()
    )
    dispatch_delta_spec = _load_dispatch_delta_spec(dispatch_delta_spec_path)
    initial_analysis = None
    if mode == "confirm":
        if initial_analysis_path is None:
            raise CandidateGateError("confirmation requires --initial-analysis")
        initial_analysis = _validate_initial_analysis(
            initial_analysis_path.resolve(),
            candidate_commit=candidate_commit,
            candidate_branch=candidate_branch,
            dispatch_delta_spec=dispatch_delta_spec,
        )
    elif initial_analysis_path is not None:
        raise CandidateGateError("initial mode must not receive --initial-analysis")

    runs: list[dict[str, Any]] = []
    run_contracts: list[dict[str, Any]] = []
    run_results: list[dict[str, Any]] = []
    run_signatures: dict[str, list[dict[str, Any]]] = {
        label: [] for label in LABELS
    }
    for run_name in run_names:
        run_dir = candidate_run_root.resolve() / run_name
        profile, profile_path, result_path = _load_run(run_dir)
        _runtime_contract(
            profile,
            commit=candidate_commit,
            branch=candidate_branch,
            exactness=False,
        )
        contract = _profile_contract(profile)
        result_contract = _result_contract(profile, result_path)
        metrics = _metrics(profile, run_dir)
        signatures = {
            label: _label_signature(profile, label) for label in LABELS
        }
        for label, signature in signatures.items():
            run_signatures[label].append(signature)
        run_contracts.append(contract)
        run_results.append(result_contract)
        runs.append(
            {
                "run_id": run_name,
                "profile_path": str(profile_path),
                "result_path": str(result_path),
                **metrics,
                "output_sha256": result_contract["sha256"],
                "output_structure": result_contract["structure"],
                "dispatch_and_graph_contract_sha256": _sha256(
                    contract["dispatch_and_graph"]
                ),
                "graph_signature_sha256": contract[
                    "graph_signature_sha256"
                ],
                "token_signatures": {
                    label: {
                        "token_stream_sha256": signature["token_stream_sha256"],
                        "stopping_sha256": signature["stopping_sha256"],
                        "generated_tokens": signature["generated_tokens"],
                    }
                    for label, signature in signatures.items()
                },
            }
        )

    audit_dir = candidate_run_root.resolve() / "exactness-audit"
    audit_profile, audit_profile_path, audit_result_path = _load_run(audit_dir)
    _runtime_contract(
        audit_profile,
        commit=candidate_commit,
        branch=candidate_branch,
        exactness=True,
    )
    audit_contract = _profile_contract(audit_profile)
    audit_result = _result_contract(audit_profile, audit_result_path)
    try:
        audit_process_wall = baseline._wall_artifact(audit_dir)["elapsed_seconds"]
    except baseline.BaselineError as exc:
        raise CandidateGateError(str(exc)) from exc
    audit_signatures = {
        label: _label_signature(audit_profile, label) for label in LABELS
    }
    strict_exactness = _strict_exactness(audit_profile)
    declared_dispatch_delta = _declared_policy_delta(
        baseline_evidence["control_contract"],
        run_contracts[0],
        dispatch_delta_spec,
    )

    checks: dict[str, dict[str, Any]] = {
        "candidate_runs_share_contract": _comparison(
            "candidate_runs_share_contract",
            [run_contracts[0]] * len(run_contracts),
            run_contracts,
        ),
        "candidate_runs_share_output": _comparison(
            "candidate_runs_share_output",
            [run_results[0]] * len(run_results),
            run_results,
        ),
        "baseline_workload": _comparison(
            "baseline_workload",
            baseline_evidence["control_contract"]["workload"],
            run_contracts[0]["workload"],
        ),
        "declared_dispatch_metadata_delta": {
            "name": "declared_dispatch_metadata_delta",
            "pass": declared_dispatch_delta["pass"],
            "observed_paths": declared_dispatch_delta["observed_paths"],
        },
        "baseline_output_bytes_and_structure": _comparison(
            "baseline_output_bytes_and_structure",
            baseline_evidence["result_contract"],
            run_results[0],
        ),
        "audit_workload": _comparison(
            "audit_workload",
            baseline_evidence["audit_contract"]["workload"],
            audit_contract["workload"],
        ),
        "audit_matches_candidate_policy": _comparison(
            "audit_matches_candidate_policy",
            {
                "preset": run_contracts[0]["preset"],
                "dispatch_and_graph": run_contracts[0]["dispatch_and_graph"],
            },
            {
                "preset": audit_contract["preset"],
                "dispatch_and_graph": audit_contract["dispatch_and_graph"],
            },
        ),
        "audit_matches_candidate_graph_signature": _comparison(
            "audit_matches_candidate_graph_signature",
            run_contracts[0]["graph_signature_sha256"],
            audit_contract["graph_signature_sha256"],
        ),
        "audit_output_bytes_and_structure": _comparison(
            "audit_output_bytes_and_structure",
            baseline_evidence["result_contract"],
            audit_result,
        ),
        "rng_progression_and_cache_writes": _comparison(
            "rng_progression_and_cache_writes",
            baseline_evidence["strict_exactness"],
            strict_exactness,
        ),
    }
    for label in LABELS:
        baseline_signature = baseline_evidence["label_signatures"][label]
        checks[f"{label}_tokens"] = _comparison(
            f"{label}_tokens",
            [baseline_signature["token_stream_sha256"]] * len(run_names),
            [value["token_stream_sha256"] for value in run_signatures[label]],
        )
        checks[f"{label}_stopping"] = _comparison(
            f"{label}_stopping",
            [baseline_signature["stopping_sha256"]] * len(run_names),
            [value["stopping_sha256"] for value in run_signatures[label]],
        )
        baseline_audit = baseline_evidence["audit_label_signatures"][label]
        checks[f"audit_{label}_tokens"] = _comparison(
            f"audit_{label}_tokens",
            baseline_audit["token_stream_sha256"],
            audit_signatures[label]["token_stream_sha256"],
        )
        checks[f"audit_{label}_stopping"] = _comparison(
            f"audit_{label}_stopping",
            baseline_audit["stopping_sha256"],
            audit_signatures[label]["stopping_sha256"],
        )
    failed = sorted(name for name, check in checks.items() if not check["pass"])
    if failed:
        raise CandidateGateError(
            "strict FP32 candidate parity failed: " + ", ".join(failed)
        )

    aggregates = {
        "process_wall_seconds": _metric_aggregate(
            [run["process_wall_seconds"] for run in runs]
        ),
        "request_to_final_osu_wall_seconds": _metric_aggregate(
            [run["request_to_final_osu_wall_seconds"] for run in runs]
        ),
        "combined_synchronized_model_seconds": _metric_aggregate(
            [run["combined_synchronized_model_seconds"] for run in runs]
        ),
        "timing_synchronized_model_seconds": _metric_aggregate(
            [
                run["generation"]["timing_context"][
                    "synchronized_model_seconds"
                ]
                for run in runs
            ]
        ),
        "main_synchronized_model_seconds": _metric_aggregate(
            [
                run["generation"]["main_generation"][
                    "synchronized_model_seconds"
                ]
                for run in runs
            ]
        ),
        "peak_cuda_memory_mb": _metric_aggregate(
            [run["peak_cuda_memory_mb"] for run in runs]
        ),
    }
    deltas = {
        key: _delta(
            baseline_evidence["medians"][key], aggregates[key]["median"]
        )
        for key in (
            "process_wall_seconds",
            "request_to_final_osu_wall_seconds",
            "combined_synchronized_model_seconds",
            "timing_synchronized_model_seconds",
            "main_synchronized_model_seconds",
        )
    }
    promotion_eligible = (
        deltas["request_to_final_osu_wall_seconds"]["saved_seconds"] > 0
        or deltas["combined_synchronized_model_seconds"]["saved_seconds"] > 0
    )
    status = (
        ("PASS_INITIAL" if mode == "initial" else "PASS_CONFIRMED")
        if promotion_eligible
        else "STOP_NO_GAIN"
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "mode": mode,
        "promotion_eligible": promotion_eligible,
        "exact_fixed_seed_parity": True,
        "candidate": {
            "commit": candidate_commit,
            "branch": candidate_branch,
            "run_count": len(runs),
            "fresh_processes": True,
        },
        "baseline": {
            "commit": baseline_evidence["commit"],
            "branch": baseline_evidence["branch"],
            "run_root": str(baseline_run_root.resolve()),
            "analysis_path": str(baseline_analysis_path.resolve()),
            "medians": baseline_evidence["medians"],
        },
        "initial_analysis_path": (
            str(initial_analysis_path.resolve())
            if initial_analysis_path is not None
            else None
        ),
        "checks": checks,
        "declared_dispatch_delta": declared_dispatch_delta,
        "runs": runs,
        "exactness_audit": {
            "authoritative_performance": False,
            "profile_path": str(audit_profile_path),
            "result_path": str(audit_result_path),
            "process_wall_seconds": audit_process_wall,
            "output_sha256": audit_result["sha256"],
            "output_structure": audit_result["structure"],
            "strict_exactness": strict_exactness,
        },
        "aggregates": aggregates,
        "deltas": deltas,
    }
    if initial_analysis is not None:
        result["initial_analysis_sha256"] = initial_analysis["analysis_sha256"]
    result["analysis_sha256"] = _sha256(result)
    return result


def _text_report(result: Mapping[str, Any]) -> str:
    lines = [
        f"status={result['status']}",
        f"mode={result['mode']}",
        f"promotion_eligible={str(result['promotion_eligible']).lower()}",
        "precision=fp32",
        "tf32=disabled",
        "exact_fixed_seed_parity=true",
        f"candidate_commit={result['candidate']['commit']}",
        f"candidate_branch={result['candidate']['branch']}",
        f"candidate_run_count={result['candidate']['run_count']}",
    ]
    for key in (
        "combined_synchronized_model_seconds",
        "request_to_final_osu_wall_seconds",
        "process_wall_seconds",
        "timing_synchronized_model_seconds",
        "main_synchronized_model_seconds",
    ):
        delta = result["deltas"][key]
        lines.extend(
            (
                f"{key}_baseline={delta['baseline_seconds']:.6f}",
                f"{key}_candidate={delta['candidate_seconds']:.6f}",
                f"{key}_saved={delta['saved_seconds']:.6f}",
                f"{key}_improvement_pct={delta['improvement_pct']:.6f}",
            )
        )
    return "\n".join(lines) + "\n"


def _error_payload(exc: Exception, *, mode: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ERROR",
        "mode": mode,
        "promotion_eligible": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    payload["analysis_sha256"] = _sha256(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run-root", type=Path, required=True)
    parser.add_argument("--baseline-analysis", type=Path, required=True)
    parser.add_argument("--candidate-run-root", type=Path, required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--candidate-branch", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--initial-analysis", type=Path)
    parser.add_argument("--dispatch-delta-spec", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    cli = parser.parse_args(argv)
    try:
        result = analyze(
            cli.baseline_run_root,
            cli.baseline_analysis,
            cli.candidate_run_root,
            candidate_commit=cli.candidate_commit,
            candidate_branch=cli.candidate_branch,
            mode=cli.mode,
            initial_analysis_path=cli.initial_analysis,
            dispatch_delta_spec_path=cli.dispatch_delta_spec,
        )
    except (CandidateGateError, baseline.BaselineError, nsight.NsightProfileError) as exc:
        result = _error_payload(exc, mode=cli.mode)
        exit_code = 2
    else:
        exit_code = 0 if result["promotion_eligible"] else 3
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    cli.text_output.parent.mkdir(parents=True, exist_ok=True)
    if result["status"] == "ERROR":
        cli.text_output.write_text(
            f"status=ERROR\nerror_type={result['error_type']}\nerror={result['error']}\n",
            encoding="utf-8",
        )
    else:
        cli.text_output.write_text(_text_report(result), encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
