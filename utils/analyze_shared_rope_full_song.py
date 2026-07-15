from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

try:
    from . import nsight_agent_profile as nsight
    from .analyze_fp32_fresh_baseline import (
        LABELS,
        _canonical_json,
        _label_metrics,
        _path_from_pointer,
        _peak_memory_mb,
        _preset_contract,
        _record_contract,
        _request_wall_seconds,
        _required_file,
        _sha256_bytes,
        _sha256_file,
        _strict_exactness_contract,
        _wall_artifact,
        _workload_contract,
    )
except ImportError:
    import nsight_agent_profile as nsight
    from analyze_fp32_fresh_baseline import (
        LABELS,
        _canonical_json,
        _label_metrics,
        _path_from_pointer,
        _peak_memory_mb,
        _preset_contract,
        _record_contract,
        _request_wall_seconds,
        _required_file,
        _sha256_bytes,
        _sha256_file,
        _strict_exactness_contract,
        _wall_artifact,
        _workload_contract,
    )


RUN_NAMES = ("run-01", "run-02", "run-03")
PRECISIONS = ("fp32", "fp16")
SIDES = ("baseline", "candidate")
CANDIDATE_VERSION = "strict-shared-rope-dtype-generic-v2"


class FullSongGateError(RuntimeError):
    pass


def _runtime_contract(
    profile: dict[str, Any],
    *,
    precision: str,
    commit: str,
    branch: str,
    audit: bool,
) -> None:
    metadata = profile["metadata"]
    expected = {
        "precision": precision,
        "inference_engine": "optimized",
        "attn_implementation": "sdpa",
        "profile_pass_kind": "exactness_audit" if audit else "untraced_control",
        "authoritative_performance": not audit,
        "strict_exactness_evidence": audit,
        "nvidia_tf32_override": "0",
        "git_commit": commit,
        "git_branch": branch,
    }
    if precision == "fp32":
        expected.update(
            {
                "float32_matmul_precision": "highest",
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
            }
        )
    failures = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if failures:
        raise FullSongGateError(f"runtime metadata mismatch: {failures}")
    if "2080 Ti" not in str(metadata.get("cuda_device_name", "")):
        raise FullSongGateError("full-song gate did not run on an RTX 2080 Ti")
    if metadata.get("cuda_device_capability") != [7, 5]:
        raise FullSongGateError("full-song gate requires SM75")
    effective = metadata.get("optimized_effective_config")
    if not isinstance(effective, dict) or effective.get("precision") != precision:
        raise FullSongGateError("optimized preset precision changed")
    generated = {
        row.get("precision")
        for row in profile["generation"]
        if isinstance(row, dict)
    }
    if generated != {precision}:
        raise FullSongGateError(f"generation precision changed: {generated}")


def _candidate_evidence_contract(
    candidate_evidence: Any,
    *,
    precision: str,
) -> dict[str, Any]:
    if not isinstance(candidate_evidence, dict):
        raise FullSongGateError("candidate evidence must be an object")
    payload = candidate_evidence.get("candidate")
    if not isinstance(payload, dict):
        raise FullSongGateError("candidate evidence is missing candidate data")
    required = {
        "version": CANDIDATE_VERSION,
        "precision": precision,
        "dtype_generic": True,
        "original_decoder_forward_required": True,
        "production_wiring": False,
        "full_song_exactness_claim": False,
    }
    failures = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in required.items()
        if payload.get(key) != value
    }
    retained = payload.get("retained_specialized_dispatch")
    if retained != {
        "q1_bmm_cross_attention": True,
        "native_q1_rope_cache_self_attention": True,
        "native_q1_self_attention": True,
    }:
        failures["retained_specialized_dispatch"] = retained
    stats = payload.get("stats")
    if not isinstance(stats, dict) or int(stats.get("reuses", 0)) <= 0:
        failures["stats.reuses"] = (
            None if not isinstance(stats, dict) else stats.get("reuses")
        )
    if failures:
        raise FullSongGateError(f"candidate evidence mismatch: {failures}")
    return payload


def _load_run(
    run_dir: Path,
    *,
    precision: str,
    commit: str,
    branch: str,
    audit: bool,
    candidate: bool,
) -> dict[str, Any]:
    for name, allow_empty in (
        ("command.txt", False),
        ("stdout.txt", False),
        ("stderr.txt", True),
        ("profile-path.txt", False),
        ("result-path.txt", False),
    ):
        _required_file(run_dir / name, allow_empty=allow_empty)
    profile_path = _path_from_pointer(run_dir, "profile-path.txt")
    result_path = _path_from_pointer(run_dir, "result-path.txt")
    profile = nsight._load_inference_profile(profile_path)
    _runtime_contract(
        profile,
        precision=precision,
        commit=commit,
        branch=branch,
        audit=audit,
    )
    result_sha = _sha256_file(result_path)
    if profile["metadata"].get("result_file_sha256") != result_sha:
        raise FullSongGateError(f"profile/output SHA mismatch: {run_dir}")
    structure = nsight.summarize_osu_structure(result_path)
    if structure["malformed_by_section"] != {"TimingPoints": 0, "HitObjects": 0}:
        raise FullSongGateError(f"malformed .osu output: {run_dir}")
    if structure["nonfinite_values"] != 0:
        raise FullSongGateError(f"non-finite .osu output: {run_dir}")
    metrics = {label: _label_metrics(profile, label) for label in LABELS}
    signatures = {
        label: nsight._profile_label_signature(profile, label) for label in LABELS
    }
    if any(
        signature["status"] != "available" or not signature["self_consistent"]
        for signature in signatures.values()
    ):
        raise FullSongGateError(f"token/stopping signature unavailable: {run_dir}")
    graph = nsight._profile_graph_cache_signature(profile, LABELS)
    if graph["status"] != "available":
        raise FullSongGateError(f"graph signature unavailable: {run_dir}")
    candidate_evidence = None
    if candidate:
        evidence_path = _required_file(
            run_dir / "candidate-evidence.json",
            allow_empty=False,
        )
        candidate_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        _candidate_evidence_contract(candidate_evidence, precision=precision)
    return {
        "profile_path": str(profile_path),
        "result_path": str(result_path),
        "process_wall_seconds": _wall_artifact(run_dir)["elapsed_seconds"],
        "request_wall_seconds": _request_wall_seconds(profile),
        "generation": metrics,
        "peak_cuda_memory_mb": _peak_memory_mb(profile),
        "output_sha256": result_sha,
        "output_structure": structure,
        "workload": _workload_contract(profile),
        "preset": _preset_contract(profile),
        "records": _record_contract(profile),
        "graph_sha256": graph["sha256"],
        "signatures": {
            label: {
                "token_stream_sha256": signature["token_stream_sha256"],
                "stopping_sha256": signature["stopping_sha256"],
                "generated_tokens": signature["generated_tokens"],
            }
            for label, signature in signatures.items()
        },
        "strict_exactness": _strict_exactness_contract(profile) if audit else None,
        "candidate_evidence": candidate_evidence,
    }


def _aggregate(values: list[float]) -> dict[str, float]:
    if len(values) != len(RUN_NAMES):
        raise FullSongGateError("performance aggregate requires three runs")
    return {
        "median": float(statistics.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _comparison(baseline: float, candidate: float) -> dict[str, float]:
    saved = baseline - candidate
    return {
        "baseline_seconds": baseline,
        "candidate_seconds": candidate,
        "saved_seconds": saved,
        "candidate_delta_pct": (candidate - baseline) / baseline * 100.0,
    }


def _same(values: list[Any]) -> bool:
    hashes = {_sha256_bytes(_canonical_json(value)) for value in values}
    return len(hashes) == 1


def _analyze_precision(
    root: Path,
    *,
    precision: str,
    baseline_commit: str,
    baseline_branch: str,
    candidate_commit: str,
    candidate_branch: str,
) -> dict[str, Any]:
    expected = {
        "baseline": (baseline_commit, baseline_branch),
        "candidate": (candidate_commit, candidate_branch),
    }
    runs: dict[str, list[dict[str, Any]]] = {}
    audits: dict[str, dict[str, Any]] = {}
    for side in SIDES:
        commit, branch = expected[side]
        runs[side] = [
            _load_run(
                root / precision / side / run_name,
                precision=precision,
                commit=commit,
                branch=branch,
                audit=False,
                candidate=side == "candidate",
            )
            for run_name in RUN_NAMES
        ]
        audits[side] = _load_run(
            root / precision / side / "exactness-audit",
            precision=precision,
            commit=commit,
            branch=branch,
            audit=True,
            candidate=side == "candidate",
        )

    combined = runs["baseline"] + runs["candidate"] + list(audits.values())
    checks = {
        "output_bytes": _same([row["output_sha256"] for row in combined]),
        "output_structure": _same([row["output_structure"] for row in combined]),
        "workload": _same([row["workload"] for row in combined]),
        "preset": _same([row["preset"] for row in combined]),
        "dispatch_and_graph": _same([row["records"] for row in combined]),
        "graph_signature": _same([row["graph_sha256"] for row in combined]),
        "token_and_stopping": _same([row["signatures"] for row in combined]),
        "rng_and_cache": _same(
            [audits[side]["strict_exactness"] for side in SIDES]
        ),
        "candidate_evidence_consistent": _same(
            [
                row["candidate_evidence"]["candidate"]
                for row in runs["candidate"] + [audits["candidate"]]
            ]
        ),
    }
    aggregates: dict[str, Any] = {}
    for side in SIDES:
        aggregates[side] = {
            "process_wall_seconds": _aggregate(
                [row["process_wall_seconds"] for row in runs[side]]
            ),
            "request_wall_seconds": _aggregate(
                [row["request_wall_seconds"] for row in runs[side]]
            ),
            "main_model_seconds": _aggregate(
                [
                    row["generation"]["main_generation"][
                        "synchronized_model_seconds"
                    ]
                    for row in runs[side]
                ]
            ),
            "timing_model_seconds": _aggregate(
                [
                    row["generation"]["timing_context"][
                        "synchronized_model_seconds"
                    ]
                    for row in runs[side]
                ]
            ),
            "peak_cuda_memory_mb": max(
                row["peak_cuda_memory_mb"] for row in runs[side]
            ),
        }
    comparisons = {
        name: _comparison(
            aggregates["baseline"][name]["median"],
            aggregates["candidate"][name]["median"],
        )
        for name in (
            "process_wall_seconds",
            "request_wall_seconds",
            "main_model_seconds",
            "timing_model_seconds",
        )
    }
    checks.update(
        {
            "process_wall_no_more_than_one_percent_slower": comparisons[
                "process_wall_seconds"
            ]["candidate_delta_pct"]
            <= 1.0,
            "request_wall_no_more_than_one_percent_slower": comparisons[
                "request_wall_seconds"
            ]["candidate_delta_pct"]
            <= 1.0,
            "main_model_no_more_than_one_percent_slower": comparisons[
                "main_model_seconds"
            ]["candidate_delta_pct"]
            <= 1.0,
            "timing_model_no_more_than_one_percent_slower": comparisons[
                "timing_model_seconds"
            ]["candidate_delta_pct"]
            <= 1.0,
        }
    )
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "same_precision_drift": {
            "byte_identical_output": checks["output_bytes"],
            "identical_tokens_and_stopping": checks["token_and_stopping"],
            "identical_rng_and_cache": checks["rng_and_cache"],
            "identical_dispatch_and_graph": (
                checks["dispatch_and_graph"] and checks["graph_signature"]
            ),
        },
        "runs": runs,
        "exactness_audits": audits,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    if args.precision not in PRECISIONS:
        raise FullSongGateError("precision must be fp16 or fp32")
    result = _analyze_precision(
        args.run_root,
        precision=args.precision,
        baseline_commit=args.baseline_commit,
        baseline_branch=args.baseline_branch,
        candidate_commit=args.candidate_commit,
        candidate_branch=args.candidate_branch,
    )
    return {
        "schema_version": 1,
        "precision": args.precision,
        "status": result["status"],
        "result": result,
    }


def _text(report: dict[str, Any]) -> str:
    rows = [f"status={report['status']}"]
    precision = report["precision"]
    result = report["result"]
    rows.append(f"precision={precision}")
    for check, passed in result["checks"].items():
        rows.append(f"check.{check}={str(passed).lower()}")
    for name, comparison in result["comparisons"].items():
        rows.append(
            f"{precision}.{name}.baseline_seconds="
            f"{comparison['baseline_seconds']:.6f}"
        )
        rows.append(
            f"{precision}.{name}.candidate_seconds="
            f"{comparison['candidate_seconds']:.6f}"
        )
        rows.append(
            f"{precision}.{name}.saved_seconds="
            f"{comparison['saved_seconds']:.6f}"
        )
        rows.append(
            f"{precision}.{name}.candidate_delta_pct="
            f"{comparison['candidate_delta_pct']:.3f}"
        )
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--precision", choices=PRECISIONS, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--baseline-branch", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--candidate-branch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.text_output.write_text(_text(report))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
