"""Fail-loud analysis for five fresh accepted-FP16 controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils import analyze_fp32_fresh_baseline as common
from utils import nsight_agent_profile as nsight


SCHEMA_VERSION = "mapperatorinator.fp16-fresh-baseline.v1"
LABELS = ("timing_context", "main_generation")
EXPECTED_PRESET_VERSION = "accepted-fp16-all-fused-v2"
SPLIT_KV_PREFIXES = tuple(range(192, 833, 64))


class FP16BaselineError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FP16BaselineError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        raise FP16BaselineError(f"{name} must be finite and non-negative")
    return result


def _runtime_contract(
    profile: Mapping[str, Any],
    *,
    expected_commit: str,
    expected_branch: str,
    expected_preset_version: str = EXPECTED_PRESET_VERSION,
    expected_split_kv: bool = False,
    audit: bool,
) -> None:
    metadata = profile.get("metadata")
    if not isinstance(metadata, dict):
        raise FP16BaselineError("profile metadata is missing")
    expected = {
        "precision": "fp16",
        "inference_engine": "optimized",
        "attn_implementation": "sdpa",
        "profile_pass_kind": "exactness_audit" if audit else "untraced_control",
        "authoritative_performance": not audit,
        "strict_exactness_evidence": audit,
        "nvidia_tf32_override": "0",
        "git_commit": expected_commit,
        "git_branch": expected_branch,
    }
    failures = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if failures:
        raise FP16BaselineError(f"FP16 runtime contract changed: {failures}")
    if "2080 Ti" not in str(metadata.get("cuda_device_name", "")):
        raise FP16BaselineError("FP16 baseline did not run on an RTX 2080 Ti")
    if metadata.get("cuda_device_capability") != [7, 5]:
        raise FP16BaselineError("FP16 baseline requires SM75")
    effective = metadata.get("optimized_effective_config")
    if not isinstance(effective, dict):
        raise FP16BaselineError("optimized effective config is missing")
    required_config = {
        "version": expected_preset_version,
        "precision": "fp16",
        "attn_implementation": "sdpa",
        "decoder_loop_backend": "active_prefix_cuda_graph",
        "batch_size": 1,
        "q1_bmm_cross_attention": True,
        "native_q1_self_attention": True,
        "native_q1_rope_cache_self_attention": True,
        "native_cross_mlp_tail": True,
    }
    config_failures = {
        key: {"expected": value, "actual": effective.get(key)}
        for key, value in required_config.items()
        if effective.get(key) != value
    }
    if config_failures:
        raise FP16BaselineError(
            f"accepted shared-specialized FP16 preset changed: {config_failures}"
        )
    if expected_split_kv:
        expected_split_config = {
            "native_q1_rope_cache_split_kv": True,
            "native_q1_rope_cache_split_kv_split_count": 8,
            "native_q1_rope_cache_split_kv_prefix_buckets": list(
                SPLIT_KV_PREFIXES
            ),
        }
        split_failures = {
            key: {"expected": value, "actual": effective.get(key)}
            for key, value in expected_split_config.items()
            if effective.get(key) != value
        }
        if split_failures:
            raise FP16BaselineError(
                f"FP16 split-KV config changed: {split_failures}"
            )
    precisions = {
        row.get("precision")
        for row in profile.get("generation", [])
        if isinstance(row, dict) and row.get("profile_label") in LABELS
    }
    if precisions != {"fp16"}:
        raise FP16BaselineError(f"generation precision changed: {precisions}")
    if expected_split_kv:
        records = [
            row
            for row in profile.get("generation", [])
            if isinstance(row, dict) and row.get("profile_label") in LABELS
        ]
        hit_maps = [
            row.get("optimized_dispatch_capture_hits")
            for row in records
            if isinstance(row.get("optimized_dispatch_capture_hits"), dict)
        ]
        aggregate_key = "native_q1_rope_cache_self_attention_split_kv_8"
        aggregate_hits = sum(int(value.get(aggregate_key, 0)) for value in hit_maps)
        prefix_hits = {
            prefix: sum(
                int(
                    value.get(
                        f"{aggregate_key}_prefix_{prefix}",
                        0,
                    )
                )
                for value in hit_maps
            )
            for prefix in SPLIT_KV_PREFIXES
        }
        if aggregate_hits <= 0 or sum(prefix_hits.values()) != aggregate_hits:
            raise FP16BaselineError(
                "FP16 split-KV hit metadata is missing or inconsistent: "
                f"aggregate={aggregate_hits}, prefixes={prefix_hits}"
            )


def _profile_and_result(run_dir: Path) -> tuple[dict[str, Any], Path]:
    try:
        profile_path = common._path_from_pointer(run_dir, "profile-path.txt")
        result_path = common._path_from_pointer(run_dir, "result-path.txt")
        profile = nsight._load_inference_profile(profile_path)
    except (common.BaselineError, ValueError) as exc:
        raise FP16BaselineError(str(exc)) from exc
    return profile, result_path


def _signature(profile: Mapping[str, Any], label: str) -> dict[str, Any]:
    value = nsight._profile_label_signature(profile, label)
    if value.get("status") != "available" or not value.get("self_consistent"):
        raise FP16BaselineError(f"{label} token/stopping signature is unavailable")
    return value


def _token_groups(profile: Mapping[str, Any], label: str) -> list[list[list[int]]]:
    material: list[list[list[int]]] = []
    for index, record in enumerate(profile.get("generation", [])):
        if not isinstance(record, dict) or record.get("profile_label") != label:
            continue
        groups = nsight._record_token_groups(record)
        if groups is None:
            raise FP16BaselineError(f"{label} record {index} has no token IDs")
        material.append(groups)
    if not material:
        raise FP16BaselineError(f"{label} token material is empty")
    return material


def _metrics(profile: Mapping[str, Any], label: str) -> dict[str, Any]:
    try:
        return common._label_metrics(profile, label)
    except common.BaselineError as exc:
        raise FP16BaselineError(str(exc)) from exc


def _aggregate(values: Sequence[float], *, expected_count: int) -> dict[str, float]:
    if len(values) != expected_count:
        raise FP16BaselineError(
            f"aggregate requires {expected_count} fresh values"
        )
    return {
        "median": float(statistics.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
        "range": float(max(values) - min(values)),
    }


def _process_wall(run_dir: Path) -> float:
    try:
        return float(common._wall_artifact(run_dir)["elapsed_seconds"])
    except common.BaselineError as exc:
        raise FP16BaselineError(str(exc)) from exc


def _request_wall(profile: Mapping[str, Any]) -> float:
    try:
        return common._request_wall_seconds(profile)
    except common.BaselineError as exc:
        raise FP16BaselineError(str(exc)) from exc


def analyze(
    run_root: Path,
    *,
    expected_commit: str,
    expected_branch: str,
    expected_preset_version: str = EXPECTED_PRESET_VERSION,
    expected_split_kv: bool = False,
    run_count: int = 5,
) -> dict[str, Any]:
    if len(expected_commit) != 40:
        raise FP16BaselineError("expected commit must be a full SHA")
    if run_count not in {1, 5}:
        raise FP16BaselineError("run_count must be one initial or five confirmation runs")
    run_names = tuple(f"run-{index:02d}" for index in range(1, run_count + 1))
    runs: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    results: list[Path] = []
    signatures: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}
    workloads: list[dict[str, Any]] = []
    presets: list[dict[str, Any]] = []
    records: list[list[dict[str, Any]]] = []
    structures: list[dict[str, Any]] = []
    output_hashes: list[str] = []
    token_counts: dict[str, list[int]] = {label: [] for label in LABELS}

    for run_name in run_names:
        run_dir = run_root / run_name
        profile, result = _profile_and_result(run_dir)
        _runtime_contract(
            profile,
            expected_commit=expected_commit,
            expected_branch=expected_branch,
            expected_preset_version=expected_preset_version,
            expected_split_kv=expected_split_kv,
            audit=False,
        )
        profile_hash = profile["metadata"].get("result_file_sha256")
        output_hash = _file_hash(result)
        if profile_hash != output_hash:
            raise FP16BaselineError(f"{run_name} profile/output hash mismatch")
        structure = nsight.summarize_osu_structure(result)
        if structure.get("malformed_by_section") != {
            "TimingPoints": 0,
            "HitObjects": 0,
        } or structure.get("nonfinite_values") != 0:
            raise FP16BaselineError(f"{run_name} produced an invalid map")
        label_metrics = {label: _metrics(profile, label) for label in LABELS}
        label_signatures = {
            label: _signature(profile, label) for label in LABELS
        }
        for label in LABELS:
            signatures[label].append(label_signatures[label])
            token_counts[label].append(label_metrics[label]["generated_tokens"])
        try:
            workload = common._workload_contract(profile)
            preset = common._preset_contract(profile)
            record = common._record_contract(profile)
            peak_memory = common._peak_memory_mb(profile)
        except common.BaselineError as exc:
            raise FP16BaselineError(str(exc)) from exc
        workloads.append(workload)
        presets.append(preset)
        records.append(record)
        structures.append(structure)
        output_hashes.append(output_hash)
        profiles.append(profile)
        results.append(result)
        runs.append(
            {
                "run_id": run_name,
                "process_wall_seconds": _process_wall(run_dir),
                "request_to_final_osu_wall_seconds": _request_wall(profile),
                "generation": label_metrics,
                "peak_cuda_memory_mb": peak_memory,
                "output_sha256": output_hash,
                "output_structure": structure,
                "dispatch_and_graph_contract_sha256": _hash(record),
                "token_signatures": {
                    label: {
                        "generated_tokens": label_signatures[label][
                            "generated_tokens"
                        ],
                        "token_stream_sha256": label_signatures[label][
                            "token_stream_sha256"
                        ],
                        "stopping_sha256": label_signatures[label][
                            "stopping_sha256"
                        ],
                        "token_groups_by_record": _token_groups(profile, label),
                    }
                    for label in LABELS
                },
            }
        )

    audits: list[dict[str, Any]] = []
    for audit_index in (1, 2):
        audit_dir = run_root / f"exactness-audit-{audit_index:02d}"
        audit_profile, audit_result = _profile_and_result(audit_dir)
        _runtime_contract(
            audit_profile,
            expected_commit=expected_commit,
            expected_branch=expected_branch,
            expected_preset_version=expected_preset_version,
            expected_split_kv=expected_split_kv,
            audit=True,
        )
        try:
            exactness = common._strict_exactness_contract(audit_profile)
            record = common._record_contract(audit_profile)
        except common.BaselineError as exc:
            raise FP16BaselineError(str(exc)) from exc
        audits.append(
            {
                "run_id": audit_dir.name,
                "run_dir": audit_dir,
                "profile": audit_profile,
                "output_sha256": _file_hash(audit_result),
                "structure": nsight.summarize_osu_structure(audit_result),
                "signatures": {
                    label: {
                        **_signature(audit_profile, label),
                        "token_groups_by_record": _token_groups(
                            audit_profile,
                            label,
                        ),
                    }
                    for label in LABELS
                },
                "strict_exactness": exactness,
                "record_contract": record,
            }
        )

    checks: dict[str, bool] = {
        "workload_metadata": len({_hash(value) for value in workloads}) == 1,
        "preset_metadata": len({_hash(value) for value in presets}) == 1,
        "dispatch_and_graph_contract": len({_hash(value) for value in records}) == 1,
        "fixed_work_timing": len(set(token_counts["timing_context"])) == 1,
        "fixed_work_main": len(set(token_counts["main_generation"])) == 1,
        "output_bytes": len(set(output_hashes)) == 1,
        "output_structure": len({_hash(value) for value in structures}) == 1,
        "audit_workload": all(
            common._workload_contract(audit["profile"]) == workloads[0]
            for audit in audits
        ),
        "audit_preset": all(
            common._preset_contract(audit["profile"]) == presets[0]
            for audit in audits
        ),
        "audit_dispatch_and_graph": all(
            audit["record_contract"] == records[0] for audit in audits
        ),
        "audit_output_bytes": all(
            audit["output_sha256"] == output_hashes[0] for audit in audits
        ),
        "audit_output_structure": all(
            audit["structure"] == structures[0] for audit in audits
        ),
        "audit_rng_cache_repeat": (
            audits[0]["strict_exactness"] == audits[1]["strict_exactness"]
        ),
    }
    for label in LABELS:
        checks[f"{label}_tokens"] = len(
            {value["token_stream_sha256"] for value in signatures[label]}
        ) == 1
        checks[f"{label}_stopping"] = len(
            {value["stopping_sha256"] for value in signatures[label]}
        ) == 1
        checks[f"audit_{label}_tokens"] = all(
            audit["signatures"][label]["token_stream_sha256"]
            == signatures[label][0]["token_stream_sha256"]
            for audit in audits
        )
        checks[f"audit_{label}_stopping"] = all(
            audit["signatures"][label]["stopping_sha256"]
            == signatures[label][0]["stopping_sha256"]
            for audit in audits
        )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise FP16BaselineError(f"FP16 fresh baseline checks failed: {failed}")

    aggregates = {
        "process_wall_seconds": _aggregate(
            [run["process_wall_seconds"] for run in runs],
            expected_count=run_count,
        ),
        "request_to_final_osu_wall_seconds": _aggregate(
            [run["request_to_final_osu_wall_seconds"] for run in runs],
            expected_count=run_count,
        ),
        "peak_cuda_memory_mb": _aggregate(
            [run["peak_cuda_memory_mb"] for run in runs],
            expected_count=run_count,
        ),
    }
    for label in LABELS:
        prefix = "timing" if label == "timing_context" else "main"
        aggregates[f"{prefix}_synchronized_model_seconds"] = _aggregate(
            [
                run["generation"][label]["synchronized_model_seconds"]
                for run in runs
            ],
            expected_count=run_count,
        )
        aggregates[f"{prefix}_outer_wall_seconds"] = _aggregate(
            [run["generation"][label]["outer_wall_seconds"] for run in runs],
            expected_count=run_count,
        )
        aggregates[f"{prefix}_tokens_per_second"] = _aggregate(
            [run["generation"][label]["tokens_per_second"] for run in runs],
            expected_count=run_count,
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "precision": "fp16",
        "runtime_identity": {
            "commit": expected_commit,
            "branch": expected_branch,
        },
        "fresh_processes": True,
        "run_count": len(runs),
        "authoritative_performance": True,
        "preset_version": expected_preset_version,
        "split_kv": expected_split_kv,
        "fixed_work": {
            "timing_tokens": token_counts["timing_context"][0],
            "main_tokens": token_counts["main_generation"][0],
        },
        "checks": checks,
        "runs": runs,
        "aggregates": aggregates,
        "exactness_audits": [
            {
                "run_id": audit["run_id"],
                "authoritative_performance": False,
                "process_wall_seconds": _process_wall(audit["run_dir"]),
                "output_sha256": audit["output_sha256"],
                "strict_exactness": audit["strict_exactness"],
                "token_signatures": {
                    label: {
                        "generated_tokens": audit["signatures"][label][
                            "generated_tokens"
                        ],
                        "token_stream_sha256": audit["signatures"][label][
                            "token_stream_sha256"
                        ],
                        "stopping_sha256": audit["signatures"][label][
                            "stopping_sha256"
                        ],
                        "token_groups_by_record": audit["signatures"][label][
                            "token_groups_by_record"
                        ],
                    }
                    for label in LABELS
                },
            }
            for audit in audits
        ],
    }


def _text(result: Mapping[str, Any]) -> str:
    aggregates = result["aggregates"]
    return "\n".join(
        (
            "status=PASS",
            "precision=fp16",
            f"preset={result['preset_version']}",
            f"fixed_work_timing_tokens={result['fixed_work']['timing_tokens']}",
            f"fixed_work_main_tokens={result['fixed_work']['main_tokens']}",
            f"timing_model_seconds_median={aggregates['timing_synchronized_model_seconds']['median']:.6f}",
            f"timing_tps_median={aggregates['timing_tokens_per_second']['median']:.6f}",
            f"main_model_seconds_median={aggregates['main_synchronized_model_seconds']['median']:.6f}",
            f"main_tps_median={aggregates['main_tokens_per_second']['median']:.6f}",
            f"request_wall_seconds_median={aggregates['request_to_final_osu_wall_seconds']['median']:.6f}",
            f"process_wall_seconds_median={aggregates['process_wall_seconds']['median']:.6f}",
        )
    ) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument(
        "--expected-preset-version",
        default=EXPECTED_PRESET_VERSION,
    )
    parser.add_argument("--expected-split-kv", action="store_true")
    parser.add_argument("--run-count", type=int, choices=(1, 5), default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(
        args.run_root,
        expected_commit=args.expected_commit,
        expected_branch=args.expected_branch,
        expected_preset_version=args.expected_preset_version,
        expected_split_kv=args.expected_split_kv,
        run_count=args.run_count,
    )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.text_output.write_text(_text(result), encoding="utf-8")
    print(_text(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
