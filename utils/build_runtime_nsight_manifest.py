"""Build the strict Nsight manifest for one fixed-work selected runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import nsight_agent_profile as nsight


RUN_IDS = (
    "accepted_record",
    "selected_control",
    "selected_graph",
    "selected_node",
)
PASS_KINDS = {
    "accepted_record": "untraced_control",
    "selected_control": "untraced_control",
    "selected_graph": "nsys_graph",
    "selected_node": "nsys_node",
}
PAIRED_CONTROL = {
    "selected_graph": "selected_control",
    "selected_node": "selected_control",
}
EXPECTED_MAIN_STEPS = 8294


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise ValueError(f"artifact is outside run root: {path}") from exc


def _artifact(
    path: Path,
    root: Path,
    role: str,
    *,
    stage: str | None = None,
    allow_empty: bool = False,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if not allow_empty and path.stat().st_size <= 0:
        raise ValueError(f"required artifact is empty: {path}")
    item: dict[str, Any] = {
        "role": role,
        "path": _relative(path, root),
        "required": True,
        "allow_empty": allow_empty,
        "sha256": _sha256(path),
    }
    if stage is not None:
        item["stage"] = stage
    return item


def _profile_stage(
    profile: dict[str, Any],
    evidence: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    signature = nsight._profile_label_signature(profile, label)
    if signature["status"] != "available" or not signature["self_consistent"]:
        raise ValueError(f"profile label is unavailable or inconsistent: {label}")
    summary = profile.get("summary", {}).get("generation_by_label", {}).get(label)
    if not isinstance(summary, dict):
        raise ValueError(f"profile lacks generation summary for {label}")
    label_evidence = evidence.get("labels", {}).get(label)
    if not isinstance(label_evidence, dict):
        raise ValueError(f"evidence lacks fixed-work counts for {label}")
    logical_steps = int(label_evidence.get("logical_steps", 0))
    if logical_steps <= 0:
        raise ValueError(f"evidence has no logical work for {label}")
    graph = nsight._profile_graph_cache_signature(profile, [label])
    return {
        "outer_wall_seconds": float(summary["wall_seconds"]),
        "synchronized_model_seconds": float(summary["model_elapsed_seconds"]),
        "generated_tokens": logical_steps,
        "token_ids_sha256": signature["token_stream_sha256"],
        "stopping_sha256": signature["stopping_sha256"],
        "cache_behavior_sha256": graph["sha256"],
        "graph_attribution": {
            "method": "none",
            "verified_on_replay": False,
            "kernel_regions": {},
        },
    }


def _pipeline(profile: dict[str, Any]) -> dict[str, int]:
    raw = profile.get("summary", {}).get("stage_wall_seconds")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("profile lacks stage wall timing")
    result = {}
    for name, value in raw.items():
        seconds = float(value)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"invalid stage wall for {name}: {value}")
        result[str(name)] = int(round(seconds * 1_000_000_000))
    return result


def _run(
    run_id: str,
    evidence_path: Path,
    *,
    root: Path,
    commit: str,
    audio_sha256: str,
    fixed_manifest_sha256: str,
    runtime_spec_sha256: str,
) -> dict[str, Any]:
    evidence = _load(evidence_path)
    if int(evidence.get("expected_main_steps", 0)) != EXPECTED_MAIN_STEPS:
        raise ValueError(f"{run_id} is not fixed to {EXPECTED_MAIN_STEPS} main steps")
    if int(evidence.get("labels", {}).get("main_generation", {}).get("logical_steps", 0)) != EXPECTED_MAIN_STEPS:
        raise ValueError(f"{run_id} did not execute {EXPECTED_MAIN_STEPS} main steps")
    profile_path = Path(evidence["profile_path"]).resolve()
    result_path = Path(evidence["result_path"]).resolve()
    if _sha256(profile_path) != evidence.get("profile_sha256"):
        raise ValueError(f"{run_id} profile digest changed")
    if _sha256(result_path) != evidence.get("result_sha256"):
        raise ValueError(f"{run_id} result digest changed")
    profile = nsight._load_inference_profile(profile_path)
    pass_kind = PASS_KINDS[run_id]
    if profile.get("metadata", {}).get("profile_pass_kind") != pass_kind:
        raise ValueError(f"{run_id} profile pass kind is not {pass_kind}")
    candidate = run_id != "accepted_record"
    if bool(evidence.get("candidate")) != candidate:
        raise ValueError(f"{run_id} candidate identity is inconsistent")
    if _sha256(Path(evidence["manifest_path"])) != fixed_manifest_sha256:
        raise ValueError(f"{run_id} fixed-work manifest changed")
    if candidate:
        if evidence.get("runtime_spec_sha256") != runtime_spec_sha256:
            raise ValueError(f"{run_id} runtime spec changed")
        runtime_name = str(evidence.get("runtime", {}).get("name"))
    else:
        runtime_name = "accepted-optimized"

    run_dir = evidence_path.parent
    artifacts = [
        _artifact(profile_path, root, "inference_profile"),
        _artifact(result_path, root, "result_osu"),
        _artifact(evidence_path, root, "run_evidence"),
        _artifact(run_dir / "stdout.txt", root, "process_stdout", allow_empty=True),
        _artifact(run_dir / "stderr.txt", root, "process_stderr", allow_empty=True),
    ]
    if candidate:
        runtime_evidence_path = run_dir / "runtime-evidence.json"
        runtime_evidence = _load(runtime_evidence_path)
        if runtime_evidence != evidence.get("runtime"):
            raise ValueError(f"{run_id} runtime evidence diverged from run evidence")
        artifacts.append(
            _artifact(runtime_evidence_path, root, "runtime_initialization_evidence")
        )
    stages = {
        "timing_generation": _profile_stage(
            profile,
            evidence,
            label="timing_context",
        ),
        "main_generation": _profile_stage(
            profile,
            evidence,
            label="main_generation",
        ),
    }
    if pass_kind in {"nsys_graph", "nsys_node"}:
        nsys_dir = run_dir / "nsys"
        artifacts.extend(
            [
                _artifact(nsys_dir / "trace.nsys-rep", root, "nsys_rep"),
                _artifact(nsys_dir / "trace.sqlite", root, "sqlite"),
                _artifact(nsys_dir / "cuda_api_sum.csv", root, "nsys_cuda_api_sum"),
                _artifact(nsys_dir / "nvtx_sum.csv", root, "nsys_nvtx_sum"),
            ]
        )
        extraction_path = nsys_dir / "stages" / "extraction.json"
        extraction = _load(extraction_path)
        artifacts.append(_artifact(extraction_path, root, "stage_sqlite_extraction"))
        for extracted in extraction.get("artifacts", []):
            path = Path(extracted["path"])
            item = _artifact(
                path,
                root,
                str(extracted["role"]),
                stage=str(extracted["stage"]),
                allow_empty=bool(extracted.get("allow_empty_rows", False)),
            )
            item["row_count"] = int(extracted["row_count"])
            item["allow_empty_rows"] = bool(extracted.get("allow_empty_rows", False))
            artifacts.append(item)
        for stage, extracted in extraction.get("stages", {}).items():
            stages[stage]["trace_window_start_ns"] = int(extracted["start_ns"])
            stages[stage]["trace_window_end_ns"] = int(extracted["end_ns"])
            stages[stage]["diagnostic_counts"] = dict(extracted["counts"])

    workload = {
        "commit": commit,
        "audio_sha256": audio_sha256,
        "config_name": "profile_salvalai",
        "precision": "fp32",
        "seed": 12345,
        "engine": "optimized",
        "attn_implementation": "sdpa",
        "expected_main_steps": EXPECTED_MAIN_STEPS,
        "fixed_manifest_sha256": fixed_manifest_sha256,
        "runtime_spec_sha256": runtime_spec_sha256 if candidate else None,
        "runtime_name": runtime_name,
    }
    metadata = profile["metadata"]
    return {
        "run_id": run_id,
        "run_group_id": "selected-runtime-fixed8294" if candidate else "accepted-fixed8294",
        "precision": "fp32",
        "engine_variant": runtime_name,
        "engine_preset_version": metadata.get("optimized_effective_config_version"),
        "pass_kind": pass_kind,
        "authoritative_performance": pass_kind == "untraced_control",
        "paired_control_run_id": PAIRED_CONTROL.get(run_id),
        "workload_contract": workload,
        "output_sha256": _sha256(result_path),
        "output_size_bytes": result_path.stat().st_size,
        "output_structure": nsight.summarize_osu_structure(result_path),
        "pipeline_stage_wall_ns": _pipeline(profile),
        "identity": {
            "source_evidence": _relative(evidence_path, root),
            "source_profile": _relative(profile_path, root),
            "fixed_work_main_steps": EXPECTED_MAIN_STEPS,
            "nsys_capture_scope": (
                "timing_and_main_generation"
                if pass_kind in {"nsys_graph", "nsys_node"}
                else None
            ),
            "stage_kernel_extraction": (
                "exact NVTX wall containment" if pass_kind == "nsys_node" else None
            ),
        },
        "stages": stages,
        "artifacts": artifacts,
    }


def build(
    *,
    run_root: Path,
    evidence_paths: dict[str, Path],
    commit: str,
    branch: str,
    remote_ref: str,
    audio: Path,
    runtime_spec: Path,
    fixed_manifest: Path,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    runtime_spec_sha256 = _sha256(runtime_spec)
    fixed_manifest_sha256 = _sha256(fixed_manifest)
    audio_sha256 = _sha256(audio)
    runs = [
        _run(
            run_id,
            evidence_paths[run_id],
            root=run_root,
            commit=commit,
            audio_sha256=audio_sha256,
            fixed_manifest_sha256=fixed_manifest_sha256,
            runtime_spec_sha256=runtime_spec_sha256,
        )
        for run_id in RUN_IDS
    ]
    return {
        "schema_version": nsight.MANIFEST_SCHEMA_VERSION,
        "provenance": {
            "commit": commit,
            "branch": branch,
            "remote_ref": remote_ref,
            "slurm_job_id": str(__import__("os").environ.get("SLURM_JOB_ID", "unknown")),
            "gpu": "NVIDIA GeForce RTX 2080 Ti",
            "runtime_spec_sha256": runtime_spec_sha256,
            "fixed_manifest_sha256": fixed_manifest_sha256,
            "expected_main_steps": EXPECTED_MAIN_STEPS,
        },
        "tool_capabilities": {},
        "runs": runs,
        # Paired-control comparisons are derived from each traced run's
        # ``paired_control_run_id`` by the analyzer.  The explicit comparison
        # list is reserved for cross-precision requests.
        "comparisons": [],
        "ncu_probe": {"attempted": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--remote-ref", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--runtime-spec", type=Path, required=True)
    parser.add_argument("--fixed-manifest", type=Path, required=True)
    parser.add_argument("--accepted-evidence", type=Path, required=True)
    parser.add_argument("--selected-control-evidence", type=Path, required=True)
    parser.add_argument("--selected-graph-evidence", type=Path, required=True)
    parser.add_argument("--selected-node-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = {
        "accepted_record": args.accepted_evidence,
        "selected_control": args.selected_control_evidence,
        "selected_graph": args.selected_graph_evidence,
        "selected_node": args.selected_node_evidence,
    }
    payload = build(
        run_root=args.run_root,
        evidence_paths=evidence,
        commit=args.commit,
        branch=args.branch,
        remote_ref=args.remote_ref,
        audio=args.audio,
        runtime_spec=args.runtime_spec,
        fixed_manifest=args.fixed_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
