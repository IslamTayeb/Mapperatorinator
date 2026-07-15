"""Summarize fixed-work runtime timing, divergence, and Nsight hot spots."""

from __future__ import annotations

import argparse
import csv
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


EXPECTED_MAIN_STEPS = 7149


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile(evidence: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = Path(evidence["profile_path"])
    if _sha256(path) != evidence.get("profile_sha256"):
        raise ValueError(f"profile digest changed: {path}")
    return path, nsight._load_inference_profile(path)


def _logical_steps(evidence: dict[str, Any], label: str) -> int:
    value = int(evidence.get("labels", {}).get(label, {}).get("logical_steps", 0))
    if value <= 0:
        raise ValueError(f"missing fixed logical work for {label}")
    return value


def _stage(
    profile: dict[str, Any],
    evidence: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    summary = profile.get("summary", {}).get("generation_by_label", {}).get(label)
    if not isinstance(summary, dict):
        raise ValueError(f"profile is missing generation summary {label}")
    model = float(summary["model_elapsed_seconds"])
    outer = float(summary["wall_seconds"])
    logical = _logical_steps(evidence, label)
    if not math.isfinite(model) or model <= 0 or not math.isfinite(outer) or outer <= 0:
        raise ValueError(f"invalid timing for {label}")
    return {
        "records": int(summary["records"]),
        "logical_steps": logical,
        "consumer_tokens": int(summary["generated_tokens"]),
        "outer_wall_seconds": outer,
        "synchronized_model_seconds": model,
        "post_model_seconds": max(0.0, outer - model),
        "fixed_work_tokens_per_second": logical / model,
    }


def _pipeline(profile: dict[str, Any]) -> dict[str, Any]:
    walls = profile.get("summary", {}).get("stage_wall_seconds")
    if not isinstance(walls, dict) or not walls:
        raise ValueError("profile has no pipeline stage timing")
    stages = {str(name): float(value) for name, value in walls.items()}
    timing_post = sum(stages.get(name, 0.0) for name in nsight.TIMING_POSTPROCESS_STAGES)
    main_post = sum(stages.get(name, 0.0) for name in nsight.MAIN_POSTPROCESS_STAGES)
    request = sum(stages.values())
    generation = stages.get("timing_context_generation", 0.0) + stages.get(
        "main_generation", 0.0
    )
    return {
        "stage_wall_seconds": dict(sorted(stages.items())),
        "request_profiled_stage_wall_seconds": request,
        "timing_postprocessing_seconds": timing_post,
        "main_postprocessing_seconds": main_post,
        "generation_stage_wall_seconds": generation,
        "other_profiled_stage_wall_seconds": max(
            0.0,
            request - generation - timing_post - main_post,
        ),
    }


def _token_groups(profile: dict[str, Any], label: str) -> list[list[int]]:
    groups: list[list[int]] = []
    for record in profile.get("generation", []):
        if isinstance(record, dict) and record.get("profile_label") == label:
            record_groups = nsight._record_token_groups(record)
            if record_groups is None:
                raise ValueError(f"profile lacks token IDs for {label}")
            groups.extend([[int(value) for value in group] for group in record_groups])
    if not groups:
        raise ValueError(f"profile has no token groups for {label}")
    return groups


def _divergence(
    accepted_profile: dict[str, Any],
    selected_profile: dict[str, Any],
    accepted_result: Path,
    selected_result: Path,
) -> dict[str, Any]:
    labels = {}
    for label in nsight.PROFILE_LABELS:
        left = _token_groups(accepted_profile, label)
        right = _token_groups(selected_profile, label)
        flattened_left = [token for group in left for token in group]
        flattened_right = [token for group in right for token in group]
        aligned = min(len(flattened_left), len(flattened_right))
        mismatches = sum(
            left_token != right_token
            for left_token, right_token in zip(
                flattened_left[:aligned],
                flattened_right[:aligned],
                strict=True,
            )
        )
        left_signature = nsight._profile_label_signature(accepted_profile, label)
        right_signature = nsight._profile_label_signature(selected_profile, label)
        labels[label] = {
            "accepted_consumer_tokens": len(flattened_left),
            "selected_consumer_tokens": len(flattened_right),
            "aligned_tokens": aligned,
            "aligned_mismatches": mismatches,
            "aligned_mismatch_fraction": mismatches / aligned if aligned else None,
            "token_stream_equal": (
                left_signature["token_stream_sha256"]
                == right_signature["token_stream_sha256"]
            ),
            "stopping_equal": (
                left_signature["stopping_sha256"]
                == right_signature["stopping_sha256"]
            ),
        }
    accepted_structure = nsight.summarize_osu_structure(accepted_result)
    selected_structure = nsight.summarize_osu_structure(selected_result)
    return {
        "classification": "relaxed_runtime_vs_accepted_fp32",
        "labels": labels,
        "result_byte_identical": _sha256(accepted_result) == _sha256(selected_result),
        "accepted_result_sha256": _sha256(accepted_result),
        "selected_result_sha256": _sha256(selected_result),
        "accepted_structure": accepted_structure,
        "selected_structure": selected_structure,
        "finite_and_well_formed": bool(
            selected_structure.get("finite_and_well_formed", False)
        ),
    }


def _write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def _export_csvs(
    report: dict[str, Any],
    nsight_analysis: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    stage_rows = []
    for run_id, run in report["runs"].items():
        for stage, timing in run["generation"].items():
            stage_rows.append(
                [
                    run_id,
                    stage,
                    timing["logical_steps"],
                    timing["consumer_tokens"],
                    timing["outer_wall_seconds"],
                    timing["synchronized_model_seconds"],
                    timing["post_model_seconds"],
                    timing["fixed_work_tokens_per_second"],
                ]
            )
    stage_path = output_dir / "stage_budget.csv"
    _write_csv(
        stage_path,
        [
            "run_id",
            "stage",
            "logical_steps",
            "consumer_tokens",
            "outer_wall_seconds",
            "synchronized_model_seconds",
            "post_model_seconds",
            "fixed_work_tokens_per_second",
        ],
        stage_rows,
    )

    request_rows = []
    for run_id, run in report["runs"].items():
        timing = run["generation"]["timing_generation"]
        main = run["generation"]["main_generation"]
        pipeline = run["pipeline"]
        request_rows.append(
            [
                run_id,
                pipeline["request_profiled_stage_wall_seconds"],
                timing["outer_wall_seconds"],
                timing["synchronized_model_seconds"],
                timing["post_model_seconds"],
                pipeline["timing_postprocessing_seconds"],
                main["outer_wall_seconds"],
                main["synchronized_model_seconds"],
                main["post_model_seconds"],
                pipeline["main_postprocessing_seconds"],
                pipeline["other_profiled_stage_wall_seconds"],
            ]
        )
    request_path = output_dir / "request_pipeline.csv"
    _write_csv(
        request_path,
        [
            "run_id",
            "complete_profiled_request_wall_seconds",
            "timing_generation_outer_wall_seconds",
            "timing_generation_synchronized_model_seconds",
            "timing_generation_post_model_seconds",
            "timing_postprocessing_seconds",
            "main_generation_outer_wall_seconds",
            "main_generation_synchronized_model_seconds",
            "main_generation_post_model_seconds",
            "main_postprocessing_seconds",
            "other_profiled_stage_wall_seconds",
        ],
        request_rows,
    )

    node = nsight_analysis.get("runs", {}).get("selected_node", {})
    family_rows = []
    kernel_rows = []
    transition_rows = []
    for stage_name, stage in node.get("stages", {}).items():
        if not isinstance(stage, dict) or stage.get("status") != "available":
            continue
        for rank, family in enumerate(stage.get("kernel_families", []), start=1):
            family_rows.append(
                [
                    stage_name,
                    rank,
                    family["family"],
                    family["kernel_count"],
                    family["calls"],
                    family["total_ns"],
                    family["share_of_accumulated_kernel_time"].get("percent"),
                ]
            )
        for rank, kernel in enumerate(stage.get("top_kernels", []), start=1):
            kernel_rows.append(
                [
                    stage_name,
                    rank,
                    kernel["raw_name"],
                    kernel["family"],
                    kernel["calls"],
                    kernel["total_ns"],
                    kernel.get("average_ns"),
                    json.dumps(kernel.get("launch_shapes", []), sort_keys=True),
                ]
            )
        trace = stage.get("trace_window", {})
        cuda_api = stage.get("cuda_api_report", {})
        sync = cuda_api.get("synchronization", {}) if isinstance(cuda_api, dict) else {}
        memory = stage.get("cuda_memory_size_report", {})
        transition_rows.append(
            [
                stage_name,
                trace.get("wall_ns"),
                trace.get("gpu_busy_union_ns"),
                trace.get("gpu_idle_gap_union_ns"),
                sync.get("calls"),
                sync.get("total_ns"),
                memory.get("total_calls"),
                memory.get("total_bytes"),
                stage.get("diagnostic_counts", {}).get("kernel_rows"),
                stage.get("diagnostic_counts", {}).get("runtime_rows"),
            ]
        )
    family_path = output_dir / "kernel_families.csv"
    _write_csv(
        family_path,
        ["stage", "rank", "family", "kernel_count", "calls", "total_ns", "share_percent"],
        family_rows,
    )
    kernel_path = output_dir / "top_kernels.csv"
    _write_csv(
        kernel_path,
        ["stage", "rank", "raw_name", "family", "calls", "total_ns", "average_ns", "launch_shapes_json"],
        kernel_rows,
    )
    transition_path = output_dir / "copy_sync_launch_gaps.csv"
    _write_csv(
        transition_path,
        [
            "stage",
            "trace_wall_ns",
            "gpu_busy_union_ns",
            "gpu_idle_gap_union_ns",
            "cuda_sync_calls",
            "cuda_sync_total_ns",
            "memory_operations",
            "memory_bytes",
            "kernel_launch_rows",
            "cuda_runtime_rows",
        ],
        transition_rows,
    )
    return {
        "stage_budget_csv": str(stage_path.resolve()),
        "request_pipeline_csv": str(request_path.resolve()),
        "kernel_families_csv": str(family_path.resolve()),
        "top_kernels_csv": str(kernel_path.resolve()),
        "copy_sync_launch_gaps_csv": str(transition_path.resolve()),
    }


def summarize(
    *,
    evidence_paths: dict[str, Path],
    topology_contract_path: Path,
    transparency_paths: dict[str, Path],
    nsight_analysis_path: Path,
    csv_dir: Path,
) -> dict[str, Any]:
    evidence = {name: _load(path) for name, path in evidence_paths.items()}
    profiles = {name: _profile(item)[1] for name, item in evidence.items()}
    for name, item in evidence.items():
        if int(item.get("expected_main_steps", 0)) != EXPECTED_MAIN_STEPS:
            raise ValueError(f"{name} is not a fixed-{EXPECTED_MAIN_STEPS} run")
        if _logical_steps(item, "main_generation") != EXPECTED_MAIN_STEPS:
            raise ValueError(f"{name} main logical work changed")
    selected_spec_hashes = {
        evidence[name].get("runtime_spec_sha256")
        for name in (
            "selected_control",
            "selected_graph",
            "selected_node",
        )
    }
    if len(selected_spec_hashes) != 1 or None in selected_spec_hashes:
        raise ValueError("selected runtime spec differs across runs")
    fixed_manifest_hashes = {
        evidence[name].get("manifest_sha256") for name in evidence
    }
    if len(fixed_manifest_hashes) != 1:
        raise ValueError("fixed-work manifest differs across runs")

    topology_contract = _load(topology_contract_path)
    if topology_contract.get("schema_version") != (
        "mapperatorinator.selected-runtime-nsight-topology.v1"
    ):
        raise ValueError("selected runtime topology contract has the wrong schema")
    if topology_contract.get("pass") is not True:
        raise ValueError("selected runtime topology contract did not pass")
    transparencies = {name: _load(path) for name, path in transparency_paths.items()}
    transparency_pass = all(bool(item.get("pass")) for item in transparencies.values())
    nsight_analysis = _load(nsight_analysis_path)
    nsight_comparisons_pass = all(
        bool(item.get("pass")) for item in nsight_analysis.get("comparisons", [])
    )

    runs = {}
    for name, profile in profiles.items():
        runs[name] = {
            "profile_pass_kind": profile["metadata"].get("profile_pass_kind"),
            "authoritative_performance": name in {"accepted_record", "selected_control"},
            "generation": {
                "timing_generation": _stage(profile, evidence[name], "timing_context"),
                "main_generation": _stage(profile, evidence[name], "main_generation"),
            },
            "pipeline": _pipeline(profile),
            "runtime": evidence[name].get("runtime"),
            "cuda_memory": evidence[name].get("cuda_memory"),
        }
    divergence = _divergence(
        profiles["accepted_record"],
        profiles["selected_control"],
        Path(evidence["accepted_record"]["result_path"]),
        Path(evidence["selected_control"]["result_path"]),
    )
    report = {
        "schema_version": "mapperatorinator.runtime-reprofile-summary.v1",
        "expected_main_steps": EXPECTED_MAIN_STEPS,
        "runtime_spec_sha256": next(iter(selected_spec_hashes)),
        "fixed_manifest_sha256": next(iter(fixed_manifest_hashes)),
        "runs": runs,
        "selected_runtime_topology": topology_contract,
        "profiler_transparency": transparencies,
        "accepted_vs_selected_divergence": divergence,
        "nsight_analysis_path": str(nsight_analysis_path.resolve()),
        "pass": transparency_pass and nsight_comparisons_pass,
    }
    report["csv_artifacts"] = _export_csvs(report, nsight_analysis, csv_dir)
    return report


def _text(report: dict[str, Any]) -> str:
    lines = [
        f"pass={str(report['pass']).lower()}",
        f"expected_main_steps={report['expected_main_steps']}",
        f"runtime_spec_sha256={report['runtime_spec_sha256']}",
    ]
    for run_id in ("accepted_record", "selected_control", "selected_graph", "selected_node"):
        run = report["runs"][run_id]
        pipeline = run["pipeline"]
        lines.append(
            f"{run_id}.request="
            f"profiled_wall_s:{pipeline['request_profiled_stage_wall_seconds']:.9f},"
            f"timing_post_s:{pipeline['timing_postprocessing_seconds']:.9f},"
            f"main_post_s:{pipeline['main_postprocessing_seconds']:.9f},"
            f"other_s:{pipeline['other_profiled_stage_wall_seconds']:.9f}"
        )
        for stage in ("timing_generation", "main_generation"):
            values = run["generation"][stage]
            lines.append(
                f"{run_id}.{stage}=logical:{values['logical_steps']},"
                f"model_s:{values['synchronized_model_seconds']:.9f},"
                f"outer_s:{values['outer_wall_seconds']:.9f},"
                f"fixed_tps:{values['fixed_work_tokens_per_second']:.6f}"
            )
    divergence = report["accepted_vs_selected_divergence"]
    lines.append(
        "accepted_vs_selected="
        f"byte_identical:{str(divergence['result_byte_identical']).lower()},"
        f"well_formed:{str(divergence['finite_and_well_formed']).lower()}"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-evidence", type=Path, required=True)
    parser.add_argument("--selected-control-evidence", type=Path, required=True)
    parser.add_argument("--selected-graph-evidence", type=Path, required=True)
    parser.add_argument("--selected-node-evidence", type=Path, required=True)
    parser.add_argument("--topology-contract", type=Path, required=True)
    parser.add_argument("--graph-transparency", type=Path, required=True)
    parser.add_argument("--node-transparency", type=Path, required=True)
    parser.add_argument("--nsight-analysis", type=Path, required=True)
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(
        evidence_paths={
            "accepted_record": args.accepted_evidence,
            "selected_control": args.selected_control_evidence,
            "selected_graph": args.selected_graph_evidence,
            "selected_node": args.selected_node_evidence,
        },
        topology_contract_path=args.topology_contract,
        transparency_paths={
            "graph": args.graph_transparency,
            "node": args.node_transparency,
        },
        nsight_analysis_path=args.nsight_analysis,
        csv_dir=args.csv_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.text_output.write_text(_text(report), encoding="utf-8")
    if not report["pass"]:
        raise SystemExit("STOP_RUNTIME_REPROFILE_GATE")


if __name__ == "__main__":
    main()
