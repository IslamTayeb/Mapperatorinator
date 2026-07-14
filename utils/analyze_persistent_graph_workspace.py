"""Analyze one same-process cold/warm persistent graph workspace pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


LABELS = ("timing_context", "main_generation")


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _list(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _profile_metrics(path: Path) -> dict[str, Any]:
    root = _object(json.loads(path.read_text(encoding="utf-8")), name=str(path))
    records = _list(root.get("generation"), name=f"{path}.generation")
    stages = _list(root.get("stages"), name=f"{path}.stages")
    by_label: dict[str, list[dict[str, Any]]] = {}
    for label in LABELS:
        rows = [
            _object(row, name=f"{path}.generation[]")
            for row in records
            if isinstance(row, dict) and row.get("profile_label") == label
        ]
        rows.sort(key=lambda row: int(row.get("sequence_index", -1)))
        if not rows:
            raise ValueError(f"{path} contains no {label} records")
        by_label[label] = rows

    stage_by_name = {
        str(stage["name"]): _object(stage, name=f"{path}.stage")
        for stage in stages
        if isinstance(stage, dict) and "name" in stage
    }
    validate = stage_by_name.get("validate_inputs")
    final = stage_by_name.get("write_osu") or stage_by_name.get("write_osz")
    if validate is None or final is None:
        raise ValueError(f"{path} is missing request boundary stages")
    complete = _number(
        final["finished_at_perf_counter_seconds"],
        name="final.finished",
    ) - _number(
        validate["started_at_perf_counter_seconds"],
        name="validate.started",
    )
    if complete <= 0.0:
        raise ValueError(f"{path} complete request wall must be positive")

    labels: dict[str, Any] = {}
    for label, rows in by_label.items():
        last_persistent = _object(
            _object(
                _object(
                    rows[-1].get("optimized_cuda_graphs"),
                    name=f"{label}.optimized_cuda_graphs",
                ).get("persistent_workspace"),
                name=f"{label}.persistent_workspace",
            ),
            name=f"{label}.persistent_workspace",
        )
        labels[label] = {
            "records": len(rows),
            "model_seconds": sum(
                _number(row["model_elapsed_seconds"], name="model_elapsed_seconds")
                for row in rows
            ),
            "outer_wall_seconds": _number(
                stage_by_name[
                    "timing_context_generation"
                    if label == "timing_context"
                    else "main_generation"
                ]["wall_seconds"],
                name=f"{label}.stage.wall_seconds",
            ),
            "tokens": sum(int(row["generated_tokens"]) for row in rows),
            "capture_seconds": sum(
                _number(
                    row.get("decode_graph_capture_seconds_delta", 0.0),
                    name="decode_graph_capture_seconds_delta",
                )
                for row in rows
            ),
            "graph_count_delta": sum(
                int(row.get("decode_graph_count_delta", 0)) for row in rows
            ),
            "token_streams": [row.get("generated_token_ids") for row in rows],
            "stopping": [
                (
                    int(row["generated_tokens"]),
                    int(row["output_tokens"]),
                    int(row["prompt_tokens"]),
                )
                for row in rows
            ],
            "dispatch_modes": [row.get("optimized_dispatch_mode") for row in rows],
            "dispatch_policies": [row.get("optimized_dispatch_policy") for row in rows],
            "cache_storage": last_persistent.get("cache_storage"),
            "cross_request_graph_hits": max(
                int(
                    _object(
                        _object(row["optimized_cuda_graphs"], name="graphs").get(
                            "persistent_workspace"
                        ),
                        name="persistent_workspace",
                    ).get("cross_request_graph_hits", 0)
                )
                for row in rows
            ),
        }
        labels[label]["tps"] = (
            labels[label]["tokens"] / labels[label]["model_seconds"]
        )

    peak_vram = max(
        _number(row["cuda_max_memory_allocated_mb"], name="peak_vram")
        for row in [*stages, *records]
        if isinstance(row, dict) and "cuda_max_memory_allocated_mb" in row
    )
    current_vram = max(
        _number(row["cuda_memory_allocated_mb"], name="current_vram")
        for row in [*stages, *records]
        if isinstance(row, dict) and "cuda_memory_allocated_mb" in row
    )
    metadata = _object(root.get("metadata"), name=f"{path}.metadata")
    result_path = Path(str(metadata.get("result_path", "")))
    if not result_path.is_file():
        raise ValueError(f"{path} result artifact is missing: {result_path}")
    return {
        "profile_path": str(path),
        "result_path": str(result_path),
        "result_sha256": _sha256(result_path),
        "complete_request_wall_seconds": complete,
        "peak_cuda_memory_allocated_mb": peak_vram,
        "max_current_cuda_memory_allocated_mb": current_vram,
        "labels": labels,
    }


def analyze(manifest_path: Path) -> dict[str, Any]:
    manifest = _object(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        name="manifest",
    )
    results = _object(manifest.get("results"), name="manifest.results")
    cold_result = _object(results.get("cold"), name="cold")
    warm_result = _object(results.get("warm"), name="warm")
    cold = _profile_metrics(Path(cold_result["profile_path"]))
    warm = _profile_metrics(Path(warm_result["profile_path"]))
    cold["process_call_wall_seconds"] = _number(
        cold_result.get("process_call_wall_seconds"),
        name="cold.process_call_wall_seconds",
    )
    warm["process_call_wall_seconds"] = _number(
        warm_result.get("process_call_wall_seconds"),
        name="warm.process_call_wall_seconds",
    )
    exactness: dict[str, Any] = {
        "final_osu": cold["result_sha256"] == warm["result_sha256"],
        "labels": {},
    }
    for label in LABELS:
        left = cold["labels"][label]
        right = warm["labels"][label]
        exactness["labels"][label] = {
            "tokens": left["token_streams"] == right["token_streams"],
            "stopping": left["stopping"] == right["stopping"],
            "dispatch_mode": left["dispatch_modes"] == right["dispatch_modes"],
            "dispatch_policy": left["dispatch_policies"] == right["dispatch_policies"],
            "cache_storage": left["cache_storage"] == right["cache_storage"],
        }
    exactness_pass = exactness["final_osu"] and all(
        all(checks.values()) for checks in exactness["labels"].values()
    )
    cold_capture = sum(
        cold["labels"][label]["capture_seconds"] for label in LABELS
    )
    warm_capture = sum(
        warm["labels"][label]["capture_seconds"] for label in LABELS
    )
    warm_graph_delta = sum(
        warm["labels"][label]["graph_count_delta"] for label in LABELS
    )
    warm_hits = sum(
        warm["labels"][label]["cross_request_graph_hits"] for label in LABELS
    )
    final_pool = _object(
        manifest.get("final_pool_summary"),
        name="manifest.final_pool_summary",
    )
    pool_checks = {}
    for role in ("main", "timing"):
        summary = _object(final_pool.get(role), name=f"final_pool.{role}")
        pool_checks[role] = {
            "one_workspace": summary.get("workspaces_created") == 1,
            "no_eviction": summary.get("workspaces_evicted") == 0,
            "bounded_resident": summary.get("max_resident_slots") == 1,
            "two_requests": summary.get("request_count") == 2,
            "closed_after_manifest": summary.get("closed") is False,
        }
    pool_pass = all(all(checks.values()) for checks in pool_checks.values())
    close_pass = manifest.get("close_completed") is True
    capture_pass = cold_capture > 0.0 and warm_capture <= 1e-9 and warm_graph_delta == 0
    hits_pass = warm_hits > 0
    report = {
        "schema_version": 1,
        "manifest_path": str(manifest_path),
        "cold": cold,
        "warm": warm,
        "savings": {
            "process_call_wall_seconds": (
                cold["process_call_wall_seconds"]
                - warm["process_call_wall_seconds"]
            ),
            "complete_request_wall_seconds": (
                cold["complete_request_wall_seconds"]
                - warm["complete_request_wall_seconds"]
            ),
            "timing_model_seconds": (
                cold["labels"]["timing_context"]["model_seconds"]
                - warm["labels"]["timing_context"]["model_seconds"]
            ),
            "timing_outer_wall_seconds": (
                cold["labels"]["timing_context"]["outer_wall_seconds"]
                - warm["labels"]["timing_context"]["outer_wall_seconds"]
            ),
            "main_model_seconds": (
                cold["labels"]["main_generation"]["model_seconds"]
                - warm["labels"]["main_generation"]["model_seconds"]
            ),
            "main_outer_wall_seconds": (
                cold["labels"]["main_generation"]["outer_wall_seconds"]
                - warm["labels"]["main_generation"]["outer_wall_seconds"]
            ),
            "graph_capture_seconds": cold_capture - warm_capture,
        },
        "exactness": exactness,
        "exactness_pass": exactness_pass,
        "capture_gate": {
            "cold_capture_seconds": cold_capture,
            "warm_capture_seconds": warm_capture,
            "warm_graph_count_delta": warm_graph_delta,
            "pass": capture_pass,
        },
        "cross_request_graph_hits": warm_hits,
        "cross_request_hits_pass": hits_pass,
        "pool_checks": pool_checks,
        "pool_pass": pool_pass,
        "close_pass": close_pass,
        "pass": (
            exactness_pass
            and capture_pass
            and hits_pass
            and pool_pass
            and close_pass
        ),
    }
    return report


def _text(report: dict[str, Any]) -> str:
    savings = report["savings"]
    return "\n".join(
        (
            f"pass={str(report['pass']).lower()}",
            f"exactness_pass={str(report['exactness_pass']).lower()}",
            f"capture_pass={str(report['capture_gate']['pass']).lower()}",
            f"cold_capture_seconds={report['capture_gate']['cold_capture_seconds']:.9f}",
            f"warm_capture_seconds={report['capture_gate']['warm_capture_seconds']:.9f}",
            f"cross_request_graph_hits={report['cross_request_graph_hits']}",
            f"process_call_saving_seconds={savings['process_call_wall_seconds']:.9f}",
            f"complete_request_saving_seconds={savings['complete_request_wall_seconds']:.9f}",
            f"timing_model_saving_seconds={savings['timing_model_seconds']:.9f}",
            f"timing_outer_saving_seconds={savings['timing_outer_wall_seconds']:.9f}",
            f"main_model_saving_seconds={savings['main_model_seconds']:.9f}",
            f"main_outer_saving_seconds={savings['main_outer_wall_seconds']:.9f}",
        )
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-text", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.manifest)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_text.write_text(_text(report), encoding="utf-8")
    if not report["pass"]:
        raise SystemExit("STOP_PERSISTENT_GRAPH_WORKSPACE")


if __name__ == "__main__":
    main()
