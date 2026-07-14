"""Analyze one same-process cold/warm persistent graph workspace pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


LABELS = ("timing_context", "main_generation")
TOPOLOGY_VERSION = "selected-k4-k1-int8-fp16-cross-persistent-graphs-v1"
VRAM_GROWTH_TOLERANCE_BYTES = 16 * 1024 * 1024


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


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


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
        sequence_indices = [
            _integer(row.get("sequence_index"), name=f"{label}.sequence_index")
            for row in rows
        ]
        if sequence_indices != sorted(set(sequence_indices)):
            raise ValueError(f"{path} contains duplicate or unordered {label} records")
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
        token_streams = []
        stopping = []
        dispatch_modes = []
        dispatch_policies = []
        for index, row in enumerate(rows):
            tokens = _list(
                row.get("generated_token_ids"),
                name=f"{label}[{index}].generated_token_ids",
            )
            if not all(
                isinstance(token, int) and not isinstance(token, bool)
                for token in tokens
            ):
                raise ValueError(f"{label}[{index}] token ids must be integers")
            generated = _integer(
                row.get("generated_tokens"),
                name=f"{label}[{index}].generated_tokens",
            )
            if generated != len(tokens):
                raise ValueError(
                    f"{label}[{index}] generated-token count does not match token ids"
                )
            token_streams.append(tokens)
            stopping.append(
                (
                    generated,
                    _integer(
                        row.get("output_tokens"),
                        name=f"{label}[{index}].output_tokens",
                    ),
                    _integer(
                        row.get("prompt_tokens"),
                        name=f"{label}[{index}].prompt_tokens",
                    ),
                )
            )
            mode = row.get("optimized_dispatch_mode")
            if not isinstance(mode, str) or not mode:
                raise ValueError(f"{label}[{index}] is missing dispatch mode")
            dispatch_modes.append(mode)
            dispatch_policies.append(
                _object(
                    row.get("optimized_dispatch_policy"),
                    name=f"{label}[{index}].optimized_dispatch_policy",
                )
            )
        cache_storage = _list(
            last_persistent.get("cache_storage"),
            name=f"{label}.persistent_workspace.cache_storage",
        )
        encoder_storage = _list(
            last_persistent.get("encoder_storage"),
            name=f"{label}.persistent_workspace.encoder_storage",
        )
        if not cache_storage or not encoder_storage:
            raise ValueError(f"{label} persistent storage evidence must be non-empty")
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
            "tokens": sum(len(tokens) for tokens in token_streams),
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
            "token_streams": token_streams,
            "stopping": stopping,
            "dispatch_modes": dispatch_modes,
            "dispatch_policies": dispatch_policies,
            "cache_storage": cache_storage,
            "encoder_storage": encoder_storage,
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
    if manifest.get("schema_version") != 1:
        raise ValueError("persistent graph manifest schema_version must be 1")
    if manifest.get("topology_version") != TOPOLOGY_VERSION:
        raise ValueError("persistent graph manifest topology version changed")
    initialization_path = Path(str(manifest.get("initialization_path", "")))
    if not initialization_path.is_file() or initialization_path.stat().st_size <= 0:
        raise ValueError("persistent graph initialization artifact is missing")
    initialization = _object(
        json.loads(initialization_path.read_text(encoding="utf-8")),
        name="initialization",
    )
    if initialization.get("schema_version") != 1:
        raise ValueError("persistent graph initialization schema_version must be 1")
    if initialization.get("topology_version") != TOPOLOGY_VERSION:
        raise ValueError("persistent graph initialization topology version changed")
    for role in ("main", "timing"):
        if not _object(initialization.get(role), name=f"initialization.{role}"):
            raise ValueError(f"persistent graph {role} initialization is empty")
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
            "encoder_storage": left["encoder_storage"] == right["encoder_storage"],
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
    cold_pool = _object(
        cold_result.get("pool_summary"),
        name="cold.pool_summary",
    )
    warm_pool = _object(
        warm_result.get("pool_summary"),
        name="warm.pool_summary",
    )
    pool_checks = {}
    for role in ("main", "timing"):
        cold_summary = _object(cold_pool.get(role), name=f"cold.pool_summary.{role}")
        warm_summary = _object(warm_pool.get(role), name=f"warm.pool_summary.{role}")
        summary = _object(final_pool.get(role), name=f"final_pool.{role}")
        cold_workspaces = _list(
            cold_summary.get("workspaces"),
            name=f"cold.pool_summary.{role}.workspaces",
        )
        warm_workspaces = _list(
            warm_summary.get("workspaces"),
            name=f"warm.pool_summary.{role}.workspaces",
        )
        final_workspaces = _list(
            summary.get("workspaces"),
            name=f"final_pool.{role}.workspaces",
        )
        if not len(cold_workspaces) == len(warm_workspaces) == len(final_workspaces) == 1:
            raise ValueError(f"persistent graph {role} must retain exactly one workspace")
        cold_workspace = _object(cold_workspaces[0], name=f"cold.workspace.{role}")
        warm_workspace = _object(warm_workspaces[0], name=f"warm.workspace.{role}")
        final_workspace = _object(final_workspaces[0], name=f"final.workspace.{role}")
        stable_workspace_fields = (
            "signature",
            "graph_count",
            "encoder_slot_count",
            "encoder_storage",
            "cache_count",
            "cache_storage",
        )
        pool_checks[role] = {
            "one_workspace": (
                cold_summary.get("workspaces_created") == 1
                and warm_summary.get("workspaces_created") == 1
                and summary.get("workspaces_created") == 1
            ),
            "no_eviction": all(
                item.get("workspaces_evicted") == 0
                for item in (cold_summary, warm_summary, summary)
            ),
            "bounded_resident": all(
                item.get("max_resident_slots") == 1
                and item.get("resident_slots") == 1
                for item in (cold_summary, warm_summary, summary)
            ),
            "request_progression": (
                cold_summary.get("request_count") == 1
                and warm_summary.get("request_count") == 2
                and summary.get("request_count") == 2
            ),
            "stable_workspace_storage": all(
                cold_workspace.get(field) == warm_workspace.get(field)
                for field in stable_workspace_fields
            ),
            "final_snapshot_matches_warm": all(
                warm_workspace.get(field) == final_workspace.get(field)
                for field in stable_workspace_fields
            ),
            "graphs_and_storage_nonempty": (
                int(cold_workspace.get("graph_count", 0)) > 0
                and int(cold_workspace.get("encoder_slot_count", 0)) > 0
                and int(cold_workspace.get("cache_count", 0)) > 0
                and bool(cold_workspace.get("encoder_storage"))
                and bool(cold_workspace.get("cache_storage"))
            ),
            "inactive_at_boundaries": all(
                item.get("in_use") is False
                for item in (cold_workspace, warm_workspace, final_workspace)
            ),
            "open_before_verified_close": all(
                item.get("closed") is False
                for item in (cold_summary, warm_summary, summary)
            ),
        }
    pool_pass = all(all(checks.values()) for checks in pool_checks.values())
    close_pass = manifest.get("close_completed") is True
    capture_pass = cold_capture > 0.0 and warm_capture <= 1e-9 and warm_graph_delta == 0
    hits_pass = warm_hits > 0
    tolerance_mb = VRAM_GROWTH_TOLERANCE_BYTES / (1024 * 1024)
    profile_memory_checks = {
        "current_allocated_bounded": (
            warm["max_current_cuda_memory_allocated_mb"]
            <= cold["max_current_cuda_memory_allocated_mb"] + tolerance_mb
        ),
        "peak_allocated_bounded": (
            warm["peak_cuda_memory_allocated_mb"]
            <= cold["peak_cuda_memory_allocated_mb"] + tolerance_mb
        ),
    }
    cuda_memory_checks: dict[str, bool] = {}
    cold_cuda = cold_result.get("cuda_memory")
    warm_cuda = warm_result.get("cuda_memory")
    if cold_cuda is not None or warm_cuda is not None:
        cold_cuda = _object(cold_cuda, name="cold.cuda_memory")
        warm_cuda = _object(warm_cuda, name="warm.cuda_memory")
        for field in (
            "allocated_bytes",
            "reserved_bytes",
            "max_allocated_bytes",
            "max_reserved_bytes",
        ):
            cold_value = _integer(
                cold_cuda.get(field),
                name=f"cold.cuda_memory.{field}",
            )
            warm_value = _integer(
                warm_cuda.get(field),
                name=f"warm.cuda_memory.{field}",
            )
            cuda_memory_checks[f"{field}_bounded"] = (
                warm_value <= cold_value + VRAM_GROWTH_TOLERANCE_BYTES
            )
    memory_pass = all(profile_memory_checks.values()) and all(
        cuda_memory_checks.values()
    )
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
        "memory_gate": {
            "tolerance_bytes": VRAM_GROWTH_TOLERANCE_BYTES,
            "profile_checks": profile_memory_checks,
            "allocator_checks": cuda_memory_checks,
            "pass": memory_pass,
        },
        "close_pass": close_pass,
        "pass": (
            exactness_pass
            and capture_pass
            and hits_pass
            and pool_pass
            and memory_pass
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
            f"pool_pass={str(report['pool_pass']).lower()}",
            f"memory_pass={str(report['memory_gate']['pass']).lower()}",
            f"close_pass={str(report['close_pass']).lower()}",
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
