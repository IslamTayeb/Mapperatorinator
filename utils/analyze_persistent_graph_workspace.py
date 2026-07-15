"""Analyze a same-process cold/warm-one/warm-two graph workspace plateau."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


LABELS = ("timing_context", "main_generation")
PASSES = ("cold", "warm1", "warm2")
WARM_PASSES = ("warm1", "warm2")
TOPOLOGY_VERSION = (
    "selected-k4-k1-int8-fp16-cross-shared-arena-persistent-graphs-v2"
)
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


def _arena_storage_identity(value: Any, *, name: str) -> tuple[Any, ...]:
    rows = _list(value, name=name)
    result = []
    for index, raw in enumerate(rows):
        row = _object(raw, name=f"{name}[{index}]")
        refreshed = _list(
            row.get("refreshed_window_identity"),
            name=f"{name}[{index}].refreshed_window_identity",
        )
        if len(refreshed) != 2:
            raise ValueError(f"{name}[{index}] refresh identity must have two values")
        if row.get("content_match") is not True:
            raise ValueError(f"{name}[{index}] arena content did not match")
        for part_index, part in enumerate(refreshed):
            _integer(part, name=f"{name}[{index}].refresh[{part_index}]")
        addresses = _list(
            row.get("tensor_addresses"),
            name=f"{name}[{index}].tensor_addresses",
        )
        if not addresses:
            raise ValueError(f"{name}[{index}] tensor addresses must be non-empty")
        address_identity = []
        for address_index, raw_address in enumerate(addresses):
            address = _object(
                raw_address,
                name=f"{name}[{index}].tensor_addresses[{address_index}]",
            )
            tensor_name = address.get("name")
            if not isinstance(tensor_name, str) or not tensor_name:
                raise ValueError(f"{name}[{index}] tensor name must be non-empty")
            address_identity.append(
                (
                    tensor_name,
                    _integer(
                        address.get("data_ptr"),
                        name=f"{name}[{index}].data_ptr",
                        minimum=1,
                    ),
                )
            )
        result.append(
            (
                row.get("state_signature"),
                row.get("slot_signature"),
                row.get("input_signature"),
                tuple(address_identity),
            )
        )
    return tuple(result)


def _arena_refresh_identities(value: Any, *, name: str) -> tuple[tuple[int, int], ...]:
    rows = _list(value, name=name)
    result = []
    for index, raw in enumerate(rows):
        row = _object(raw, name=f"{name}[{index}]")
        refreshed = _list(
            row.get("refreshed_window_identity"),
            name=f"{name}[{index}].refreshed_window_identity",
        )
        if len(refreshed) != 2:
            raise ValueError(f"{name}[{index}] refresh identity must have two values")
        result.append(
            (
                _integer(refreshed[0], name=f"{name}[{index}].request", minimum=1),
                _integer(refreshed[1], name=f"{name}[{index}].window", minimum=1),
            )
        )
    return tuple(result)


def _dispatch_topology_pass(
    label: str,
    modes: list[str],
    policies: list[dict[str, Any]],
) -> bool:
    expected_mode = (
        "fp32_timing_native_self_batch1"
        if label == "timing_context"
        else "approximate_weight_only_batch1"
    )
    if not modes or any(mode != expected_mode for mode in modes):
        return False
    for policy in policies:
        q1 = policy.get("q1_bmm_cross_attention")
        effective_self = policy.get("effective_native_q1_rope_cache_self_attention")
        if not isinstance(q1, dict) or q1.get("enabled") is not True:
            return False
        if not isinstance(effective_self, dict) or effective_self.get("enabled") is not True:
            return False
        if effective_self.get("kernel") != "native_q1_rope_cache_attention":
            return False
        if label == "timing_context":
            timing = policy.get("timing_native_self")
            if (
                effective_self.get("owner") != "accepted_attention_hook"
                or not isinstance(timing, dict)
                or timing.get("enabled") is not True
                or timing.get("exactness_claim") is not True
            ):
                return False
        else:
            weight = policy.get("approximate_weight_only")
            if (
                effective_self.get("owner") != "approximate_weight_only"
                or not isinstance(weight, dict)
                or weight.get("enabled") is not True
                or weight.get("exactness_claim") is not False
            ):
                return False
    return True


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
        k8_topology_checks = []
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
            graphs = _object(
                row.get("optimized_cuda_graphs"),
                name=f"{label}[{index}].optimized_cuda_graphs",
            )
            k8 = _object(
                graphs.get("k8_candidate"),
                name=f"{label}[{index}].k8_candidate",
            )
            k8_topology_checks.append(
                k8.get("block_size") == 4
                and k8.get("remainder_backend") == "cuda_graph"
                and k8.get("shared_static_input_arena") is True
                and k8.get("static_input_arena_content_match") is True
                and _integer(
                    k8.get("static_input_arena_content_checks"),
                    name=f"{label}[{index}].arena_content_checks",
                )
                >= 0
            )
        cache_storage = _list(
            last_persistent.get("cache_storage"),
            name=f"{label}.persistent_workspace.cache_storage",
        )
        encoder_storage = _list(
            last_persistent.get("encoder_storage"),
            name=f"{label}.persistent_workspace.encoder_storage",
        )
        arena_storage = _list(
            last_persistent.get("arena_storage"),
            name=f"{label}.persistent_workspace.arena_storage",
        )
        if not cache_storage or not encoder_storage or not arena_storage:
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
            "arena_storage": arena_storage,
            "arena_storage_identity": _arena_storage_identity(
                arena_storage,
                name=f"{label}.persistent_workspace.arena_storage",
            ),
            "arena_refresh_identities": _arena_refresh_identities(
                arena_storage,
                name=f"{label}.persistent_workspace.arena_storage",
            ),
            "dispatch_topology_pass": _dispatch_topology_pass(
                label,
                dispatch_modes,
                dispatch_policies,
            ),
            "k8_topology_pass": all(k8_topology_checks),
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
    if manifest.get("schema_version") != 2:
        raise ValueError("persistent graph manifest schema_version must be 2")
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
    initialization_roles = {}
    for role in ("main", "timing"):
        initialization_roles[role] = _object(
            initialization.get(role),
            name=f"initialization.{role}",
        )
        if not initialization_roles[role]:
            raise ValueError(f"persistent graph {role} initialization is empty")
    main_initialization = initialization_roles["main"]
    timing_initialization = initialization_roles["timing"]
    main_cross = _object(
        main_initialization.get("cross_candidate"),
        name="initialization.main.cross_candidate",
    )
    main_int8 = _object(
        main_initialization.get("int8_mlp_overlay"),
        name="initialization.main.int8_mlp_overlay",
    )
    initialization_topology_pass = (
        main_initialization.get("result_class") == "documented-drift"
        and main_initialization.get("exactness_claim") is False
        and main_cross.get("mode") == "fp16_packed_projections"
        and main_cross.get("accepted_q1_bmm") is True
        and main_cross.get("incremental_exactness_required") is True
        and main_int8.get("dispatch_counter") == "int8_weight_mlp_tail"
        and timing_initialization.get("result_class") == "exact"
        and timing_initialization.get("exactness_claim") is True
        and timing_initialization.get("precision") == "fp32"
        and timing_initialization.get("native_q1_self_attention") is True
        and timing_initialization.get("native_q1_rope_cache_self_attention") is True
        and timing_initialization.get("q1_bmm_cross_attention_retained") is True
        and timing_initialization.get("original_decoder_forward_retained") is True
    )
    if not initialization_topology_pass:
        raise ValueError("persistent graph initialization topology is wrong")
    pool_initial = _object(
        initialization.get("pool_initial"),
        name="initialization.pool_initial",
    )
    results = _object(manifest.get("results"), name="manifest.results")
    if set(results) != set(PASSES):
        raise ValueError(f"persistent graph results must be exactly {list(PASSES)}")
    result_rows = {
        pass_name: _object(results.get(pass_name), name=pass_name)
        for pass_name in PASSES
    }
    metrics = {
        pass_name: _profile_metrics(Path(result_rows[pass_name]["profile_path"]))
        for pass_name in PASSES
    }
    for pass_name in PASSES:
        metrics[pass_name]["process_call_wall_seconds"] = _number(
            result_rows[pass_name].get("process_call_wall_seconds"),
            name=f"{pass_name}.process_call_wall_seconds",
        )
    profile_topology_pass = all(
        metrics[pass_name]["labels"][label]["dispatch_topology_pass"]
        and metrics[pass_name]["labels"][label]["k8_topology_pass"]
        for pass_name in PASSES
        for label in LABELS
    )
    cold = metrics["cold"]
    warm2 = metrics["warm2"]
    exactness: dict[str, Any] = {"reference": "cold", "passes": {}}
    for pass_name in WARM_PASSES:
        candidate = metrics[pass_name]
        comparison: dict[str, Any] = {
            "final_osu": cold["result_sha256"] == candidate["result_sha256"],
            "labels": {},
        }
        for label in LABELS:
            left = cold["labels"][label]
            right = candidate["labels"][label]
            comparison["labels"][label] = {
                "tokens": left["token_streams"] == right["token_streams"],
                "stopping": left["stopping"] == right["stopping"],
                "dispatch_mode": left["dispatch_modes"] == right["dispatch_modes"],
                "dispatch_policy": (
                    left["dispatch_policies"] == right["dispatch_policies"]
                ),
                "cache_storage": left["cache_storage"] == right["cache_storage"],
                "encoder_storage": (
                    left["encoder_storage"] == right["encoder_storage"]
                ),
                "arena_storage": (
                    left["arena_storage_identity"]
                    == right["arena_storage_identity"]
                ),
            }
        exactness["passes"][pass_name] = comparison
    exactness_pass = all(
        comparison["final_osu"]
        and all(
            all(checks.values())
            for checks in comparison["labels"].values()
        )
        for comparison in exactness["passes"].values()
    )
    capture_seconds = {
        pass_name: sum(
            metrics[pass_name]["labels"][label]["capture_seconds"]
            for label in LABELS
        )
        for pass_name in PASSES
    }
    graph_deltas = {
        pass_name: sum(
            metrics[pass_name]["labels"][label]["graph_count_delta"]
            for label in LABELS
        )
        for pass_name in WARM_PASSES
    }
    cross_request_hits = {
        pass_name: sum(
            metrics[pass_name]["labels"][label]["cross_request_graph_hits"]
            for label in LABELS
        )
        for pass_name in WARM_PASSES
    }
    final_pool = _object(
        manifest.get("final_pool_summary"),
        name="manifest.final_pool_summary",
    )
    pools = {
        pass_name: _object(
            result_rows[pass_name].get("pool_summary"),
            name=f"{pass_name}.pool_summary",
        )
        for pass_name in PASSES
    }
    pool_checks = {}
    for role in ("main", "timing"):
        expected_topology = (
            TOPOLOGY_VERSION,
            role,
            "block_size=4",
            "graph_remainders=true",
            "shared_rope=true" if role == "main" else "timing_native_self=true",
            "shared_static_input_arena=true",
        )
        expected_topology_repr = repr(expected_topology)
        initial_summary = _object(
            pool_initial.get(role),
            name=f"initialization.pool_initial.{role}",
        )
        summaries = {
            pass_name: _object(
                pools[pass_name].get(role),
                name=f"{pass_name}.pool_summary.{role}",
            )
            for pass_name in PASSES
        }
        summary = _object(final_pool.get(role), name=f"final_pool.{role}")
        workspaces = {
            pass_name: _list(
                summaries[pass_name].get("workspaces"),
                name=f"{pass_name}.pool_summary.{role}.workspaces",
            )
            for pass_name in PASSES
        }
        final_workspaces = _list(
            summary.get("workspaces"),
            name=f"final_pool.{role}.workspaces",
        )
        if not all(len(items) == 1 for items in (*workspaces.values(), final_workspaces)):
            raise ValueError(f"persistent graph {role} must retain exactly one workspace")
        workspace_rows = {
            pass_name: _object(
                workspaces[pass_name][0],
                name=f"{pass_name}.workspace.{role}",
            )
            for pass_name in PASSES
        }
        final_workspace = _object(final_workspaces[0], name=f"final.workspace.{role}")
        arena_identities = {
            pass_name: _arena_storage_identity(
                workspace_rows[pass_name].get("arena_storage"),
                name=f"{pass_name}.workspace.{role}.arena_storage",
            )
            for pass_name in PASSES
        }
        final_arena_identity = _arena_storage_identity(
            final_workspace.get("arena_storage"),
            name=f"final.workspace.{role}.arena_storage",
        )
        arena_refreshes = {
            pass_name: _arena_refresh_identities(
                workspace_rows[pass_name].get("arena_storage"),
                name=f"{pass_name}.workspace.{role}.arena_storage",
            )
            for pass_name in PASSES
        }
        final_arena_refreshes = _arena_refresh_identities(
            final_workspace.get("arena_storage"),
            name=f"final.workspace.{role}.arena_storage",
        )
        stable_workspace_fields = (
            "signature",
            "graph_count",
            "encoder_slot_count",
            "encoder_storage",
            "cache_count",
            "cache_storage",
        )
        pool_checks[role] = {
            "topology_signature": all(
                item.get("topology_signature") == expected_topology_repr
                for item in (
                    initial_summary,
                    *summaries.values(),
                    summary,
                )
            ),
            "one_workspace": (
                all(item.get("workspaces_created") == 1 for item in summaries.values())
                and summary.get("workspaces_created") == 1
            ),
            "no_eviction": all(
                item.get("workspaces_evicted") == 0
                for item in (*summaries.values(), summary)
            ),
            "bounded_resident": all(
                item.get("max_resident_slots") == 1
                and item.get("resident_slots") == 1
                for item in (*summaries.values(), summary)
            ),
            "request_progression": (
                summaries["cold"].get("request_count") == 1
                and summaries["warm1"].get("request_count") == 2
                and summaries["warm2"].get("request_count") == 3
                and summary.get("request_count") == 3
            ),
            "stable_workspace_storage": all(
                workspace_rows["cold"].get(field)
                == workspace_rows["warm1"].get(field)
                == workspace_rows["warm2"].get(field)
                for field in stable_workspace_fields
            ),
            "stable_arena_storage": (
                arena_identities["cold"]
                == arena_identities["warm1"]
                == arena_identities["warm2"]
            ),
            "arena_refresh_progression": all(
                identities
                and all(request_serial == expected_request for request_serial, _ in identities)
                for expected_request, identities in (
                    (1, arena_refreshes["cold"]),
                    (2, arena_refreshes["warm1"]),
                    (3, arena_refreshes["warm2"]),
                )
            ),
            "final_snapshot_matches_warm2": all(
                workspace_rows["warm2"].get(field) == final_workspace.get(field)
                for field in stable_workspace_fields
            )
            and arena_identities["warm2"] == final_arena_identity
            and arena_refreshes["warm2"] == final_arena_refreshes,
            "graphs_and_storage_nonempty": (
                int(workspace_rows["cold"].get("graph_count", 0)) > 0
                and int(workspace_rows["cold"].get("encoder_slot_count", 0)) > 0
                and int(workspace_rows["cold"].get("cache_count", 0)) > 0
                and bool(workspace_rows["cold"].get("encoder_storage"))
                and bool(workspace_rows["cold"].get("cache_storage"))
                and bool(arena_identities["cold"])
            ),
            "inactive_at_boundaries": all(
                item.get("in_use") is False
                for item in (*workspace_rows.values(), final_workspace)
            ),
            "open_before_verified_close": all(
                item.get("closed") is False
                for item in (*summaries.values(), summary)
            ),
        }
    pool_pass = all(all(checks.values()) for checks in pool_checks.values())
    close_pass = manifest.get("close_completed") is True
    capture_pass = (
        capture_seconds["cold"] > 0.0
        and all(capture_seconds[name] <= 1e-9 for name in WARM_PASSES)
        and all(graph_deltas[name] == 0 for name in WARM_PASSES)
    )
    hits_progression_by_label = {
        label: (
            metrics["warm1"]["labels"][label]["cross_request_graph_hits"] > 0
            and metrics["warm2"]["labels"][label]["cross_request_graph_hits"]
            > metrics["warm1"]["labels"][label]["cross_request_graph_hits"]
        )
        for label in LABELS
    }
    hits_pass = (
        cross_request_hits["warm1"] > 0
        and cross_request_hits["warm2"] > cross_request_hits["warm1"]
        and all(hits_progression_by_label.values())
    )
    tolerance_mb = VRAM_GROWTH_TOLERANCE_BYTES / (1024 * 1024)
    warm1 = metrics["warm1"]
    profile_memory_checks = {
        "current_allocated_bounded": (
            warm2["max_current_cuda_memory_allocated_mb"]
            <= warm1["max_current_cuda_memory_allocated_mb"] + tolerance_mb
        ),
        "peak_allocated_bounded": (
            warm2["peak_cuda_memory_allocated_mb"]
            <= warm1["peak_cuda_memory_allocated_mb"] + tolerance_mb
        ),
    }
    cuda_memory_checks: dict[str, bool] = {}
    cuda_rows = {
        pass_name: result_rows[pass_name].get("cuda_memory")
        for pass_name in PASSES
    }
    peak_reset_checks: dict[str, bool] = {}
    if any(value is not None for value in cuda_rows.values()):
        parsed_cuda = {
            pass_name: _object(value, name=f"{pass_name}.cuda_memory")
            for pass_name, value in cuda_rows.items()
        }
        peak_reset_checks = {
            pass_name: result_rows[pass_name].get("cuda_peak_stats_reset") is True
            for pass_name in PASSES
        }
        for field in (
            "allocated_bytes",
            "reserved_bytes",
            "max_allocated_bytes",
            "max_reserved_bytes",
        ):
            warm1_value = _integer(
                parsed_cuda["warm1"].get(field),
                name=f"warm1.cuda_memory.{field}",
            )
            warm2_value = _integer(
                parsed_cuda["warm2"].get(field),
                name=f"warm2.cuda_memory.{field}",
            )
            cuda_memory_checks[f"{field}_bounded"] = (
                warm2_value <= warm1_value + VRAM_GROWTH_TOLERANCE_BYTES
            )
    memory_pass = all(profile_memory_checks.values()) and all(
        cuda_memory_checks.values()
    ) and all(peak_reset_checks.values())
    report = {
        "schema_version": 2,
        "manifest_path": str(manifest_path),
        "cold": cold,
        "warm1": warm1,
        "warm2": warm2,
        "savings": {
            "process_call_wall_seconds": (
                cold["process_call_wall_seconds"]
                - warm2["process_call_wall_seconds"]
            ),
            "complete_request_wall_seconds": (
                cold["complete_request_wall_seconds"]
                - warm2["complete_request_wall_seconds"]
            ),
            "timing_model_seconds": (
                cold["labels"]["timing_context"]["model_seconds"]
                - warm2["labels"]["timing_context"]["model_seconds"]
            ),
            "timing_outer_wall_seconds": (
                cold["labels"]["timing_context"]["outer_wall_seconds"]
                - warm2["labels"]["timing_context"]["outer_wall_seconds"]
            ),
            "main_model_seconds": (
                cold["labels"]["main_generation"]["model_seconds"]
                - warm2["labels"]["main_generation"]["model_seconds"]
            ),
            "main_outer_wall_seconds": (
                cold["labels"]["main_generation"]["outer_wall_seconds"]
                - warm2["labels"]["main_generation"]["outer_wall_seconds"]
            ),
            "graph_capture_seconds": (
                capture_seconds["cold"] - capture_seconds["warm2"]
            ),
        },
        "exactness": exactness,
        "exactness_pass": exactness_pass,
        "capture_gate": {
            "capture_seconds": capture_seconds,
            "warm_graph_count_delta": graph_deltas,
            "pass": capture_pass,
        },
        "cross_request_graph_hits": cross_request_hits,
        "cross_request_hit_progression_by_label": hits_progression_by_label,
        "cross_request_hits_pass": hits_pass,
        "initialization_topology_pass": initialization_topology_pass,
        "profile_topology_pass": profile_topology_pass,
        "pool_checks": pool_checks,
        "pool_pass": pool_pass,
        "memory_gate": {
            "tolerance_bytes": VRAM_GROWTH_TOLERANCE_BYTES,
            "profile_checks": profile_memory_checks,
            "allocator_checks": cuda_memory_checks,
            "peak_reset_checks": peak_reset_checks,
            "pass": memory_pass,
        },
        "close_pass": close_pass,
        "pass": (
            exactness_pass
            and capture_pass
            and hits_pass
            and initialization_topology_pass
            and profile_topology_pass
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
            f"cross_request_hits_pass={str(report['cross_request_hits_pass']).lower()}",
            f"initialization_topology_pass={str(report['initialization_topology_pass']).lower()}",
            f"profile_topology_pass={str(report['profile_topology_pass']).lower()}",
            f"pool_pass={str(report['pool_pass']).lower()}",
            f"memory_pass={str(report['memory_gate']['pass']).lower()}",
            f"close_pass={str(report['close_pass']).lower()}",
            f"cold_capture_seconds={report['capture_gate']['capture_seconds']['cold']:.9f}",
            f"warm1_capture_seconds={report['capture_gate']['capture_seconds']['warm1']:.9f}",
            f"warm2_capture_seconds={report['capture_gate']['capture_seconds']['warm2']:.9f}",
            f"warm1_cross_request_graph_hits={report['cross_request_graph_hits']['warm1']}",
            f"warm2_cross_request_graph_hits={report['cross_request_graph_hits']['warm2']}",
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
