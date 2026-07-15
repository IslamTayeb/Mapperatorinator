from __future__ import annotations

import json
import copy
from pathlib import Path

import pytest

from utils.analyze_persistent_graph_workspace import analyze


def _stage(name: str, start: float, wall: float) -> dict:
    return {
        "name": name,
        "started_at_perf_counter_seconds": start,
        "finished_at_perf_counter_seconds": start + wall,
        "wall_seconds": wall,
        "cuda_memory_allocated_mb": 100.0,
        "cuda_max_memory_allocated_mb": 200.0,
    }


def _profile(
    root: Path,
    label: str,
    *,
    capture_seconds: float,
    graph_delta: int,
    cross_request_hits: int,
) -> Path:
    request_serial = {"cold": 1, "warm1": 2, "warm2": 3}[label]
    result = root / f"{label}.osu"
    result.write_text("same map\n", encoding="utf-8")
    cache_storage = [
        {
            "state_signature": "sig",
            "kind": "self_attention_cache",
            "layer": 0,
            "name": "keys",
            "data_ptr": 123,
            "shape": [1, 2, 3, 4],
            "dtype": "torch.float32",
            "device": "cuda:0",
        }
    ]
    arena_storage = [
        {
            "state_signature": "sig",
            "slot_signature": "slot",
            "refreshed_window_identity": [request_serial, 44],
            "input_signature": "inputs",
            "content_match": True,
            "tensor_addresses": [
                {"name": "decoder_input_ids", "data_ptr": 789}
            ],
        }
    ]
    generation = []
    for index, profile_label in enumerate(("timing_context", "main_generation")):
        timing = profile_label == "timing_context"
        generation.append(
            {
                "profile_label": profile_label,
                "sequence_index": index,
                "model_elapsed_seconds": 2.0 + index,
                "generated_tokens": 2,
                "output_tokens": 2 + index,
                "prompt_tokens": 2,
                "generated_token_ids": [index, index + 1],
                "decode_graph_capture_seconds_delta": capture_seconds / 2,
                "decode_graph_count_delta": graph_delta,
                "optimized_dispatch_mode": (
                    "fp32_timing_native_self_batch1"
                    if timing
                    else "approximate_weight_only_batch1"
                ),
                "optimized_dispatch_policy": {
                    "q1_bmm_cross_attention": {"enabled": True},
                    "effective_native_q1_rope_cache_self_attention": {
                        "enabled": True,
                        "owner": (
                            "accepted_attention_hook"
                            if timing
                            else "approximate_weight_only"
                        ),
                        "kernel": "native_q1_rope_cache_attention",
                    },
                    **(
                        {
                            "timing_native_self": {
                                "enabled": True,
                                "exactness_claim": True,
                            }
                        }
                        if timing
                        else {
                            "approximate_weight_only": {
                                "enabled": True,
                                "exactness_claim": False,
                            }
                        }
                    ),
                },
                "optimized_cuda_graphs": {
                    "k8_candidate": {
                        "block_size": 4,
                        "remainder_backend": "cuda_graph",
                        "shared_static_input_arena": True,
                        "static_input_arena_content_match": True,
                        "static_input_arena_content_checks": 4,
                    },
                    "persistent_workspace": {
                        "cross_request_graph_hits": cross_request_hits,
                        "cache_storage": cache_storage,
                        "encoder_storage": [
                            {
                                "state_signature": "sig",
                                "slot_signature": "encoder",
                                "data_ptr": 456,
                                "shape": [1, 8, 4],
                                "stride": [32, 4, 1],
                                "dtype": "torch.float32",
                                "device": "cuda:0",
                            }
                        ],
                        "arena_storage": arena_storage,
                    }
                },
                "cuda_memory_allocated_mb": 150.0,
                "cuda_max_memory_allocated_mb": 250.0,
            }
        )
    profile = root / f"{label}.profile.json"
    profile.write_text(
        json.dumps(
            {
                "metadata": {"result_path": str(result)},
                "generation": generation,
                "stages": [
                    _stage("validate_inputs", 10.0, 0.1),
                    _stage("timing_context_generation", 11.0, 2.5),
                    _stage("main_generation", 14.0, 3.5),
                    _stage("write_osu", 18.0, 0.2),
                ],
            }
        ),
        encoding="utf-8",
    )
    return profile


def _manifest(tmp_path: Path, *, warm_capture: float = 0.0) -> Path:
    cold = _profile(
        tmp_path,
        "cold",
        capture_seconds=0.4,
        graph_delta=1,
        cross_request_hits=0,
    )
    warm1 = _profile(
        tmp_path,
        "warm1",
        capture_seconds=warm_capture,
        graph_delta=0,
        cross_request_hits=9,
    )
    warm2 = _profile(
        tmp_path,
        "warm2",
        capture_seconds=warm_capture,
        graph_delta=0,
        cross_request_hits=18,
    )
    workspace = {
        "signature": "sig",
        "graph_count": 2,
        "encoder_slot_count": 1,
        "encoder_storage": [
            {
                "data_ptr": 456,
                "shape": [1, 8, 4],
                "dtype": "torch.float32",
                "device": "cuda:0",
            }
        ],
        "cache_count": 1,
        "cache_storage": [
            {
                "data_ptr": 123,
                "shape": [1, 2, 3, 4],
                "dtype": "torch.float32",
                "device": "cuda:0",
            }
        ],
        "arena_storage": [
            {
                "state_signature": "sig",
                "slot_signature": "slot",
                "refreshed_window_identity": [1, 44],
                "input_signature": "inputs",
                "content_match": True,
                "tensor_addresses": [
                    {"name": "decoder_input_ids", "data_ptr": 789}
                ],
            }
        ],
        "in_use": False,
        "closed": False,
    }

    def pool(request_count: int, role: str) -> dict:
        workspace_row = copy.deepcopy(workspace)
        workspace_row["arena_storage"][0]["refreshed_window_identity"] = [
            request_count,
            44,
        ]
        topology = (
            "selected-k4-k1-int8-fp16-cross-shared-arena-persistent-graphs-v2",
            role,
            "block_size=4",
            "graph_remainders=true",
            "shared_rope=true" if role == "main" else "timing_native_self=true",
            "shared_static_input_arena=true",
        )
        return {
            "topology_signature": repr(topology),
            "workspaces_created": 1,
            "workspaces_evicted": 0,
            "max_resident_slots": 1,
            "resident_slots": 1,
            "request_count": request_count,
            "closed": False,
            "workspaces": [workspace_row],
        }

    initialization = tmp_path / "initialization.json"
    initialization.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "topology_version": (
                    "selected-k4-k1-int8-fp16-cross-shared-arena-"
                    "persistent-graphs-v2"
                ),
                "main": {
                    "result_class": "documented-drift",
                    "exactness_claim": False,
                    "cross_candidate": {
                        "mode": "fp16_packed_projections",
                        "accepted_q1_bmm": True,
                        "incremental_exactness_required": True,
                    },
                    "int8_mlp_overlay": {
                        "dispatch_counter": "int8_weight_mlp_tail"
                    },
                },
                "timing": {
                    "result_class": "exact",
                    "exactness_claim": True,
                    "precision": "fp32",
                    "native_q1_self_attention": True,
                    "native_q1_rope_cache_self_attention": True,
                    "q1_bmm_cross_attention_retained": True,
                    "original_decoder_forward_retained": True,
                },
                "pool_initial": {
                    "main": pool(0, "main"),
                    "timing": pool(0, "timing"),
                },
            }
        ),
        encoding="utf-8",
    )
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "topology_version": (
                    "selected-k4-k1-int8-fp16-cross-shared-arena-"
                    "persistent-graphs-v2"
                ),
                "initialization_path": str(initialization),
                "results": {
                    "cold": {
                        "profile_path": str(cold),
                        "process_call_wall_seconds": 7.0,
                        "pool_summary": {
                            "main": pool(1, "main"),
                            "timing": pool(1, "timing"),
                        },
                        "cuda_peak_stats_reset": True,
                        "cuda_memory": {
                            "allocated_bytes": 100,
                            "reserved_bytes": 200,
                            "max_allocated_bytes": 300,
                            "max_reserved_bytes": 400,
                        },
                    },
                    "warm1": {
                        "profile_path": str(warm1),
                        "process_call_wall_seconds": 6.0,
                        "pool_summary": {
                            "main": pool(2, "main"),
                            "timing": pool(2, "timing"),
                        },
                        "cuda_peak_stats_reset": True,
                        "cuda_memory": {
                            "allocated_bytes": 100,
                            "reserved_bytes": 200,
                            "max_allocated_bytes": 300,
                            "max_reserved_bytes": 400,
                        },
                    },
                    "warm2": {
                        "profile_path": str(warm2),
                        "process_call_wall_seconds": 5.9,
                        "pool_summary": {
                            "main": pool(3, "main"),
                            "timing": pool(3, "timing"),
                        },
                        "cuda_peak_stats_reset": True,
                        "cuda_memory": {
                            "allocated_bytes": 100,
                            "reserved_bytes": 200,
                            "max_allocated_bytes": 300,
                            "max_reserved_bytes": 400,
                        },
                    },
                },
                "final_pool_summary": {
                    "main": pool(3, "main"),
                    "timing": pool(3, "timing"),
                },
                "close_completed": True,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_analyzer_accepts_exact_capture_free_warm_request(tmp_path: Path) -> None:
    report = analyze(_manifest(tmp_path))

    assert report["pass"] is True
    assert report["exactness_pass"] is True
    assert report["capture_gate"]["pass"] is True
    assert report["cross_request_graph_hits"] == {"warm1": 18, "warm2": 36}
    assert report["pool_pass"] is True
    assert report["memory_gate"]["pass"] is True


def test_analyzer_rejects_warm_recapture(tmp_path: Path) -> None:
    report = analyze(_manifest(tmp_path, warm_capture=0.01))

    assert report["pass"] is False
    assert report["capture_gate"]["pass"] is False


def test_analyzer_fails_loudly_without_token_evidence(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    warm_profile_path = Path(manifest["results"]["warm1"]["profile_path"])
    warm_profile = json.loads(warm_profile_path.read_text(encoding="utf-8"))
    del warm_profile["generation"][0]["generated_token_ids"]
    warm_profile_path.write_text(json.dumps(warm_profile), encoding="utf-8")

    with pytest.raises(ValueError, match="generated_token_ids"):
        analyze(manifest_path)


def test_analyzer_rejects_warm_workspace_address_growth(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["results"]["warm1"]["pool_summary"]["main"]["workspaces"][0][
        "encoder_storage"
    ][0]["data_ptr"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = analyze(manifest_path)

    assert report["pool_pass"] is False
    assert report["pass"] is False


def test_analyzer_rejects_warm_arena_address_change(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["results"]["warm1"]["pool_summary"]["main"]["workspaces"][0][
        "arena_storage"
    ][0]["tensor_addresses"][0]["data_ptr"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = analyze(manifest_path)

    assert report["pool_checks"]["main"]["stable_arena_storage"] is False
    assert report["pool_pass"] is False


def test_analyzer_rejects_constant_cross_request_arena_identity(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for pass_name in ("warm1", "warm2"):
        for role in ("main", "timing"):
            manifest["results"][pass_name]["pool_summary"][role]["workspaces"][0][
                "arena_storage"
            ][0]["refreshed_window_identity"][0] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = analyze(manifest_path)

    assert report["pool_checks"]["main"]["arena_refresh_progression"] is False
    assert report["pool_pass"] is False


def test_analyzer_rejects_nonincreasing_cross_request_hits(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    warm2_path = Path(manifest["results"]["warm2"]["profile_path"])
    warm2 = json.loads(warm2_path.read_text(encoding="utf-8"))
    for row in warm2["generation"]:
        row["optimized_cuda_graphs"]["persistent_workspace"][
            "cross_request_graph_hits"
        ] = 9
    warm2_path.write_text(json.dumps(warm2), encoding="utf-8")

    report = analyze(manifest_path)

    assert report["cross_request_hits_pass"] is False
    assert report["pass"] is False


def test_analyzer_rejects_wrong_common_dispatch_topology(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for pass_name in ("cold", "warm1", "warm2"):
        profile_path = Path(manifest["results"][pass_name]["profile_path"])
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        for row in profile["generation"]:
            row["optimized_dispatch_mode"] = "wrong-topology"
        profile_path.write_text(json.dumps(profile), encoding="utf-8")

    report = analyze(manifest_path)

    assert report["profile_topology_pass"] is False
    assert report["pass"] is False


def test_analyzer_rejects_wrong_pool_topology_signature(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["results"]["warm2"]["pool_summary"]["main"][
        "topology_signature"
    ] = "wrong"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = analyze(manifest_path)

    assert report["pool_checks"]["main"]["topology_signature"] is False
    assert report["pool_pass"] is False


def test_analyzer_rejects_memory_growth_between_warm_passes(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["results"]["warm2"]["cuda_memory"]["reserved_bytes"] += (
        17 * 1024 * 1024
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = analyze(manifest_path)

    assert report["memory_gate"]["allocator_checks"][
        "reserved_bytes_bounded"
    ] is False
    assert report["memory_gate"]["pass"] is False
    assert report["pass"] is False


def test_analyzer_requires_peak_reset_evidence(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["results"]["warm2"]["cuda_peak_stats_reset"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = analyze(manifest_path)

    assert report["memory_gate"]["peak_reset_checks"]["warm2"] is False
    assert report["memory_gate"]["pass"] is False
