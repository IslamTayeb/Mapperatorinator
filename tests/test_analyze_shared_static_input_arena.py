from __future__ import annotations

import copy

import pytest

from utils.analyze_shared_static_input_arena import (
    EXPECTED_DISPATCH_DELTA_PATTERNS,
    analyze,
)


STAGES = (
    "prepare_inputs_for_generation",
    "graph_key_lookup",
    "static_input_refresh",
    "static_input_copies",
    "graph_replay",
    "status_d2h",
    "sync_model_kwargs",
)


def _runtime(*, candidate: bool) -> dict:
    stage_seconds = {
        name: (0.001 if name != "status_d2h" else 0.090)
        for name in STAGES
    }
    accounted = sum(stage_seconds.values())
    model_wall = accounted + 0.001
    return {
        "shared_static_input_arena": candidate,
        "static_input_arena_refreshes": 1 if candidate else 0,
        "static_input_arena_refresh_copy_calls": 4 if candidate else 0,
        "static_input_arena_refresh_copy_bytes": 256 if candidate else 0,
        "static_input_arena_graph_entries": 2 if candidate else 0,
        "copy_calls": 4 if candidate else 120,
        "transition_timing": {
            "schema_version": 1,
            "enabled": True,
            "model_wall_seconds": model_wall,
            "accounted_wall_seconds": accounted,
            "unattributed_wall_seconds": 0.001,
            "reconciliation_error_fraction": 0.001 / model_wall,
            "stages": {
                name: {"calls": 1, "wall_seconds": seconds}
                for name, seconds in stage_seconds.items()
            },
            "device": {
                "static_input_copies": {
                    "calls": 1 if candidate else 30,
                    "seconds": 0.0002 if candidate else 0.006,
                },
                "graph_replay": {"calls": 30, "seconds": 0.080},
            },
        },
    }


def _profile(*, candidate: bool) -> dict:
    return {
        "generation": [
            {
                "profile_label": label,
                "optimized_dispatch_mode": "approximate_weight_only_batch1",
                "optimized_dispatch_policy": {
                    "effective_native_q1_rope_cache_self_attention": {
                        "enabled": label == "main_generation",
                    },
                },
                "optimized_dispatch_capture_hits": {
                    "native_q1_rope_cache_self_attention": (
                        24 if label == "main_generation" else 0
                    ),
                    "q1_bmm_cross_attention": (
                        24 if label == "main_generation" else 0
                    ),
                    "int8_weight_mlp_tail": (
                        24 if label == "main_generation" else 0
                    ),
                    "fp16_packed_cross_projection_candidate": (
                        24 if label == "main_generation" else 0
                    ),
                },
                "optimized_cuda_graphs": {
                    "graph_count": 2,
                    "decode_replays": 100,
                    "buckets": {
                        "128": {"graph_count": 2, "decode_replays": 100}
                    },
                    "k8_candidate": {
                        **_runtime(candidate=candidate),
                        "block_size": 4,
                        "prefill_steps": 1,
                        "eligible_steps": 96,
                        "block_replays": 24,
                        "remainder_steps": 3,
                        "remainder_backend": "cuda_graph",
                        "remainder_graph_replays": 3,
                        "eager_remainder_steps": 0,
                        "physical_steps": 100,
                        "logical_steps": 100,
                        "wasted_steps": 0,
                        "rng_policy": "counter_request_seed_window_prompt_v2",
                        "rng_window_identity": 1,
                    },
                },
            }
            for label in ("timing_context", "main_generation")
        ]
    }


def _reciprocal(*, saving: float = 0.35) -> dict:
    return {
        "schema_version": "mapperatorinator.reciprocal-full-song-candidate.v1",
        "parity": {
            "cross_candidate_exact": True,
            "required_exact_labels": ["timing_context", "main_generation"],
            "required_exact_labels_pass": True,
            "dispatch_cache_topology": {
                "pass": True,
                "declared_patterns": list(EXPECTED_DISPATCH_DELTA_PATTERNS),
            },
            "output_divergence": {"final_map_equal": True},
        },
        "metrics": {
            "fixed_8294_main_seconds": {
                "baseline_median": 18.0,
                "candidate_median": 18.0 - saving,
                "improvement": saving,
            }
        },
    }


def _profiles() -> dict:
    return {
        "baseline_first": _profile(candidate=False),
        "candidate_first": _profile(candidate=True),
        "candidate_second": _profile(candidate=True),
        "baseline_second": _profile(candidate=False),
    }


def _append_prefill_eos_window(profiles: dict) -> None:
    for profile in profiles.values():
        source = next(
            record
            for record in profile["generation"]
            if record["profile_label"] == "main_generation"
        )
        record = copy.deepcopy(source)
        runtime = record["optimized_cuda_graphs"]["k8_candidate"]
        runtime.update({
            "eligible_steps": 0,
            "block_replays": 0,
            "remainder_steps": 0,
            "remainder_graph_replays": 0,
            "physical_steps": 1,
            "logical_steps": 1,
            "static_input_arena_refreshes": 0,
            "static_input_arena_refresh_copy_calls": 0,
            "static_input_arena_refresh_copy_bytes": 0,
            "static_input_arena_graph_entries": 0,
            "copy_calls": 0,
        })
        timing = runtime["transition_timing"]
        timing.update({
            "model_wall_seconds": 0.0,
            "accounted_wall_seconds": 0.0,
            "unattributed_wall_seconds": 0.0,
            "reconciliation_error_fraction": 0.0,
        })
        for stage in timing["stages"].values():
            stage.update({"calls": 0, "wall_seconds": 0.0})
        for device in timing["device"].values():
            device.update({"calls": 0, "seconds": 0.0})
        profile["generation"].append(record)


def test_shared_arena_gate_requires_exact_parity_and_point_three_seconds():
    report = analyze(_reciprocal(), _profiles())

    assert report["pass"] is True
    assert report["decision"] == "PROMOTE"
    assert report["fixed_main_steps"] == 8_294
    assert report["main_copy_calls"] == {
        "baseline_median": 120.0,
        "candidate_median": 4.0,
    }
    assert report["dispatch_metadata_exact"] is True


def test_shared_arena_gate_accepts_prefill_eos_zero_refresh_window():
    profiles = _profiles()
    _append_prefill_eos_window(profiles)

    report = analyze(_reciprocal(), profiles)

    main = report["transitions"]["candidate_first"]["main_generation"]
    assert main["records"] == 2
    assert main["measured_records"] == 1
    assert main["static_input_arena_refreshes"] == 1
    assert main["static_input_arena_refresh_copy_calls"] == 4


def test_shared_arena_gate_rejects_refresh_for_prefill_eos_window():
    profiles = _profiles()
    _append_prefill_eos_window(profiles)
    runtime = profiles["candidate_first"]["generation"][-1][
        "optimized_cuda_graphs"
    ]["k8_candidate"]
    runtime["static_input_arena_refreshes"] = 1
    runtime["static_input_arena_refresh_copy_calls"] = 4
    runtime["static_input_arena_refresh_copy_bytes"] = 256

    with pytest.raises(ValueError, match="zero graph replays require zero"):
        analyze(_reciprocal(), profiles)


def test_shared_arena_gate_stops_below_declared_saving():
    report = analyze(_reciprocal(saving=0.299), _profiles())

    assert report["pass"] is False
    assert report["decision"] == "STOP_SHARED_STATIC_INPUT_ARENA"


def test_shared_arena_gate_rejects_extra_refresh_or_reconciliation_gap():
    profiles = _profiles()
    runtime = profiles["candidate_first"]["generation"][0]["optimized_cuda_graphs"][
        "k8_candidate"
    ]
    runtime["static_input_arena_refreshes"] = 2
    with pytest.raises(ValueError, match="exactly four tensors once"):
        analyze(_reciprocal(), profiles)

    profiles = _profiles()
    timing = profiles["candidate_first"]["generation"][0]["optimized_cuda_graphs"][
        "k8_candidate"
    ]["transition_timing"]
    timing["unattributed_wall_seconds"] = 0.021 * timing["model_wall_seconds"]
    timing["accounted_wall_seconds"] = (
        timing["model_wall_seconds"] - timing["unattributed_wall_seconds"]
    )
    other_stage_seconds = sum(
        stage["wall_seconds"]
        for name, stage in timing["stages"].items()
        if name != "status_d2h"
    )
    timing["stages"]["status_d2h"]["wall_seconds"] = (
        timing["accounted_wall_seconds"] - other_stage_seconds
    )
    timing["reconciliation_error_fraction"] = 0.021
    with pytest.raises(ValueError, match="2% reconciliation"):
        analyze(_reciprocal(), profiles)


def test_shared_arena_gate_rejects_token_or_final_map_drift():
    reciprocal = _reciprocal()
    reciprocal["parity"]["cross_candidate_exact"] = False
    with pytest.raises(ValueError, match="changed tokens"):
        analyze(reciprocal, _profiles())

    reciprocal = _reciprocal()
    reciprocal["parity"]["output_divergence"]["final_map_equal"] = False
    with pytest.raises(ValueError, match="not byte-identical"):
        analyze(reciprocal, _profiles())


def test_shared_arena_gate_rejects_broad_delta_or_dispatch_drift():
    reciprocal = _reciprocal()
    reciprocal["parity"]["dispatch_cache_topology"]["declared_patterns"] = [
        "records.*[[]*].optimized_cuda_graphs.*"
    ]
    with pytest.raises(ValueError, match="too broad"):
        analyze(reciprocal, _profiles())

    profiles = _profiles()
    profiles["candidate_first"]["generation"][1][
        "optimized_dispatch_capture_hits"
    ]["q1_bmm_cross_attention"] -= 1
    with pytest.raises(ValueError, match="dispatch metadata"):
        analyze(_reciprocal(), profiles)


def test_shared_arena_gate_requires_arena_graph_capture_evidence():
    profiles = _profiles()
    for record in profiles["candidate_first"]["generation"]:
        record["optimized_cuda_graphs"]["k8_candidate"][
            "static_input_arena_graph_entries"
        ] = 0
    with pytest.raises(ValueError, match="captured no shared-arena graph"):
        analyze(_reciprocal(), profiles)
