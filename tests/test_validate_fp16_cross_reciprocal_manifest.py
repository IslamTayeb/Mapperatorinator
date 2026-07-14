from __future__ import annotations

import json

import pytest

from utils.validate_fp16_cross_reciprocal_manifest import (
    CONTROL_RUNTIME,
    CrossReciprocalManifestError,
    ROLES,
    validate_run_root,
)


MODE = "fp16_packed_projections"


def _initialization(role: str) -> dict:
    candidate = role.startswith("candidate")
    payload = {
        "combined_runtime": (
            f"{CONTROL_RUNTIME.removesuffix('-v1')}-{MODE}-v1"
            if candidate
            else CONTROL_RUNTIME
        ),
        "shared_rope": {
            "version": "shared-decoder-rope-v1",
            "scope": "main-model-only",
            "incremental_exactness_claim": True,
            "original_decoder_forward_required": True,
            "stats": {
                "forwards": 101,
                "computes": 101,
                "expected_computes": 101,
                "reuses": 1111,
                "expected_reuses": 1111,
            },
        },
        "int8_mlp_overlay": {"dispatch_counter": "int8_weight_mlp_tail"},
        "cross_candidate": {
            "mode": MODE if candidate else "accepted",
            "accepted_q1_bmm": True,
            "attention_accumulation": "fp32",
            "production_selector_unchanged": True,
        },
    }
    if candidate:
        payload["cross_runtime"] = {
            "mode": MODE,
            "incremental_control": CONTROL_RUNTIME,
            "incremental_exactness_required": True,
            "packed_projection_delta_only": True,
            "accepted_q1_bmm_required": True,
            "original_decoder_forward_required": True,
        }
    return payload


def _weight(role: str) -> dict:
    candidate = role.startswith("candidate")
    return {
        "role": "candidate",
        "cross_mode": MODE if candidate else "accepted",
        "candidate_enabled_for_main": True,
        "candidate_disabled_for_timing": True,
        "main_weight_only_dispatch_counts": {
            "weight_only_self_attention_block": 168,
            "weight_only_mlp_tail": 168,
            "weight_only_final_projection": 14,
        },
        "main_q1_bmm_cross_attention_count": 168,
        "main_cross_candidate_dispatch_counts": {
            "fp16_packed_cross_projection_candidate": 168 if candidate else 0,
        },
        "main_effective_self_attention_counts": {
            "native_q1_rope_cache_self_attention": 168,
            "native_q1_rope_cache_self_attention_split_kv_8": 144,
            "split_kv_prefix_counts": {"192": 12, "640": 12},
            "accepted_fallback": 24,
        },
    }


def _int8() -> dict:
    return {
        "role": "candidate",
        "overlay_present": True,
        "pass": True,
        "dispatch_totals": {"timing_context": 0, "main_generation": 168},
        "mixed_mlp_hook_totals": {"timing_context": 0, "main_generation": 168},
    }


def _k1() -> dict:
    return {
        "role": "candidate",
        "remainder_backend": "cuda_graph",
        "pass": True,
        "labels": {
            "timing_context": {
                "remainder_steps": 42,
                "remainder_graph_replays": 42,
                "eager_remainder_steps": 0,
            },
            "main_generation": {
                "remainder_steps": 98,
                "remainder_graph_replays": 98,
                "eager_remainder_steps": 0,
            },
        },
    }


def _k4() -> dict:
    return {
        "role": "candidate",
        "block_size": 4,
        "rng_policy": "counter_request_seed_window_prompt_v2",
        "pass": True,
        "labels": {
            "timing_context": {"block_replays": 185, "eligible_steps": 740},
            "main_generation": {"block_replays": 2116, "eligible_steps": 8464},
        },
    }


def _write_manifest(tmp_path, mutation=None):
    initializations = {role: _initialization(role) for role in ROLES}
    reports = {
        role: {
            "weight": _weight(role),
            "int8": _int8(),
            "k1": _k1(),
            "k4": _k4(),
        }
        for role in ROLES
    }
    if mutation is not None:
        mutation(initializations, reports)
    init_dir = tmp_path / "candidate-initialization"
    init_dir.mkdir()
    for role in ROLES:
        (init_dir / f"{role}.json").write_text(
            json.dumps(initializations[role]), encoding="utf-8"
        )
        for key, suffix in (
            ("weight", "weight-only-validation"),
            ("int8", "int8-mlp-validation"),
            ("k1", "k1-remainder-validation"),
            ("k4", "k4-validation"),
        ):
            (tmp_path / f"{role}.{suffix}.json").write_text(
                json.dumps(reports[role][key]), encoding="utf-8"
            )


def test_live_four_arm_manifest_retains_every_common_runtime_component(
    tmp_path,
) -> None:
    _write_manifest(tmp_path)

    report = validate_run_root(tmp_path, cross_mode=MODE)

    assert report["pass"] is True
    assert report["incremental_delta"] == "fp16-packed-cross-projections-only"
    common = report["common_topology"]
    assert common["q1_bmm_cross_attention"] == 168
    assert (
        common["self_attention"][
            "native_q1_rope_cache_self_attention_split_kv_8"
        ]
        == 144
    )
    assert common["self_attention"]["accepted_fallback"] == 24
    assert common["int8_dispatches"]["main_generation"] == 168
    assert common["k1_labels"]["main_generation"]["eager_remainder_steps"] == 0
    assert common["k4_labels"]["main_generation"]["block_replays"] == 2116
    assert report["roles"]["baseline_first"]["cross_mode"] == "accepted"
    assert report["roles"]["candidate_first"]["cross_mode"] == MODE


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda init, reports: init["baseline_first"].pop("shared_rope"),
            "baseline_first.shared_rope",
        ),
        (
            lambda init, reports: reports["baseline_first"]["k1"]["labels"][
                "main_generation"
            ].update({"eager_remainder_steps": 1}),
            "baseline_first.k1.main_generation.eager_steps",
        ),
        (
            lambda init, reports: reports["candidate_first"]["weight"].update(
                {"main_q1_bmm_cross_attention_count": 167}
            ),
            "candidate_first.weight.q1_bmm",
        ),
        (
            lambda init, reports: reports["candidate_second"]["weight"][
                "main_effective_self_attention_counts"
            ].update({"accepted_fallback": 0}),
            "candidate_second.weight.self_attention.accepted_fallback",
        ),
        (
            lambda init, reports: reports["candidate_second"]["int8"].update(
                {"overlay_present": False}
            ),
            "candidate_second.int8.overlay_present",
        ),
    ),
)
def test_manifest_fails_loudly_when_common_topology_is_lost(
    tmp_path, mutation, message
) -> None:
    _write_manifest(tmp_path, mutation=mutation)

    with pytest.raises(CrossReciprocalManifestError, match=message):
        validate_run_root(tmp_path, cross_mode=MODE)
