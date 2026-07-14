from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from utils.validate_weight_only_full_song_profile import (
    WeightOnlyProfileError,
    validate_profile,
)
from utils.validate_fp32_timing_native_self_profile import (
    COMPOSITION_VERSION as TIMING_COMPOSITION_VERSION,
    VERSION as TIMING_SCOUT_VERSION,
    validate as validate_timing_scout,
)


def _metadata() -> dict:
    return {
        "version": "approximate-fp16-weights-fp32-state-v2",
        "result_class": "documented-drift",
        "exactness_claim": False,
        "fp32_activations_caches_reductions_logits": True,
        "fp16_weight_regions": [
            "self_qkv",
            "self_output",
            "mlp_fc1",
            "mlp_fc2",
            "final_logits",
        ],
        "fp32_selected_decode_matrix_regions": ["cross_query", "cross_output"],
        "cross_candidate": {
            "mode": "accepted",
            "scope": "main-model-only",
            "attention_accumulation": "fp32",
            "production_selector_unchanged": True,
            "result_class": "control",
            "exactness_claim": False,
            "accepted_q1_bmm": True,
        },
    }


def _record(label: str, *, candidate: bool) -> dict:
    timing = label == "timing_context"
    hits = {
        "native_q1_rope_cache_self_attention": 0 if timing else 12,
        "q1_bmm_cross_attention": 10,
        "native_cross_mlp_tail": 0 if timing else 12,
    }
    policy = {
        "q1_bmm_cross_attention": {"requested": True, "enabled": True},
    }
    record = {
        "profile_label": label,
        "optimized_dispatch_mode": "accepted_batch1",
        "optimized_dispatch_policy": policy,
        "optimized_dispatch_capture_hits": hits,
    }
    policy["effective_native_q1_rope_cache_self_attention"] = (
        {
            "requested": False,
            "enabled": False,
            "owner": None,
            "kernel": None,
            "standard_attention_hook_enabled": False,
            "split_kv_selector_enabled": False,
            "disabled_reason": "timing_context",
        }
        if timing
        else {
            "requested": True,
            "enabled": True,
            "owner": (
                "approximate_weight_only" if candidate else "accepted_attention_hook"
            ),
            "kernel": "native_q1_rope_cache_attention",
            "standard_attention_hook_enabled": not candidate,
            "split_kv_selector_enabled": True,
            "disabled_reason": None,
        }
    )
    if not timing:
        hits.update(
            {
                "native_q1_rope_cache_self_attention_split_kv_8": 10,
                "native_q1_rope_cache_self_attention_split_kv_8_prefix_192": 3,
                "native_q1_rope_cache_self_attention_split_kv_8_prefix_640": 7,
            }
        )
    if candidate and not timing:
        policy["approximate_weight_only"] = {
            "requested": True,
            "enabled": True,
            "disabled_reason": None,
            "result_class": "documented-drift",
            "exactness_claim": False,
        }
    if candidate:
        hits.update(
            {
                key: 0
                for key in (
                    "weight_only_self_attention_block",
                    "weight_only_mlp_tail",
                    "weight_only_final_projection",
                )
            }
        )
    if candidate and not timing:
        record["optimized_dispatch_mode"] = "approximate_weight_only_batch1"
        hits.update(
            {
                "native_q1_rope_cache_self_attention": 12,
                "q1_bmm_cross_attention": 12,
                "native_cross_mlp_tail": 0,
                "weight_only_self_attention_block": 12,
                "weight_only_mlp_tail": 12,
                "weight_only_final_projection": 1,
            }
        )
    return record


def _profile(*, candidate: bool) -> dict:
    metadata = {}
    if candidate:
        metadata["optimized_approximate_weight_only"] = _metadata()
    return {
        "schema_version": 1,
        "metadata": metadata,
        "generation": [
            _record("timing_context", candidate=candidate),
            _record("main_generation", candidate=candidate),
        ],
    }


def _cross_profile(mode: str) -> dict:
    profile = _profile(candidate=True)
    metadata = profile["metadata"]["optimized_approximate_weight_only"]
    cross = metadata["cross_candidate"]
    hits = profile["generation"][1]["optimized_dispatch_capture_hits"]
    if mode == "fp16_packed_projections":
        metadata["fp16_weight_regions"].extend(("cross_query", "cross_output"))
        metadata["fp32_selected_decode_matrix_regions"] = []
        cross.clear()
        cross.update(
            {
                "mode": mode,
                "scope": "main-model-only",
                "attention_accumulation": "fp32",
                "production_selector_unchanged": True,
                "result_class": "documented-drift",
                "exactness_claim": False,
                "packed_projection_weights": True,
                "projection_delta_only": True,
                "accepted_q1_bmm": True,
                "incremental_exactness_required": True,
            }
        )
        hits["fp16_packed_cross_projection_candidate"] = 12
    else:
        raise AssertionError(mode)
    return profile


def test_candidate_requires_enabled_main_and_disabled_timing_dispatch() -> None:
    report = validate_profile(_profile(candidate=True), role="candidate")

    assert report["candidate_enabled_for_main"] is True
    assert report["candidate_disabled_for_timing"] is True
    assert report["timing_effective_self_attention_counts"] == {
        "native_q1_rope_cache_self_attention": 0,
        "native_q1_rope_cache_self_attention_split_kv_8": 0,
        "split_kv_prefix_counts": {},
    }
    assert report["main_weight_only_dispatch_counts"] == {
        "weight_only_self_attention_block": 12,
        "weight_only_mlp_tail": 12,
        "weight_only_final_projection": 1,
    }
    assert report["main_effective_self_attention_counts"] == {
        "native_q1_rope_cache_self_attention": 12,
        "native_q1_rope_cache_self_attention_split_kv_8": 10,
        "split_kv_prefix_counts": {192: 3, 640: 7},
        "accepted_fallback": 2,
    }


def test_selected_cross_candidate_requires_truthful_metadata_and_dispatch() -> None:
    mode = "fp16_packed_projections"
    report = validate_profile(
        _cross_profile(mode),
        role="candidate",
        cross_mode=mode,
    )

    assert report["cross_mode"] == mode
    assert report["main_cross_candidate_dispatch_counts"] == {
        "fp16_packed_cross_projection_candidate": 12,
    }


def _timing_scout_metadata() -> dict:
    return {
        "version": TIMING_SCOUT_VERSION,
        "scope": "timing-context-only",
        "precision": "fp32",
        "torch_dtype": "torch.float32",
        "result_class": "documented-drift",
        "exactness_claim": False,
        "native_q1_self_attention": True,
        "native_q1_rope_cache_self_attention": True,
        "split_kv_selector": True,
        "q1_bmm_cross_attention_retained": True,
        "native_cross_mlp_tail_retained_disabled": True,
        "approximate_weight_only_retained_disabled": True,
        "original_decoder_forward_retained": True,
        "production_selector_unchanged": True,
    }


def _timing_scout_profile() -> tuple[dict, dict]:
    profile = _cross_profile("fp16_packed_projections")
    for record in profile["generation"]:
        record["precision"] = "fp32"
    timing = profile["generation"][0]
    policy = timing["optimized_dispatch_policy"]
    hits = timing["optimized_dispatch_capture_hits"]
    timing["optimized_dispatch_mode"] = "fp32_timing_native_self_batch1"
    policy["native_q1_self_attention"] = {"enabled": True}
    policy["native_q1_rope_cache_self_attention"] = {"enabled": True}
    policy["native_cross_mlp_tail"] = {"enabled": False}
    policy["effective_native_q1_rope_cache_self_attention"] = {
        "requested": True,
        "enabled": True,
        "owner": "accepted_attention_hook",
        "kernel": "native_q1_rope_cache_attention",
        "standard_attention_hook_enabled": True,
        "split_kv_selector_enabled": True,
        "disabled_reason": None,
    }
    policy["timing_native_self_scout"] = {
        "requested": True,
        "enabled": True,
        "disabled_reason": None,
        "result_class": "documented-drift",
        "exactness_claim": False,
    }
    timing["optimized_timing_native_self_scout"] = _timing_scout_metadata()
    hits["native_q1_rope_cache_self_attention"] = 12
    hits["native_q1_self_attention"] = 0
    initialization = {
        "timing_native_self_scout": {
            **_timing_scout_metadata(),
            "composition_version": TIMING_COMPOSITION_VERSION,
            "incremental_control": "selected-k1-int8-fp16-cross",
            "timing_model_distinct_from_main": True,
            "fixed_timing_work_required": True,
            "complete_request_wall_required": True,
        }
    }
    return profile, initialization


def test_timing_native_self_contract_retains_cross_and_main_composition() -> None:
    profile, initialization = _timing_scout_profile()
    cached_record = copy.deepcopy(profile["generation"][0])
    cached_record["optimized_dispatch_capture_hits"][
        "native_q1_rope_cache_self_attention"
    ] = 0
    cached_record["optimized_dispatch_capture_hits"]["q1_bmm_cross_attention"] = 0
    profile["generation"].insert(1, cached_record)

    report = validate_timing_scout(profile, initialization, role="candidate")

    timing = report["dispatch_totals"]["timing_context"]
    assert timing["native_rope_cache_self"] == 12
    assert timing["split_kv"] == 0
    assert timing["q1_bmm_cross"] == 10
    assert timing["native_cross_mlp"] == 0
    assert report["base_composition"]["cross_mode"] == "fp16_packed_projections"
    assert report["fixed_timing_work_required"] is True
    assert report["complete_request_wall_required"] is True


def test_timing_native_self_contract_requires_at_least_one_capture_hit() -> None:
    profile, _ = _timing_scout_profile()
    profile["generation"][0]["optimized_dispatch_capture_hits"][
        "native_q1_rope_cache_self_attention"
    ] = 0

    with pytest.raises(
        WeightOnlyProfileError,
        match="did not execute native self-attention across any record",
    ):
        validate_profile(
            profile,
            role="candidate",
            cross_mode="fp16_packed_projections",
            timing_native_self=True,
        )


def test_timing_native_self_contract_rejects_cross_mlp_or_missing_marker() -> None:
    profile, initialization = _timing_scout_profile()
    profile["generation"][0]["optimized_dispatch_capture_hits"][
        "native_cross_mlp_tail"
    ] = 1
    with pytest.raises(WeightOnlyProfileError, match="native cross/MLP"):
        validate_timing_scout(profile, initialization, role="candidate")

    profile, initialization = _timing_scout_profile()
    initialization.pop("timing_native_self_scout")
    with pytest.raises(WeightOnlyProfileError, match="must be an object"):
        validate_timing_scout(profile, initialization, role="candidate")


def test_timing_native_self_validator_runs_outside_repo_with_empty_pythonpath(
    tmp_path: Path,
) -> None:
    profile, initialization = _timing_scout_profile()
    profile_path = tmp_path / "profile.json"
    initialization_path = tmp_path / "initialization.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    initialization_path.write_text(json.dumps(initialization), encoding="utf-8")
    script = (
        Path(__file__).resolve().parents[1]
        / "utils/validate_fp32_timing_native_self_profile.py"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = ""

    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--initialization" in help_result.stdout
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--profile",
            str(profile_path),
            "--initialization",
            str(initialization_path),
            "--role",
            "candidate",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["pass"] is True


def test_cross_candidate_mode_mismatch_fails_loudly() -> None:
    with pytest.raises(WeightOnlyProfileError, match="cross_mode must"):
        validate_profile(
            _cross_profile("fp16_packed_projections"),
            role="candidate",
            cross_mode="split8_attention",
        )


def test_candidate_rejects_zero_main_or_nonzero_timing_dispatch() -> None:
    zero_main = _profile(candidate=True)
    zero_main["generation"][1]["optimized_dispatch_capture_hits"][
        "weight_only_final_projection"
    ] = 0
    with pytest.raises(WeightOnlyProfileError, match="did not execute every"):
        validate_profile(zero_main, role="candidate")

    timing_dispatch = _profile(candidate=True)
    timing_dispatch["generation"][0]["optimized_dispatch_capture_hits"][
        "weight_only_mlp_tail"
    ] = 1
    with pytest.raises(WeightOnlyProfileError, match="contains mixed-weight dispatch"):
        validate_profile(timing_dispatch, role="candidate")

    timing_policy = _profile(candidate=True)
    timing_policy["generation"][0]["optimized_dispatch_policy"][
        "approximate_weight_only"
    ] = {
        "requested": True,
        "enabled": False,
        "disabled_reason": "timing_context",
        "result_class": "documented-drift",
        "exactness_claim": False,
    }
    with pytest.raises(WeightOnlyProfileError, match="invalid mixed-weight dispatch"):
        validate_profile(timing_policy, role="candidate")


def test_candidate_rejects_wrong_enabled_policy_or_legacy_metadata_name() -> None:
    wrong_policy = _profile(candidate=True)
    wrong_policy["generation"][1]["optimized_dispatch_policy"][
        "approximate_weight_only"
    ]["enabled"] = False
    with pytest.raises(
        WeightOnlyProfileError, match="invalid mixed-weight dispatch policy"
    ):
        validate_profile(wrong_policy, role="candidate")

    legacy = _profile(candidate=True)
    metadata = legacy["metadata"]["optimized_approximate_weight_only"]
    metadata["fp32_weight_regions"] = metadata.pop(
        "fp32_selected_decode_matrix_regions"
    )
    with pytest.raises(WeightOnlyProfileError, match="invalid mixed-weight metadata"):
        validate_profile(legacy, role="candidate")


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda record: record["optimized_dispatch_policy"].pop(
                "effective_native_q1_rope_cache_self_attention"
            ),
            "effective native q1 policy",
        ),
        (
            lambda record: record["optimized_dispatch_policy"][
                "effective_native_q1_rope_cache_self_attention"
            ].update({"owner": "accepted_attention_hook"}),
            "effective native q1 policy",
        ),
        (
            lambda record: record["optimized_dispatch_capture_hits"].update(
                {"native_q1_rope_cache_self_attention": 11}
            ),
            "native q1 count does not match",
        ),
        (
            lambda record: record["optimized_dispatch_capture_hits"].update(
                {"q1_bmm_cross_attention": 11}
            ),
            "q1 BMM count does not match",
        ),
        (
            lambda record: record["optimized_dispatch_capture_hits"].update(
                {"native_q1_rope_cache_self_attention_split_kv_8": 9}
            ),
            "aggregate does not equal",
        ),
        (
            lambda record: record["optimized_dispatch_capture_hits"].update(
                {
                    "native_q1_rope_cache_self_attention_split_kv_8_prefix_128": 1,
                    "native_q1_rope_cache_self_attention_split_kv_8": 11,
                }
            ),
            "unsupported split-KV prefix",
        ),
    ),
)
def test_candidate_rejects_inconsistent_effective_self_attention(
    mutation,
    message,
) -> None:
    profile = _profile(candidate=True)
    mutation(profile["generation"][1])
    with pytest.raises(WeightOnlyProfileError, match=message):
        validate_profile(profile, role="candidate")


def test_candidate_requires_both_split_kv_and_accepted_fallback() -> None:
    no_split = _profile(candidate=True)
    hits = no_split["generation"][1]["optimized_dispatch_capture_hits"]
    hits["native_q1_rope_cache_self_attention_split_kv_8"] = 0
    hits["native_q1_rope_cache_self_attention_split_kv_8_prefix_192"] = 0
    hits["native_q1_rope_cache_self_attention_split_kv_8_prefix_640"] = 0
    with pytest.raises(WeightOnlyProfileError, match="did not execute split-KV"):
        validate_profile(no_split, role="candidate")

    no_fallback = _profile(candidate=True)
    hits = no_fallback["generation"][1]["optimized_dispatch_capture_hits"]
    hits["native_q1_rope_cache_self_attention_split_kv_8"] = 12
    hits["native_q1_rope_cache_self_attention_split_kv_8_prefix_640"] = 9
    with pytest.raises(WeightOnlyProfileError, match="accepted native q1 fallback"):
        validate_profile(no_fallback, role="candidate")


def test_baseline_requires_candidate_metadata_policy_and_counters_absent() -> None:
    report = validate_profile(_profile(candidate=False), role="baseline")
    assert report["main_weight_only_dispatch_counts"] == {
        "weight_only_self_attention_block": 0,
        "weight_only_mlp_tail": 0,
        "weight_only_final_projection": 0,
    }
    assert report["main_effective_self_attention_counts"] == {
        "native_q1_rope_cache_self_attention": 12,
        "native_q1_rope_cache_self_attention_split_kv_8": 10,
        "split_kv_prefix_counts": {192: 3, 640: 7},
        "accepted_fallback": 2,
    }

    for mutation, message in (
        (
            lambda profile: profile["metadata"].update(
                {"optimized_approximate_weight_only": _metadata()}
            ),
            "initialization metadata",
        ),
        (
            lambda profile: profile["generation"][1][
                "optimized_dispatch_capture_hits"
            ].update({"weight_only_mlp_tail": 0}),
            "dispatch counters",
        ),
        (
            lambda profile: profile["generation"][1][
                "optimized_dispatch_policy"
            ].update({"approximate_weight_only": {"enabled": False}}),
            "must not request",
        ),
    ):
        profile = copy.deepcopy(_profile(candidate=False))
        mutation(profile)
        with pytest.raises(WeightOnlyProfileError, match=message):
            validate_profile(profile, role="baseline")


def test_baseline_requires_exact_split_kv_and_accepted_fallback() -> None:
    no_split = _profile(candidate=False)
    hits = no_split["generation"][1]["optimized_dispatch_capture_hits"]
    hits["native_q1_rope_cache_self_attention_split_kv_8"] = 0
    hits["native_q1_rope_cache_self_attention_split_kv_8_prefix_192"] = 0
    hits["native_q1_rope_cache_self_attention_split_kv_8_prefix_640"] = 0
    with pytest.raises(WeightOnlyProfileError, match="exact split-KV"):
        validate_profile(no_split, role="baseline")

    no_fallback = _profile(candidate=False)
    hits = no_fallback["generation"][1]["optimized_dispatch_capture_hits"]
    hits["native_q1_rope_cache_self_attention_split_kv_8"] = 12
    hits["native_q1_rope_cache_self_attention_split_kv_8_prefix_640"] = 9
    with pytest.raises(WeightOnlyProfileError, match="accepted native q1 fallback"):
        validate_profile(no_fallback, role="baseline")


def test_baseline_allows_zero_dispatch_windows_when_stage_aggregate_is_valid() -> None:
    profile = _profile(candidate=False)
    zero_decode_window = copy.deepcopy(profile["generation"][1])
    hits = zero_decode_window["optimized_dispatch_capture_hits"]
    for key in tuple(hits):
        if key.startswith("native_q1_rope_cache_self_attention") or key in {
            "q1_bmm_cross_attention",
            "native_cross_mlp_tail",
        }:
            hits[key] = 0
    profile["generation"].insert(1, zero_decode_window)

    report = validate_profile(profile, role="baseline")

    assert report["baseline_main_dispatch_counts"] == {
        "q1_bmm_cross_attention": 10,
        "native_cross_mlp_tail": 12,
    }


@pytest.mark.parametrize(
    "dispatch",
    ["q1_bmm_cross_attention", "native_cross_mlp_tail"],
)
def test_baseline_requires_accepted_dispatch_across_main_stage(dispatch: str) -> None:
    profile = _profile(candidate=False)
    profile["generation"][1]["optimized_dispatch_capture_hits"][dispatch] = 0

    with pytest.raises(WeightOnlyProfileError, match="every accepted region"):
        validate_profile(profile, role="baseline")
