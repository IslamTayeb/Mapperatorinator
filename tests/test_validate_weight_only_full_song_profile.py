from __future__ import annotations

import copy

import pytest

from utils.validate_weight_only_full_song_profile import (
    WeightOnlyProfileError,
    validate_profile,
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
    }


def _record(label: str, *, candidate: bool) -> dict:
    timing = label == "timing_context"
    hits = {
        "native_q1_rope_cache_self_attention": 10,
        "q1_bmm_cross_attention": 10,
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
    if candidate and not timing:
        policy["approximate_weight_only"] = {
            "requested": True,
            "enabled": True,
            "disabled_reason": None,
            "result_class": "documented-drift",
            "exactness_claim": False,
        }
        policy["effective_native_q1_rope_cache_self_attention"] = {
            "requested": True,
            "enabled": True,
            "owner": "approximate_weight_only",
            "kernel": "native_q1_rope_cache_attention",
            "standard_attention_hook_enabled": False,
            "split_kv_selector_enabled": True,
            "disabled_reason": None,
        }
        record["optimized_dispatch_mode"] = "approximate_weight_only_batch1"
        hits.update(
            {
                "native_q1_rope_cache_self_attention": 12,
                "q1_bmm_cross_attention": 12,
                "weight_only_self_attention_block": 12,
                "weight_only_mlp_tail": 12,
                "weight_only_final_projection": 1,
                "native_q1_rope_cache_self_attention_split_kv_8": 10,
                "native_q1_rope_cache_self_attention_split_kv_8_prefix_192": 3,
                "native_q1_rope_cache_self_attention_split_kv_8_prefix_640": 7,
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


def test_candidate_requires_enabled_main_and_disabled_timing_dispatch() -> None:
    report = validate_profile(_profile(candidate=True), role="candidate")

    assert report["candidate_enabled_for_main"] is True
    assert report["candidate_disabled_for_timing"] is True
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
    ] = {"enabled": False}
    with pytest.raises(WeightOnlyProfileError, match="must not request"):
        validate_profile(timing_policy, role="candidate")


def test_candidate_rejects_wrong_enabled_policy_or_legacy_metadata_name() -> None:
    wrong_policy = _profile(candidate=True)
    wrong_policy["generation"][1]["optimized_dispatch_policy"][
        "approximate_weight_only"
    ]["enabled"] = False
    with pytest.raises(WeightOnlyProfileError, match="invalid mixed-weight dispatch policy"):
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
        "native_q1_rope_cache_self_attention": 0,
        "native_q1_rope_cache_self_attention_split_kv_8": 0,
        "split_kv_prefix_counts": {},
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
