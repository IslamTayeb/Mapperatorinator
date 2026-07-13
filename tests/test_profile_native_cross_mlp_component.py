from __future__ import annotations

from utils.profile_native_cross_mlp_component import summarize_component


def _entry(*, accepted: float, candidate: float, count: int = 100):
    return {
        "decode_replays": count,
        "accepted_ms_per_call": accepted,
        "candidate_ms_per_call": candidate,
        "candidate_checks": {
            "finite_outputs": True,
            "cache_shapes_valid": True,
            "active_slot_writes_valid": True,
            "future_slots_untouched": True,
            "cross_cache_unchanged": True,
            "storage_ownership_valid": True,
            "candidate_repeat_deterministic": True,
            "graph_repeat_deterministic": True,
            "memory_stable": True,
        },
        "drift": {
            "cache_key_slot_max_abs": 0.0,
            "cache_value_slot_max_abs": 0.0,
            "logits_max_abs": 1e-6,
        },
    }


def test_summary_weights_live_counts_and_separates_sizing_from_correctness() -> None:
    report = summarize_component(
        {
            "128": _entry(accepted=2.0, candidate=1.0, count=1000),
            "640": _entry(accepted=3.0, candidate=1.0, count=250),
        },
        total_replays=2500,
    )

    assert report["correctness_pass"]
    assert report["measured_replays"] == 1250
    assert report["coverage_fraction"] == 0.5
    assert report["measured_saving_seconds"] == 1.5
    assert report["sizing_pass"]


def test_summary_fails_nonexact_self_cache_and_large_logit_drift() -> None:
    entry = _entry(accepted=2.0, candidate=1.0)
    entry["drift"]["cache_key_slot_max_abs"] = 1e-8
    entry["drift"]["logits_max_abs"] = 0.00101

    report = summarize_component({"128": entry}, total_replays=100)

    assert not report["correctness_pass"]
    assert report["correctness_failures"] == {
        "128": [
            "cache_key_slot_not_exact",
            "logits_drift_above_limit",
        ]
    }
