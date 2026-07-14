from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import utils.profile_split_kv_q1_component as split_profile
from utils.profile_split_kv_q1_component import (
    FIXED_MAIN_DECODE_REPLAYS,
    FIXED_MAIN_GENERATED_TOKENS,
    MAX_ABS_DRIFT,
    _paired_graph_rounds,
    select_live_buckets,
    split_counts_for_prefix,
    summarize_split_kv,
    validate_fixed_main_work,
    validate_live_graph_cache,
    validate_split_mask,
)


def _paired(
    accepted: float,
    candidate: float,
    *,
    conservative_delta: float | None = None,
    faster: bool | None = None,
):
    delta = accepted - candidate
    conservative = delta if conservative_delta is None else conservative_delta
    return {
        "accepted_median_ms_per_call": accepted,
        "candidate_median_ms_per_call": candidate,
        "median_paired_delta_ms_per_call": delta,
        "paired_delta_mad_ms_per_call": 0.0,
        "robust_delta_lower_bound_ms_per_call": delta,
        "noise_floor_ms_per_call": 0.0,
        "conservative_delta_lower_bound_ms_per_call": conservative,
        "faster_than_noise": conservative > 0 if faster is None else faster,
        "samples": [],
    }


def _candidate(
    ms: float,
    *,
    accepted: float,
    conservative_delta: float | None = None,
    faster: bool | None = None,
    drift: float = 0.0,
    passed: bool = True,
):
    return {
        "ms_per_call": ms,
        "paired_timing": _paired(
            accepted,
            ms,
            conservative_delta=conservative_delta,
            faster=faster,
        ),
        "memory_stable": True,
        "verifier": {"pass": passed},
        "drift": {
            "output_max_abs": drift,
            "cache_key_slot_max_abs": 0.0,
            "cache_value_slot_max_abs": 0.0,
        },
    }


def _bucket(accepted: float, count: int, candidates: dict):
    return {
        "accepted_ms_per_call": accepted,
        "accepted_memory_stable": True,
        "accepted_verifier": {"pass": True},
        "decode_replays": count,
        "candidates": candidates,
    }


def _graph_entry(prefix: int, count: int) -> dict:
    return {
        "active_prefix_length": prefix,
        "decode_replays": count,
        "graph": object(),
        "outputs": object(),
        "static_inputs": {},
    }


def test_split_policy_keeps_128_unsplit_and_sweeps_larger_buckets() -> None:
    assert split_counts_for_prefix(128) == (1,)
    assert split_counts_for_prefix(576) == (2, 4, 8)
    assert split_counts_for_prefix(640) == (2, 4, 8)


def test_live_graph_cache_accepts_dynamic_counts_and_all_buckets() -> None:
    cache = {
        ("large",): _graph_entry(896, 2),
        ("small",): _graph_entry(128, 3),
    }

    entries = validate_live_graph_cache(cache)

    assert list(entries) == [128, 896]
    assert select_live_buckets(entries, "all") == entries
    with pytest.raises(ValueError, match="missing sentinel"):
        select_live_buckets(entries, "sentinel")

    cache[("duplicate",)] = _graph_entry(128, 1)
    with pytest.raises(ValueError, match="duplicate"):
        validate_live_graph_cache(cache)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda entry: entry.update(decode_replays=0), "invalid replay count"),
        (lambda entry: entry.pop("static_inputs"), "missing"),
    ],
)
def test_live_graph_cache_fails_loudly_on_invalid_entries(mutation, error) -> None:
    entry = _graph_entry(128, 3)
    mutation(entry)

    with pytest.raises(ValueError, match=error):
        validate_live_graph_cache({("entry",): entry})


def test_fixed_main_work_reconciles_tokens_decode_and_prefill_windows() -> None:
    entries = {
        128: _graph_entry(128, 22),
        640: _graph_entry(640, FIXED_MAIN_DECODE_REPLAYS - 22),
    }
    run = {
        "processor": SimpleNamespace(
            last_generation_stats={
                "generated_tokens": FIXED_MAIN_GENERATED_TOKENS,
            }
        )
    }

    manifest = validate_fixed_main_work(run, entries)

    assert manifest["decode_replays"] == FIXED_MAIN_DECODE_REPLAYS
    assert manifest["prefill_windows"] == 87
    assert manifest["reconciled"]

    run["processor"].last_generation_stats["generated_tokens"] -= 1
    with pytest.raises(RuntimeError, match="token count changed"):
        validate_fixed_main_work(run, entries)

    run["processor"].last_generation_stats[
        "generated_tokens"
    ] = FIXED_MAIN_GENERATED_TOKENS
    entries[640]["decode_replays"] -= 1
    with pytest.raises(RuntimeError, match="replay count changed"):
        validate_fixed_main_work(run, entries)


def test_paired_rounds_run_repeated_abba_baab_with_restore(monkeypatch) -> None:
    timings = {"accepted": 0.2, "split_8": 0.1}
    restores = []

    def fake_time(graph, *, warmup, iters):
        assert warmup == 3
        assert iters == 7
        return {
            "ms_per_call": timings[graph],
            "allocated_before_bytes": 10,
            "allocated_after_bytes": 10,
            "memory_stable": True,
        }

    monkeypatch.setattr(split_profile, "_time_graph", fake_time)
    medians, paired, rounds, memory = _paired_graph_rounds(
        {"accepted": "accepted", "split_8": "split_8"},
        restore=lambda: restores.append(True),
        warmup=3,
        iters=7,
        cycles=2,
    )

    assert medians == pytest.approx({"accepted": 0.2, "split_8": 0.1})
    assert len(rounds) == 16
    assert len(restores) == 16
    assert {row["pattern"] for row in rounds} == {"ABBA", "BAAB"}
    assert paired["split_8"]["median_paired_delta_ms_per_call"] == pytest.approx(0.1)
    assert paired["split_8"]["conservative_delta_lower_bound_ms_per_call"] > 0
    assert memory == {"accepted": True, "split_8": True}


def test_summary_selects_largest_conservative_delta_and_weights_live_counts() -> None:
    report = summarize_split_kv(
        {
            "128": _bucket(
                0.10,
                1000,
                {
                    "split_1": _candidate(
                        0.04,
                        accepted=0.10,
                        conservative_delta=0.05,
                    )
                },
            ),
            "640": _bucket(
                0.20,
                500,
                {
                    "split_2": _candidate(
                        0.11,
                        accepted=0.20,
                        conservative_delta=0.07,
                    ),
                    "split_4": _candidate(
                        0.08,
                        accepted=0.20,
                        conservative_delta=0.10,
                    ),
                    "split_8": _candidate(
                        0.09,
                        accepted=0.20,
                        conservative_delta=0.08,
                    ),
                },
            ),
        },
        total_replays=1500,
        require_full_coverage=True,
    )

    assert report["correctness_pass"]
    assert report["coverage_pass"]
    assert report["selected"]["640"]["candidate"] == "split_4"
    assert report["measured_replays"] == 1500
    assert report["coverage_fraction"] == 1.0
    assert report["projected_main_saving_seconds"] == pytest.approx(1.2)
    assert report["component_pass"]


def test_summary_falls_back_to_accepted_for_drift_slower_or_noisy_split() -> None:
    report = summarize_split_kv(
        {
            "128": _bucket(
                1.0,
                100,
                {
                    "split_1": _candidate(
                        0.1,
                        accepted=1.0,
                        conservative_delta=0.8,
                        drift=MAX_ABS_DRIFT + 1e-7,
                    )
                },
            ),
            "192": _bucket(
                0.5,
                100,
                {
                    "split_2": _candidate(
                        0.6,
                        accepted=0.5,
                        conservative_delta=-0.1,
                    )
                },
            ),
            "256": _bucket(
                0.5,
                100,
                {
                    "split_4": _candidate(
                        0.49,
                        accepted=0.5,
                        conservative_delta=-0.001,
                        faster=False,
                    )
                },
            ),
        },
        total_replays=300,
        require_full_coverage=True,
    )

    assert report["correctness_pass"]
    assert not report["component_pass"]
    assert all(
        entry["candidate"] == "accepted" for entry in report["selected"].values()
    )
    assert report["projected_main_saving_seconds"] == 0.0
    assert report["candidate_failures"]["128"] == {
        "split_1": [f"output_max_abs_above_{MAX_ABS_DRIFT}"]
    }
    assert report["candidate_failures"]["192"] == {"split_2": ["not_faster_than_noise"]}


def test_full_coverage_is_required_for_component_pass() -> None:
    report = summarize_split_kv(
        {
            "640": _bucket(
                1.0,
                300,
                {
                    "split_8": _candidate(
                        0.25,
                        accepted=1.0,
                        conservative_delta=0.7,
                    )
                },
            )
        },
        total_replays=600,
        require_full_coverage=True,
    )

    assert report["projected_main_saving_seconds"] > 0
    assert not report["coverage_pass"]
    assert not report["component_pass"]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("accepted_memory_stable", False, "memory stable"),
        ("accepted_verifier", {"pass": False}, "cache verifier"),
    ],
)
def test_summary_rejects_an_invalid_accepted_control(field, value, error) -> None:
    bucket = _bucket(
        1.0,
        300,
        {
            "split_8": _candidate(
                0.25,
                accepted=1.0,
                conservative_delta=0.7,
            )
        },
    )
    bucket[field] = value

    with pytest.raises(ValueError, match=error):
        summarize_split_kv({"640": bucket}, total_replays=300)


def test_split_mask_rejects_an_all_masked_partition() -> None:
    mask = torch.zeros(8)
    mask[:2] = -torch.inf

    with pytest.raises(RuntimeError, match="all-masked partition"):
        validate_split_mask(mask, prefix=8, split_counts=(4,))

    validate_split_mask(torch.zeros(8), prefix=8, split_counts=(2, 4))
    validate_split_mask(None, prefix=8, split_counts=(2, 4))
