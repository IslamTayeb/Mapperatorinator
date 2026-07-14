from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from utils.profile_coalesced_split_kv_component import (
    FIXED_MAIN_DECODE_REPLAYS,
    FIXED_MAIN_GENERATED_TOKENS,
    SAVING_TARGET_SECONDS,
    summarize_component,
    validate_fixed_main_work,
    validate_live_graph_cache,
)


def _entry(*, accepted: float, candidate: float, count: int, drift: float = 0.0):
    return {
        "decode_replays": count,
        "accepted_ms_per_call": accepted,
        "candidate_ms_per_call": candidate,
        "accepted_memory_stable": True,
        "accepted_verifier": {"pass": True},
        "candidate_memory_stable": True,
        "candidate_verifier": {"pass": True},
        "drift": {
            "output_max_abs": drift,
            "cache_key_slot_max_abs": 0.0,
            "cache_value_slot_max_abs": 0.0,
        },
    }


def _graph_entry(prefix: int, count: int):
    return {
        "active_prefix_length": prefix,
        "decode_replays": count,
        "graph": object(),
        "outputs": object(),
        "static_inputs": {},
    }


def test_component_wrapper_fits_one_backfill_window() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts/dcc/profile_coalesced_split_kv_component.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --gres=gpu:2080:1" in source
    assert "#SBATCH --time=00:30:00" in source


def test_summary_requires_full_coverage_correctness_and_half_second_saving() -> None:
    # Twelve layers * 1000 calls * 0.05 ms = 0.6 seconds.
    report = summarize_component(
        {"640": _entry(accepted=0.10, candidate=0.05, count=1000)},
        total_replays=1000,
    )

    assert SAVING_TARGET_SECONDS == 0.5
    assert report["coverage_pass"]
    assert report["correctness_pass"]
    assert report["projected_main_saving_seconds"] == pytest.approx(0.6)
    assert report["promotion_pass"]

    incomplete = summarize_component(
        {"640": _entry(accepted=0.10, candidate=0.05, count=500)},
        total_replays=1000,
    )
    assert not incomplete["coverage_pass"]
    assert not incomplete["promotion_pass"]

    drifted = _entry(accepted=0.10, candidate=0.01, count=1000, drift=0.00101)
    failed = summarize_component({"640": drifted}, total_replays=1000)
    assert not failed["correctness_pass"]
    assert not failed["promotion_pass"]


def test_live_work_manifest_fails_loudly_if_fixed_work_changes() -> None:
    entries = validate_live_graph_cache(
        {
            "small": _graph_entry(128, 87),
            "large": _graph_entry(640, FIXED_MAIN_DECODE_REPLAYS - 87),
        }
    )
    run = {
        "processor": SimpleNamespace(
            last_generation_stats={"generated_tokens": FIXED_MAIN_GENERATED_TOKENS}
        )
    }

    manifest = validate_fixed_main_work(run, entries)
    assert manifest["decode_replays"] == FIXED_MAIN_DECODE_REPLAYS

    run["processor"].last_generation_stats["generated_tokens"] -= 1
    with pytest.raises(RuntimeError, match="fixed SALVALAI work changed"):
        validate_fixed_main_work(run, entries)


def test_live_graph_manifest_rejects_duplicates_and_missing_state() -> None:
    duplicate = {
        "a": _graph_entry(128, 1),
        "b": _graph_entry(128, 2),
    }
    with pytest.raises(ValueError, match="duplicate"):
        validate_live_graph_cache(duplicate)

    missing = _graph_entry(128, 1)
    missing.pop("static_inputs")
    with pytest.raises(ValueError, match="missing"):
        validate_live_graph_cache({"a": missing})
