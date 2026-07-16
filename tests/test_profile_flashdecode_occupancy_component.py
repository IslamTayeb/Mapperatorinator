from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from utils.profile_flashdecode_occupancy_component import (
    BASE_TIP_CHOICE,
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
        root / "scripts/dcc/profile_flashdecode_occupancy_component.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --gres=gpu:2080:1" in source
    assert "#SBATCH --time=00:30:00" in source
    assert "profile_flashdecode_occupancy_component.py" in source
    assert "EXCLUDE_NODES" in source or "--exclude=" in source


def test_base_tip_choice_documents_coalesced_over_compiled_cross() -> None:
    assert "coalesced-split-kv" in BASE_TIP_CHOICE["selected"]
    assert "compiled-cross" in BASE_TIP_CHOICE["alternative"]
    assert "attention-only" in BASE_TIP_CHOICE["reason"]


def test_summary_requires_full_coverage_correctness_and_scout_saving() -> None:
    # Twelve layers * 1000 calls * 0.01 ms = 0.12 seconds.
    report = summarize_component(
        {"640": _entry(accepted=0.04, candidate=0.03, count=1000)},
        total_replays=1000,
    )

    assert SAVING_TARGET_SECONDS == 0.1
    assert report["coverage_pass"]
    assert report["correctness_pass"]
    assert report["projected_main_saving_seconds"] == pytest.approx(0.12)
    assert report["promotion_pass"]

    incomplete = summarize_component(
        {"640": _entry(accepted=0.04, candidate=0.03, count=500)},
        total_replays=1000,
    )
    assert not incomplete["coverage_pass"]
    assert not incomplete["promotion_pass"]

    weak = summarize_component(
        {"640": _entry(accepted=0.04, candidate=0.039, count=1000)},
        total_replays=1000,
    )
    assert not weak["sizing_pass"]
    assert not weak["promotion_pass"]

    drifted = _entry(accepted=0.04, candidate=0.01, count=1000, drift=0.00101)
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
