from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import torch

from utils.profile_accepted_decoder_regions import REGIONS
from utils.profile_batched_decoder_regions import (
    CANDIDATE_FAMILIES,
    PROMOTION_THRESHOLD_FRACTION,
    SENTINEL_PREFIXES,
    _framework_context,
    _validate_workload_report,
    select_live_graph_entries,
    summarize_batched_region_ceiling,
)


def _entry(batch: int, prefix: int, replays: int = 10) -> dict:
    return {
        "graph": object(),
        "outputs": object(),
        "static_inputs": {
            "decoder_input_ids": torch.ones((batch, 1), dtype=torch.long),
        },
        "active_prefix_length": prefix,
        "decode_replays": replays,
    }


def _cache(batches=(10, 11)) -> dict:
    return {
        (batch, prefix): _entry(batch, prefix)
        for batch in batches
        for prefix in SENTINEL_PREFIXES
    }


def _profile(batch: int, prefix: int, *, replays: int, region_ms: float) -> dict:
    return {
        "batch_size": batch,
        "active_prefix_length": prefix,
        "decode_replays": replays,
        "production_calibrated_regions_ms_per_call": {
            region: region_ms for region in REGIONS
        },
    }


def test_framework_context_enables_only_prefix_and_detail_ranges(monkeypatch) -> None:
    from osuT5.osuT5 import runtime_profiling

    sentinel = object()
    calls = []

    def fake_context(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(runtime_profiling, "generation_profile_context", fake_context)

    assert _framework_context(640, detail_ranges=True) is sentinel
    assert calls == [{
        "detail_ranges": True,
        "active_prefix_self_attention_length": 640,
        "q1_bmm_cross_attention": False,
        "native_q1_self_attention": False,
        "native_q1_rope_cache_self_attention": False,
        "native_cross_mlp_tail": False,
        "optimized_expected_dtype": torch.float32,
    }]


def test_selects_exact_live_batch_and_sentinel_cross_product() -> None:
    selected = select_live_graph_entries(
        _cache(),
        actual_batch_sizes={10, 11},
    )

    assert tuple(selected) == tuple(
        (batch, prefix)
        for batch in (10, 11)
        for prefix in SENTINEL_PREFIXES
    )


def test_live_graph_selection_fails_on_missing_or_duplicate_signature() -> None:
    cache = _cache()
    cache.pop((11, 640))
    with pytest.raises(RuntimeError, match="missing"):
        select_live_graph_entries(cache, actual_batch_sizes={10, 11})

    cache = _cache()
    cache["duplicate"] = _entry(10, 128)
    with pytest.raises(RuntimeError, match="repeats live signature"):
        select_live_graph_entries(cache, actual_batch_sizes={10, 11})


def test_weighting_uses_replays_without_batch_or_layer_multiplier() -> None:
    profiles = {
        "b10-p128": _profile(10, 128, replays=2, region_ms=100.0),
        "b11-p128": _profile(11, 128, replays=3, region_ms=100.0),
    }

    weighted = summarize_batched_region_ceiling(
        profiles,
        complete_super_timing_seconds=20.0,
        total_graph_replays=10,
    )

    assert weighted["regions_seconds"] == pytest.approx(
        {region: 0.5 for region in REGIONS}
    )
    assert weighted["promotion_threshold_seconds"] == pytest.approx(1.0)
    assert weighted["measured_graph_replays"] == 5
    assert weighted["replay_coverage_fraction"] == pytest.approx(0.5)
    assert weighted["candidate_family_decisions"]["q1_bmm_cross_attention"][
        "optimistic_seconds"
    ] == pytest.approx(0.5)
    native_regions = CANDIDATE_FAMILIES["native_cross_mlp_tail"]
    assert weighted["candidate_family_decisions"]["native_cross_mlp_tail"][
        "optimistic_seconds"
    ] == pytest.approx(len(native_regions) * 0.5)
    assert weighted["passing_candidate_families"] == ["native_cross_mlp_tail"]
    assert weighted["promotion_pass"]
    assert weighted["promotion_threshold_fraction"] == PROMOTION_THRESHOLD_FRACTION


def _workload_report() -> dict:
    return {
        "metadata": {
            "mode": "graph",
            "batch_size": 16,
            "precision": "fp32",
            "seed": 12345,
            "timer_iterations": 20,
            "timer_num_beams": 1,
            "timer_cfg_scale": 1.0,
            "public_wiring": False,
            "audio_sha256": (
                "f16f06549e732ef5fb0fceb4dcec95efe5beb6167efe13f6998491b5a16826fe"
            ),
        },
        "gates": {"pass": True},
        "dispatch": {
            "modes": ["framework_batch"],
            "capture_hits": {
                "native_q1_rope_cache_self_attention": 0,
                "native_q1_self_attention": 0,
                "q1_bmm_cross_attention": 0,
                "native_cross_mlp_tail": 0,
            },
        },
        "graphs": {
            "cache_ownership_pass": True,
            "observed_cache_batch_sizes": [10, 11],
        },
        "workload": {
            "batches_by_iteration": {"0": [10], "1": [11]},
        },
    }


def test_workload_gate_requires_exact_contract_dispatch_and_cache_batches() -> None:
    assert _validate_workload_report(_workload_report()) == {10, 11}

    changed = _workload_report()
    changed["metadata"]["timer_iterations"] = 19
    with pytest.raises(ValueError, match="contract changed"):
        _validate_workload_report(changed)

    changed = _workload_report()
    changed["dispatch"]["capture_hits"].pop("q1_bmm_cross_attention")
    with pytest.raises(RuntimeError, match="framework-only dispatch"):
        _validate_workload_report(changed)

    changed = _workload_report()
    changed["graphs"]["observed_cache_batch_sizes"] = [10]
    with pytest.raises(RuntimeError, match="do not match workload"):
        _validate_workload_report(changed)


def test_direct_cli_imports_from_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "utils" / "profile_batched_decoder_regions.py"),
            "--help",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--report-path" in completed.stdout
