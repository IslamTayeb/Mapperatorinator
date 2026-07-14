import csv

import pytest

from utils.profile_dp4a_projection_component import (
    CONFIGS,
    _write_csv,
    summarize_component,
)


def _bucket(*, baseline_ms: float = 1.0, candidate_ms: float = 0.5):
    regions = {}
    for region in ("self_norm_qkv", "cross_norm_q"):
        candidates = {
            config: {
                "finite": True,
                "repeat_bitwise_equal": True,
                "memory_stable": True,
                "kernel_vs_quantized_reference_pass": True,
                "active_warps": warps,
                "persistent_ctas": ctas,
                "candidate_vs_current_baseline": {"max_abs": 0.1},
                "kernel_vs_quantized_reference": {"max_abs": 0.0},
            }
            for config, (warps, ctas) in CONFIGS.items()
        }
        regions[region] = {
            "baseline_finite": True,
            "baseline_repeat_bitwise_equal": True,
            "baseline_memory_stable": True,
            "cross_configuration_bitwise_equal": True,
            "candidates": candidates,
            "l2_hot": {
                "baseline_ms_per_call": baseline_ms,
                "candidates": {
                    config: {"ms_per_call": candidate_ms}
                    for config in CONFIGS
                },
            },
            "cache_cold": {
                "baseline_ms_per_call": baseline_ms,
                "candidates": {
                    config: {"ms_per_call": candidate_ms}
                    for config in CONFIGS
                },
            },
        }
    return {"decode_replays": 50, "regions": regions}


def test_both_hot_and_cold_must_clear_fixed_main_gate() -> None:
    buckets = {"128": _bucket()}

    report = summarize_component(buckets, total_replays=100)

    assert report["modes"]["l2_hot"]["fixed_main_saving_seconds"] == pytest.approx(1.2)
    assert report["modes"]["cache_cold"]["fixed_main_saving_seconds"] == pytest.approx(1.2)
    assert report["conservative_fixed_main_saving_seconds"] == pytest.approx(1.2)
    assert report["component_pass"] is True
    assert report["retained_full_graph_required"] is True
    assert report["promotion_pass"] is False

    for region in buckets["128"]["regions"].values():
        region["l2_hot"]["candidates"] = {
            config: {"ms_per_call": 0.99} for config in CONFIGS
        }
    failed = summarize_component(buckets, total_replays=100)
    assert failed["modes"]["l2_hot"]["passes_saving_gate"] is False
    assert failed["component_pass"] is False
    assert failed["retained_full_graph_required"] is False


def test_implementation_or_cross_configuration_drift_blocks_component() -> None:
    buckets = {"128": _bucket()}
    buckets["128"]["regions"]["self_norm_qkv"]["candidates"]["w8_c68"][
        "kernel_vs_quantized_reference_pass"
    ] = False

    report = summarize_component(buckets, total_replays=100)

    assert report["invariants_pass"] is False
    assert report["component_pass"] is False


def test_csv_contains_both_timing_modes_and_every_candidate(tmp_path) -> None:
    report = {"buckets": {"128": _bucket()}}
    path = tmp_path / "component.csv"

    _write_csv(report, path)

    rows = list(csv.DictReader(path.open()))
    assert {row["mode"] for row in rows} == {"l2_hot", "cache_cold"}
    assert {row["variant"] for row in rows} == {"baseline", *CONFIGS}
    assert len(rows) == 2 * 2 * (1 + len(CONFIGS))
