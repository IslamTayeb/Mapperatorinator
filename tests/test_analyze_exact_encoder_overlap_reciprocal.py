import copy

import pytest

from osuT5.osuT5.inference.optimized.scout.encoder_overlap import OVERLAP_VERSION
from utils.analyze_exact_encoder_overlap_reciprocal import (
    _exact_signatures,
    _text,
    _validate_candidate,
)


def _profile():
    graph = {
        "graph_count": 2,
        "decode_replays": 9,
        "capture_seconds": 0.1,
        "buckets": {
            "64": {"graph_count": 1, "decode_replays": 7, "capture_seconds": 0.05},
            "128": {"graph_count": 1, "decode_replays": 2, "capture_seconds": 0.05},
        },
    }
    return {
        "metadata": {"optimized_effective_config_version": "accepted"},
        "generation": [
            {
                "profile_label": label,
                "context_type": "timing" if label == "timing_context" else "map",
                "mode": "sequential",
                "sequence_index": 0,
                "generated_tokens": 3,
                "generated_token_ids": [1, 2, 3],
                "optimized_cuda_graphs": graph,
                "decoder_loop_backend": "active_prefix_cuda_graph",
            }
            for label in ("timing_context", "main_generation")
        ],
    }


def _overlap():
    return {
        "version": OVERLAP_VERSION,
        "production_default": False,
        "batch_size": 1,
        "separate_timing_main_models": True,
        "labels_completed": ["timing_context", "main_generation"],
        "manifest_changed_after_launch": False,
        "live_window_count": 2,
        "main_model_generate_calls": 2,
        "launch_after_timing_window": 1,
        "all_encoder_outputs_finite": True,
        "launch_manifest": {"graph_count": 2, "buckets": {"64": 1, "128": 1}},
        "pinned_input_setup_seconds": 0.01,
        "launch_host_seconds": 0.01,
        "encoder_elapsed_seconds": 2.0,
        "encoder_overlap_with_timing_seconds": 1.9,
        "encoder_after_timing_seconds": 0.1,
        "main_join_wait_seconds": 0.1,
        "per_row_main_join_wait_seconds": [0.05, 0.05],
        "output_store_bytes": 100,
        "peak_allocated_vram_bytes": 1000,
        "peak_reserved_vram_bytes": 1200,
        "incremental_peak_allocated_vram_bytes": 200,
        "incremental_peak_reserved_vram_bytes": 300,
    }


def test_exact_signatures_cover_tokens_stopping_and_graph_cache():
    baseline = _profile()
    candidate = copy.deepcopy(baseline)
    report = _exact_signatures(baseline, candidate)
    assert report["pass"]
    candidate["generation"][1]["generated_token_ids"][-1] = 9
    assert not _exact_signatures(baseline, candidate)["pass"]


def test_candidate_validation_requires_complete_exact_b1_overlap():
    overlap = _overlap()
    # The production gate has 87 windows; use 3 here so launch after window 2 is valid.
    overlap["live_window_count"] = 3
    overlap["main_model_generate_calls"] = 3
    overlap["launch_after_timing_window"] = 2
    overlap["per_row_main_join_wait_seconds"] = [0.03, 0.03, 0.04]
    profile = {
        "metadata": {"optimized_exact_main_encoder_overlap": copy.deepcopy(overlap)}
    }
    evidence = {"mode": "candidate", "encoder_overlap": overlap}
    assert _validate_candidate(profile, evidence) is overlap
    overlap["manifest_changed_after_launch"] = True
    with pytest.raises(ValueError, match="contract changed"):
        _validate_candidate(profile, evidence)


def test_text_report_contains_overlap_and_timing_costs():
    report = {
        "decision": "PASS_EXACT_ENCODER_OVERLAP",
        "mean_complete_request_seconds_saved": 1.2,
        "minimum_request_saving_seconds": 1.0,
        "exactness_pass": True,
        "performance_pass": True,
        "no_order_regression_pass": True,
        "orders": [
            {
                "complete_request_seconds_saved": 1.1,
                "timing_stage_slowdown_seconds": 0.2,
            }
        ],
        "candidate_overlap": {
            "candidate_first": {
                "encoder_elapsed_seconds": 2.0,
                "encoder_overlap_with_timing_seconds": 1.8,
                "main_join_wait_seconds": 0.1,
                "incremental_peak_allocated_vram_bytes": 123,
            }
        },
    }
    text = _text(report)
    assert "mean_complete_request_seconds_saved=1.200000" in text
    assert "order_0_timing_slowdown_seconds=0.200000" in text
    assert "candidate_first_encoder_overlap_seconds=1.800000" in text
