import pytest

from osuT5.osuT5.inference.optimized.scout.encoder_store import STORE_VERSION
from utils.analyze_encoder_store_reciprocal import (
    _text,
    _token_comparison,
    _runner_complete_wall,
    _validate_candidate_store,
    _validate_store_charges,
)


def _store():
    entry = {
        "entry_index": 0,
        "labels": ["timing_context", "main_generation"],
        "live_window_count": 87,
        "output_store_bytes": 100,
        "conditioning_sha256": "a" * 64,
        "complete_precompute_seconds": 2.0,
        "batch_setup_seconds": 0.1,
        "input_copy_seconds": 0.2,
        "encoder_synchronized_seconds": 1.5,
        "storage_allocation_seconds": 0.05,
        "output_store_copy_seconds": 0.05,
        "baseline_allocated_vram_bytes": 1000,
        "baseline_reserved_vram_bytes": 2000,
        "peak_allocated_vram_bytes": 1200,
        "peak_reserved_vram_bytes": 2300,
        "incremental_peak_allocated_vram_bytes": 200,
        "incremental_peak_reserved_vram_bytes": 300,
        "uses": {
            "timing_context": {
                "rows_reused": 87,
                "per_window_device_copy": False,
            },
            "main_generation": {
                "rows_reused": 87,
                "per_window_device_copy": False,
            },
        },
    }
    return {
        "version": STORE_VERSION,
        "production_default": False,
        "batch_size": 16,
        "store_count": 1,
        "shared_across_timing_main": True,
        "total_complete_precompute_seconds": 2.0,
        "total_output_store_bytes": 100,
        "labels_completed": ["timing_context", "main_generation"],
        "entries": [entry],
    }


def test_token_comparison_reports_aligned_and_count_drift():
    row = _token_comparison([1, 2, 3], [1, 9, 3, 4])
    assert row["count_delta"] == 1
    assert row["aligned_mismatch_count"] == 1
    assert row["first_mismatch"] == 1
    assert not row["exact"]


def test_candidate_store_requires_matching_profile_and_external_evidence():
    store = _store()
    profile = {"metadata": {"optimized_batched_encoder_store": store}}
    evidence = {"encoder_store": store}
    assert _validate_candidate_store(profile, evidence, batch_size=16) is store

    changed = _store()
    changed["entries"][0]["conditioning_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="conditioning_sha256"):
        _validate_candidate_store(
            profile,
            {"encoder_store": changed},
            batch_size=16,
        )


def test_store_charges_setup_copies_storage_and_vram() -> None:
    charges = _validate_store_charges(_store())

    assert charges["total_complete_precompute_seconds"] == 2.0
    assert charges["total_accounted_precompute_seconds"] == pytest.approx(1.9)
    assert charges["total_output_store_bytes"] == 100
    assert charges["total_incremental_peak_allocated_vram_bytes"] == 200

    changed = _store()
    changed["entries"][0]["incremental_peak_allocated_vram_bytes"] = 199
    with pytest.raises(ValueError, match="allocated VRAM delta changed"):
        _validate_store_charges(changed)


def test_complete_request_wall_uses_outer_runner_evidence() -> None:
    evidence = {"run_wall_seconds": 42.75}

    assert _runner_complete_wall(evidence, role="candidate_first") == 42.75
    with pytest.raises(ValueError, match="candidate_first.run_wall_seconds"):
        _runner_complete_wall({}, role="candidate_first")


def test_text_report_contains_both_model_audits():
    report = {
        "decision": "PASS_FULL_SONG_GATE",
        "batch_size": 16,
        "mean_complete_request_seconds_saved": 0.5,
        "minimum_complete_request_saving_seconds": 0.3,
        "audits": {
            "main": {
                "max_abs_encoder_drift": 1e-4,
                "complete_seconds_saved_vs_b1": 0.3,
            },
            "timing": {
                "max_abs_encoder_drift": 2e-4,
                "complete_seconds_saved_vs_b1": 0.2,
            },
        },
        "repeatability_pass": True,
        "output_repeatability_pass": True,
        "complete_wall_saving_pass": True,
        "drift_pass": True,
        "storage_pass": True,
        "candidate_vram_pass": True,
        "orders": [
            {
                "complete_request_seconds_saved": 0.4,
                "main_tokens": {"count_delta": 0},
            }
        ],
    }
    text = _text(report)
    assert "main_encoder_max_abs_drift=0.0001" in text
    assert "timing_encoder_max_abs_drift=0.0002" in text
