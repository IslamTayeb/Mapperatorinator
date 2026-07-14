import pytest

from osuT5.osuT5.inference.optimized.scout.encoder_store import STORE_VERSION
from utils.analyze_encoder_store_reciprocal import (
    _text,
    _token_comparison,
    _validate_candidate_store,
)


def _store():
    entry = {
        "entry_index": 0,
        "labels": ["timing_context", "main_generation"],
        "live_window_count": 87,
        "output_store_bytes": 1234,
        "conditioning_sha256": "a" * 64,
        "complete_precompute_seconds": 2.0,
        "uses": {
            "timing_context": {"rows_reused": 87},
            "main_generation": {"rows_reused": 87},
        },
    }
    return {
        "version": STORE_VERSION,
        "production_default": False,
        "batch_size": 16,
        "store_count": 1,
        "shared_across_timing_main": True,
        "total_complete_precompute_seconds": 2.0,
        "total_output_store_bytes": 1234,
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


def test_text_report_contains_both_model_audits():
    report = {
        "decision": "PASS_FULL_SONG_GATE",
        "batch_size": 16,
        "mean_complete_request_seconds_saved": 0.5,
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
        "request_nonregression_pass": True,
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
