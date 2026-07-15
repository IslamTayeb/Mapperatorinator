from copy import deepcopy

import pytest

from osuT5.osuT5.inference.optimized.scout.encoder_store import STORE_VERSION
from utils.analyze_encoder_store_reciprocal import (
    EncoderGateError,
    _drift_row,
    _store_contract,
    _text,
)


def _store(batch_size=16, precision="fp32"):
    output_store_bytes = 87 * 2 * 3 * (4 if precision == "fp32" else 2)
    entry = {
        "entry_index": 0,
        "labels": ["timing_context", "main_generation"],
        "live_window_count": 87,
        "batch_size": batch_size,
        "batch_count": 6,
        "frame_contract": {
            "shape": [87, 80, 3000],
            "stride": [240000, 3000, 1],
            "dtype": "torch.float32",
            "device": "cpu",
        },
        "output_shape": [87, 2, 3],
        "output_dtype": (
            "torch.float32" if precision == "fp32" else "torch.float16"
        ),
        "output_store_bytes": output_store_bytes,
        "conditioning_sha256": "a" * 64,
        "batch_setup_seconds": 0.01,
        "input_copy_seconds": 0.02,
        "encoder_synchronized_seconds": 0.5,
        "storage_allocation_seconds": 0.01,
        "output_store_copy_seconds": 0.01,
        "complete_precompute_seconds": 0.56,
        "baseline_allocated_vram_bytes": 1000,
        "baseline_reserved_vram_bytes": 2000,
        "peak_allocated_vram_bytes": 1000 + output_store_bytes,
        "peak_reserved_vram_bytes": 4000,
        "incremental_peak_allocated_vram_bytes": output_store_bytes,
        "incremental_peak_reserved_vram_bytes": 2000,
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
        "result_class": "same-precision-gate-with-documented-encoder-drift",
        "production_default": False,
        "selector_changes": False,
        "decoder_changes": False,
        "model_forward_changes": False,
        "server_changes": False,
        "precision": precision,
        "strict_fp32": precision == "fp32",
        "strict_exactness_required_for_promotion": True,
        "batch_size": batch_size,
        "store_count": 1,
        "shared_across_timing_main": True,
        "total_complete_precompute_seconds": 0.56,
        "total_output_store_bytes": output_store_bytes,
        "labels_completed": ["timing_context", "main_generation"],
        "active_processor_count": 0,
        "entries": [entry],
    }


def test_store_contract_charges_every_time_copy_storage_and_vram_field():
    store = _store()
    profile = {"metadata": {"optimized_batched_encoder_store": store}}
    evidence = {"encoder_store": store}

    charged = _store_contract(profile, evidence, batch_size=16)

    assert charged["complete_precompute_seconds"] == pytest.approx(0.56)
    assert charged["output_store_bytes"] == 2088
    assert charged["incremental_peak_allocated_vram_bytes"] == 2088


def test_store_contract_is_same_precision_and_rejects_cross_precision_evidence():
    store = _store(precision="fp16")
    profile = {"metadata": {"optimized_batched_encoder_store": store}}
    evidence = {"encoder_store": store}

    charged = _store_contract(
        profile,
        evidence,
        batch_size=16,
        precision="fp16",
    )
    assert charged["output_store_bytes"] == 1044

    with pytest.raises(EncoderGateError, match="contract changed"):
        _store_contract(
            profile,
            evidence,
            batch_size=16,
            precision="fp32",
        )


def test_store_contract_rejects_unmatched_evidence_and_unowned_decoder_changes():
    store = _store()
    profile = {"metadata": {"optimized_batched_encoder_store": store}}
    changed = deepcopy(store)
    changed["decoder_changes"] = True
    with pytest.raises(EncoderGateError, match="profile and external"):
        _store_contract(profile, {"encoder_store": changed}, batch_size=16)

    profile = {"metadata": {"optimized_batched_encoder_store": changed}}
    with pytest.raises(EncoderGateError, match="contract changed"):
        _store_contract(profile, {"encoder_store": changed}, batch_size=16)


def test_drift_row_classifies_exact_and_nonexact_batches():
    variants = {}
    for batch_size in (1, 2, 4, 8, 16):
        max_abs = 0.0 if batch_size == 1 else 0.001
        variants[str(batch_size)] = {
            "complete_precompute_seconds": 1.0,
            "complete_seconds_saved_vs_b1": 0.0,
            "batch_setup_seconds": 0.01,
            "input_copy_seconds": 0.1,
            "encoder_synchronized_seconds": 0.7,
            "storage_allocation_seconds": 0.1,
            "output_store_copy_seconds": 0.1,
            "output_store_bytes": 1024,
            "baseline_allocated_vram_bytes": 1024,
            "peak_allocated_vram_bytes": 3072,
            "incremental_peak_vram_bytes": 2048,
            "verification_device_to_cpu_seconds_excluded": 0.05,
            "encoder_drift_vs_b1": {
                "window_count": 3,
                "exact_window_count": 3 if batch_size == 1 else 0,
                "max_abs": max_abs,
                "mean_window_max_abs": max_abs,
            },
        }
    audit = {
        "metadata": {
            "batch_sizes": [1, 2, 4, 8, 16],
            "source_precision": "fp32",
        },
        "variants": variants,
    }
    assert _drift_row(audit, 1, name="main")["encoder_exact"]
    assert not _drift_row(audit, 16, name="main")["encoder_exact"]


def test_text_report_exposes_every_batch_without_creating_a_document():
    variants = {
        str(batch_size): {
            "classification": "documented-drift-not-promotion-eligible",
            "determinism_pass": True,
            "strict_exactness_pass": False,
            "performance": {"request_seconds_saved": 0.5},
            "encoder_drift": {
                "main": {"max_abs": 0.1},
                "timing": {"max_abs": 0.2},
            },
        }
        for batch_size in (1, 2, 4, 8, 16)
    }
    text = _text(
        {
            "decision": "STOP_ENCODER_PRECOMPUTE",
            "precision": "fp16",
            "baseline_request_median_seconds": 42.0,
            "minimum_request_saving_seconds": 2.1,
            "deterministic_batch_sizes": [1, 2, 4, 8, 16],
            "strict_exact_batch_sizes": [],
            "promotion_batch_sizes": [],
            "variants": variants,
        }
    )
    for batch_size in (1, 2, 4, 8, 16):
        assert f"b{batch_size}_classification=" in text
        assert f"b{batch_size}_deterministic=true" in text
    assert "precision=fp16" in text
