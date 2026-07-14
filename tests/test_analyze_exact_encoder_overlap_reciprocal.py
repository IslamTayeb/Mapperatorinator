import copy
import hashlib

import pytest

from osuT5.osuT5.inference.optimized.scout.encoder_overlap import (
    MATERIAL_AUDIT_VERSION,
    OVERLAP_VERSION,
)
from utils.analyze_exact_encoder_overlap_reciprocal import (
    _exact_signatures,
    _material_exactness,
    _overlap_material_exactness,
    _text,
    _validate_candidate,
    _validate_material_audit,
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
    encoder_digests = ["e" * 64, "f" * 64]
    aggregate = hashlib.sha256()
    for digest in encoder_digests:
        aggregate.update(digest.encode("ascii"))
    return {
        "version": OVERLAP_VERSION,
        "production_default": False,
        "batch_size": 1,
        "separate_timing_main_models": True,
        "labels_completed": ["timing_context", "main_generation"],
        "manifest_changed_after_launch": False,
        "live_window_count": 2,
        "timing_model_generate_calls": 2,
        "main_model_generate_calls": 2,
        "encoder_launch_chunks": 2,
        "encoder_rows_per_chunk": [1, 1],
        "launch_after_timing_window": 1,
        "all_encoder_outputs_finite": True,
        "launch_manifest": {"graph_count": 2, "buckets": {"64": 1, "128": 1}},
        "final_timing_manifest": {
            "graph_count": 2,
            "buckets": {"64": 1, "128": 1},
        },
        "pre_capture_barrier_count": 0,
        "pre_capture_barrier_wait_seconds": 0.0,
        "guarded_manifest_growth": [],
        "encoder_synchronized_before_main": True,
        "pinned_input_setup_seconds": 0.01,
        "launch_host_seconds": 0.01,
        "encoder_elapsed_seconds": 2.0,
        "encoder_overlap_with_timing_seconds": 1.9,
        "encoder_after_timing_seconds": 0.1,
        "main_join_wait_seconds": 0.1,
        "per_row_main_join_wait_seconds": [0.05, 0.05],
        "pinned_input_bytes": 80,
        "device_input_copy_bytes": 80,
        "output_store_bytes": 100,
        "encoder_output_sha256_per_window": encoder_digests,
        "encoder_output_aggregate_sha256": aggregate.hexdigest(),
        "peak_allocated_vram_bytes": 1000,
        "peak_reserved_vram_bytes": 1200,
        "incremental_peak_allocated_vram_bytes": 200,
        "incremental_peak_reserved_vram_bytes": 300,
    }


def _material_audit(digest: str = "a" * 64):
    tensor = {
        "name": "encoder.last_hidden_state",
        "sha256": digest,
        "contract": {
            "shape": [1, 2, 3],
            "stride": [6, 3, 1],
            "dtype": "torch.float32",
            "device": "cuda:0",
        },
        "bytes": 24,
    }
    stages = {
        "timing_context": {
            "profile_label": "timing_context",
            "captured": False,
            "reason": "avoid_synchronizing_main_encoder_overlap_stream",
            "copied_bytes": 0,
            "copy_and_hash_seconds": 0.0,
        },
        "main_generation": {
            "profile_label": "main_generation",
            "captured": True,
            "encoder": tensor,
            "cross_kv": [{"layer": 0, "keys": tensor, "values": tensor}],
            "aggregate_sha256": digest,
            "copied_bytes": 72,
            "copy_and_hash_seconds": 0.01,
            "rng_after_stage": {"cpu": "b" * 64, "cuda": "c" * 64},
            "cross_layer_count": 1,
        },
    }
    return {
        "version": MATERIAL_AUDIT_VERSION,
        "production_default": False,
        "labels_completed": ["timing_context", "main_generation"],
        "processor_count": 2,
        "stages": stages,
        "total_copied_bytes": 72,
        "total_copy_and_hash_seconds": 0.01,
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
    overlap["timing_model_generate_calls"] = 3
    overlap["encoder_launch_chunks"] = 3
    overlap["encoder_rows_per_chunk"] = [1, 1, 1]
    overlap["launch_after_timing_window"] = 2
    overlap["per_row_main_join_wait_seconds"] = [0.03, 0.03, 0.04]
    overlap["encoder_output_sha256_per_window"] = ["e" * 64, "f" * 64, "0" * 64]
    aggregate = hashlib.sha256()
    for digest in overlap["encoder_output_sha256_per_window"]:
        aggregate.update(digest.encode("ascii"))
    overlap["encoder_output_aggregate_sha256"] = aggregate.hexdigest()
    profile = {
        "metadata": {"optimized_exact_main_encoder_overlap": copy.deepcopy(overlap)}
    }
    evidence = {"mode": "candidate", "encoder_overlap": overlap}
    assert _validate_candidate(profile, evidence) is overlap
    overlap["manifest_changed_after_launch"] = True
    with pytest.raises(ValueError, match="contract changed"):
        _validate_candidate(profile, evidence)


def test_candidate_validation_rejects_unguarded_or_broken_manifest_growth():
    overlap = _overlap()
    overlap["live_window_count"] = 3
    overlap["main_model_generate_calls"] = 3
    overlap["timing_model_generate_calls"] = 3
    overlap["encoder_launch_chunks"] = 3
    overlap["encoder_rows_per_chunk"] = [1, 1, 1]
    overlap["launch_after_timing_window"] = 2
    overlap["per_row_main_join_wait_seconds"] = [0.03, 0.03, 0.04]
    overlap["encoder_output_sha256_per_window"] = ["e" * 64, "f" * 64, "0" * 64]
    aggregate = hashlib.sha256()
    for digest in overlap["encoder_output_sha256_per_window"]:
        aggregate.update(digest.encode("ascii"))
    overlap["encoder_output_aggregate_sha256"] = aggregate.hexdigest()
    overlap["pre_capture_barrier_count"] = 1
    overlap["final_timing_manifest"] = {
        "graph_count": 3,
        "buckets": {"64": 1, "128": 1, "192": 1},
    }
    overlap["guarded_manifest_growth"] = [
        {
            "after_timing_window": 2,
            "capture_barriers": 1,
            "before": overlap["launch_manifest"],
            "after": overlap["final_timing_manifest"],
        }
    ]
    profile = {
        "metadata": {"optimized_exact_main_encoder_overlap": copy.deepcopy(overlap)}
    }
    evidence = {"mode": "candidate", "encoder_overlap": overlap}
    assert _validate_candidate(profile, evidence) is overlap

    overlap["guarded_manifest_growth"][0]["before"] = {
        "graph_count": 1,
        "buckets": {"64": 1},
    }
    profile["metadata"]["optimized_exact_main_encoder_overlap"] = copy.deepcopy(overlap)
    with pytest.raises(ValueError, match="chain broke"):
        _validate_candidate(profile, evidence)


def test_material_audit_requires_profile_identity_and_raw_kv_rng_exactness():
    audit = _material_audit()
    profile = {
        "metadata": {"optimized_encoder_material_audit": copy.deepcopy(audit)}
    }
    evidence = {"material_audit": audit}
    assert _validate_material_audit(profile, evidence, role="baseline_first") is audit

    audits = {
        role: copy.deepcopy(audit)
        for role in (
            "baseline_first",
            "candidate_second",
            "candidate_first",
            "baseline_second",
        )
    }
    assert _material_exactness(audits)["pass"]
    audits["candidate_first"]["stages"]["main_generation"][
        "aggregate_sha256"
    ] = "d" * 64
    assert not _material_exactness(audits)["pass"]


def test_overlap_digest_matches_the_encoder_output_consumed_by_main():
    overlap = _overlap()
    consumed_digest = overlap["encoder_output_sha256_per_window"][-1]
    candidate_evidence = {
        role: copy.deepcopy(overlap)
        for role in ("candidate_first", "candidate_second")
    }
    material_audits = {
        role: _material_audit(consumed_digest)
        for role in ("candidate_first", "candidate_second")
    }

    assert _overlap_material_exactness(candidate_evidence, material_audits)["pass"]

    material_audits["candidate_second"]["stages"]["main_generation"][
        "encoder"
    ]["sha256"] = "0" * 64
    report = _overlap_material_exactness(candidate_evidence, material_audits)
    assert not report["last_overlap_matches_consumed_encoder"]
    assert not report["pass"]


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
