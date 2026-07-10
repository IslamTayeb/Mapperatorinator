from __future__ import annotations

import math

import pytest
import torch

from osuT5.osuT5.inference.optimized.batch.lane_one_token import (
    DEFAULT_SEPARATE_GRAPH_POOL,
    PYTORCH_PER_STREAM_CUBLAS_WORKSPACE,
    LaneResourceEvidence,
    OneLaneCaptureConfig,
    compare_lane_logits,
    normalized_one_token_request_ids,
    normalized_one_token_workload_contract,
    reciprocal_lane_orders,
    summarize_lane_scaling,
    summarize_reciprocal_l2_performance,
    validate_l2_baseline_report,
    validate_lane_gate_order,
    validate_lane_resource_ownership,
)
from utils.verify_optimized_b1_lane_capture import (
    _cache_position_is_overwritten,
    _cache_position_is_zero,
    _zero_cache_position,
    build_parser,
)
from utils.verify_optimized_l2_lane_capture import (
    REVIEWED_B1_COMPLETE_TPS,
    _observation,
    _resolve_row_seeds,
    build_parser as build_l2_parser,
)


def _evidence(
        lane_id: int,
        *,
        shared_model_id: int = 10,
        pointer_offset: int = 0,
) -> LaneResourceEvidence:
    return LaneResourceEvidence(
        lane_id=lane_id,
        shared_model_id=shared_model_id,
        stream_id=100 + lane_id,
        graph_id=200 + lane_id,
        graph_pool_id=f"pool-{lane_id}",
        graph_pool_policy=DEFAULT_SEPARATE_GRAPH_POOL,
        graph_pool_sharing_requested=False,
        session_id=300 + lane_id,
        cache_id=400 + lane_id,
        encoder_output_storage_ptr=500 + pointer_offset,
        generator_id=600 + lane_id,
        logits_processor_id=700 + lane_id,
        static_input_storage_ptrs={"decoder_input_ids": 800 + pointer_offset},
        self_cache_storage_ptrs=(900 + pointer_offset, 901 + pointer_offset),
        cross_cache_storage_ptrs=(902 + pointer_offset, 903 + pointer_offset),
        cublas_workspace_policy=PYTORCH_PER_STREAM_CUBLAS_WORKSPACE,
        cublas_workspace_owner_stream_id=100 + lane_id,
        cublas_workspace_config=":4096:8",
    )


def test_one_lane_config_requires_warmed_positive_gate():
    config = OneLaneCaptureConfig(seed=12345, active_prefix_length=128)

    assert config.capture_warmup_repeats == 1
    with pytest.raises(ValueError, match="active_prefix_length"):
        OneLaneCaptureConfig(seed=12345, active_prefix_length=0)
    with pytest.raises(ValueError, match="capture_warmup_repeats"):
        OneLaneCaptureConfig(
            seed=12345,
            active_prefix_length=128,
            capture_warmup_repeats=0,
        )


def test_lane_ownership_allows_shared_model_but_requires_every_mutable_resource_private():
    lanes = [
        _evidence(0, pointer_offset=0),
        _evidence(1, pointer_offset=1000),
    ]

    validate_lane_resource_ownership(lanes, expected_lane_count=2)

    aliased_stream = LaneResourceEvidence(
        **{
            **lanes[1].__dict__,
            "stream_id": lanes[0].stream_id,
            "cublas_workspace_owner_stream_id": lanes[0].stream_id,
        }
    )
    with pytest.raises(ValueError, match="stream_id"):
        validate_lane_resource_ownership(
            [lanes[0], aliased_stream],
            expected_lane_count=2,
        )

    aliased_static_input = LaneResourceEvidence(
        **{
            **lanes[1].__dict__,
            "static_input_storage_ptrs": lanes[0].static_input_storage_ptrs,
        }
    )
    with pytest.raises(ValueError, match="tensor storage aliases"):
        validate_lane_resource_ownership(
            [lanes[0], aliased_static_input],
            expected_lane_count=2,
        )


def test_lane_ownership_rejects_shared_graph_pool_and_fake_workspace_pointer():
    values = _evidence(0).__dict__
    with pytest.raises(ValueError, match="must not share CUDA graph pools"):
        LaneResourceEvidence(**{**values, "graph_pool_sharing_requested": True})
    with pytest.raises(ValueError, match="does not expose"):
        LaneResourceEvidence(**{**values, "cublas_workspace_pointer_exposed": True})


def test_lane_gate_order_stops_concurrency_before_b1_parity_and_before_five_percent_signal():
    validate_lane_gate_order(1, None)
    with pytest.raises(ValueError, match="requires a passed L=1"):
        validate_lane_gate_order(2, None)

    b1_report = {
        "lane_count": 1,
        "pass": True,
        "exactness_pass": True,
        "resource_ownership_pass": True,
        "capture_pass": True,
        "performance_gate_pass": None,
    }
    validate_lane_gate_order(2, b1_report)

    b2_below_bar = {**b1_report, "lane_count": 2, "performance_gate_pass": False}
    with pytest.raises(ValueError, match="5% B1 keep bar"):
        validate_lane_gate_order(3, b2_below_bar)

    b2_pass = {**b2_below_bar, "performance_gate_pass": True}
    validate_lane_gate_order(3, b2_pass)


def test_l2_baseline_is_pinned_to_reviewed_normalized_l1_report():
    tps = 414.9046973558729
    report = {
        "lane_count": 1,
        "pass": True,
        "exactness_pass": True,
        "resource_ownership_pass": True,
        "capture_pass": True,
        "timing": {"complete_wall_tokens_per_second": tps},
        "observation": {
            "execution_family": "b1_lane_pool",
            "parallelism": 1,
            "scheduler_wall_tokens_per_second": tps,
        },
    }

    validate_l2_baseline_report(
        report,
        expected_complete_tokens_per_second=tps,
    )
    with pytest.raises(ValueError, match="differs from the reviewed value"):
        validate_l2_baseline_report(
            report,
            expected_complete_tokens_per_second=tps + 0.01,
        )


def test_l2_reciprocal_order_gate_uses_slower_order_and_five_percent_bar():
    assert reciprocal_lane_orders(2) == ((0, 1), (1, 0))
    with pytest.raises(ValueError, match="only for L=2"):
        reciprocal_lane_orders(3)

    baseline = 414.9046973558729
    result = summarize_reciprocal_l2_performance(
        b1_tokens_per_second=baseline,
        order_tokens_per_second={"0,1": 500.0, "1,0": 440.0},
    )

    assert result["worst_order_tokens_per_second"] == 440.0
    assert result["best_order_tokens_per_second"] == 500.0
    assert result["clears_five_percent_b1_keep_bar"] is True
    below = summarize_reciprocal_l2_performance(
        b1_tokens_per_second=baseline,
        order_tokens_per_second={"0,1": 435.0, "1,0": 500.0},
    )
    assert below["clears_five_percent_b1_keep_bar"] is False


def test_l2_serial_and_concurrent_observation_histograms_describe_same_workload():
    common = {
        "seeds": (12345, 23456),
        "timing_repeats": 50,
        "workload_contract_hash": "same-two-row-workload",
        "token_hashes": {"row-0": "tokens-a", "row-1": "tokens-b"},
        "rng_hashes": {"row-0": "rng-a", "row-1": "rng-b"},
        "model_timing": {"cuda_seconds": 0.15},
        "complete_timing": {
            "wall_seconds": 0.20,
            "cuda_seconds": 0.19,
            "peak_allocated_bytes": 1024,
        },
    }
    serial = _observation(parallelism=1, **common)
    concurrent = _observation(parallelism=2, **common)

    assert serial.generated_tokens == concurrent.generated_tokens == 100
    assert serial.active_batch_size_histogram == {1: 100}
    assert concurrent.active_batch_size_histogram == {2: 50}
    assert serial.workload_contract_hash == concurrent.workload_contract_hash


def test_lane_scaling_uses_b1_denominator_and_five_percent_keep_bar():
    below = summarize_lane_scaling(
        b1_tokens_per_second=300.0,
        candidate_lane_count=2,
        candidate_tokens_per_second=314.0,
    )
    above = summarize_lane_scaling(
        b1_tokens_per_second=300.0,
        candidate_lane_count=2,
        candidate_tokens_per_second=330.0,
    )

    assert below["clears_five_percent_b1_keep_bar"] is False
    assert above["clears_five_percent_b1_keep_bar"] is True
    assert math.isclose(above["fraction_of_ideal_linear_capacity"], 0.55)


def test_lane_workload_schema_and_request_ids_match_merged_reports():
    contract = normalized_one_token_workload_contract(
        batch_size=2,
        seeds=[12345, 23456],
        do_sample=True,
        prompt_sha256="prompt",
        prompt_attention_mask_sha256="mask",
        frames_sha256="frames",
        condition_tensor_hashes={"beatmap_idx": "condition"},
        active_prefix_prefill=False,
        active_prefix_decode=True,
        active_prefix_decode_length=128,
        runtime_contract={"config_name": "profile_salvalai_smoke15"},
    )

    assert normalized_one_token_request_ids(2) == ("row-0", "row-1")
    assert list(contract) == [
        "batch_size",
        "seeds",
        "do_sample",
        "prompt_sha256",
        "prompt_attention_mask_sha256",
        "frames_sha256",
        "condition_tensor_hashes",
        "active_prefix_prefill",
        "active_prefix_decode",
        "active_prefix_decode_length",
        "runtime_contract",
    ]


def test_logit_comparison_checks_nonfinite_layout_and_topk():
    reference = torch.tensor([[3.0, 2.0, -float("inf"), 1.0]])
    close = torch.tensor([[3.0 + 1e-5, 2.0, -float("inf"), 1.0]])
    wrong_nonfinite = torch.tensor([[3.0, 2.0, 0.0, 1.0]])

    passing = compare_lane_logits(reference, close, atol=1e-4, rtol=1e-4, top_k=2)
    failing = compare_lane_logits(
        reference,
        wrong_nonfinite,
        atol=1e-4,
        rtol=1e-4,
        top_k=2,
    )

    assert passing["allclose"] is True
    assert passing["topk_match"] is True
    assert failing["allclose"] is False
    assert failing["nonfinite_match"] is False


def test_cache_position_sentinel_zeros_only_the_target_slot():
    class Layer:
        def __init__(self):
            self.keys = torch.ones(1, 2, 4, 3)
            self.values = torch.full((1, 2, 4, 3), 2.0)

    class CachePart:
        def __init__(self):
            self.layers = [Layer(), Layer()]

    cache = CachePart()
    _zero_cache_position(cache, 2)

    assert _cache_position_is_zero(cache, 2) is True
    assert _cache_position_is_overwritten(cache, 2) is False
    for layer in cache.layers:
        layer.keys[..., 2, :].fill_(3.0)
        layer.values[..., 2, :].fill_(4.0)
    assert _cache_position_is_overwritten(cache, 2) is True
    assert torch.count_nonzero(cache.layers[0].keys[..., 1, :]).item() > 0
    assert torch.count_nonzero(cache.layers[1].values[..., 3, :]).item() > 0


def test_cli_exposes_only_incremental_lane_gate_inputs():
    parser = build_parser()
    args = parser.parse_args([
        "--report-path",
        "report.json",
        "--lane-count",
        "1",
        "inference_generation_compile=false",
    ])

    assert args.lane_count == 1
    assert args.previous_gate_report is None
    assert args.overrides == ["inference_generation_compile=false"]

    l2_args = build_l2_parser().parse_args([
        "--previous-gate-report",
        "l1.json",
        "--report-path",
        "l2.json",
        "--row-seed",
        "12345",
        "--row-seed",
        "23456",
        "inference_generation_compile=false",
    ])
    assert l2_args.expected_b1_complete_tps == REVIEWED_B1_COMPLETE_TPS
    assert _resolve_row_seeds(l2_args.row_seed, 1) == (12345, 23456)
    with pytest.raises(ValueError, match="exactly two"):
        _resolve_row_seeds([12345], 1)
