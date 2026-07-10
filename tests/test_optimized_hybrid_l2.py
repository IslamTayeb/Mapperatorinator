from __future__ import annotations

import copy
import inspect
from types import SimpleNamespace

import pytest

from osuT5.osuT5.inference.optimized.batch.hybrid_l2 import (
    FIVE_PERCENT_KEEP_TPS,
    FIXED_ACTIVE_PREFIX_BUCKET_SIZE,
    FIXED_ACTIVE_PREFIX_LENGTH,
    FIXED_ROW_SEEDS,
    FIXED_SEQUENCE_INDEX,
    FIXED_TIMING_REPEATS,
    HYBRID_EVENT_ORDER,
    HYBRID_L2_GATE,
    HYBRID_L2_SCHEMA_VERSION,
    REVIEWED_L2_COMMIT,
    REVIEWED_L2_ORDER_TPS,
    REVIEWED_L2_REPORT_SHA256,
    REVIEWED_L2_WORST_ORDER_TPS,
    REVIEWED_FRAMES_SHA256,
    REVIEWED_MODEL_PATH,
    REVIEWED_PROMPT_ATTENTION_MASK_SHA256,
    REVIEWED_PROMPT_SHA256,
    REVIEWED_WORKLOAD_CONTRACT,
    REVIEWED_WORKLOAD_CONTRACT_SHA256,
    HybridBridgeResourceEvidence,
    HybridSamplingResourceEvidence,
    summarize_hybrid_l2_performance,
    validate_hybrid_l2_report,
    validate_hybrid_processor_family,
    validate_hybrid_resource_ownership,
    validate_reviewed_l2_report,
)
from osuT5.osuT5.inference.optimized.batch.lane_one_token import (
    DEFAULT_SEPARATE_GRAPH_POOL,
    PYTORCH_PER_STREAM_CUBLAS_WORKSPACE,
    LaneResourceEvidence,
)
from utils.verify_optimized_hybrid_l2 import (
    _enqueue_hybrid_steps,
    _run_complete_output_discard_timing,
    _validate_fixed_runtime_contract,
    build_parser,
)


def _lane(lane_id: int, pointer_base: int) -> LaneResourceEvidence:
    return LaneResourceEvidence(
        lane_id=lane_id,
        shared_model_id=10,
        stream_id=100 + lane_id,
        graph_id=200 + lane_id,
        graph_pool_id=f"model-pool-{lane_id}",
        graph_pool_policy=DEFAULT_SEPARATE_GRAPH_POOL,
        graph_pool_sharing_requested=False,
        session_id=300 + lane_id,
        cache_id=400 + lane_id,
        encoder_output_storage_ptr=pointer_base,
        generator_id=500 + lane_id,
        logits_processor_id=600 + lane_id,
        static_input_storage_ptrs={"input": pointer_base + 1},
        self_cache_storage_ptrs=(pointer_base + 2, pointer_base + 3),
        cross_cache_storage_ptrs=(pointer_base + 4, pointer_base + 5),
        cublas_workspace_policy=PYTORCH_PER_STREAM_CUBLAS_WORKSPACE,
        cublas_workspace_owner_stream_id=100 + lane_id,
        cublas_workspace_config=None,
    )


def _bridge() -> HybridBridgeResourceEvidence:
    sampling = tuple(
        HybridSamplingResourceEvidence(
            lane_id=row,
            sampling_graph_id=700 + row,
            sampling_graph_pool_id=f"sampling-pool-{row}",
            processed_scores_storage_ptr=800 + row * 10,
            probabilities_storage_ptr=801 + row * 10,
            sampled_token_storage_ptr=802 + row * 10,
            generator_id=900 + row,
            logits_warper_id=910 + row,
        )
        for row in range(2)
    )
    return HybridBridgeResourceEvidence(
        control_stream_id=1000,
        processor_graph_id=1001,
        processor_graph_pool_id="processor-pool",
        prefix_batch_storage_ptr=1002,
        raw_logits_bridge_storage_ptr=1003,
        processed_scores_storage_ptr=1004,
        source_release_event_ids=(1100, 1101),
        lane_ready_event_ids=(1200, 1201),
        sample_done_event_ids=(1300, 1301),
        processor_done_event_id=1400,
        sampling_lanes=sampling,
    )


def _reviewed_l2_report() -> dict:
    return {
        "lane_count": 2,
        "exactness_pass": True,
        "resource_ownership_pass": True,
        "capture_pass": True,
        "runtime_metadata": {
            "git_commit": REVIEWED_L2_COMMIT,
            "seeds": list(FIXED_ROW_SEEDS),
            "active_prefix_length": FIXED_ACTIVE_PREFIX_LENGTH,
            "probe": {"sequence_index": FIXED_SEQUENCE_INDEX},
        },
        "performance": {
            "worst_order_tokens_per_second": REVIEWED_L2_WORST_ORDER_TPS,
        },
        "orders": {
            order: {
                "complete_timing": {"wall_tokens_per_second": tps},
            }
            for order, tps in REVIEWED_L2_ORDER_TPS.items()
        },
        "workload_contract": copy.deepcopy(REVIEWED_WORKLOAD_CONTRACT),
    }


def _runtime_args(**overrides) -> SimpleNamespace:
    values = {
        "model_path": REVIEWED_MODEL_PATH,
        "device": "cuda",
        "precision": "fp32",
        "attn_implementation": "sdpa",
        "inference_engine": "v32",
        "optimized_inference_mode": "single",
        "profile_sdpa_backend": None,
        "lora_path": None,
        "auto_select_gamemode_model": True,
        "use_server": False,
        "parallel": False,
        "do_sample": True,
        "inference_generation_compile": False,
        "inference_active_prefix_decode_loop": True,
        "inference_active_prefix_decode_cuda_graph": True,
        "inference_stateful_monotonic_logits_processor": True,
        "inference_q1_bmm_cross_attention": True,
        "inference_decode_session_runtime": True,
        "inference_decode_session_cuda_graph": True,
        "inference_native_decode_kernels": True,
        "inference_native_q1_self_attention": True,
        "inference_native_q1_rope_cache_self_attention": True,
        "seed": 12345,
        "gamemode": 0,
        "num_beams": 1,
        "top_k": 0,
        "inference_active_prefix_decode_bucket_size": 64,
        "inference_active_prefix_decode_cuda_graph_warmup": 0,
        "inference_active_prefix_decode_cuda_graph_min_decode_steps": 1,
        "inference_decode_session_chunk_size": 1,
        "cfg_scale": 1.0,
        "temperature": 0.9,
        "timeshift_bias": 0.0,
        "top_p": 0.9,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _order_report(order: str, tps: float) -> dict:
    token_hashes = {"row-0": "token-a", "row-1": "token-b"}
    rng_hashes = {"row-0": "rng-a", "row-1": "rng-b"}
    generated_tokens = 2 * FIXED_TIMING_REPEATS
    wall_seconds = generated_tokens / tps
    cuda_seconds = wall_seconds * 0.95
    return {
        "graph_and_sampling_order": [int(item) for item in order.split(",")],
        "processed_rows_bitwise_match": True,
        "processed_rows_topk_match": True,
        "processed_rows": [
            {
                "row": row,
                "bitwise_match": True,
                "topk_match": True,
                "private_sha256": f"score-{row}",
                "shared_sha256": f"score-{row}",
                "comparison": {
                    "shape_match": True,
                    "allclose": True,
                    "finite_allclose": True,
                    "nonfinite_match": True,
                    "positive_inf_match": True,
                    "negative_inf_match": True,
                    "topk_match": True,
                    "reference_shape": [1, 4069],
                    "candidate_shape": [1, 4069],
                    "reference_topk": [1, 2],
                    "candidate_topk": [1, 2],
                },
                "pass": True,
            }
            for row in range(2)
        ],
        "transcript_match": True,
        "final_rng_state_match": True,
        "timed_final_rng_state_match": True,
        "candidate_token_hashes": dict(token_hashes),
        "candidate_final_rng_state_hashes": dict(rng_hashes),
        "timed_final_rng_state_hashes": dict(rng_hashes),
        "exactness_pass": True,
        "complete_output_discard_timing": {
            "measurement": "complete_output_discard",
            "outputs_discarded": True,
            "timing_repeats": FIXED_TIMING_REPEATS,
            "generated_tokens": generated_tokens,
            "wall_seconds": wall_seconds,
            "cuda_seconds": cuda_seconds,
            "wall_tokens_per_second": tps,
            "cuda_tokens_per_second": generated_tokens / cuda_seconds,
            "memory_before_allocated_bytes": 100,
            "memory_before_reserved_bytes": 200,
            "peak_allocated_bytes": 150,
            "peak_reserved_bytes": 250,
            "peak_allocated_delta_bytes": 50,
            "peak_reserved_delta_bytes": 50,
            "per_step_global_synchronize": False,
            "per_step_stack_or_cat": False,
            "per_step_python_tensor_allocation": False,
            "static_tensor_storage": True,
            "sampling_tail_in_cuda_graph": True,
            "explicit_generators_registered_with_sampling_graphs": True,
        },
    }


def _report(tps_a: float, tps_b: float) -> dict:
    performance = summarize_hybrid_l2_performance({"0,1": tps_a, "1,0": tps_b})
    promotion = performance["five_percent_keep_bar_pass"]
    graduation = promotion and performance["target_500_bar_pass"]
    return {
        "schema_version": HYBRID_L2_SCHEMA_VERSION,
        "gate": HYBRID_L2_GATE,
        "lane_count": 2,
        "pass": graduation,
        "exactness_pass": True,
        "resource_ownership_pass": True,
        "capture_pass": True,
        "safe_event_order_pass": True,
        "promotion_gate_pass": promotion,
        "graduation_to_next_shape_pass": graduation,
        "base_processor_classes": [
            "MonotonicTimeShiftLogitsProcessor",
            "TemperatureLogitsWarper",
        ],
        "private_warper_classes": ["TopPLogitsWarper"],
        "event_order": list(HYBRID_EVENT_ORDER),
        "fixed_workload_contract": {
            "config_name": "profile_salvalai_smoke15",
            "sequence_index": FIXED_SEQUENCE_INDEX,
            "row_seeds": list(FIXED_ROW_SEEDS),
            "timing_repeats": FIXED_TIMING_REPEATS,
            "active_prefix_length": FIXED_ACTIVE_PREFIX_LENGTH,
            "active_prefix_bucket_size": FIXED_ACTIVE_PREFIX_BUCKET_SIZE,
            "model_path": REVIEWED_MODEL_PATH,
            "device": "cuda",
            "precision": "fp32",
            "attn_implementation": "sdpa",
            "profile_sdpa_backend": None,
            "inference_engine": "v32",
            "optimized_inference_mode": "single",
            "do_sample": True,
            "temperature": 0.9,
            "timeshift_bias": 0.0,
            "top_p": 0.9,
            "top_k": 0,
            "cfg_scale": 1.0,
            "num_beams": 1,
            "use_server": False,
            "parallel": False,
            "generation_compile": False,
            "active_prefix_decode_loop": True,
            "active_prefix_decode_cuda_graph": True,
            "active_prefix_decode_cuda_graph_warmup": 0,
            "active_prefix_decode_cuda_graph_min_decode_steps": 1,
            "stateful_monotonic_logits_processor": True,
            "q1_bmm_cross_attention": True,
            "native_q1_self_attention": True,
            "native_q1_rope_cache_self_attention": True,
            "decode_session_runtime": True,
            "decode_session_cuda_graph": True,
            "decode_session_chunk_size": 1,
            "native_decode_kernels": True,
            "prompt_sha256": REVIEWED_PROMPT_SHA256,
            "prompt_attention_mask_sha256": REVIEWED_PROMPT_ATTENTION_MASK_SHA256,
            "frames_sha256": REVIEWED_FRAMES_SHA256,
            "condition_tensor_hashes": {},
            "probe": copy.deepcopy(REVIEWED_WORKLOAD_CONTRACT["runtime_contract"]["probe"]),
            "identical_base_prompt": True,
            "row_private_anchor_and_sampling_state": True,
        },
        "workload_contract": copy.deepcopy(REVIEWED_WORKLOAD_CONTRACT),
        "workload_contract_sha256": REVIEWED_WORKLOAD_CONTRACT_SHA256,
        "reference_token_hashes": {"row-0": "token-a", "row-1": "token-b"},
        "reference_final_rng_state_hashes": {"row-0": "rng-a", "row-1": "rng-b"},
        "orders": {
            "0,1": _order_report("0,1", tps_a),
            "1,0": _order_report("1,0", tps_b),
        },
        "performance": performance,
        "resource_ownership": {
            "model_lanes": [_lane(0, 2000).as_dict(), _lane(1, 3000).as_dict()],
            "hybrid_bridge": _bridge().as_dict(),
        },
        "capture": {
            "model_graph_count": 2,
            "processor_graph_count": 1,
            "private_sampling_graph_count": 2,
            "total_graph_count": 5,
            "explicit_generator_registration": True,
        },
        "reviewed_l2_source": {
            "job_id": "49548733",
            "commit": REVIEWED_L2_COMMIT,
            "report_sha256": REVIEWED_L2_REPORT_SHA256,
            "workload_contract_sha256": REVIEWED_WORKLOAD_CONTRACT_SHA256,
            "worst_order_tokens_per_second": REVIEWED_L2_WORST_ORDER_TPS,
        },
        "runtime_metadata": {"git_commit": "test-commit"},
    }


def test_reviewed_l2_source_and_processor_family_are_pinned():
    validate_reviewed_l2_report(_reviewed_l2_report())
    validate_hybrid_processor_family(
        base_processor_classes=(
            "MonotonicTimeShiftLogitsProcessor",
            "TemperatureLogitsWarper",
        ),
        private_warper_classes=("TopPLogitsWarper",),
    )

    wrong = _reviewed_l2_report()
    wrong["runtime_metadata"]["git_commit"] = "wrong"
    with pytest.raises(ValueError, match="commit"):
        validate_reviewed_l2_report(wrong)
    with pytest.raises(ValueError, match="base processor family"):
        validate_hybrid_processor_family(
            base_processor_classes=("MonotonicTimeShiftLogitsProcessor",),
            private_warper_classes=("TopPLogitsWarper",),
        )


def test_hybrid_resource_contract_allows_only_the_shared_bridge():
    lanes = [_lane(0, 2000), _lane(1, 3000)]
    bridge = _bridge()

    validate_hybrid_resource_ownership(lanes, bridge)

    aliased = HybridBridgeResourceEvidence(
        **{
            **bridge.__dict__,
            "raw_logits_bridge_storage_ptr": lanes[0].encoder_output_storage_ptr,
        }
    )
    with pytest.raises(ValueError, match="bridge tensors"):
        validate_hybrid_resource_ownership(lanes, aliased)

    aliased_events = HybridBridgeResourceEvidence(
        **{
            **bridge.__dict__,
            "source_release_event_ids": (
                bridge.lane_ready_event_ids[0],
                bridge.source_release_event_ids[1],
            ),
        }
    )
    with pytest.raises(ValueError, match="event IDs must be pairwise distinct"):
        validate_hybrid_resource_ownership(lanes, aliased_events)

    duplicate_bridge_pointer = HybridBridgeResourceEvidence(
        **{
            **bridge.__dict__,
            "processed_scores_storage_ptr": bridge.raw_logits_bridge_storage_ptr,
        }
    )
    with pytest.raises(ValueError, match="bridge tensor pointers must be pairwise distinct"):
        validate_hybrid_resource_ownership(lanes, duplicate_bridge_pointer)

    duplicate_sampling_lane = HybridSamplingResourceEvidence(
        **{
            **bridge.sampling_lanes[0].__dict__,
            "probabilities_storage_ptr": (
                bridge.sampling_lanes[0].processed_scores_storage_ptr
            ),
        }
    )
    duplicate_sampling_pointer = HybridBridgeResourceEvidence(
        **{
            **bridge.__dict__,
            "sampling_lanes": (duplicate_sampling_lane, bridge.sampling_lanes[1]),
        }
    )
    with pytest.raises(ValueError, match="sampling lane 0 tensor pointers"):
        validate_hybrid_resource_ownership(lanes, duplicate_sampling_pointer)

    cross_group_alias = HybridBridgeResourceEvidence(
        **{
            **bridge.__dict__,
            "processed_scores_storage_ptr": (
                bridge.sampling_lanes[0].processed_scores_storage_ptr
            ),
        }
    )
    with pytest.raises(ValueError, match="bridge and private sampling tensors"):
        validate_hybrid_resource_ownership(lanes, cross_group_alias)


def test_performance_reports_five_percent_and_500_bars_separately():
    keep_only = summarize_hybrid_l2_performance({"0,1": 450.0, "1,0": 470.0})
    target = summarize_hybrid_l2_performance({"0,1": 501.0, "1,0": 510.0})

    assert FIVE_PERCENT_KEEP_TPS == pytest.approx(435.64993222366656)
    assert keep_only["five_percent_keep_bar_pass"] is True
    assert keep_only["target_500_bar_pass"] is False
    assert target["five_percent_keep_bar_pass"] is True
    assert target["target_500_bar_pass"] is True


def test_direct_runtime_contract_requires_reviewed_numeric_and_native_path():
    kwargs = {
        "config_name": "profile_salvalai_smoke15",
        "q1_bmm_cross_attention": True,
        "native_q1_self_attention": True,
        "native_q1_rope_cache_self_attention": True,
    }
    _validate_fixed_runtime_contract(_runtime_args(), **kwargs)

    for field in (
            "q1_bmm_cross_attention",
            "native_q1_self_attention",
            "native_q1_rope_cache_self_attention",
    ):
        changed = {**kwargs, field: False}
        with pytest.raises(ValueError, match=f"CLI {field}=true"):
            _validate_fixed_runtime_contract(_runtime_args(), **changed)

    for args, message in (
            (_runtime_args(attn_implementation="eager"), "attn_implementation"),
            (_runtime_args(temperature=1.0), "temperature"),
            (_runtime_args(timeshift_bias=1.0), "timeshift_bias"),
            (_runtime_args(inference_q1_bmm_cross_attention=False), "inference_q1"),
            (_runtime_args(use_server=True), "use_server"),
    ):
        with pytest.raises(ValueError, match=message):
            _validate_fixed_runtime_contract(args, **kwargs)


def test_report_requires_exact_reciprocal_complete_output_discard_evidence():
    passing = _report(501.0, 505.0)
    validate_hybrid_l2_report(passing)

    keep_only = _report(450.0, 460.0)
    validate_hybrid_l2_report(keep_only)
    assert keep_only["promotion_gate_pass"] is True
    assert keep_only["graduation_to_next_shape_pass"] is False
    assert keep_only["pass"] is False

    malformed = _report(501.0, 505.0)
    malformed["orders"]["1,0"]["complete_output_discard_timing"][
        "per_step_stack_or_cat"
    ] = True
    with pytest.raises(ValueError, match="concatenate"):
        validate_hybrid_l2_report(malformed)


def test_structurally_valid_negative_exactness_reports_are_preserved():
    for failure in ("processed", "transcript", "rng"):
        rejected = _report(501.0, 505.0)
        order = rejected["orders"]["0,1"]
        if failure == "processed":
            order["processed_rows"][0]["bitwise_match"] = False
            order["processed_rows"][0]["shared_sha256"] = "different-score-hash"
            order["processed_rows"][0]["pass"] = False
            order["processed_rows_bitwise_match"] = False
        elif failure == "transcript":
            order["candidate_token_hashes"]["row-0"] = "different-token-hash"
            order["transcript_match"] = False
        else:
            order["candidate_final_rng_state_hashes"]["row-0"] = "different-rng-hash"
            order["final_rng_state_match"] = False
        order["exactness_pass"] = False
        rejected["exactness_pass"] = False
        rejected["promotion_gate_pass"] = False
        rejected["graduation_to_next_shape_pass"] = False
        rejected["pass"] = False

        validate_hybrid_l2_report(rejected)


def test_processed_score_booleans_must_match_hash_and_nested_comparison_evidence():
    tampered_hash = _report(501.0, 505.0)
    row = tampered_hash["orders"]["0,1"]["processed_rows"][0]
    row["shared_sha256"] = "different-score-hash"
    with pytest.raises(ValueError, match="bitwise_match does not match"):
        validate_hybrid_l2_report(tampered_hash)

    tampered_topk = _report(501.0, 505.0)
    row = tampered_topk["orders"]["1,0"]["processed_rows"][1]
    row["comparison"]["candidate_topk"] = [9, 10]
    with pytest.raises(ValueError, match="comparison top-k is inconsistent"):
        validate_hybrid_l2_report(tampered_topk)


def test_report_recomputes_tps_from_measured_seconds_before_promotion():
    tampered = _report(501.0, 505.0)
    tampered["orders"]["0,1"]["complete_output_discard_timing"]["wall_seconds"] = 999.0

    with pytest.raises(ValueError, match="wall_tokens_per_second does not self-validate"):
        validate_hybrid_l2_report(tampered)

    tampered_cuda = _report(501.0, 505.0)
    tampered_cuda["orders"]["1,0"]["complete_output_discard_timing"][
        "cuda_tokens_per_second"
    ] = 999.0
    with pytest.raises(ValueError, match="cuda_tokens_per_second does not self-validate"):
        validate_hybrid_l2_report(tampered_cuda)

    fractional_tokens = _report(501.0, 505.0)
    fractional_tokens["orders"]["1,0"]["complete_output_discard_timing"][
        "generated_tokens"
    ] = 100.9
    with pytest.raises(ValueError, match="exactly two tokens per repeat"):
        validate_hybrid_l2_report(fractional_tokens)


def test_report_pins_source_workload_and_runtime_commit():
    wrong_source = _report(501.0, 505.0)
    wrong_source["reviewed_l2_source"]["job_id"] = "wrong"
    with pytest.raises(ValueError, match="source field job_id"):
        validate_hybrid_l2_report(wrong_source)

    wrong_contract = _report(501.0, 505.0)
    wrong_contract["fixed_workload_contract"]["precision"] = "amp"
    with pytest.raises(ValueError, match="precision"):
        validate_hybrid_l2_report(wrong_contract)

    missing_commit = _report(501.0, 505.0)
    missing_commit["runtime_metadata"]["git_commit"] = ""
    with pytest.raises(ValueError, match="git commit must be non-empty"):
        validate_hybrid_l2_report(missing_commit)

    wrong_workload = _report(501.0, 505.0)
    wrong_workload["workload_contract"]["prompt_sha256"] = "different"
    with pytest.raises(ValueError, match="workload contract differs"):
        validate_hybrid_l2_report(wrong_workload)


def test_report_recomputes_and_validates_memory_deltas():
    tampered = _report(501.0, 505.0)
    timing = tampered["orders"]["1,0"]["complete_output_discard_timing"]
    timing["peak_allocated_delta_bytes"] += 1

    with pytest.raises(ValueError, match="allocated memory delta"):
        validate_hybrid_l2_report(tampered)


def test_timed_enqueue_source_contains_no_per_step_global_sync_or_tensor_assembly():
    enqueue_source = inspect.getsource(_enqueue_hybrid_steps)
    timing_source = inspect.getsource(_run_complete_output_discard_timing)

    assert "torch.cuda.synchronize" not in enqueue_source
    assert "torch.stack" not in enqueue_source
    assert "torch.cat" not in enqueue_source
    assert "torch.empty" not in enqueue_source
    assert "row:row" not in enqueue_source
    assert "torch.stack" not in timing_source
    assert "torch.cat" not in timing_source


def test_cli_locks_the_incremental_fixed_shape_defaults():
    args = build_parser().parse_args([
        "--reviewed-l2-report",
        "l2.json",
        "--report-path",
        "hybrid.json",
        "--row-seed",
        "12345",
        "--row-seed",
        "23456",
        "inference_generation_compile=false",
    ])

    assert args.sequence_index == FIXED_SEQUENCE_INDEX
    assert args.row_seed == list(FIXED_ROW_SEEDS)
    assert args.timing_repeats == FIXED_TIMING_REPEATS
    assert args.active_prefix_bucket_size == FIXED_ACTIVE_PREFIX_BUCKET_SIZE
    assert args.overrides == ["inference_generation_compile=false"]
