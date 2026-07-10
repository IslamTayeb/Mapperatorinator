from __future__ import annotations

import copy
import inspect

import pytest

from osuT5.osuT5.inference.optimized.batch.hybrid_changing_prefix import (
    ACTIVE_PREFIX_LENGTH,
    CHANGING_PREFIX_EVENT_ORDER,
    CHANGING_PREFIX_GATE,
    CHANGING_PREFIX_SCHEMA_VERSION,
    FIXED_HYBRID_SOURCE_COMMIT,
    FIXED_HYBRID_SOURCE_JOB,
    FIXED_HYBRID_SOURCE_REPORT_SHA256,
    FIXED_HYBRID_SOURCE_WORST_TPS,
    PHASE_A_HORIZON,
    PHASE_B_HORIZON,
    PHASE_B_TIMING_TRIALS,
    PRODUCTION_BUCKET128_DECODE_TOKENS,
    PRODUCTION_TOTAL_DECODE_TOKENS,
    summarize_changing_prefix_performance,
    validate_changing_prefix_report,
)
from osuT5.osuT5.inference.optimized.batch.hybrid_l2 import (
    HybridBridgeResourceEvidence,
    HybridSamplingResourceEvidence,
    REVIEWED_FRAMES_SHA256,
    REVIEWED_PROMPT_ATTENTION_MASK_SHA256,
    REVIEWED_PROMPT_SHA256,
    REVIEWED_WORKLOAD_CONTRACT_SHA256,
)
from osuT5.osuT5.inference.optimized.batch.lane_one_token import (
    DEFAULT_SEPARATE_GRAPH_POOL,
    PYTORCH_PER_STREAM_CUBLAS_WORKSPACE,
    LaneResourceEvidence,
)
from utils.verify_optimized_hybrid_changing_prefix import (
    _enqueue_timed_trial,
    build_parser,
)


def _lane(row: int, base: int) -> LaneResourceEvidence:
    return LaneResourceEvidence(
        lane_id=row,
        shared_model_id=1,
        stream_id=100 + row,
        graph_id=200 + row,
        graph_pool_id=f"model-pool-{row}",
        graph_pool_policy=DEFAULT_SEPARATE_GRAPH_POOL,
        graph_pool_sharing_requested=False,
        session_id=300 + row,
        cache_id=400 + row,
        encoder_output_storage_ptr=base,
        generator_id=500 + row,
        logits_processor_id=600 + row,
        static_input_storage_ptrs={
            "cache_position": base + 1,
            "decoder_attention_mask": base + 2,
            "decoder_input_ids": base + 3,
            "decoder_position_ids": base + 4,
        },
        self_cache_storage_ptrs=(base + 5, base + 6),
        cross_cache_storage_ptrs=(base + 7, base + 8),
        cublas_workspace_policy=PYTORCH_PER_STREAM_CUBLAS_WORKSPACE,
        cublas_workspace_owner_stream_id=100 + row,
        cublas_workspace_config=None,
    )


def _bridge() -> HybridBridgeResourceEvidence:
    sampling = tuple(
        HybridSamplingResourceEvidence(
            lane_id=row,
            sampling_graph_id=700 + row,
            sampling_graph_pool_id=f"sample-pool-{row}",
            processed_scores_storage_ptr=800 + 10 * row,
            probabilities_storage_ptr=801 + 10 * row,
            sampled_token_storage_ptr=802 + 10 * row,
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


def _comparison(*, reference_topk=None, candidate_topk=None) -> dict:
    reference_topk = [1, 2] if reference_topk is None else reference_topk
    candidate_topk = list(reference_topk) if candidate_topk is None else candidate_topk
    return {
        "shape_match": True,
        "allclose": True,
        "topk_match": reference_topk == candidate_topk,
        "reference_shape": [1, 4069],
        "candidate_shape": [1, 4069],
        "reference_topk": reference_topk,
        "candidate_topk": candidate_topk,
        "max_abs": 0.0,
        "reference_sha256": "comparison",
        "candidate_sha256": "comparison",
    }


def _hash_pair(value: str) -> dict:
    return {
        "reference_sha256": value,
        "candidate_sha256": value,
        "bitwise_match": True,
    }


def _row_step(row: int, step: int) -> dict:
    raw_hash = f"raw-{row}-{step}"
    raw = {
        **_comparison(),
        "reference_sha256": f"reference-{raw_hash}",
        "candidate_sha256": raw_hash,
        "changed_from_previous": None if step == 0 else True,
    }
    processed = {**_hash_pair(f"processed-{row}-{step}"), **_comparison()}
    static = {
        key: _hash_pair(f"static-{row}-{step}-{key}")
        for key in (
            "decoder_input_ids",
            "cache_position",
            "decoder_position_ids",
            "decoder_attention_mask",
        )
    }
    cross = {
        **_hash_pair(f"cross-{row}"),
        "reference_comparison": _comparison(),
    }
    padded = {
        **_hash_pair(f"padded-prefix-{row}-{step}"),
        "real_length": 85 + step,
        "full_length": ACTIVE_PREFIX_LENGTH,
        "model_cache_position": 84 + step,
        "pad_token_id": 0,
        "suffix_all_pad": True,
        "pad_outside_sos_ids": True,
        "pad_outside_time_shift_range": True,
        "suffix": _hash_pair(f"zero-suffix-{step}"),
    }
    post_state = {
        "rng_state": _hash_pair(f"rng-{row}-{step}"),
        "prefix": _hash_pair(f"post-prefix-{row}-{step}"),
        "next_static_inputs": {
            key: _hash_pair(f"post-static-{row}-{step}-{key}")
            for key in (
                "decoder_input_ids",
                "cache_position",
                "decoder_position_ids",
                "decoder_attention_mask",
            )
        },
        "self_cache": {
            **_comparison(),
            "reference_sha256": f"post-self-reference-{row}-{step}",
            "candidate_sha256": f"post-self-candidate-{row}-{step}",
        },
        "cross_cache": {
            **_comparison(),
            "reference_sha256": f"cross-{row}",
            "candidate_sha256": f"cross-{row}",
        },
        "executed_cache_position": 84 + step,
        "next_static_cache_position": 85 + step,
        "prefix_real_length": 86 + step,
    }
    return {
        "row": row,
        "raw_logits": raw,
        "processed_scores": processed,
        "sampled_token": {
            "reference": 1000 + row * 100 + step,
            "candidate": 1000 + row * 100 + step,
            "match": True,
        },
        "rng_state": _hash_pair(f"rng-{row}-{step}"),
        "stop": {
            "reference": False,
            "candidate": False,
            "match": True,
            "sampled_after_stop": False,
        },
        "padded_prefix": padded,
        "static_inputs": static,
        "cache": {
            "active_prefix": _comparison(),
            "current_slot": {
                **_comparison(),
                "candidate_nonzero": True,
                "changed_from_reset": True,
            },
            "future_suffix": _hash_pair(f"future-{row}"),
            "cross": cross,
        },
        "post_step_state": post_state,
    }


def _reset_contract(lanes: list[LaneResourceEvidence]) -> dict:
    return {
        "self_cache_snapshot_hashes": {"row-0": "self-0", "row-1": "self-1"},
        "post_restore_self_cache_hashes": {"row-0": "self-0", "row-1": "self-1"},
        "cross_cache_snapshot_hashes": {"row-0": "cross-0", "row-1": "cross-1"},
        "post_restore_cross_cache_hashes": {"row-0": "cross-0", "row-1": "cross-1"},
        "initial_seed_rng_state_hashes": {"row-0": "seed-0", "row-1": "seed-1"},
        "post_anchor_rng_state_hashes": {"row-0": "anchor-0", "row-1": "anchor-1"},
        "anchor_tokens": {
            "row-0": {"reference": 100, "candidate": 100, "match": True},
            "row-1": {"reference": 101, "candidate": 101, "match": True},
        },
        "post_anchor_rng_states": {
            "row-0": _hash_pair("anchor-0"),
            "row-1": _hash_pair("anchor-1"),
        },
        "static_input_storage_ptrs": {
            f"row-{row}": lane.static_input_storage_ptrs
            for row, lane in enumerate(lanes)
        },
        "self_cache_storage_ptrs": {
            f"row-{row}": list(lane.self_cache_storage_ptrs)
            for row, lane in enumerate(lanes)
        },
        "cross_cache_storage_ptrs": {
            f"row-{row}": list(lane.cross_cache_storage_ptrs)
            for row, lane in enumerate(lanes)
        },
        "in_place_restore": True,
        "captured_pointer_swap": False,
        "cache_snapshot_stage": "post_prefill_before_anchor_or_capture",
        "restore_verified_after_graph_capture": True,
    }


def _timing(
        horizon: int,
        trials: int,
        reset: dict,
        tps: float,
        final_state_fingerprints: dict,
) -> dict:
    trial_tokens = 2 * horizon
    trial_wall = trial_tokens / tps
    trial_cuda = trial_wall * 0.95
    trial_values = [
        {
            "trial": trial,
            "generated_tokens": trial_tokens,
            "wall_seconds": trial_wall,
            "cuda_seconds": trial_cuda,
            "wall_tokens_per_second": tps,
            "cuda_tokens_per_second": trial_tokens / trial_cuda,
            "reset_evidence": copy.deepcopy(reset),
            "initial_rng_state_hashes": copy.deepcopy(reset["post_anchor_rng_state_hashes"]),
            "final_stop_flags": {"row-0": False, "row-1": False},
            "final_state_fingerprints": copy.deepcopy(final_state_fingerprints),
        }
        for trial in range(trials)
    ]
    tokens = trial_tokens * trials
    wall = trial_wall * trials
    cuda = trial_cuda * trials
    return {
        "horizon": horizon,
        "trial_count": trials,
        "generated_tokens": tokens,
        "wall_seconds": wall,
        "cuda_seconds": cuda,
        "wall_tokens_per_second": tokens / wall,
        "cuda_tokens_per_second": tokens / cuda,
        "trials": trial_values,
        "memory_before_allocated_bytes": 100,
        "memory_before_reserved_bytes": 200,
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 200,
        "peak_allocated_delta_bytes": 0,
        "peak_reserved_delta_bytes": 0,
        "reset_outside_timing": True,
        "initial_state_copies_inside_timing": True,
        "per_step_updates_inside_timing": True,
        "device_stop_flags": True,
        "static_tensor_storage": True,
        "sampling_tail_in_cuda_graph": True,
        "explicit_generators_registered_with_sampling_graphs": True,
        "no_eos_in_timed_horizon": True,
        "per_step_global_synchronize": False,
        "per_step_stack_or_cat": False,
        "per_step_python_tensor_allocation": False,
        "graph_recapture_between_trials": False,
        "dummy_or_stopped_row_sampled": False,
    }


def _order(order: str, horizon: int, trials: int, reset: dict, candidate_tps: float) -> dict:
    final_state_fingerprints = {
        f"row-{row}": _row_step(row, horizon - 1)["post_step_state"]
        for row in range(2)
    }
    timing = _timing(horizon, trials, reset, candidate_tps, final_state_fingerprints)
    control = _timing(horizon, trials, reset, 400.0, final_state_fingerprints)
    control["measurement"] = "ordinary_private_B1_changing_prefix_control"
    control["promotion_gate_uses_control"] = False
    relative = candidate_tps / 400.0 - 1.0
    return {
        "graph_and_sampling_order": [int(value) for value in order.split(",")],
        "committed_steps": horizon,
        "reset_evidence": copy.deepcopy(reset),
        "initial_rng_state_hashes": copy.deepcopy(reset["post_anchor_rng_state_hashes"]),
        "raw_logits_allclose": True,
        "raw_logits_topk_match": True,
        "raw_logits_changed_each_step": True,
        "processed_rows_bitwise_match": True,
        "processed_rows_topk_match": True,
        "sampled_tokens_match": True,
        "per_step_rng_state_match": True,
        "final_rng_state_match": True,
        "stop_behavior_match": True,
        "static_inputs_bitwise_match": True,
        "cache_active_prefix_allclose": True,
        "cache_current_slot_written": True,
        "cache_future_suffix_unchanged": True,
        "cross_cache_unchanged_bitwise": True,
        "cross_cache_reference_allclose": True,
        "neutral_padding_proof_pass": True,
        "no_dummy_or_stopped_row_sampled": True,
        "no_eos_in_horizon": True,
        "exactness_pass": True,
        "steps": [
            {"step": step, "rows": [_row_step(0, step), _row_step(1, step)]}
            for step in range(horizon)
        ],
        "final_rng_state_hashes": {
            "row-0": f"rng-0-{horizon - 1}",
            "row-1": f"rng-1-{horizon - 1}",
        },
        "final_state_fingerprints": final_state_fingerprints,
        "complete_timing": timing,
        "same_job_private_b1_control": control,
        "relative_gain_vs_same_job_control": relative,
        "five_percent_vs_same_job_control": relative > 0.05,
    }


def _phase(horizon: int, trials: int, reset: dict, tps_a: float, tps_b: float) -> dict:
    orders = {
        "0,1": _order("0,1", horizon, trials, reset, tps_a),
        "1,0": _order("1,0", horizon, trials, reset, tps_b),
    }
    performance = summarize_changing_prefix_performance({
        order: values["complete_timing"]
        for order, values in orders.items()
    })
    return {
        "horizon": horizon,
        "timing_trials_per_order": trials,
        "orders": orders,
        "performance": performance,
        "exactness_pass": True,
        "target_500_gate_pass": performance["target_500_bar_pass"],
    }


def _report() -> dict:
    lanes = [_lane(0, 2000), _lane(1, 3000)]
    bridge = _bridge()
    reset = _reset_contract(lanes)
    return {
        "schema_version": CHANGING_PREFIX_SCHEMA_VERSION,
        "gate": CHANGING_PREFIX_GATE,
        "lane_count": 2,
        "pass": True,
        "fixed_hybrid_source": {
            "job_id": FIXED_HYBRID_SOURCE_JOB,
            "commit": FIXED_HYBRID_SOURCE_COMMIT,
            "report_sha256": FIXED_HYBRID_SOURCE_REPORT_SHA256,
            "worst_order_tokens_per_second": FIXED_HYBRID_SOURCE_WORST_TPS,
        },
        "fixed_workload_contract": {
            "phase_a_horizon": PHASE_A_HORIZON,
            "phase_b_horizon": PHASE_B_HORIZON,
            "phase_b_timing_trials": PHASE_B_TIMING_TRIALS,
            "active_prefix_length": ACTIVE_PREFIX_LENGTH,
            "prompt_length": 84,
            "neutral_pad_token_id": 0,
            "model_path": "OliBomby/Mapperatorinator-v32",
            "precision": "fp32",
            "attn_implementation": "sdpa",
            "sequence_index": 9,
            "row_seeds": [12345, 23456],
            "processor_prefix_shape": [2, ACTIVE_PREFIX_LENGTH],
            "reference_prefix_policy": "private_unpadded_B1",
            "candidate_prefix_policy": "shared_neutral_padded_B2",
            "prompt_sha256": REVIEWED_PROMPT_SHA256,
            "prompt_attention_mask_sha256": REVIEWED_PROMPT_ATTENTION_MASK_SHA256,
            "frames_sha256": REVIEWED_FRAMES_SHA256,
            "condition_tensor_hashes": {},
            "source_workload_contract_sha256": REVIEWED_WORKLOAD_CONTRACT_SHA256,
        },
        "event_order": list(CHANGING_PREFIX_EVENT_ORDER),
        "resource_ownership_pass": True,
        "capture_pass": True,
        "pointer_stability_pass": True,
        "reset_in_place_pass": True,
        "processor_family_pass": True,
        "processor_contract": {
            "shared_base_processor_classes": [
                "MonotonicTimeShiftLogitsProcessor",
                "TemperatureLogitsWarper",
            ],
            "private_warper_classes": ["TopPLogitsWarper"],
            "neutral_pad_token_id": 0,
            "sos_token_ids": [1, 3],
            "time_shift_start": 10,
            "time_shift_end": 20,
        },
        "resource_ownership": {
            "model_lanes": [lane.as_dict() for lane in lanes],
            "hybrid_bridge": bridge.as_dict(),
            "source_logit_views": [
                {
                    "row": row,
                    "graph_output_storage_ptr": 4000 + row,
                    "view_storage_ptr": 4000 + row,
                    "zero_copy_alias": True,
                    "captured_lane_logits_property_used": False,
                }
                for row in range(2)
            ],
            "state_ready_event_ids": [1500, 1501],
            "stop_flag_storage_ptrs": [1600, 1601],
            "sampling_eos_token_storage_ptrs": [1700, 1701],
            "sampling_stop_output_storage_ptrs": [1800, 1801],
            "capture": {
                "model_graph_count": 2,
                "processor_graph_count": 1,
                "private_sampling_and_stop_graph_count": 2,
                "total_graph_count": 5,
                "explicit_generator_registration": True,
                "graph_recapture_between_orders_or_trials": False,
            },
        },
        "reset_contract": reset,
        "phase_a": _phase(PHASE_A_HORIZON, 1, reset, 540.0, 530.0),
        "phase_b": _phase(
            PHASE_B_HORIZON,
            PHASE_B_TIMING_TRIALS,
            reset,
            525.0,
            515.0,
        ),
        "graduation_scope": {
            "production_bucket128_decode_tokens": PRODUCTION_BUCKET128_DECODE_TOKENS,
            "production_total_decode_tokens": PRODUCTION_TOTAL_DECODE_TOKENS,
            "production_bucket128_fraction": (
                PRODUCTION_BUCKET128_DECODE_TOKENS / PRODUCTION_TOTAL_DECODE_TOKENS
            ),
            "authorized_next_step": "weighted_active_prefix_bucket_sweep",
            "scheduler_or_runtime_wiring_authorized": False,
        },
    }


def test_passing_report_self_validates_and_pins_scope():
    report = _report()
    validate_changing_prefix_report(report)

    assert report["graduation_scope"]["production_bucket128_fraction"] == pytest.approx(
        305 / 5905
    )
    assert report["graduation_scope"]["scheduler_or_runtime_wiring_authorized"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["phase_a"]["orders"]["0,1"]["steps"][3]["rows"][0][
                "processed_scores"
            ].__setitem__("candidate_sha256", "tampered"),
            "processed scores bitwise_match",
        ),
        (
            lambda report: report["phase_b"]["orders"]["1,0"]["steps"][2]["rows"][1][
                "sampled_token"
            ].__setitem__("candidate", -1),
            "token IDs must be non-negative integers",
        ),
        (
            lambda report: report["phase_b"]["orders"]["0,1"]["complete_timing"]["trials"][0].__setitem__(
                "generated_tokens", 1
            ),
            "each trial",
        ),
        (
            lambda report: report["resource_ownership"]["source_logit_views"][0].__setitem__(
                "view_storage_ptr", 9999
            ),
            "alias graph output",
        ),
        (
            lambda report: (
                report["resource_ownership"]["source_logit_views"][1].__setitem__(
                    "graph_output_storage_ptr", 4000
                ),
                report["resource_ownership"]["source_logit_views"][1].__setitem__(
                    "view_storage_ptr", 4000
                ),
            ),
            "distinct graph-output views",
        ),
        (
            lambda report: report["reset_contract"]["post_anchor_rng_state_hashes"].__setitem__(
                "row-0", "seed-0"
            ),
            "anchor consumed RNG",
        ),
    ],
)
def test_report_rejects_tampered_step_order_trial_resource_and_rng_evidence(mutation, message):
    report = _report()
    mutation(report)
    with pytest.raises(ValueError, match=message):
        validate_changing_prefix_report(report)


def test_reciprocal_request_evidence_must_be_order_invariant():
    report = _report()
    row = report["phase_a"]["orders"]["1,0"]["steps"][4]["rows"][0]
    row["sampled_token"]["reference"] = 777
    row["sampled_token"]["candidate"] = 777

    with pytest.raises(ValueError, match="order-dependent"):
        validate_changing_prefix_report(report)


def test_stale_raw_logits_and_final_step_eos_cannot_authorize_timing():
    stale = _report()
    row = stale["phase_a"]["orders"]["0,1"]["steps"][1]["rows"][0]
    row["raw_logits"]["candidate_sha256"] = "raw-0-0"
    row["raw_logits"]["changed_from_previous"] = False
    stale["phase_a"]["orders"]["0,1"]["raw_logits_changed_each_step"] = False
    stale["phase_a"]["orders"]["0,1"]["exactness_pass"] = False
    with pytest.raises(ValueError, match="order-dependent"):
        validate_changing_prefix_report(stale)

    stopped = _report()
    for order in ("0,1", "1,0"):
        values = stopped["phase_a"]["orders"][order]
        stop = values["steps"][-1]["rows"][0]["stop"]
        stop["reference"] = True
        stop["candidate"] = True
        values["no_eos_in_horizon"] = False
        values["exactness_pass"] = False
        values["complete_timing"] = None
        values["same_job_private_b1_control"] = None
    stopped["phase_a"]["exactness_pass"] = False
    stopped["phase_a"]["target_500_gate_pass"] = False
    stopped["phase_a"]["performance"] = None
    stopped["phase_b"] = None
    stopped["pass"] = False
    validate_changing_prefix_report(stopped)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["phase_a"]["orders"]["0,1"]["steps"][0]["rows"][0][
                "sampled_token"
            ].update(reference=None, candidate=None),
            "token IDs must be non-negative integers",
        ),
        (
            lambda report: report["phase_a"]["orders"]["0,1"]["steps"][0]["rows"][0][
                "stop"
            ].__setitem__("reference", None),
            "stop field reference must be boolean",
        ),
        (
            lambda report: report["reset_contract"]["self_cache_snapshot_hashes"].__setitem__(
                "row-0", None
            ),
            "requires nonempty hashes",
        ),
        (
            lambda report: report["phase_a"]["orders"]["0,1"]["steps"][0]["rows"][0][
                "raw_logits"
            ].update(shape_match=False, candidate_shape=[2, 4069]),
            "cannot match values when tensor shapes differ",
        ),
        (
            lambda report: report["phase_a"]["orders"]["1,0"]["steps"][2]["rows"][0][
                "static_inputs"
            ]["decoder_input_ids"].update(
                reference_sha256="order-drift", candidate_sha256="order-drift"
            ),
            "order-dependent",
        ),
        (
            lambda report: report["phase_b"]["orders"]["0,1"]["complete_timing"]["trials"][0][
                "final_state_fingerprints"
            ]["row-0"]["rng_state"].__setitem__("candidate_sha256", "wrong"),
            "final state differs from exact decode",
        ),
        (
            lambda report: report["phase_a"]["orders"]["0,1"]["steps"][3]["rows"][0][
                "post_step_state"
            ]["prefix"].__setitem__("candidate_sha256", "forged"),
            "post-step prefix bitwise_match",
        ),
        (
            lambda report: report["reset_contract"]["anchor_tokens"]["row-0"].update(
                candidate=999, match=False
            ),
            "anchor differs from independent reference",
        ),
    ],
)
def test_report_rejects_missing_or_drifted_exact_and_timed_state(mutation, message):
    report = _report()
    mutation(report)
    with pytest.raises(ValueError, match=message):
        validate_changing_prefix_report(report)


def test_phase_b_is_required_only_after_phase_a_exact_target_pass():
    report = _report()
    report["phase_b"] = None
    with pytest.raises(ValueError, match="Phase B is required"):
        validate_changing_prefix_report(report)

    rejected = _report()
    for order in ("0,1", "1,0"):
        timing = rejected["phase_a"]["orders"][order]["complete_timing"]
        trial = timing["trials"][0]
        trial["wall_seconds"] = trial["generated_tokens"] / 450.0
        trial["wall_tokens_per_second"] = 450.0
        timing["wall_seconds"] = trial["wall_seconds"]
        timing["wall_tokens_per_second"] = 450.0
        rejected["phase_a"]["orders"][order][
            "relative_gain_vs_same_job_control"
        ] = 450.0 / 400.0 - 1.0
        rejected["phase_a"]["orders"][order][
            "five_percent_vs_same_job_control"
        ] = True
    rejected["phase_a"]["performance"] = summarize_changing_prefix_performance({
        order: rejected["phase_a"]["orders"][order]["complete_timing"]
        for order in ("0,1", "1,0")
    })
    rejected["phase_a"]["target_500_gate_pass"] = False
    rejected["phase_b"] = None
    rejected["pass"] = False
    validate_changing_prefix_report(rejected)


def test_timed_candidate_loop_has_no_per_step_tensor_assembly_or_sync():
    source = inspect.getsource(_enqueue_timed_trial)

    assert "torch.cat" not in source
    assert "torch.stack" not in source
    assert "torch.empty" not in source
    assert ".reshape(" not in source
    assert source.count(".synchronize(") == 1
    assert "prefix_batch[row:" not in source


def test_changing_prefix_cli_keeps_the_staged_shape_fixed():
    cli = build_parser().parse_args([
        "--fixed-hybrid-report",
        "fixed.json",
        "--report-path",
        "changing.json",
        "--row-seed",
        "12345",
        "--row-seed",
        "23456",
        "device=cuda",
    ])

    assert cli.sequence_index == 9
    assert cli.row_seed == [12345, 23456]
    assert cli.capture_warmup_repeats == 1
    assert cli.overrides == ["device=cuda"]
