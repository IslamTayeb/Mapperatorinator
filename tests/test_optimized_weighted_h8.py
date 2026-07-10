from __future__ import annotations

import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.inference.optimized.batch import weighted_h8 as _MODULE
from utils import verify_optimized_weighted_h8 as _VERIFIER


def _digest(value: int = 0) -> str:
    return f"{value:064x}"


def _hash_pair(value: int = 0) -> dict:
    digest = _digest(value)
    return {
        "reference_sha256": digest,
        "candidate_sha256": digest,
        "bitwise_match": True,
    }


def _comparison(value: int = 0) -> dict:
    digest = _digest(value)
    return {
        "shape_match": True,
        "allclose": True,
        "topk_match": True,
        "reference_shape": [1, 10],
        "candidate_shape": [1, 10],
        "reference_topk": [1, 2, 3],
        "candidate_topk": [1, 2, 3],
        "max_abs": 0.0,
        "reference_sha256": digest,
        "candidate_sha256": digest,
    }


def _timing(measurement: str, tps: float) -> dict:
    tokens_per_trial = 2 * _MODULE.HORIZON
    trial_wall = tokens_per_trial / tps
    trial_cuda = trial_wall * 0.98
    trials = [
        {
            "trial": index,
            "generated_tokens": tokens_per_trial,
            "wall_seconds": trial_wall,
            "cuda_seconds": trial_cuda,
            "wall_tokens_per_second": tokens_per_trial / trial_wall,
            "cuda_tokens_per_second": tokens_per_trial / trial_cuda,
            "final_state_matches_exact": True,
            "memory_before_allocated_bytes": 1000,
            "memory_before_reserved_bytes": 2000,
            "peak_allocated_bytes": 1000,
            "peak_reserved_bytes": 2000,
            "peak_allocated_delta_bytes": 0,
            "peak_reserved_delta_bytes": 0,
        }
        for index in range(_MODULE.TIMING_TRIALS_PER_ORDER)
    ]
    tokens = tokens_per_trial * _MODULE.TIMING_TRIALS_PER_ORDER
    wall = sum(item["wall_seconds"] for item in trials)
    cuda = sum(item["cuda_seconds"] for item in trials)
    return {
        "measurement": measurement,
        "horizon": _MODULE.HORIZON,
        "trial_count": _MODULE.TIMING_TRIALS_PER_ORDER,
        "generated_tokens": tokens,
        "wall_seconds": wall,
        "cuda_seconds": cuda,
        "wall_tokens_per_second": tokens / wall,
        "cuda_tokens_per_second": tokens / cuda,
        "trials": trials,
        "no_per_step_synchronize": True,
        "no_per_step_allocation": True,
        "no_graph_recapture": True,
        "reset_outside_timing": True,
        "initial_state_copy_inside_timing": True,
        "final_state_matches_exact": True,
    }


def _exactness() -> dict:
    rows = []
    for step in range(_MODULE.HORIZON):
        step_rows = []
        for row in range(2):
            token_id = 1000 + step * 2 + row
            raw = _comparison(10 + step * 2 + row)
            processed = {
                **_hash_pair(30 + step * 2 + row),
                "topk_match": True,
            }
            probabilities = _hash_pair(50 + step * 2 + row)
            rng = _hash_pair(70 + step * 2 + row)
            prefix = _hash_pair(90 + step * 2 + row)
            static = {
                "before": {"cache_position": _hash_pair(110 + step * 2 + row)},
                "after": {"cache_position": _hash_pair(130 + step * 2 + row)},
            }
            active = _comparison(150 + step * 2 + row)
            current = _comparison(170 + step * 2 + row)
            cross = _comparison(190 + step * 2 + row)
            future = _digest(210 + step * 2 + row)
            step_rows.append({
                "row": row,
                "accepted_output_offset": (
                    _MODULE.ROW_FIRST_OUTPUT_OFFSETS[row] + step
                ),
                "accepted_token_id": token_id,
                "reference_token_id": token_id,
                "candidate_token_id": token_id,
                "reference_stop": False,
                "candidate_stop": False,
                "accepted_token_match": True,
                "candidate_token_match": True,
                "stop_match": True,
                "rng_match": True,
                "raw_allclose": True,
                "raw_topk_match": True,
                "raw_bitwise_match": True,
                "processed_bitwise_match": True,
                "processed_topk_match": True,
                "probabilities_bitwise_match": True,
                "static_inputs_bitwise_match": True,
                "padded_prefix_bitwise_match": True,
                "active_cache_allclose": True,
                "active_cache_bitwise_match": True,
                "current_cache_written": True,
                "future_cache_unchanged": True,
                "cross_cache_unchanged": True,
                "cross_cache_bitwise_match": True,
                "raw_logits": raw,
                "processed_scores": processed,
                "sampling_probabilities": probabilities,
                "rng_state": rng,
                "prefix": prefix,
                "static_inputs": static,
                "cache": {
                    "active": active,
                    "current": current,
                    "future_sha256": future,
                    "future_expected_sha256": future,
                    "current_nonzero": True,
                    "cross": cross,
                    "cross_expected_sha256": cross["candidate_sha256"],
                },
            })
        rows.append({"step": step, "rows": step_rows})
    exactness = {
        "steps": rows,
        "accepted_tokens_match": True,
        "candidate_tokens_match": True,
        "stop_behavior_match": True,
        "per_step_rng_match": True,
        "final_rng_match": True,
        "raw_logits_allclose": True,
        "raw_logits_topk_match": True,
        "raw_logits_bitwise_match": True,
        "processed_scores_bitwise_match": True,
        "processed_scores_topk_match": True,
        "sampling_probabilities_bitwise_match": True,
        "static_inputs_bitwise_match": True,
        "padded_prefixes_bitwise_match": True,
        "active_cache_allclose": True,
        "active_cache_bitwise_match": True,
        "current_cache_written": True,
        "future_cache_unchanged": True,
        "cross_cache_unchanged": True,
        "cross_cache_bitwise_match": True,
        "source_handoff_match": True,
        "exactness_pass": True,
        "signature": _MODULE.weighted_h8_steps_signature(rows),
    }
    return exactness


def _private_exactness() -> dict:
    starting_states = [
        {
            "row": row,
            "current_output_offset": _MODULE.ROW_FIRST_OUTPUT_OFFSETS[row] - 1,
            "current_token_id": 900 + row,
            "has_time_shift": _hash_pair(300 + row),
            "last_time_shift_value": _hash_pair(310 + row),
            "state_match": True,
        }
        for row in range(2)
    ]
    steps = []
    for step in range(_MODULE.HORIZON):
        rows = []
        for row in range(2):
            token_id = 1000 + step * 2 + row
            rows.append({
                "row": row,
                "accepted_output_offset": (
                    _MODULE.ROW_FIRST_OUTPUT_OFFSETS[row] + step
                ),
                "accepted_token_id": token_id,
                "candidate_token_id": token_id,
                "processed_scores": {
                    **_hash_pair(330 + step * 2 + row),
                    "topk_match": True,
                },
                "sampling_probabilities": _hash_pair(350 + step * 2 + row),
                "rng_state": _hash_pair(370 + step * 2 + row),
                "has_time_shift": _hash_pair(390 + step * 2 + row),
                "last_time_shift_value": _hash_pair(410 + step * 2 + row),
                "feedback_has_time_shift": _hash_pair(430 + step * 2 + row),
                "feedback_last_time_shift_value": _hash_pair(
                    450 + step * 2 + row
                ),
                "processed_bitwise_match": True,
                "processed_topk_match": True,
                "probabilities_bitwise_match": True,
                "candidate_token_match": True,
                "rng_match": True,
                "monotonic_state_match": True,
                "state_feedback_match": True,
            })
        steps.append({"step": step, "rows": rows})
    return {
        "starting_states": starting_states,
        "steps": steps,
        "starting_monotonic_state_match": True,
        "offset41_time_shift_retention_match": True,
        "processed_scores_bitwise_match": True,
        "processed_scores_topk_match": True,
        "sampling_probabilities_bitwise_match": True,
        "candidate_tokens_match": True,
        "per_step_rng_match": True,
        "monotonic_state_match": True,
        "state_feedback_match": True,
        "final_state_matches_exact": True,
        "exactness_pass": True,
        "signature": _MODULE.weighted_h8_steps_signature(
            [*starting_states, *steps]
        ),
    }


def _report(candidate_tps: float = 650.0, control_tps: float = 600.0) -> dict:
    candidate = {
        order: _timing("complete_hybrid_b2_h8", candidate_tps)
        for order in ("0,1", "1,0")
    }
    control = {
        order: _timing("serial_private_b1_graph_h8", control_tps)
        for order in ("0,1", "1,0")
    }
    performance = _MODULE.summarize_weighted_h8_performance(candidate, control)
    return {
        "schema_version": _MODULE.WEIGHTED_H8_SCHEMA_VERSION,
        "gate": _MODULE.WEIGHTED_H8_GATE,
        "claim_scope": "verifier_only_real_prefix_bucket576_phase_a_not_runtime",
        "pass": performance["performance_pass"],
        "exactness_pass": True,
        "resource_ownership_pass": True,
        "result": {
            "scope": "h8_sampled_prefix_scout_not_full_exact_output_or_osu",
            "classification": "bitwise-calculation-exact",
            "raw_logits_bitwise_match": True,
            "active_cache_bitwise_match": True,
            "cross_cache_bitwise_match": True,
            "full_exact_output_claimed": False,
        },
        "source_contract": {
            "job_id": _MODULE.REVIEWED_SOURCE_CAPTURE_JOB_ID,
            "commit": _MODULE.REVIEWED_SOURCE_CAPTURE_COMMIT,
            "contract_sha256": _MODULE.REVIEWED_SOURCE_CONTRACT_SHA256,
            "report_file_sha256": _MODULE.REVIEWED_SOURCE_REPORT_FILE_SHA256,
            "digest_override_allowed": False,
        },
        "workload": {
            "active_prefix_length": _MODULE.ACTIVE_PREFIX_LENGTH,
            "horizon": _MODULE.HORIZON,
            "timing_trials_per_order": _MODULE.TIMING_TRIALS_PER_ORDER,
            "row_seeds": [_MODULE.ROW_SEED, _MODULE.ROW_SEED],
            "row_source_offsets": list(_MODULE.ROW_SOURCE_OFFSETS),
            "row_initial_prefix_lengths": list(
                _MODULE.ROW_INITIAL_PREFIX_LENGTHS
            ),
            "row_initial_cache_positions": list(
                _MODULE.ROW_INITIAL_CACHE_POSITIONS
            ),
            "row_first_output_offsets": list(_MODULE.ROW_FIRST_OUTPUT_OFFSETS),
            "processor_prefix_shape": [2, _MODULE.ACTIVE_PREFIX_LENGTH],
            "neutral_pad_token_id": _MODULE.NEUTRAL_PAD_TOKEN_ID,
            "request_shape": "same_seed_staggered_reachable_production_states",
            "distinct_seed_rejected_reason": (
                "source contract authorizes only seed12345; resetting a future row to "
                "seed23456 would fabricate an unreachable request state"
            ),
            "weighted_ceiling_report_sha256": (
                _MODULE.WEIGHTED_CEILING_REPORT_SHA256
            ),
            "required_weighted_complete_tokens_per_second": (
                _MODULE.REQUIRED_WEIGHTED_COMPLETE_TPS
            ),
        },
        "event_order": list(_MODULE.EVENT_ORDER),
        "resources": {
            "model_graph_count": 2,
            "shared_b2_processor_graph_count": 1,
            "private_b1_processor_graph_count": 2,
            "private_sampling_stop_graph_count": 2,
            "total_graph_count": 7,
            "private_b1_processors": [
                {
                    "row": row,
                    "graph_id": 100 + row,
                    "graph_pool_id": f"pool-{row}",
                    "processor_id": 200 + row,
                    "processor_input_storage_ptr": 300 + row,
                    "output_prefix_storage_ptr": 400 + row,
                    "raw_logits_storage_ptr": 500 + row,
                    "processed_scores_storage_ptr": 600 + row,
                    "monotonic_input_has_time_shift_storage_ptr": 700 + row,
                    "monotonic_input_last_time_shift_storage_ptr": 800 + row,
                    "monotonic_output_has_time_shift_storage_ptr": 900 + row,
                    "monotonic_output_last_time_shift_storage_ptr": 1000 + row,
                }
                for row in range(2)
            ],
            "private_b1_ownership_pass": True,
            "private_b1_shared_bridge_non_alias": True,
            "private_b1_state_feedback_pass": True,
            "private_b1_state_feedback_inside_timed_interval": True,
            "private_b1_state_feedback_policy": (
                "captured_output_copied_to_captured_input_after_every_private_processor_replay"
            ),
        },
        "reset_contract": {
            "in_place_cache_restore": True,
            "exact_generator_state_restore": True,
            "static_pointer_stability": True,
            "restore_verified_after_capture": True,
        },
        "orders": {
            order: {
                "execution_order": [int(value) for value in order.split(",")],
                "exactness": _exactness(),
                "private_control_exactness": _private_exactness(),
                "candidate_timing": candidate[order],
                "control_timing": control[order],
                "same_order_gain_fraction": performance[
                    "same_order_gain_fraction"
                ][order],
            }
            for order in ("0,1", "1,0")
        },
        "performance": performance,
        "setup": {
            "excluded_from_performance": True,
            "excluded_components": [
                "source_reconstruction", "prefill", "graph_warmup", "graph_capture"
            ],
            "total_gate_wall_seconds": 1.0,
            "graph_capture_count": 7,
            "model_graphs": [
                {"warmup_seconds": 0.1, "capture_seconds": 0.1}
                for _ in range(2)
            ],
            "shared_b2_processor_graph": {
                "warmup_seconds": 0.1, "capture_seconds": 0.1
            },
            "private_b1_processor_graphs": [
                {"warmup_seconds": 0.1, "capture_seconds": 0.1}
                for _ in range(2)
            ],
            "private_sampling_graphs": [
                {"warmup_seconds": 0.1, "capture_seconds": 0.1}
                for _ in range(2)
            ],
            "extension_cache_before": {
                "path": "/tmp/extensions",
                "exists": True,
                "entry_count": 1,
                "file_count": 2,
            },
            "extension_cache_after": {
                "path": "/tmp/extensions",
                "exists": True,
                "entry_count": 1,
                "file_count": 2,
            },
            "memory_current_allocated_bytes": 1,
            "memory_current_reserved_bytes": 2,
            "memory_peak_allocated_bytes": 3,
            "memory_peak_reserved_bytes": 4,
        },
        "decision": {
            "phase_a_only": True,
            "phase_b_authorized": False,
            "weighted_bucket_sweep_authorized": performance["performance_pass"],
            "scheduler_or_runtime_wiring_authorized": False,
            "server_authorized": False,
        },
        "runtime_metadata": {
            "device": "cuda",
            "precision": "fp32",
            "gpu_name": "NVIDIA GeForce RTX 2080 Ti",
            "gpu_capability": [7, 5],
            "model_max_target_positions": 2560,
            "torch_version": "test",
            "cuda_version": "test",
            "git_commit": "a" * 40,
            "slurm_job_id": "12345",
            "h8_executed": True,
            "scheduler_or_runtime_wiring_authorized": False,
        },
    }


def _replace_timings(
    report: dict,
    *,
    candidate_tps: dict[str, float],
    control_tps: dict[str, float],
) -> None:
    candidates = {
        order: _timing("complete_hybrid_b2_h8", candidate_tps[order])
        for order in ("0,1", "1,0")
    }
    controls = {
        order: _timing("serial_private_b1_graph_h8", control_tps[order])
        for order in ("0,1", "1,0")
    }
    performance = _MODULE.summarize_weighted_h8_performance(candidates, controls)
    for order in ("0,1", "1,0"):
        report["orders"][order]["candidate_timing"] = candidates[order]
        report["orders"][order]["control_timing"] = controls[order]
        report["orders"][order]["same_order_gain_fraction"] = performance[
            "same_order_gain_fraction"
        ][order]
    report["performance"] = performance
    report["pass"] = performance["performance_pass"]
    report["decision"]["weighted_bucket_sweep_authorized"] = report["pass"]


def test_weighted_h8_report_passes_strict_contract() -> None:
    report = _report()
    _MODULE.validate_weighted_h8_report(report)
    assert report["pass"] is True
    assert report["performance"]["worst_candidate_wall_tokens_per_second"] == pytest.approx(
        650.0
    )


def test_weighted_h8_rejects_below_absolute_or_relative_bar() -> None:
    absolute = _report(candidate_tps=620.0, control_tps=580.0)
    assert absolute["pass"] is False
    assert absolute["performance"]["absolute_weighted_bar_pass"] is False

    relative = _report(candidate_tps=650.0, control_tps=630.0)
    assert relative["pass"] is False
    assert relative["performance"]["five_percent_same_order_bar_pass"] is False


def test_weighted_h8_rejects_exact_threshold_equalities() -> None:
    absolute = _report()
    _replace_timings(
        absolute,
        candidate_tps={
            "0,1": _MODULE.REQUIRED_WEIGHTED_COMPLETE_TPS,
            "1,0": _MODULE.REQUIRED_WEIGHTED_COMPLETE_TPS,
        },
        control_tps={"0,1": 500.0, "1,0": 500.0},
    )
    _MODULE.validate_weighted_h8_report(absolute)
    assert absolute["performance"]["absolute_weighted_bar_pass"] is False

    relative = _report()
    _replace_timings(
        relative,
        candidate_tps={"0,1": 630.0, "1,0": 630.0},
        control_tps={"0,1": 600.0, "1,0": 600.0},
    )
    _MODULE.validate_weighted_h8_report(relative)
    assert relative["performance"]["five_percent_same_order_bar_pass"] is False


def test_weighted_h8_uses_worst_reciprocal_order() -> None:
    report = _report()
    _replace_timings(
        report,
        candidate_tps={"0,1": 650.0, "1,0": 620.0},
        control_tps={"0,1": 600.0, "1,0": 580.0},
    )
    _MODULE.validate_weighted_h8_report(report)
    assert report["pass"] is False
    assert report["performance"]["worst_candidate_wall_tokens_per_second"] == (
        pytest.approx(620.0)
    )


def test_weighted_h8_rejects_state_or_timing_mutation() -> None:
    report = _report()
    bad_state = copy.deepcopy(report)
    bad_state["orders"]["0,1"]["exactness"]["steps"][3]["rows"][1][
        "rng_match"
    ] = False
    with pytest.raises(ValueError, match="step 3 row 1 failed"):
        _MODULE.validate_weighted_h8_report(bad_state)

    bad_timing = copy.deepcopy(report)
    bad_timing["orders"]["1,0"]["candidate_timing"]["wall_seconds"] *= 2
    with pytest.raises(ValueError, match="throughput is inconsistent"):
        _MODULE.validate_weighted_h8_report(bad_timing)

    bad_scope = copy.deepcopy(report)
    bad_scope["decision"]["scheduler_or_runtime_wiring_authorized"] = True
    with pytest.raises(ValueError, match="graduation boundary changed"):
        _MODULE.validate_weighted_h8_report(bad_scope)


def test_weighted_h8_rejects_evidence_or_reciprocal_signature_mutation() -> None:
    bad_hash = _report()
    bad_hash["orders"]["0,1"]["exactness"]["steps"][0]["rows"][0][
        "rng_state"
    ]["candidate_sha256"] = _digest(999)
    with pytest.raises(ValueError, match="bitwise result is inconsistent"):
        _MODULE.validate_weighted_h8_report(bad_hash)

    bad_signature = _report()
    bad_signature["orders"]["0,1"]["exactness"]["signature"] = _digest(999)
    with pytest.raises(ValueError, match="exact signature is inconsistent"):
        _MODULE.validate_weighted_h8_report(bad_signature)

    reciprocal = _report()
    exactness = reciprocal["orders"]["1,0"]["exactness"]
    row = exactness["steps"][0]["rows"][0]
    row["accepted_token_id"] += 1
    row["reference_token_id"] += 1
    row["candidate_token_id"] += 1
    exactness["signature"] = _MODULE.weighted_h8_steps_signature(exactness["steps"])
    with pytest.raises(ValueError, match="reciprocal-order dependent"):
        _MODULE.validate_weighted_h8_report(reciprocal)

    private = _report()
    private_row = private["orders"]["0,1"]["private_control_exactness"][
        "steps"
    ][0]["rows"][0]
    private_row["sampling_probabilities"]["candidate_sha256"] = _digest(999)
    with pytest.raises(ValueError, match="bitwise result is inconsistent"):
        _MODULE.validate_weighted_h8_report(private)


def test_weighted_h8_scopes_nonbitwise_math_to_sampled_prefix() -> None:
    report = _report()
    for order in ("0,1", "1,0"):
        report["orders"][order]["exactness"]["raw_logits_bitwise_match"] = False
        exactness = report["orders"][order]["exactness"]
        exactness["steps"][0]["rows"][0]["raw_bitwise_match"] = False
        exactness["steps"][0]["rows"][0]["raw_logits"][
            "candidate_sha256"
        ] = _digest(999)
        exactness["signature"] = _MODULE.weighted_h8_steps_signature(
            exactness["steps"]
        )
    report["result"] = {
        "scope": "h8_sampled_prefix_scout_not_full_exact_output_or_osu",
        "classification": "exact-sampled-prefix",
        "raw_logits_bitwise_match": False,
        "active_cache_bitwise_match": True,
        "cross_cache_bitwise_match": True,
        "full_exact_output_claimed": False,
    }
    _MODULE.validate_weighted_h8_report(report)
    assert report["pass"] is True


def test_weighted_h8_rejects_private_processor_alias() -> None:
    report = _report()
    report["resources"]["private_b1_processors"][1][
        "processor_input_storage_ptr"
    ] = report["resources"]["private_b1_processors"][0][
        "processor_input_storage_ptr"
    ]
    with pytest.raises(ValueError, match="aliases rows"):
        _MODULE.validate_weighted_h8_report(report)


def test_weighted_h8_rejects_source_phase_b_setup_memory_or_provenance_drift() -> None:
    source = _report()
    source["source_contract"]["contract_sha256"] = _digest(999)
    with pytest.raises(ValueError, match="reviewed source changed"):
        _MODULE.validate_weighted_h8_report(source)

    phase_b = _report()
    phase_b["decision"]["phase_b_authorized"] = True
    with pytest.raises(ValueError, match="graduation boundary changed"):
        _MODULE.validate_weighted_h8_report(phase_b)

    setup = _report()
    setup["setup"]["total_gate_wall_seconds"] = float("nan")
    with pytest.raises(ValueError, match="finite and positive"):
        _MODULE.validate_weighted_h8_report(setup)

    memory = _report()
    memory["setup"]["memory_peak_allocated_bytes"] = 0
    with pytest.raises(ValueError, match="peak memory is below current"):
        _MODULE.validate_weighted_h8_report(memory)

    extension = _report()
    del extension["setup"]["extension_cache_before"]["file_count"]
    with pytest.raises(ValueError, match="extension-cache evidence changed"):
        _MODULE.validate_weighted_h8_report(extension)

    provenance = _report()
    provenance["runtime_metadata"]["git_commit"] = "not-a-commit"
    with pytest.raises(ValueError, match="hardware/runtime contract changed"):
        _MODULE.validate_weighted_h8_report(provenance)


def test_weighted_h8_wrapper_hashes_rejections_before_failing() -> None:
    script = (
        REPO_ROOT / "scripts" / "dcc" / "verify_weighted_h8.sbatch"
    ).read_text(encoding="utf-8")
    validation = script.index("weighted_h8_report_validation=PASS")
    performance_status = script.index("performance_gate_status=")
    final_hash = script.rindex('sha256sum "$REPORT"')
    keep_bar_failure = script.index(
        "weighted H8 exact report failed the absolute or 5% keep bar"
    )
    assert validation < performance_status < final_hash < keep_bar_failure

    failure_validation = script.index("weighted_h8_failure_report_validation=PASS")
    first_hash = script.index('sha256sum "$REPORT"')
    failure_exit = script.index('exit "$VERIFY_STATUS"')
    assert failure_validation < first_hash < failure_exit


def test_weighted_h8_capture_keeps_full_inputs_separate_from_mutable_static(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_inputs = {
        "encoder_outputs": object(),
        "past_key_values": object(),
        "decoder_input_ids": torch.ones((1, 1), dtype=torch.long),
        "cache_position": torch.ones((1,), dtype=torch.long),
        "decoder_position_ids": torch.ones((1, 1), dtype=torch.long),
        "decoder_attention_mask": torch.ones((1, 1, 1, 1)),
    }
    static = _VERIFIER._static_inputs(full_inputs)
    assert set(static) == set(_VERIFIER.STATIC_KEYS)
    assert "encoder_outputs" not in static
    assert "past_key_values" not in static

    class CaptureReached(RuntimeError):
        pass

    def _capture(model, prepared, **kwargs):
        assert prepared is full_inputs
        assert prepared["encoder_outputs"] is full_inputs["encoder_outputs"]
        assert prepared["past_key_values"] is full_inputs["past_key_values"]
        raise CaptureReached

    monkeypatch.setattr(_VERIFIER, "capture_prepared_b1_lane", _capture)
    state = SimpleNamespace(capture_inputs=full_inputs)
    with pytest.raises(CaptureReached):
        _VERIFIER._capture_candidate(None, state, 1)


def test_weighted_h8_capture_fails_loudly_on_reduced_inputs() -> None:
    state = SimpleNamespace(
        capture_inputs={"decoder_input_ids": torch.ones((1, 1), dtype=torch.long)}
    )
    with pytest.raises(RuntimeError, match="complete prepared decoder inputs"):
        _VERIFIER._capture_candidate(None, state, 1)
