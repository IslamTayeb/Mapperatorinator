from __future__ import annotations

from osuT5.osuT5.inference.optimized.batch import BatchPhysicsRequest, BatchPhysicsScheduler
from osuT5.osuT5.inference.optimized.benchmark import (
    BatchPhysicsExecutionFamily,
    BatchPhysicsObservation,
    BatchPhysicsPlan,
    LANE_STATE_OWNERSHIP_CONTRACT,
    MERGED_STATE_OWNERSHIP_CONTRACT,
    SamplingComponentMeasurement,
    compare_batch_physics_observations,
    summarize_sampling_gap_target,
)
from osuT5.osuT5.inference.optimized.exactness import ExactnessResultClass


def _scores(*favored_tokens: int, vocabulary_size: int = 6) -> tuple[tuple[float, ...], ...]:
    rows = []
    for token_id in favored_tokens:
        row = [0.0] * vocabulary_size
        row[token_id] = 1.0
        rows.append(tuple(row))
    return tuple(rows)


def _request(
        request_id: str,
        tokens: tuple[int, ...],
        *,
        seed: int,
        max_new_tokens: int | None = None,
        eos_token_ids: tuple[int, ...] = (),
        arrival_step: int = 0,
) -> BatchPhysicsRequest:
    return BatchPhysicsRequest(
        request_id=request_id,
        seed=seed,
        score_steps=_scores(*tokens),
        max_new_tokens=max_new_tokens or len(tokens),
        eos_token_ids=eos_token_ids,
        arrival_step=arrival_step,
    )


def _observable(results):
    return {
        result.request_id: (
            result.generated_token_ids,
            result.stop_reason,
            result.final_rng_state_hash,
            result.sample_calls,
        )
        for result in results
    }


def _assert_raises(exc_type, fn, message_fragment):
    try:
        fn()
    except exc_type as exc:
        assert message_fragment in str(exc)
    else:
        raise AssertionError(f"Expected {exc_type.__name__}")


def test_request_order_does_not_change_tokens_stops_or_per_request_rng():
    requests = [
        _request("song-a", (1, 2, 3), seed=12345),
        _request("song-b", (4, 1), seed=23456),
        _request("song-c", (2, 5, 0, 1), seed=34567),
    ]
    forward = BatchPhysicsScheduler(requests, max_active_sequences=2).run_until_idle()
    reverse = BatchPhysicsScheduler(reversed(requests), max_active_sequences=2).run_until_idle()

    assert _observable(forward) == _observable(reverse)


def test_finished_and_empty_rows_never_advance_rng():
    scheduler = BatchPhysicsScheduler(
        [
            _request("early-eos", (5, 1, 2), seed=12345, eos_token_ids=(5,)),
            _request("long", (1, 2, 3), seed=23456),
        ],
        max_active_sequences=5,
    )
    scheduler.step()
    finished_hash = scheduler.rng_state_hash("early-eos")
    assert scheduler.sample_calls("early-eos") == 1

    scheduler.step()
    scheduler.step()
    assert scheduler.rng_state_hash("early-eos") == finished_hash
    assert scheduler.sample_calls("early-eos") == 1
    results = {result.request_id: result for result in scheduler.results()}
    assert results["early-eos"].generated_token_ids == (5,)
    assert results["early-eos"].stop_reason == "eos"
    assert results["early-eos"].exactness_evidence().generated_token_ids == (5,)
    assert results["long"].sample_calls == 3


def test_eos_max_token_staggered_arrival_and_slot_generation_reuse():
    scheduler = BatchPhysicsScheduler(
        [
            _request("eos", (5, 1), seed=1, eos_token_ids=(5,)),
            _request("max", (2, 3, 4), seed=2, max_new_tokens=2),
            _request("late", (4,), seed=3, arrival_step=3),
        ],
        max_active_sequences=1,
    )
    results = {result.request_id: result for result in scheduler.run_until_idle()}

    assert results["eos"].stop_reason == "eos"
    assert results["max"].generated_token_ids == (2, 3)
    assert results["max"].stop_reason == "max_new_tokens"
    assert results["late"].activation_step == 3
    assert results["eos"].slot_id == results["max"].slot_id == results["late"].slot_id == 0
    assert [results[name].slot_generation for name in ("eos", "max", "late")] == [1, 2, 3]
    assert scheduler.active_batch_size_histogram == {1: 4}


def test_independent_generators_isolate_a_request_from_unrelated_work():
    target = BatchPhysicsRequest(
        request_id="target",
        seed=12345,
        score_steps=((0.1, 0.2, 0.3, 0.4),) * 4,
        max_new_tokens=4,
    )
    alone = BatchPhysicsScheduler([target], max_active_sequences=1).run_until_idle()
    crowded = BatchPhysicsScheduler(
        [
            _request("noise-a", (1, 1, 1), seed=9),
            target,
            _request("noise-b", (2, 2), seed=10),
        ],
        max_active_sequences=2,
    ).run_until_idle()

    assert _observable(alone)["target"] == _observable(crowded)["target"]


def test_measurement_schema_records_required_gpu_and_exactness_fields():
    observation = BatchPhysicsObservation(
        execution_family=BatchPhysicsExecutionFamily.MERGED_BATCH,
        parallelism=5,
        state_ownership_contract=MERGED_STATE_OWNERSHIP_CONTRACT,
        workload_contract_hash="five-song-three-seed-contract",
        result_class=ExactnessResultClass.EXACT_OUTPUT,
        seeds={"a": 12345, "b": 23456},
        generated_tokens=500,
        scheduler_wall_seconds=1.0,
        model_seconds=0.8,
        cuda_seconds=0.7,
        peak_memory_bytes=1024,
        graph_capture_count=1,
        graph_replay_count=100,
        active_batch_size_histogram={5: 100},
        token_hashes={"a": "token-a", "b": "token-b"},
        final_rng_state_hashes={"a": "rng-a", "b": "rng-b"},
        stop_reasons={"a": "max_new_tokens", "b": "eos"},
    )

    assert observation.scheduler_wall_tokens_per_second == 500.0
    assert observation.as_dict()["active_batch_size_histogram"] == {"5": 100}
    assert observation.as_dict()["graph_capture_count"] == 1
    assert observation.as_dict()["graph_replay_count"] == 100
    assert BatchPhysicsPlan().as_dict() == {
        "merged_batch_sizes": [1, 2, 5, 8],
        "b1_lane_counts": [1, 2, 3, 4],
        "result_class": "exact-output",
        "status": "merged_one_token_verifier_only",
        "implemented_merged_one_token_batch_sizes": [1, 2, 5, 8],
        "planned_merged_batch_sizes": [],
        "lane_pool_status": "execution_not_implemented",
        "required_observation_fields": [
            "execution_family",
            "parallelism",
            "state_ownership_contract",
            "workload_contract_hash",
            "result_class",
            "seeds",
            "generated_tokens",
            "scheduler_wall_seconds",
            "model_seconds",
            "cuda_seconds",
            "peak_memory_bytes",
            "graph_capture_count",
            "graph_replay_count",
            "active_batch_size_histogram",
            "token_hashes",
            "final_rng_state_hashes",
            "stop_reasons",
        ],
        "bitwise_required_observation_fields": [
            "intermediate_state_hashes",
            "cache_state_hashes",
        ],
    }

    candidate_payload = observation.as_dict()
    candidate_payload["execution_family"] = "b1_lane_pool"
    candidate_payload["parallelism"] = 2
    candidate_payload["state_ownership_contract"] = LANE_STATE_OWNERSHIP_CONTRACT
    candidate_payload["active_batch_size_histogram"] = {"2": 100}
    candidate_payload["scheduler_wall_seconds"] = 0.9
    candidate = BatchPhysicsObservation.from_dict(candidate_payload)
    comparison = compare_batch_physics_observations(observation, candidate)
    assert comparison["clears_five_percent_scout_bar"] is True
    assert comparison["decision"] == "continue_incremental_gates"

    below_bar_payload = candidate.as_dict()
    below_bar_payload["scheduler_wall_seconds"] = 0.97
    below_bar = BatchPhysicsObservation.from_dict(below_bar_payload)
    _assert_raises(
        ValueError,
        lambda: compare_batch_physics_observations(observation, below_bar),
        "below 5%",
    )

    wrong_class_payload = candidate.as_dict()
    wrong_class_payload["result_class"] = "bitwise-calculation-exact"
    _assert_raises(
        ValueError,
        lambda: BatchPhysicsObservation.from_dict(wrong_class_payload),
        "requires per-request intermediate_state_hashes",
    )
    wrong_class_payload["intermediate_state_hashes"] = {
        "a": {"decoder_logits": "logits-a"},
        "b": {"decoder_logits": "logits-b"},
    }
    wrong_class_payload["cache_state_hashes"] = {
        "a": {"self_kv": "cache-a"},
        "b": {"self_kv": "cache-b"},
    }
    wrong_class = BatchPhysicsObservation.from_dict(wrong_class_payload)
    _assert_raises(
        ValueError,
        lambda: compare_batch_physics_observations(observation, wrong_class),
        "result classes differ",
    )

    wrong_contract_payload = candidate.as_dict()
    wrong_contract_payload["workload_contract_hash"] = "different-workload"
    wrong_contract = BatchPhysicsObservation.from_dict(wrong_contract_payload)
    _assert_raises(
        ValueError,
        lambda: compare_batch_physics_observations(observation, wrong_contract),
        "workload contracts differ",
    )

    bitwise_baseline_payload = observation.as_dict()
    bitwise_baseline_payload["result_class"] = "bitwise-calculation-exact"
    bitwise_baseline_payload["intermediate_state_hashes"] = {
        "a": {"decoder_logits": "logits-a"},
        "b": {"decoder_logits": "logits-b"},
    }
    bitwise_baseline_payload["cache_state_hashes"] = {
        "a": {"self_kv": "cache-a"},
        "b": {"self_kv": "cache-b"},
    }
    bitwise_candidate_payload = candidate.as_dict()
    bitwise_candidate_payload["result_class"] = "bitwise-calculation-exact"
    bitwise_candidate_payload["intermediate_state_hashes"] = {
        "a": {"decoder_logits": "changed-logits-a"},
        "b": {"decoder_logits": "logits-b"},
    }
    bitwise_candidate_payload["cache_state_hashes"] = {
        "a": {"self_kv": "cache-a"},
        "b": {"self_kv": "cache-b"},
    }
    bitwise_baseline = BatchPhysicsObservation.from_dict(bitwise_baseline_payload)
    bitwise_candidate = BatchPhysicsObservation.from_dict(bitwise_candidate_payload)
    _assert_raises(
        ValueError,
        lambda: compare_batch_physics_observations(bitwise_baseline, bitwise_candidate),
        "intermediate_state_hashes",
    )

    bitwise_candidate_payload["intermediate_state_hashes"] = (
        bitwise_baseline_payload["intermediate_state_hashes"]
    )
    bitwise_candidate_payload["cache_state_hashes"] = {
        "a": {"self_kv": "changed-cache-a"},
        "b": {"self_kv": "cache-b"},
    }
    bitwise_candidate = BatchPhysicsObservation.from_dict(bitwise_candidate_payload)
    _assert_raises(
        ValueError,
        lambda: compare_batch_physics_observations(bitwise_baseline, bitwise_candidate),
        "cache_state_hashes",
    )


def test_sampling_component_schema_and_exact_b8_target_math():
    target = summarize_sampling_gap_target(
        batch_size=8,
        target_tokens_per_second=500.0,
        complete_wall_seconds_per_step=0.018196870647370814,
        model_seconds_per_step=0.011957005615234375,
    )
    measurement = SamplingComponentMeasurement(
        component="private_generator_multinomial",
        repeats=200,
        wall_seconds=0.6,
        cuda_seconds=0.5,
        includes=("eight_private_generators", "eight_multinomial_draws"),
    )
    rendered = measurement.as_dict(
        required_saving_seconds_per_step=target["required_saving_seconds_per_step"]
    )

    assert target["target_step_seconds"] == 0.016
    assert abs(target["required_saving_seconds_per_step"] - 0.002196870647370814) < 1e-15
    assert abs(
        target["baseline_complete_minus_model_gap_seconds_per_step"]
        - 0.006239865032136439
    ) < 1e-15
    assert abs(target["required_saving_fraction_of_measured_gap"] - 0.352070219) < 1e-9
    assert target["model_only_ceiling_clears_target"] is True
    assert rendered["wall_seconds_per_step"] == 0.003
    assert rendered["cuda_seconds_per_step"] == 0.0025
    assert rendered["fantasy_free_wall_clears_required_saving"] is True
    assert rendered["fantasy_free_cuda_clears_required_saving"] is True

    _assert_raises(
        ValueError,
        lambda: SamplingComponentMeasurement(
            component="bad",
            repeats=1,
            wall_seconds=0.1,
            cuda_seconds=0.2,
            includes=("impossible",),
        ),
        "cannot exceed synchronized wall_seconds",
    )
