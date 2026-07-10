from osuT5.osuT5.inference.optimized import (
    ExactnessEvidence,
    ExactnessResultClass,
    OptimizedRequest,
    OptimizedResult,
    RequestState,
    RequestTiming,
    ThroughputMeasurement,
    active_batch_size_histogram,
)


def _request() -> OptimizedRequest:
    return OptimizedRequest(
        request_id="request-1",
        song_id="song-1",
        window_id="window-0",
        context_id="main_generation",
        model_inputs={"input_ids": object()},
        generation_settings={"top_p": 0.95},
        seed=12345,
    )


def test_contracts_keep_request_state_and_exactness_evidence_explicit():
    request = _request()
    state = RequestState(
        request=request,
        generator=object(),
        logits_processors=(),
        stopping_state={},
        encoder_output=object(),
        self_attention_cache=object(),
        cross_attention_cache=object(),
        slot_id=2,
        slot_generation=4,
    )
    exactness = ExactnessEvidence(
        result_class=ExactnessResultClass.EXACT_OUTPUT,
        generated_token_ids=[10, 11, 12],
        stop_reason="eos",
        final_rng_state_hash="rng-sha256",
        result_file_sha256="osu-sha256",
        result_file_size_bytes=42,
    )
    throughput = ThroughputMeasurement(
        generated_tokens=3,
        started_at_seconds=10.0,
        finished_at_seconds=10.5,
    )
    result = OptimizedResult(
        request=request,
        generated_output="osu text",
        exactness=exactness,
        throughput=throughput,
        timing_metadata={"queue_wait_seconds": 0.1},
        cache_metadata={"slot_generation": state.slot_generation},
        graph_metadata={"replays": 1},
    )

    assert exactness.generated_token_ids == (10, 11, 12)
    assert result.throughput.scheduler_wall_tokens_per_second == 6.0
    assert result.cache_metadata["slot_generation"] == 4


def test_metrics_keep_scheduler_wall_latency_and_batch_shape_separate():
    timing = RequestTiming(queue_wait_seconds=0.25, latency_seconds=1.5)

    assert timing.queue_wait_seconds == 0.25
    assert active_batch_size_histogram([1, 2, 2, 5]) == {1: 1, 2: 2, 5: 1}


def test_bitwise_result_class_requires_intermediate_and_cache_evidence():
    try:
        ExactnessEvidence(
            result_class=ExactnessResultClass.BITWISE_CALCULATION_EXACT,
            generated_token_ids=(10,),
            stop_reason="one_step",
            final_rng_state_hash="rng-sha256",
        )
    except ValueError as exc:
        assert "requires explicit intermediate_state_hashes and cache_state_hashes" in str(exc)
    else:
        raise AssertionError("Expected bitwise exactness without calculation evidence to fail")

    evidence = ExactnessEvidence(
        result_class=ExactnessResultClass.BITWISE_CALCULATION_EXACT,
        generated_token_ids=(10,),
        stop_reason="one_step",
        final_rng_state_hash="rng-sha256",
        intermediate_state_hashes={"decoder_logits": "logits-sha256"},
        cache_state_hashes={"self_kv": "cache-sha256"},
    )

    assert evidence.result_file_sha256 is None
