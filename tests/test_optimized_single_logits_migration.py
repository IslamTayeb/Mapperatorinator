from __future__ import annotations

import pytest
import torch

from osuT5.osuT5.event import EventType
from osuT5.osuT5.inference.logit_processors import (
    ConditionalTemperatureLogitsWarper as V32ConditionalTemperatureLogitsWarper,
    MonotonicTimeShiftLogitsProcessor as V32MonotonicTimeShiftLogitsProcessor,
)
from osuT5.osuT5.inference.optimized.single.logits import (
    ConditionalTemperatureLogitsWarper as OptimizedConditionalTemperatureLogitsWarper,
    MonotonicTimeShiftLogitsProcessor as OptimizedMonotonicTimeShiftLogitsProcessor,
)


class _FakeTokenizer:
    sos_id = 1
    context_sos = {}
    event_start = {EventType.TIME_SHIFT: 10}
    event_end = {EventType.TIME_SHIFT: 20}


def _apply(processor, input_ids, scores):
    return processor(input_ids.clone(), scores.clone())


def test_stateful_monotonic_matches_v32_full_scan_for_growing_batch1_prefixes():
    tokenizer = _FakeTokenizer()
    full_scan = V32MonotonicTimeShiftLogitsProcessor(tokenizer)
    stateful = OptimizedMonotonicTimeShiftLogitsProcessor(tokenizer)
    tokens = torch.tensor(
        [[1, 12, 5, 11, 7, 15, 3, 1, 14, 9]],
        dtype=torch.long,
    )

    for length in range(1, tokens.shape[1] + 1):
        input_ids = tokens[:, :length]
        scores = torch.arange(64, dtype=torch.float32).reshape(1, 64) / 10
        assert torch.equal(
            _apply(full_scan, input_ids, scores),
            _apply(stateful, input_ids, scores),
        )


def test_stateful_monotonic_reinitializes_after_sequence_jump():
    tokenizer = _FakeTokenizer()
    full_scan = V32MonotonicTimeShiftLogitsProcessor(tokenizer)
    stateful = OptimizedMonotonicTimeShiftLogitsProcessor(tokenizer)
    scores = torch.arange(64, dtype=torch.float32).reshape(1, 64) / 10

    _apply(stateful, torch.tensor([[1, 12, 5, 13]], dtype=torch.long), scores)
    second = torch.tensor([[1, 18, 4]], dtype=torch.long)

    assert torch.equal(
        _apply(full_scan, second, scores),
        _apply(stateful, second, scores),
    )


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_optimized_monotonic_matches_v32_full_scan_for_random_inputs(batch_size):
    tokenizer = _FakeTokenizer()
    generator = torch.Generator().manual_seed(34567 + batch_size)
    full_scan = V32MonotonicTimeShiftLogitsProcessor(tokenizer)

    for _ in range(32):
        candidate = OptimizedMonotonicTimeShiftLogitsProcessor(tokenizer)
        seq_len = int(torch.randint(1, 48, (), generator=generator).item())
        input_ids = torch.randint(
            0,
            30,
            (batch_size, seq_len),
            generator=generator,
        )
        scores = torch.randn((batch_size, 64), generator=generator)
        assert torch.equal(
            _apply(full_scan, input_ids, scores),
            _apply(candidate, input_ids, scores),
        )


def _conditional_temperature(cls):
    return cls(
        temperature=0.9,
        timing_temperature=0.1,
        mania_column_temperature=0.8,
        taiko_hit_temperature=0.7,
        types_first=True,
        beat_type_tokens=(10, 11),
        mania_type_tokens=(20, 21),
        scroll_speed_tokens=(30, 31),
    )


@pytest.mark.parametrize(
    "tokens",
    [
        [0],
        [0, 10],
        [20, 0, 0],
        [20, 0, 10],
        [30, 0, 11],
        [0, 0, 0, 0],
    ],
)
def test_device_conditional_temperature_matches_v32_batch1_exactly(tokens):
    baseline = _conditional_temperature(V32ConditionalTemperatureLogitsWarper)
    candidate = _conditional_temperature(
        OptimizedConditionalTemperatureLogitsWarper
    )
    input_ids = torch.tensor([tokens], dtype=torch.long)
    scores = torch.tensor(
        [[-3.25, -0.0, 0.1, 1.0, 19.75]],
        dtype=torch.float32,
    )

    assert torch.equal(
        _apply(baseline, input_ids, scores),
        _apply(candidate, input_ids, scores),
    )


def test_device_conditional_temperature_matches_v32_batched_fallback():
    baseline = _conditional_temperature(V32ConditionalTemperatureLogitsWarper)
    candidate = _conditional_temperature(
        OptimizedConditionalTemperatureLogitsWarper
    )
    input_ids = torch.tensor(
        [[20, 0, 10], [20, 0, 0], [0, 0, 30]],
        dtype=torch.long,
    )
    scores = torch.arange(15, dtype=torch.float32).reshape(3, 5) - 7

    assert torch.equal(
        _apply(baseline, input_ids, scores),
        _apply(candidate, input_ids, scores),
    )


def test_device_conditional_temperature_rejects_mismatched_batch1_devices():
    candidate = _conditional_temperature(
        OptimizedConditionalTemperatureLogitsWarper
    )
    with pytest.raises(ValueError, match="matching batch-1 scores"):
        candidate(torch.tensor([[10]]), torch.zeros((2, 4)))
