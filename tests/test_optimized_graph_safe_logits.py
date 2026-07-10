from __future__ import annotations

import inspect

import pytest
import torch

from osuT5.osuT5.event import EventType
from osuT5.osuT5.inference.logit_processors import (
    MonotonicTimeShiftLogitsProcessor as V32MonotonicTimeShiftLogitsProcessor,
)
from osuT5.osuT5.inference.optimized.graph_safe_logits import (
    MonotonicTimeShiftLogitsProcessor,
    make_graph_safe_logits_processor_list,
)
from osuT5.osuT5.inference.optimized.single.logits import (
    MonotonicTimeShiftLogitsProcessor as StatefulMonotonicTimeShiftLogitsProcessor,
)


class _FakeTokenizer:
    sos_id = 1
    context_sos = {}
    event_start = {EventType.TIME_SHIFT: 10}
    event_end = {EventType.TIME_SHIFT: 20}


def _random_input_ids(
    generator: torch.Generator,
    *,
    batch_size: int,
    seq_len: int,
    scenario: str,
) -> torch.LongTensor:
    input_ids = torch.randint(
        0,
        30,
        (batch_size, seq_len),
        generator=generator,
        dtype=torch.long,
    )
    is_time_shift = (input_ids >= 10) & (input_ids < 20)
    if scenario == "no_timeshift":
        input_ids[is_time_shift] = 0
        input_ids[:, 0] = 1
    elif scenario == "no_sos":
        input_ids[input_ids == 1] = 0
        input_ids[:, 0] = torch.randint(
            10,
            20,
            (batch_size,),
            generator=generator,
        )
    elif scenario == "reset":
        input_ids[is_time_shift] = 0
        input_ids[:, 0] = 1
        input_ids[:, 1] = torch.randint(
            10,
            20,
            (batch_size,),
            generator=generator,
        )
        input_ids[:, -1] = 1
    elif scenario == "mixed":
        input_ids[:, 0] = 1
        input_ids[:, seq_len // 2] = torch.randint(
            10,
            20,
            (batch_size,),
            generator=generator,
        )
        if batch_size > 1:
            input_ids[1::2, -2] = 1
    else:
        raise ValueError(f"unknown scenario {scenario}")
    return input_ids


def test_graph_safe_processor_is_bitwise_exact_for_random_b1_b2_b5_inputs():
    tokenizer = _FakeTokenizer()
    generator = torch.Generator().manual_seed(34567)
    for batch_size in (1, 2, 5):
        for scenario in ("no_timeshift", "no_sos", "reset", "mixed"):
            for _ in range(12):
                seq_len = int(torch.randint(5, 32, (), generator=generator).item())
                input_ids = _random_input_ids(
                    generator,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    scenario=scenario,
                )
                scores = torch.randn((batch_size, 64), generator=generator)
                reference = V32MonotonicTimeShiftLogitsProcessor(tokenizer)
                candidate = MonotonicTimeShiftLogitsProcessor(tokenizer)
                assert torch.equal(
                    reference(input_ids.clone(), scores.clone()),
                    candidate(input_ids.clone(), scores.clone()),
                )


def test_graph_safe_b2_matches_two_private_stateful_b1_processors_bitwise():
    tokenizer = _FakeTokenizer()
    generator = torch.Generator().manual_seed(23456)
    for scenario in ("no_timeshift", "no_sos", "reset", "mixed"):
        for _ in range(10):
            seq_len = int(torch.randint(5, 24, (), generator=generator).item())
            input_ids = _random_input_ids(
                generator,
                batch_size=2,
                seq_len=seq_len,
                scenario=scenario,
            )
            scores = torch.randn((2, 64), generator=generator)
            shared = MonotonicTimeShiftLogitsProcessor(
                tokenizer,
                stateful_batch1=True,
            )
            shared_scores = shared(input_ids.clone(), scores.clone())
            for row in range(2):
                private = StatefulMonotonicTimeShiftLogitsProcessor(
                    tokenizer,
                    stateful_batch1=True,
                )
                private_scores = private(
                    input_ids[row:row + 1].clone(),
                    scores[row:row + 1].clone(),
                )
                assert torch.equal(shared_scores[row:row + 1], private_scores)


def test_graph_safe_source_avoids_dynamic_boolean_index_writes():
    source = inspect.getsource(MonotonicTimeShiftLogitsProcessor._full_scan_call)
    arange_lines = [line for line in source.splitlines() if "torch.arange(" in line]
    assert len(arange_lines) == 3
    assert all("device=device" in line for line in arange_lines)
    assert "scores[apply_mask]" not in source
    assert "batch_mask[apply_mask]" not in source
    assert "scores.masked_fill_(batch_mask & apply_mask.unsqueeze(1), -torch.inf)" in source
    assert ".cpu(" not in source
    assert ".item(" not in source
    assert "torch.tensor(" not in source


def test_replacement_is_optimized_only_and_preserves_processor_contract():
    tokenizer = _FakeTokenizer()
    legacy = StatefulMonotonicTimeShiftLogitsProcessor(
        tokenizer,
        stateful_batch1=True,
    )
    sentinel = object()
    processors = [legacy, sentinel]

    assert make_graph_safe_logits_processor_list(processors) is processors
    assert isinstance(processors[0], MonotonicTimeShiftLogitsProcessor)
    assert type(processors[0]).__name__ == type(legacy).__name__
    assert type(processors[0]).__module__.endswith("optimized.graph_safe_logits")
    assert processors[0].stateful_batch1 is True
    assert processors[1] is sentinel


def test_replacement_fails_loudly_after_processor_state_is_initialized():
    tokenizer = _FakeTokenizer()
    legacy = StatefulMonotonicTimeShiftLogitsProcessor(
        tokenizer,
        stateful_batch1=True,
    )
    input_ids = torch.tensor([[1, 12, 5]], dtype=torch.long)
    legacy(input_ids, torch.zeros((1, 64), dtype=torch.float32))

    with pytest.raises(ValueError, match="pristine"):
        make_graph_safe_logits_processor_list([legacy])


def test_replacement_rejects_missing_duplicate_and_unknown_subclasses():
    tokenizer = _FakeTokenizer()

    with pytest.raises(ValueError, match="exactly one"):
        make_graph_safe_logits_processor_list([])
    with pytest.raises(ValueError, match="exactly one"):
        make_graph_safe_logits_processor_list([
            V32MonotonicTimeShiftLogitsProcessor(tokenizer),
            V32MonotonicTimeShiftLogitsProcessor(tokenizer),
        ])

    class _UnknownMonotonic(V32MonotonicTimeShiftLogitsProcessor):
        pass

    with pytest.raises(TypeError, match="unknown"):
        make_graph_safe_logits_processor_list([_UnknownMonotonic(tokenizer)])
