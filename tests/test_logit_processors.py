import inspect
import torch
import sys
import types
from importlib import util
from pathlib import Path

from osuT5.osuT5.event import EventType


_TRANSFORMERS_STUB = types.ModuleType("transformers")
_TRANSFORMERS_STUB.LogitsProcessor = object
sys.modules.setdefault("transformers", _TRANSFORMERS_STUB)
_DATA_UTILS_STUB = types.ModuleType("osuT5.osuT5.dataset.data_utils")
_DATA_UTILS_STUB.TIMED_EVENTS = []
sys.modules.setdefault("osuT5.osuT5.dataset.data_utils", _DATA_UTILS_STUB)
_TOKENIZER_STUB = types.ModuleType("osuT5.osuT5.tokenizer")
_TOKENIZER_STUB.Tokenizer = object
sys.modules.setdefault("osuT5.osuT5.tokenizer", _TOKENIZER_STUB)

_LOGIT_PROCESSORS_PATH = (
    Path(__file__).resolve().parents[1] / "osuT5" / "osuT5" / "inference" / "logit_processors.py"
)
_SPEC = util.spec_from_file_location("_mapperatorinator_logit_processors", _LOGIT_PROCESSORS_PATH)
_MODULE = util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
MonotonicTimeShiftLogitsProcessor = _MODULE.MonotonicTimeShiftLogitsProcessor


class _FakeTokenizer:
    sos_id = 1
    context_sos = {}
    event_start = {EventType.TIME_SHIFT: 10}
    event_end = {EventType.TIME_SHIFT: 20}


def _apply(processor, input_ids, scores):
    return processor(input_ids.clone(), scores.clone())


def _legacy_full_scan_reference(processor, input_ids, scores):
    """CPU reference for the pre-fix indexing calculation."""

    device = input_ids.device
    batch_size, seq_len = input_ids.shape
    is_time_shift = (
        (input_ids >= processor.time_shift_start)
        & (input_ids < processor.time_shift_end)
    )
    is_sos = processor._is_sos(input_ids)
    indices = torch.arange(seq_len, device=device).expand(batch_size, -1)
    last_time_shift_idx = torch.max(torch.where(is_time_shift, indices, -1), dim=1).values
    last_sos_idx = torch.max(torch.where(is_sos, indices, -1), dim=1).values
    last_time_shift_values = torch.where(
        last_time_shift_idx != -1,
        input_ids[torch.arange(batch_size), last_time_shift_idx]
        - processor.time_shift_start,
        0,
    )
    apply_mask = (last_time_shift_idx != -1) & (last_time_shift_idx > last_sos_idx)
    time_shift_vocab = torch.arange(
        processor.time_shift_start,
        processor.time_shift_end,
        device=device,
    )
    invalid_mask = time_shift_vocab.unsqueeze(0) < (
        processor.time_shift_start + last_time_shift_values
    ).unsqueeze(1)
    batch_mask = torch.full_like(scores, False, dtype=torch.bool)
    batch_mask[:, processor.time_shift_start:processor.time_shift_end] = invalid_mask
    scores[apply_mask] = scores[apply_mask].masked_fill(
        batch_mask[apply_mask],
        -torch.inf,
    )
    return scores


def _random_input_ids(generator, *, batch_size, seq_len, scenario):
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


def test_stateful_monotonic_matches_full_scan_for_growing_batch1_prefixes():
    tokenizer = _FakeTokenizer()
    full_scan = MonotonicTimeShiftLogitsProcessor(tokenizer)
    stateful = MonotonicTimeShiftLogitsProcessor(tokenizer, stateful_batch1=True)
    tokens = torch.tensor([[1, 12, 5, 11, 7, 15, 3, 1, 14, 9]], dtype=torch.long)

    for length in range(1, tokens.shape[1] + 1):
        input_ids = tokens[:, :length]
        scores = torch.arange(64, dtype=torch.float32).reshape(1, 64) / 10
        assert torch.equal(_apply(full_scan, input_ids, scores), _apply(stateful, input_ids, scores))


def test_stateful_monotonic_reinitializes_after_sequence_jump():
    tokenizer = _FakeTokenizer()
    full_scan = MonotonicTimeShiftLogitsProcessor(tokenizer)
    stateful = MonotonicTimeShiftLogitsProcessor(tokenizer, stateful_batch1=True)

    first = torch.tensor([[1, 12, 5, 13]], dtype=torch.long)
    second = torch.tensor([[1, 18, 4]], dtype=torch.long)
    scores = torch.arange(64, dtype=torch.float32).reshape(1, 64) / 10

    _apply(stateful, first, scores)
    assert torch.equal(_apply(full_scan, second, scores), _apply(stateful, second, scores))


def test_full_scan_uses_input_device_for_every_arange_and_preserves_outputs(monkeypatch):
    tokenizer = _FakeTokenizer()
    processor = MonotonicTimeShiftLogitsProcessor(tokenizer)
    input_ids = torch.tensor(
        [
            [1, 12, 5, 15, 3],
            [1, 18, 1, 14, 7],
        ],
        dtype=torch.long,
    )
    scores = torch.arange(128, dtype=torch.float32).reshape(2, 64) / 10
    expected = _legacy_full_scan_reference(
        processor,
        input_ids.clone(),
        scores.clone(),
    )

    original_arange = torch.arange
    observed_devices = []

    def recording_arange(*args, **kwargs):
        observed_devices.append(kwargs.get("device"))
        return original_arange(*args, **kwargs)

    monkeypatch.setattr(_MODULE.torch, "arange", recording_arange)
    candidate = _apply(processor, input_ids, scores)

    assert torch.equal(candidate, expected)
    assert len(observed_devices) == 3
    assert all(device == input_ids.device for device in observed_devices)
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


def test_b2_full_scan_matches_two_private_stateful_batch1_processors_bitwise():
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
            shared_scores = _apply(shared, input_ids, scores)

            for row in range(2):
                private = MonotonicTimeShiftLogitsProcessor(
                    tokenizer,
                    stateful_batch1=True,
                )
                private_scores = _apply(
                    private,
                    input_ids[row:row + 1],
                    scores[row:row + 1],
                )
                assert torch.equal(shared_scores[row:row + 1], private_scores)


def test_full_scan_rewrite_is_bitwise_exact_for_randomized_b1_b2_b5_inputs():
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
                processor = MonotonicTimeShiftLogitsProcessor(tokenizer)
                expected = _legacy_full_scan_reference(
                    processor,
                    input_ids.clone(),
                    scores.clone(),
                )
                candidate = _apply(processor, input_ids, scores)
                assert torch.equal(candidate, expected)
