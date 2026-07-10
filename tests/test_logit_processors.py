import torch
import sys
import types
import pytest
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


def test_v32_monotonic_full_scan_masks_earlier_time_shifts():
    tokenizer = _FakeTokenizer()
    processor = MonotonicTimeShiftLogitsProcessor(tokenizer)
    input_ids = torch.tensor([[1, 12, 5, 15, 3]], dtype=torch.long)
    scores = torch.arange(64, dtype=torch.float32).reshape(1, 64) / 10

    result = _apply(processor, input_ids, scores)

    assert torch.isneginf(result[0, 10:15]).all()
    assert torch.equal(result[0, 15:], scores[0, 15:])


def test_v32_monotonic_rejects_optimized_stateful_mode():
    with pytest.raises(ValueError, match="moved to optimized single inference"):
        MonotonicTimeShiftLogitsProcessor(
            _FakeTokenizer(),
            stateful_batch1=True,
        )
