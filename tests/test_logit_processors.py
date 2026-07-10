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
