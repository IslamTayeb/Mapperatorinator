from types import SimpleNamespace

import torch

from osuT5.osuT5.inference.optimized.batch import ActiveRowLedger, ActiveRowState
from osuT5.osuT5.inference.optimized.batch.merged_loop import compare_active_session_caches


def _assert_raises(exc_type, fn, message_fragment):
    try:
        fn()
    except exc_type as exc:
        assert message_fragment in str(exc)
    else:
        raise AssertionError(f"Expected {exc_type.__name__}")


def test_forced_eos_stops_row_without_later_draw_and_dummy_stays_inactive():
    ledger = ActiveRowLedger([
        ActiveRowState("early-eos", max_new_tokens=4, eos_token_ids=(9,)),
        ActiveRowState("long", max_new_tokens=3, eos_token_ids=(9,)),
        None,
    ])

    tokens0, active0 = ledger.materialize_step({0: 9, 1: 1}, pad_token_id=0)
    tokens1, active1 = ledger.materialize_step({1: 2}, pad_token_id=0)
    tokens2, active2 = ledger.materialize_step({1: 3}, pad_token_id=0)

    assert tokens0 == (9, 1, 0)
    assert active0 == (True, True, False)
    assert tokens1 == (0, 2, 0)
    assert active1 == (False, True, False)
    assert tokens2 == (0, 3, 0)
    assert active2 == (False, True, False)
    assert not ledger.has_active_rows()
    report = ledger.report()
    assert report[0]["draw_count"] == 1
    assert report[0]["stop_reason"] == "eos"
    assert report[1]["draw_count"] == 3
    assert report[1]["stop_reason"] == "max_new_tokens"
    assert report[2] is None


def test_inactive_dummy_or_missing_active_draw_fails_loudly():
    ledger = ActiveRowLedger([
        ActiveRowState("a", max_new_tokens=1),
        ActiveRowState("b", max_new_tokens=2),
        None,
    ])
    ledger.materialize_step({0: 1, 1: 2}, pad_token_id=0)

    _assert_raises(
        ValueError,
        lambda: ledger.materialize_step({0: 3, 1: 4}, pad_token_id=0),
        "inactive_or_dummy=[0]",
    )
    _assert_raises(
        ValueError,
        lambda: ledger.materialize_step({2: 4}, pad_token_id=0),
        "inactive_or_dummy=[2]",
    )
    _assert_raises(
        ValueError,
        lambda: ledger.materialize_step({}, pad_token_id=0),
        "missing=[1]",
    )


class _FakeLayer:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        self.keys = keys
        self.values = values
        self.is_initialized = True


class _FakePart:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor, sequence_length: int):
        self.layers = [_FakeLayer(keys, values)]
        self._sequence_length = sequence_length

    def get_seq_length(self) -> int:
        return self._sequence_length


def _fake_session(self_keys: torch.Tensor, cross_keys: torch.Tensor):
    cache = SimpleNamespace(
        self_attention_cache=_FakePart(self_keys, self_keys + 1, self_keys.shape[-2]),
        cross_attention_cache=_FakePart(cross_keys, cross_keys + 1, cross_keys.shape[-2]),
    )
    return SimpleNamespace(
        cache_state=SimpleNamespace(cache=cache),
        model=SimpleNamespace(device=torch.device("cpu")),
    )


def test_active_cache_comparison_selects_rows_and_used_prefix():
    row0_self = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2)
    row1_self = row0_self + 100
    row2_self = row0_self + 200
    row0_cross = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
    row1_cross = row0_cross + 100
    row2_cross = row0_cross + 200
    references = [
        _fake_session(row0_self, row0_cross),
        _fake_session(row1_self, row1_cross),
        _fake_session(row2_self, row2_cross),
    ]
    merged = _fake_session(
        torch.cat([row0_self, row1_self, row2_self], dim=0),
        torch.cat([row0_cross, row1_cross, row2_cross], dim=0),
    )

    report = compare_active_session_caches(
        references,
        merged,
        active_rows=(0, 2),
        self_sequence_length=3,
        atol=0,
        rtol=0,
    )

    assert report["pass"] is True
    assert report["active_rows"] == [0, 2]
    assert report["parts"]["self_attention"]["sequence_length_match"] is True
