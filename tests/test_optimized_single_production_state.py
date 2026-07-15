from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.inference.optimized.single import state as state_module
from osuT5.osuT5.inference.optimized.single.state import ProductionDecodeSession


class _ResettableCache:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class _FakeCache:
    def __init__(self):
        self.self_attention_cache = _ResettableCache()
        self.cross_attention_cache = _ResettableCache()
        self.is_updated = {0: True, 2: True}


def _model(*, dtype=torch.float32, device="cpu"):
    return SimpleNamespace(dtype=dtype, device=torch.device(device))


def test_production_session_reuses_and_resets_cache_between_windows(monkeypatch):
    cache = _FakeCache()
    calls = []

    def fake_get_cache(model, batch_size, num_beams, cfg_scale):
        calls.append((model, batch_size, num_beams, cfg_scale))
        return cache

    monkeypatch.setattr(state_module, "get_cache", fake_get_cache)
    session = ProductionDecodeSession()
    model = _model()

    first = session.cache_for_window(
        model,
        batch_size=1,
        num_beams=1,
        cfg_scale=1.0,
    )
    assert first is cache
    assert calls == [(model, 1, 1, 1.0)]
    assert cache.self_attention_cache.reset_calls == 0
    assert cache.cross_attention_cache.reset_calls == 0

    second = session.cache_for_window(
        model,
        batch_size=1,
        num_beams=1,
        cfg_scale=1.0,
    )
    assert second is first
    assert calls == [(model, 1, 1, 1.0)]
    assert cache.self_attention_cache.reset_calls == 1
    assert cache.cross_attention_cache.reset_calls == 1
    assert cache.is_updated == {0: False, 2: False}


def test_production_session_exposes_graph_and_encoder_state_by_identity(monkeypatch):
    cache = _FakeCache()
    monkeypatch.setattr(state_module, "get_cache", lambda *args, **kwargs: cache)
    session = ProductionDecodeSession()
    graph_entry = {
        "active_prefix_length": 128,
        "capture_seconds": 0.25,
        "decode_replays": 3,
    }
    session.graph_cache[(128,)] = graph_entry
    session.cache_for_window(_model(), batch_size=1, num_beams=1, cfg_scale=1.0)
    encoder_output = object()
    holder = session.stable_encoder_holders[session.active_state_signature]
    holder["encoder_outputs"] = encoder_output

    kwargs = session.active_prefix_decode_kwargs()

    assert kwargs["shared_graph_cache"] is session.graph_cache
    assert kwargs["stable_encoder_holder"] is holder
    assert kwargs["shared_graph_cache"][(128,)] is graph_entry
    assert kwargs["stable_encoder_holder"]["encoder_outputs"] is encoder_output
    assert session.graph_count == 1
    assert session.graph_capture_seconds == 0.25
    assert session.graph_decode_replays == 3
    assert session.graph_profile_summary() == {
        "graph_count": 1,
        "decode_replays": 3,
        "capture_seconds": 0.25,
        "buckets": {
            "128": {
                "graph_count": 1,
                "decode_replays": 3,
                "capture_seconds": 0.25,
            }
        },
    }


def test_production_sessions_do_not_share_mutable_state():
    first = ProductionDecodeSession()
    second = ProductionDecodeSession()

    first.graph_cache[(128,)] = {"decode_replays": 1}
    first.stable_encoder_holders[("shape",)] = {"encoder_outputs": object()}

    assert second.graph_cache == {}
    assert second.stable_encoder_holders == {}


def test_batch_tail_pool_reuses_exact_cache_and_encoder_identity(monkeypatch):
    allocated = []

    def fake_get_cache(model, batch_size, num_beams, cfg_scale):
        cache = _FakeCache()
        cache.batch_size = batch_size
        allocated.append(cache)
        return cache

    monkeypatch.setattr(state_module, "get_cache", fake_get_cache)
    session = ProductionDecodeSession()
    model = _model()

    full = session.cache_for_window(
        model, batch_size=4, num_beams=1, cfg_scale=1.0
    )
    full_holder = session.active_prefix_decode_kwargs()["stable_encoder_holder"]
    tail = session.cache_for_window(
        model, batch_size=1, num_beams=1, cfg_scale=1.0
    )
    tail_holder = session.active_prefix_decode_kwargs()["stable_encoder_holder"]
    full_again = session.cache_for_window(
        model, batch_size=4, num_beams=1, cfg_scale=1.0
    )
    full_holder_again = session.active_prefix_decode_kwargs()[
        "stable_encoder_holder"
    ]

    assert len(allocated) == 2
    assert full is full_again
    assert full is not tail
    assert full_holder is full_holder_again
    assert full_holder is not tail_holder
    assert full.self_attention_cache.reset_calls == 1
    assert tail.self_attention_cache.reset_calls == 0


def test_cache_signature_separates_precision_device_beams_cfg_and_model():
    first = _model()
    second = _model(dtype=torch.float16)
    base = ProductionDecodeSession._state_signature(
        first, batch_size=2, num_beams=1, cfg_scale=1.0
    )
    signatures = {
        base,
        ProductionDecodeSession._state_signature(
            first, batch_size=1, num_beams=1, cfg_scale=1.0
        ),
        ProductionDecodeSession._state_signature(
            first, batch_size=2, num_beams=2, cfg_scale=1.0
        ),
        ProductionDecodeSession._state_signature(
            first, batch_size=2, num_beams=1, cfg_scale=1.1
        ),
        ProductionDecodeSession._state_signature(
            second, batch_size=2, num_beams=1, cfg_scale=1.0
        ),
    }
    assert len(signatures) == 5
    with pytest.raises(ValueError, match="batch_size"):
        ProductionDecodeSession._state_signature(
            first, batch_size=0, num_beams=1, cfg_scale=1.0
        )


def test_sequential_generation_creates_one_session_per_unfinished_context():
    from osuT5.osuT5.inference.processor import Processor

    processor = object.__new__(Processor)
    processor.args = SimpleNamespace()
    processor.inference_runtime = object()
    processor.decode_session_state = object()
    processor._create_tokens_per_second_meter = lambda: object()
    created = []

    def new_session():
        session = object()
        created.append(session)
        return session

    processor._new_decode_session_state = new_session
    contexts = [
        {"finished": True},
        {"finished": False},
        {"finished": False},
    ]

    Processor.generate_sequential(
        processor,
        sequences=([], [], 0.0),
        in_context=[],
        out_context=contexts,
        model_kwargs={},
        req_special_tokens=[],
        verbose=False,
    )

    assert len(created) == 2
    assert created[0] is not created[1]
    assert processor.decode_session_state is created[1]


def test_session_type_does_not_own_rng_or_per_window_processing_state():
    fields = set(ProductionDecodeSession.__dataclass_fields__)
    assert fields == {
        "caches",
        "graph_cache",
        "stable_encoder_holders",
        "shared_rope_plans",
        "active_state_signature",
    }
    assert not fields.intersection(
        {"generator", "rng", "logits_processor", "stopping_criteria", "tokens"}
    )
