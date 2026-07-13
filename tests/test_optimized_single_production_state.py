from __future__ import annotations

from types import SimpleNamespace

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


def test_production_session_reuses_and_resets_cache_between_windows(monkeypatch):
    cache = _FakeCache()
    calls = []

    def fake_get_cache(model, batch_size, num_beams, cfg_scale):
        calls.append((model, batch_size, num_beams, cfg_scale))
        return cache

    monkeypatch.setattr(state_module, "get_cache", fake_get_cache)
    session = ProductionDecodeSession()
    model = object()

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


def test_production_session_exposes_graph_and_encoder_state_by_identity():
    session = ProductionDecodeSession()
    graph_entry = {"decode_replays": 3}
    session.graph_cache[(128,)] = graph_entry
    encoder_output = object()
    session.stable_encoder_holder["encoder_outputs"] = encoder_output

    kwargs = session.active_prefix_decode_kwargs()

    assert kwargs["shared_graph_cache"] is session.graph_cache
    assert kwargs["stable_encoder_holder"] is session.stable_encoder_holder
    assert kwargs["shared_graph_cache"][(128,)] is graph_entry
    assert kwargs["stable_encoder_holder"]["encoder_outputs"] is encoder_output
    assert session.graph_count == 1


def test_production_sessions_do_not_share_mutable_state():
    first = ProductionDecodeSession()
    second = ProductionDecodeSession()

    first.graph_cache[(128,)] = {"decode_replays": 1}
    first.stable_encoder_holder["encoder_outputs"] = object()

    assert second.graph_cache == {}
    assert second.stable_encoder_holder == {}


def test_session_with_existing_cache_resets_on_first_window():
    cache = _FakeCache()
    session = ProductionDecodeSession(cache=cache)

    assert session.cache_for_window(
        object(),
        batch_size=1,
        num_beams=1,
        cfg_scale=1.0,
    ) is cache
    assert cache.self_attention_cache.reset_calls == 1
    assert cache.cross_attention_cache.reset_calls == 1
    assert cache.is_updated == {0: False, 2: False}


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
    assert fields == {"cache", "graph_cache", "stable_encoder_holder"}
    assert not fields.intersection(
        {"generator", "rng", "logits_processor", "stopping_criteria", "tokens"}
    )
