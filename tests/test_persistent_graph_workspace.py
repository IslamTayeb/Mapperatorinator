from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from transformers.modeling_outputs import BaseModelOutput

from osuT5.osuT5.inference.optimized.scout.persistent_graph_workspace import (
    PersistentGraphWorkspacePool,
)
from osuT5.osuT5.inference.optimized.single import state as state_module
from osuT5.osuT5.inference.optimized.single import k8_runtime
from osuT5.osuT5.inference.optimized.single.decode_loop import (
    _stable_encoder_outputs,
)
from osuT5.osuT5.inference.optimized.single.k8_runtime import _StaticInputArena


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 4, bias=False)

    @property
    def dtype(self):
        return self.projection.weight.dtype

    @property
    def device(self):
        return self.projection.weight.device


class _Resettable:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class _Cache:
    def __init__(self):
        self.self_attention_cache = _Resettable()
        self.cross_attention_cache = _Resettable()
        self.is_updated = {0: True, 1: True}


@pytest.fixture(autouse=True)
def _active_k8_topology(monkeypatch):
    monkeypatch.setattr(k8_runtime, "_ACTIVE_BLOCK_SIZE", 4)
    monkeypatch.setattr(k8_runtime, "_ACTIVE_GRAPH_REMAINDERS", True)
    monkeypatch.setattr(k8_runtime, "_ACTIVE_SHARED_STATIC_INPUT_ARENA", True)
    monkeypatch.setattr(k8_runtime, "_ACTIVE_TRANSITION_TIMING", False)


def _pool(model: _Model, *, max_slots: int = 2):
    return PersistentGraphWorkspacePool(
        model,
        topology_signature=("k4", "k1", "selected-v1"),
        packed_state=None,
        max_slots=max_slots,
    )


def _lease(request, model, *, batch_size: int = 1):
    return request.window_lease(
        model,
        batch_size=batch_size,
        num_beams=1,
        cfg_scale=1.0,
    )


def test_a1_a2_reuse_one_workspace_and_reset_cross_cache(monkeypatch):
    cache = _Cache()
    monkeypatch.setattr(state_module, "get_cache", lambda *args, **kwargs: cache)
    model = _Model()
    pool = _pool(model)
    first = pool.new_request()

    with _lease(first, model):
        assert first.cache_for_window(model, batch_size=1, num_beams=1, cfg_scale=1.0) is cache
        holder = first.active_prefix_decode_kwargs()["stable_encoder_holder"]
        holder["__k8_runtime_slots__"] = {
            ("slot",): SimpleNamespace(
                static_input_arena=_StaticInputArena.create(
                    {"decoder_input_ids": torch.tensor([[3]])},
                    window_identity=(1, 1),
                )
            )
        }
        first._workspace().session.graph_cache[("graph",)] = {
            "active_prefix_length": 128,
            "capture_seconds": 0.2,
            "decode_replays": 3,
        }

    second = pool.new_request()
    with _lease(second, model):
        assert second.cache_for_window(model, batch_size=1, num_beams=1, cfg_scale=1.0) is cache
        assert second.graph_count == 1
        assert second.graph_capture_seconds == pytest.approx(0.2)
        assert second.graph_decode_replays == 3

    assert cache.self_attention_cache.reset_calls == 1
    assert cache.cross_attention_cache.reset_calls == 1
    assert cache.is_updated == {0: False, 1: False}
    summary = pool.summary()
    assert summary["workspaces_created"] == 1
    assert summary["resident_slots"] == 1
    assert summary["borrow_count"] == 2
    assert summary["request_count"] == 2
    arena = summary["workspaces"][0]["arena_storage"]
    assert len(arena) == 1
    assert arena[0]["refreshed_window_identity"] == [1, 1]
    assert arena[0]["content_match"] is True
    assert arena[0]["tensor_addresses"][0]["name"] == "decoder_input_ids"


def test_a_b_a_encoder_shapes_reuse_stable_shape_slots(monkeypatch):
    monkeypatch.setattr(state_module, "get_cache", lambda *args, **kwargs: _Cache())
    model = _Model()
    pool = _pool(model)
    pointers = []

    for width, fill in ((3, 1.0), (5, 2.0), (3, 3.0)):
        request = pool.new_request()
        with _lease(request, model):
            request.cache_for_window(model, batch_size=1, num_beams=1, cfg_scale=1.0)
            holder = request.active_prefix_decode_kwargs()["stable_encoder_holder"]
            current = _stable_encoder_outputs(
                holder,
                BaseModelOutput(last_hidden_state=torch.full((1, width, 4), fill)),
            )
            pointers.append(current.last_hidden_state.data_ptr())
            assert torch.equal(
                current.last_hidden_state,
                torch.full((1, width, 4), fill),
            )

    assert pointers[0] != pointers[1]
    assert pointers[0] == pointers[2]
    workspace = pool.summary()["workspaces"][0]
    assert workspace["encoder_slot_count"] == 2
    assert pool.summary()["workspaces_created"] == 1


def test_request_seed_and_window_serial_are_request_local():
    model = _Model()
    pool = _pool(model)
    first = pool.new_request()
    first.request_state["__k8_request_seed__"] = 123
    first.request_state["__k8_window_serial__"] = 87

    second = pool.new_request()

    assert second.request_state == {
        "__persistent_request_serial__": 2,
        "__k8_window_serial__": 0,
    }
    assert first.request_state["__k8_request_seed__"] == 123


def test_concurrent_borrow_fails_loudly():
    model = _Model()
    pool = _pool(model)
    first = pool.new_request()
    second = pool.new_request()

    with _lease(first, model):
        with pytest.raises(RuntimeError, match="concurrent use"):
            with _lease(second, model):
                pass


def test_concurrent_borrow_with_different_signature_also_fails_loudly():
    model = _Model()
    pool = _pool(model)
    first = pool.new_request()
    second = pool.new_request()

    with _lease(first, model, batch_size=1):
        with pytest.raises(RuntimeError, match="concurrent use"):
            with _lease(second, model, batch_size=2):
                pass


def test_bounded_slots_evict_only_inactive_workspace(monkeypatch):
    monkeypatch.setattr(state_module, "get_cache", lambda *args, **kwargs: _Cache())
    model = _Model()
    pool = _pool(model, max_slots=1)
    first = pool.new_request()
    with _lease(first, model, batch_size=1):
        first.cache_for_window(model, batch_size=1, num_beams=1, cfg_scale=1.0)

    second = pool.new_request()
    with _lease(second, model, batch_size=2):
        second.cache_for_window(model, batch_size=2, num_beams=1, cfg_scale=1.0)

    summary = pool.summary()
    assert summary["resident_slots"] == 1
    assert summary["workspaces_created"] == 2
    assert summary["workspaces_evicted"] == 1
    assert summary["max_resident_slots"] == 1


def test_failed_eviction_close_retains_graph_ownership_for_retry():
    model = _Model()
    pool = _pool(model, max_slots=1)
    first = pool.new_request()
    with _lease(first, model):
        workspace = first._workspace()

    calls = 0
    graph = SimpleNamespace(closed=False)

    def close():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic close failure")
        graph.closed = True

    graph.close = close
    workspace.lifecycle.graphs[id(graph)] = graph

    second = pool.new_request()
    with pytest.raises(RuntimeError, match="synthetic close failure"):
        with _lease(second, model, batch_size=2):
            pass

    failed_summary = pool.summary()
    assert failed_summary["resident_slots"] == 1
    assert failed_summary["workspaces_evicted"] == 0
    assert graph.closed is False

    with _lease(second, model, batch_size=2):
        pass
    assert graph.closed is True
    assert calls == 2
    assert pool.summary()["workspaces_evicted"] == 1


def test_close_is_idempotent_and_closes_child_graph_once():
    model = _Model()
    pool = _pool(model)
    request = pool.new_request()
    with _lease(request, model):
        workspace = request._workspace()

    calls = []
    graph = SimpleNamespace(closed=False)

    def close():
        calls.append("close")
        graph.closed = True

    graph.close = close
    workspace.lifecycle.graphs[id(graph)] = graph
    workspace.session.graph_cache[("graph",)] = {"runtime_entry": object()}

    pool.close()
    pool.close()

    assert calls == ["close"]
    assert pool.summary()["closed"] is True
    assert pool.summary()["resident_slots"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        pool.new_request()


def test_model_tensor_version_change_invalidates_pool():
    model = _Model()
    pool = _pool(model)
    pool.new_request()
    with torch.no_grad():
        model.projection.weight.add_(1.0)

    with pytest.raises(RuntimeError, match="addresses or versions changed"):
        pool.new_request()


def test_model_forward_topology_change_invalidates_pool():
    model = _Model()
    pool = _pool(model)

    def replacement(self, value):
        return self.projection(value)

    model.forward = MethodType(replacement, model)
    with pytest.raises(RuntimeError, match="forward topology changed"):
        pool.new_request()


def test_k8_runtime_topology_change_invalidates_pool(monkeypatch):
    model = _Model()
    pool = _pool(model)

    monkeypatch.setattr(k8_runtime, "_ACTIVE_SHARED_STATIC_INPUT_ARENA", False)

    with pytest.raises(RuntimeError, match="K8 runtime topology changed"):
        pool.new_request()


def test_foreign_model_and_mismatched_cache_signature_fail_loudly(monkeypatch):
    monkeypatch.setattr(state_module, "get_cache", lambda *args, **kwargs: _Cache())
    model = _Model()
    foreign = _Model()
    pool = _pool(model)
    request = pool.new_request()

    with pytest.raises(RuntimeError, match="differs from its pool owner"):
        with _lease(request, foreign):
            pass

    with _lease(request, model, batch_size=1):
        with pytest.raises(RuntimeError, match="active workspace signature"):
            request.cache_for_window(
                model,
                batch_size=2,
                num_beams=1,
                cfg_scale=1.0,
            )


def test_persistent_encoder_slots_reject_auxiliary_outputs():
    holder = {"__persistent_encoder_slots__": {}}
    with pytest.raises(RuntimeError, match="without auxiliary outputs"):
        _stable_encoder_outputs(
            holder,
            BaseModelOutput(
                last_hidden_state=torch.zeros((1, 3, 4)),
                hidden_states=(torch.zeros((1, 3, 4)),),
            ),
        )
