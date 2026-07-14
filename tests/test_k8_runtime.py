from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.inference.optimized.single import k8_runtime
from osuT5.osuT5.inference.optimized.single.k8_runtime import (
    K8DeviceState,
    K8LogitsProcessor,
    _ChildGraphSequence,
    _K8GraphLifecycle,
    _capture_k8_entry,
    _k8_eligible,
    _processor_signature,
    _prompt_seed,
    _request_seed,
    _restore_snapshot_tensors,
    _runtime_slot,
    _sync_model_kwargs_after_k8,
    _update_static_model_inputs,
    install_k8_candidate,
)
from osuT5.osuT5.inference.optimized.single.state import ProductionDecodeSession


class MonotonicTimeShiftLogitsProcessor:
    time_shift_start = 4
    time_shift_end = 8

    @staticmethod
    def _sos_ids(device):
        return torch.tensor([1], dtype=torch.long, device=device)


class TemperatureLogitsWarper:
    temperature = 2.0


class ConditionalTemperatureLogitsWarper:
    temperature = 2.0
    conditionals = [(0.5, (1,), 3)]


class LookbackBiasLogitsWarper:
    def __init__(self):
        self.types_first = False
        self.lookback_range = torch.tensor(
            [False, False, True, False, False, False, False, False, False, False]
        )


def _state(*, max_length=12):
    return K8DeviceState.allocate(
        torch.tensor([[1, 5]]),
        max_length=max_length,
        pad_token_id=0,
        eos_token_ids=torch.tensor([9]),
        rng_seed=123,
    )


def test_fixed_state_records_logical_stop_without_cat_or_reallocation():
    state = _state()
    pointer = state.sequence.data_ptr()

    assert state.append(torch.tensor([7])).tolist() == [7]
    assert state.append(torch.tensor([9])).tolist() == [9]
    assert state.append(torch.tensor([8])).tolist() == [0]

    assert state.sequence.data_ptr() == pointer
    assert state.sequence[0, :5].tolist() == [1, 5, 7, 9, 0]
    assert state.physical_length.tolist() == [5]
    assert state.logical_length.tolist() == [4]
    assert state.unfinished.tolist() == [False]


def test_state_reset_preserves_every_fixed_address_and_prompt_seed():
    state = _state()
    pointers = {
        descriptor.name: value.data_ptr()
        for descriptor in fields(state)
        if isinstance((value := getattr(state, descriptor.name)), torch.Tensor)
    }
    first_seed = state.rng_seed.clone()
    state.append(torch.tensor([9]))
    state.reset(
        torch.tensor([[1, 6, 7]]),
        pad_token_id=0,
        eos_token_ids=torch.tensor([9]),
        rng_seed=456,
    )

    assert {
        descriptor.name: value.data_ptr()
        for descriptor in fields(state)
        if isinstance((value := getattr(state, descriptor.name)), torch.Tensor)
    } == pointers
    assert state.sequence[0, :4].tolist() == [1, 6, 7, 0]
    assert state.physical_length.tolist() == [3]
    assert state.unfinished.tolist() == [True]
    assert not torch.equal(state.rng_seed, first_seed)


def test_prompt_stream_seed_owns_request_seed_window_and_prompt_identity():
    prompt = torch.tensor([[0, 1, 5]])
    first = _prompt_seed(prompt, request_seed=123, window_identity=7)

    assert first == _prompt_seed(prompt, request_seed=123, window_identity=7)
    assert first != _prompt_seed(prompt, request_seed=124, window_identity=7)
    assert first != _prompt_seed(prompt, request_seed=123, window_identity=8)
    assert first != _prompt_seed(
        torch.tensor([[0, 1, 6]]), request_seed=123, window_identity=7
    )


def test_later_window_seed_is_independent_of_prior_overshoot_counter():
    first_window = _state()
    for _ in range(8):
        first_window.counter_uniform()
    next_prompt = torch.tensor([[0, 1, 5]])

    after_overshoot = _prompt_seed(
        next_prompt,
        request_seed=123,
        window_identity=2,
    )
    clean_replay = _prompt_seed(
        next_prompt,
        request_seed=123,
        window_identity=2,
    )

    assert after_overshoot == clean_replay


def test_explicit_request_generator_seed_is_used_without_consuming_state():
    generator = torch.Generator().manual_seed(9876)
    before = generator.get_state().clone()

    assert _request_seed(generator, torch.device("cpu")) == 9876
    assert torch.equal(generator.get_state(), before)


def test_runtime_slot_separates_sampling_modes():
    holder = {}
    processors = [MonotonicTimeShiftLogitsProcessor()]
    signature = _processor_signature(processors)
    common = dict(
        input_ids=torch.tensor([[1, 5]]),
        max_length=12,
        pad_token_id=0,
        eos_token_ids=torch.tensor([9]),
        logits_processor=processors,
        processor_signature=signature,
        vocab_size=10,
        rng_seed=123,
    )

    sampled = _runtime_slot(holder, do_sample=True, **common)
    greedy = _runtime_slot(holder, do_sample=False, **common)

    assert sampled is not greedy
    assert sampled.signature[-1] is True
    assert greedy.signature[-1] is False


def test_graph_processor_owns_monotonic_temperature_and_lookback_state():
    state = _state()
    chain = K8LogitsProcessor.compile(
        [
            MonotonicTimeShiftLogitsProcessor(),
            TemperatureLogitsWarper(),
            LookbackBiasLogitsWarper(),
        ],
        state=state,
        vocab_size=10,
        prompt_length=2,
    )
    scores = chain(torch.arange(2.0, 22.0, 2.0).reshape(1, 10))

    assert state.monotonic_has_time_shift.tolist() == [True]
    assert state.monotonic_last_value.tolist() == [1]
    assert scores[0, 2].item() == -torch.inf
    assert scores[0, 1].item() == pytest.approx(2.0)

    chain.observe(torch.tensor([1]))
    assert state.monotonic_has_time_shift.tolist() == [False]


def test_conditional_temperature_requires_enough_physical_history():
    state = _state()
    chain = K8LogitsProcessor.compile(
        [
            MonotonicTimeShiftLogitsProcessor(),
            ConditionalTemperatureLogitsWarper(),
        ],
        state=state,
        vocab_size=10,
        prompt_length=2,
    )
    scores = torch.ones((1, 10))

    assert torch.equal(
        chain._conditional_temperature(ConditionalTemperatureLogitsWarper(), scores),
        torch.full_like(scores, 0.5),
    )
    state.append(torch.tensor([7]))
    assert torch.equal(
        chain._conditional_temperature(ConditionalTemperatureLogitsWarper(), scores),
        torch.full_like(scores, 2.0),
    )


@pytest.mark.parametrize(
    ("cur_len", "max_length", "expected"),
    ((128, 1024, False), (185, 1024, True), (1017, 1024, False)),
)
def test_k8_never_crosses_bucket_or_max_length(cur_len, max_length, expected):
    assert _k8_eligible(cur_len, max_length, 64) is expected


def test_static_handoff_updates_token_positions_and_causal_mask_in_place():
    inputs = {
        "decoder_input_ids": torch.tensor([[3]]),
        "cache_position": torch.tensor([4]),
        "decoder_position_ids": torch.tensor([[4]]),
        "decoder_attention_mask": torch.full((1, 1, 1, 12), -torch.inf),
    }
    pointers = {name: value.data_ptr() for name, value in inputs.items()}

    _update_static_model_inputs(inputs, torch.tensor([7]))

    assert inputs["decoder_input_ids"].tolist() == [[7]]
    assert inputs["cache_position"].tolist() == [5]
    assert inputs["decoder_position_ids"].tolist() == [[5]]
    assert inputs["decoder_attention_mask"][0, 0, 0, 5].item() == 0
    assert {name: value.data_ptr() for name, value in inputs.items()} == pointers


def test_capture_snapshot_restore_waits_for_side_stream_before_copy(monkeypatch):
    order = []

    class Tensor:
        def copy_(self, snapshot):
            order.append(("copy", snapshot))

    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: order.append(("synchronize", device)),
    )
    device = torch.device("cuda", 0)

    _restore_snapshot_tensors(
        [(Tensor(), "sequence"), (Tensor(), "lookback")],
        device=device,
    )

    assert order == [
        ("synchronize", device),
        ("copy", "sequence"),
        ("copy", "lookback"),
        ("synchronize", device),
    ]


def test_left_padding_survives_two_k8_handoffs_with_position_progression():
    static_inputs = {
        "decoder_input_ids": torch.tensor([[3]]),
        "cache_position": torch.tensor([3]),
        "decoder_position_ids": torch.tensor([[3]]),
        "decoder_attention_mask": torch.full((1, 1, 1, 24), -torch.inf),
    }
    entry = SimpleNamespace(static_inputs=static_inputs)
    model_kwargs = {
        "cache_position": torch.tensor([3]),
        "decoder_attention_mask": torch.tensor([[0, 0, 1, 1]]),
    }

    for expected_length in (12, 20):
        for token in range(8):
            _update_static_model_inputs(static_inputs, torch.tensor([token + 4]))
        _sync_model_kwargs_after_k8(
            model_kwargs,
            entry,
            cur_len=expected_length,
        )
        assert model_kwargs["decoder_attention_mask"][0, :2].tolist() == [0, 0]
        assert torch.all(model_kwargs["decoder_attention_mask"][0, 2:] == 1)

    assert static_inputs["cache_position"].tolist() == [19]
    assert static_inputs["decoder_position_ids"].tolist() == [[19]]
    assert model_kwargs["cache_position"] is static_inputs["cache_position"]


def test_raw_child_graph_api_fails_loudly_when_torch_does_not_expose_handle():
    with pytest.raises(RuntimeError, match="raw_cuda_graph"):
        _ChildGraphSequence.build(object(), object())


def test_child_graph_sequence_accepts_exact_cuda_binding_return_shapes(monkeypatch):
    from cuda.bindings import runtime

    success = runtime.cudaError_t.cudaSuccess
    calls = []
    device_guards = []
    node_count = 0

    def create(flags):
        calls.append(("create", flags))
        return success, "parent"

    def add(parent, dependencies, dependency_count, child):
        nonlocal node_count
        node_count += 1
        node = f"node-{node_count}"
        calls.append(("add", parent, dependencies, dependency_count, child, node))
        return success, node

    def instantiate(parent, flags):
        calls.append(("instantiate", parent, flags))
        return success, "executable"

    def launch(executable, stream):
        calls.append(("launch", executable, stream))
        return (success,)

    def destroy_executable(executable):
        calls.append(("destroy_executable", executable))
        return (success,)

    def destroy_graph(graph):
        calls.append(("destroy_graph", graph))
        return (success,)

    monkeypatch.setattr(runtime, "cudaGraphCreate", create)
    monkeypatch.setattr(runtime, "cudaGraphAddChildGraphNode", add)
    monkeypatch.setattr(runtime, "cudaGraphInstantiate", instantiate)
    monkeypatch.setattr(runtime, "cudaGraphLaunch", launch)
    monkeypatch.setattr(runtime, "cudaGraphExecDestroy", destroy_executable)
    monkeypatch.setattr(runtime, "cudaGraphDestroy", destroy_graph)
    monkeypatch.setattr(runtime, "cudaGraph_t", lambda handle: ("graph", handle))
    monkeypatch.setattr(runtime, "cudaStream_t", lambda handle: ("stream", handle))

    @contextmanager
    def device_guard(device):
        device_guards.append(torch.device(device))
        yield

    monkeypatch.setattr(torch.cuda, "device", device_guard)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: SimpleNamespace(cuda_stream=99),
    )
    model_graph = SimpleNamespace(raw_cuda_graph=lambda: 11)
    tail_graph = SimpleNamespace(raw_cuda_graph=lambda: 22)

    sequence = _ChildGraphSequence.build(
        model_graph,
        tail_graph,
        device=torch.device("cuda", 1),
    )
    sequence.replay()
    sequence.close()
    sequence.close()

    assert calls[0] == ("create", 0)
    add_calls = [row for row in calls if row[0] == "add"]
    assert len(add_calls) == 16
    assert add_calls[0][2:4] == (None, 0)
    assert add_calls[1][2:4] == (("node-1",), 1)
    assert ("instantiate", "parent", 0) in calls
    assert ("launch", "executable", ("stream", 99)) in calls
    assert calls.count(("destroy_executable", "executable")) == 1
    assert calls.count(("destroy_graph", "parent")) == 1
    assert device_guards == [
        torch.device("cuda", 1),
        torch.device("cuda", 1),
        torch.device("cuda", 1),
    ]


def test_child_graph_partial_construction_destroys_parent(monkeypatch):
    from cuda.bindings import runtime

    success = runtime.cudaError_t.cudaSuccess
    failure = runtime.cudaError_t.cudaErrorInvalidValue
    destroyed = []
    add_calls = 0

    monkeypatch.setattr(
        runtime,
        "cudaGraphCreate",
        lambda flags: (success, "parent"),
    )

    def add(*args):
        nonlocal add_calls
        add_calls += 1
        return (failure,) if add_calls == 2 else (success, "node-1")

    monkeypatch.setattr(runtime, "cudaGraphAddChildGraphNode", add)
    monkeypatch.setattr(
        runtime,
        "cudaGraphDestroy",
        lambda graph: destroyed.append(graph) or (success,),
    )
    monkeypatch.setattr(runtime, "cudaGraph_t", lambda handle: handle)

    @contextmanager
    def device_guard(device):
        yield

    monkeypatch.setattr(torch.cuda, "device", device_guard)
    model_graph = SimpleNamespace(raw_cuda_graph=lambda: 11)
    tail_graph = SimpleNamespace(raw_cuda_graph=lambda: 22)

    with pytest.raises(RuntimeError, match="cudaGraphAddChildGraphNode failed"):
        _ChildGraphSequence.build(
            model_graph,
            tail_graph,
            device=torch.device("cuda", 0),
        )

    assert destroyed == ["parent"]


def test_capture_uses_state_device_guard_and_closes_parent_on_late_failure(
    monkeypatch,
):
    device_guards = []

    @contextmanager
    def device_guard(device):
        device_guards.append(torch.device(device))
        yield

    class Graph:
        def __init__(self, keep_graph=False):
            self.keep_graph = keep_graph

    class Parent:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    parent = Parent()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", device_guard)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", lambda graph: device_guard("cuda:2"))
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: (_ for _ in ()).throw(RuntimeError("late failure")),
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    monkeypatch.setattr(k8_runtime, "_clone_static_graph_inputs", lambda values: values)
    monkeypatch.setattr(k8_runtime, "_snapshot_tensors", lambda *args: [])
    monkeypatch.setattr(k8_runtime, "_one_step_tail", lambda **kwargs: None)
    monkeypatch.setattr(
        _ChildGraphSequence,
        "build",
        lambda *args, **kwargs: parent,
    )
    state = SimpleNamespace(
        sequence=SimpleNamespace(device=torch.device("cuda", 2))
    )
    model = lambda **kwargs: SimpleNamespace(logits=torch.zeros((1, 1, 4)))

    with pytest.raises(RuntimeError, match="late failure"):
        _capture_k8_entry(
            model,
            {},
            active_prefix_length=128,
            state=state,
            processor=object(),
            do_sample=True,
        )

    assert parent.close_calls == 1
    assert device_guards[0] == torch.device("cuda", 2)


def test_k8_lifecycle_closes_graphs_at_session_transition_and_context_exit():
    class Graph:
        def __init__(self):
            self.closed = False
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            self.closed = True

    first = Graph()
    second = Graph()
    first_cache = {}
    second_cache = {}

    with install_k8_candidate():
        lifecycle = k8_runtime._ACTIVE_K8_LIFECYCLE
        assert isinstance(lifecycle, _K8GraphLifecycle)
        lifecycle.activate(first_cache)
        lifecycle.register(first)
        lifecycle.activate(first_cache)
        assert not first.closed
        lifecycle.activate(second_cache)
        assert first.closed
        lifecycle.register(second)

    assert first.close_calls == 1
    assert second.close_calls == 1
    assert k8_runtime._ACTIVE_K8_LIFECYCLE is None


def test_opt_in_install_restores_default_even_on_exception(monkeypatch):
    from osuT5.osuT5.inference.optimized.single import engine

    default = engine.active_prefix_decode_generate
    with pytest.raises(RuntimeError, match="boom"):
        with install_k8_candidate():
            assert (
                engine.active_prefix_decode_generate
                is k8_runtime.k8_active_prefix_decode_generate
            )
            raise RuntimeError("boom")
    assert engine.active_prefix_decode_generate is default


def test_session_reports_latest_k8_window_independent_of_graph_entries():
    session = ProductionDecodeSession()
    assert "k8_candidate" not in session.graph_profile_summary()
    stats = {
        "k8_candidate": True,
        "window_serial": 1,
        "block_size": 8,
        "prefill_steps": 1,
        "eligible_steps": 16,
        "block_replays": 2,
        "remainder_steps": 3,
        "status_reads": 5,
        "copy_calls": 4,
        "copy_bytes": 64,
        "capture_seconds": 0.25,
        "peak_vram_bytes": 123,
        "physical_steps": 20,
        "logical_steps": 18,
        "wasted_steps": 2,
        "rng_policy": "counter_request_seed_window_prompt_v2",
        "rng_exact": False,
        "rng_drift": "documented_counter_based_per_window",
        "rng_request_seed": 12345,
        "rng_window_identity": 1,
        "rng_early_eos_isolation": True,
        "sampling_mode": "sample",
        "prompt_seed_d2h_copy_calls": 1,
        "prompt_seed_d2h_copy_bytes": 16,
        "prompt_seed_setup_seconds": 0.01,
        "processor_signature_d2h_copy_calls": 2,
        "processor_signature_d2h_copy_bytes": 32,
        "processor_signature_setup_seconds": 0.02,
        "parent_backend": "cuda_python_child_graphs",
        "capture_state_restore_synchronized": True,
    }
    signature = ("shape",)
    session.active_state_signature = signature
    session.stable_encoder_holders[signature] = {"__k8_latest_stats__": stats}
    session.graph_cache[("candidate",)] = {
        "k8_candidate": True,
        "active_prefix_length": 640,
        "decode_replays": 16,
        "capture_seconds": 0.25,
        "k8_stats": stats,
    }

    assert session.graph_profile_summary()["k8_candidate"] == {
        "block_size": 8,
        "prefill_steps": 1,
        "eligible_steps": 16,
        "block_replays": 2,
        "remainder_steps": 3,
        "status_reads": 5,
        "copy_calls": 4,
        "copy_bytes": 64,
        "capture_seconds": 0.25,
        "peak_vram_bytes": 123,
        "physical_steps": 20,
        "logical_steps": 18,
        "wasted_steps": 2,
        "rng_policy": "counter_request_seed_window_prompt_v2",
        "rng_exact": False,
        "rng_drift": "documented_counter_based_per_window",
        "rng_request_seed": 12345,
        "rng_window_identity": 1,
        "rng_early_eos_isolation": True,
        "sampling_mode": "sample",
        "prompt_seed_d2h_copy_calls": 1,
        "prompt_seed_d2h_copy_bytes": 16,
        "prompt_seed_setup_seconds": 0.01,
        "processor_signature_d2h_copy_calls": 2,
        "processor_signature_d2h_copy_bytes": 32,
        "processor_signature_setup_seconds": 0.02,
        "parent_backend": "cuda_python_child_graphs",
        "capture_state_restore_synchronized": True,
    }


def test_k1_only_latest_window_does_not_report_stale_k8_graph_stats():
    session = ProductionDecodeSession()
    signature = ("shape",)
    session.active_state_signature = signature
    latest = {
        "k8_candidate": True,
        "window_serial": 2,
        "block_size": 8,
        "prefill_steps": 1,
        "eligible_steps": 0,
        "block_replays": 0,
        "remainder_steps": 3,
        "physical_steps": 4,
        "logical_steps": 4,
        "wasted_steps": 0,
    }
    session.stable_encoder_holders[signature] = {"__k8_latest_stats__": latest}
    session.graph_cache[("stale",)] = {
        "k8_candidate": True,
        "active_prefix_length": 640,
        "decode_replays": 16,
        "capture_seconds": 0.25,
        "k8_stats": {"window_serial": 1, "block_replays": 2},
    }

    report = session.graph_profile_summary()["k8_candidate"]

    assert report["prefill_steps"] == 1
    assert report["eligible_steps"] == 0
    assert report["block_replays"] == 0
    assert report["remainder_steps"] == 3
    assert report["physical_steps"] == report["logical_steps"] == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fixed_state_counter_and_append_are_cuda_graph_capture_compatible():
    state = K8DeviceState.allocate(
        torch.tensor([[1, 2]], device="cuda"),
        max_length=12,
        pad_token_id=0,
        eos_token_ids=torch.tensor([9], device="cuda"),
        rng_seed=123,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        uniform = state.counter_uniform()
        state.append((uniform > 0.5).to(dtype=torch.long) + 3)
    graph.replay()
    torch.cuda.synchronize()

    assert state.physical_length.cpu().tolist() == [4]
