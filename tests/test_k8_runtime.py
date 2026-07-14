from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from osuT5.osuT5.inference.optimized.single import k8_runtime
from osuT5.osuT5.inference.optimized.single.k8_runtime import (
    K8DeviceState,
    K8LogitsProcessor,
    _ChildGraphSequence,
    _k8_eligible,
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


class LookbackBiasLogitsWarper:
    def __init__(self):
        self.types_first = False
        self.lookback_range = torch.tensor([False, False, True, False, False])


def _state(*, max_length=12):
    return K8DeviceState.allocate(
        torch.tensor([[1, 5]]),
        max_length=max_length,
        pad_token_id=0,
        eos_token_ids=torch.tensor([9]),
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


def test_graph_processor_owns_monotonic_temperature_and_lookback_state():
    state = _state()
    chain = K8LogitsProcessor.compile(
        [
            MonotonicTimeShiftLogitsProcessor(),
            TemperatureLogitsWarper(),
            LookbackBiasLogitsWarper(),
        ],
        state=state,
        vocab_size=5,
        prompt_length=2,
    )
    scores = chain(torch.tensor([[2.0, 4.0, 6.0, 8.0, 10.0]]))

    assert state.monotonic_has_time_shift.tolist() == [True]
    assert state.monotonic_last_value.tolist() == [1]
    assert scores[0, 2].item() == -torch.inf
    assert scores[0, 1].item() == pytest.approx(2.0)

    chain.observe(torch.tensor([1]))
    assert state.monotonic_has_time_shift.tolist() == [False]


@pytest.mark.parametrize(
    ("cur_len", "max_length", "expected"),
    ((128, 1024, True), (185, 1024, False), (1017, 1024, False)),
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


def test_raw_child_graph_api_fails_loudly_when_torch_does_not_expose_handle():
    with pytest.raises(RuntimeError, match="raw_cuda_graph"):
        _ChildGraphSequence.build(object(), object())


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


def test_session_reports_k8_counters_only_when_candidate_entries_exist():
    session = ProductionDecodeSession()
    assert "k8_candidate" not in session.graph_profile_summary()
    stats = {
        "window_serial": 1,
        "block_size": 8,
        "eligible_steps": 16,
        "block_replays": 2,
        "remainder_steps": 3,
        "status_reads": 5,
        "copy_calls": 4,
        "copy_bytes": 64,
        "capture_seconds": 0.25,
        "peak_vram_bytes": 123,
        "wasted_steps": 2,
        "rng_policy": "counter_prompt_hash_v1",
        "parent_backend": "cuda_python_child_graphs",
    }
    session.graph_cache[("candidate",)] = {
        "k8_candidate": True,
        "active_prefix_length": 640,
        "decode_replays": 16,
        "capture_seconds": 0.25,
        "k8_stats": stats,
    }

    assert session.graph_profile_summary()["k8_candidate"] == {
        "block_size": 8,
        "eligible_steps": 16,
        "block_replays": 2,
        "remainder_steps": 3,
        "status_reads": 5,
        "copy_calls": 4,
        "copy_bytes": 64,
        "capture_seconds": 0.25,
        "peak_vram_bytes": 123,
        "wasted_steps": 2,
        "rng_policy": "counter_prompt_hash_v1",
        "parent_backend": "cuda_python_child_graphs",
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fixed_state_counter_and_append_are_cuda_graph_capture_compatible():
    state = K8DeviceState.allocate(
        torch.tensor([[1, 2]], device="cuda"),
        max_length=12,
        pad_token_id=0,
        eos_token_ids=torch.tensor([9], device="cuda"),
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        uniform = state.counter_uniform()
        state.append((uniform > 0.5).to(dtype=torch.long) + 3)
    graph.replay()
    torch.cuda.synchronize()

    assert state.physical_length.cpu().tolist() == [4]
