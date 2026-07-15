from __future__ import annotations

from dataclasses import fields
import inspect
from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.inference.optimized.single import exact_k4_runtime
from osuT5.osuT5.inference.optimized.single.exact_k4_runtime import (
    EXACT_K4_BLOCK_SIZE,
    RNG_POLICY,
    ExactK4DeviceState,
    ExactK4LogitsProcessor,
    _ExactK4GraphEntry,
    _default_cuda_generator,
    _exact_k4_eligible,
    _processor_signature,
    _restore_snapshot_tensors,
    _restore_terminal_block,
    _runtime_slot,
    _sample_exact,
    _snapshot_tensors,
    _terminal_steps,
    _update_static_model_inputs,
    _validate_block_size,
    install_exact_k4_candidate,
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


def _state(*, max_length: int = 12) -> ExactK4DeviceState:
    return ExactK4DeviceState.allocate(
        torch.tensor([[1, 5]]),
        max_length=max_length,
        pad_token_id=0,
        eos_token_ids=torch.tensor([9]),
    )


def _processor(state: ExactK4DeviceState) -> ExactK4LogitsProcessor:
    return ExactK4LogitsProcessor.compile(
        [MonotonicTimeShiftLogitsProcessor(), TemperatureLogitsWarper()],
        state=state,
        vocab_size=10,
        prompt_length=2,
    )


def _cache(max_length: int = 12, *, dtype: torch.dtype = torch.float32):
    layer = SimpleNamespace(
        keys=torch.zeros((1, 2, max_length, 3), dtype=dtype),
        values=torch.zeros((1, 2, max_length, 3), dtype=dtype),
    )
    return SimpleNamespace(
        self_attention_cache=SimpleNamespace(layers=[layer]),
    )


def _static_inputs(cache=None):
    return {
        "decoder_input_ids": torch.tensor([[5]], dtype=torch.long),
        "cache_position": torch.tensor([2], dtype=torch.long),
        "decoder_position_ids": torch.tensor([[2]], dtype=torch.long),
        "decoder_attention_mask": torch.full((1, 1, 1, 12), -torch.inf),
        "past_key_values": _cache() if cache is None else cache,
    }


def test_exact_k4_policy_is_default_generator_multinomial_only():
    assert EXACT_K4_BLOCK_SIZE == 4
    assert RNG_POLICY == "default_cuda_generator_torch_multinomial_exact_offset_rollback"
    source = inspect.getsource(_sample_exact)
    assert "torch.multinomial(probabilities, num_samples=1).squeeze(1)" in source
    assert "searchsorted" not in source
    assert "counter" not in source


def test_fixed_state_rolls_terminal_overshoot_back_without_reallocation():
    state = _state()
    pointer = state.sequence.data_ptr()

    state.append(torch.tensor([7]), snapshot_index=0)
    state.append(torch.tensor([9]), snapshot_index=1)
    state.append(torch.tensor([8]), snapshot_index=2)
    state.append(torch.tensor([8]), snapshot_index=3)
    state.restore_terminal(valid_steps=2)

    assert state.sequence.data_ptr() == pointer
    assert state.sequence[0, :6].tolist() == [1, 5, 7, 9, 0, 0]
    assert state.physical_length.tolist() == [4]
    assert state.logical_length.tolist() == [4]
    assert state.unfinished.tolist() == [False]
    assert state.stop_flags.tolist() == [False, True, False, False]
    assert state.length_snapshots.tolist() == [3, 4, 0, 0]


def test_state_reset_preserves_every_fixed_address():
    state = _state()
    pointers = {
        descriptor.name: value.data_ptr()
        for descriptor in fields(state)
        if isinstance((value := getattr(state, descriptor.name)), torch.Tensor)
    }
    state.append(torch.tensor([9]), snapshot_index=0)
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
    assert state.stop_flags.count_nonzero().item() == 0


def test_processor_snapshots_restore_terminal_state():
    state = _state()
    processor = _processor(state)
    state.monotonic_has_time_shift.fill_(False)
    state.monotonic_last_value.fill_(2)
    processor.snapshot(0)
    state.monotonic_has_time_shift.fill_(True)
    state.monotonic_last_value.fill_(3)
    processor.snapshot(1)
    state.monotonic_has_time_shift.fill_(False)
    state.monotonic_last_value.fill_(7)

    processor.restore_terminal(valid_steps=2)

    assert state.monotonic_has_time_shift.tolist() == [True]
    assert state.monotonic_last_value.tolist() == [3]
    assert processor.snapshots["monotonic_last_value"][2:].count_nonzero().item() == 0


def test_graph_processor_matches_temperature_and_monotonic_storage_policy():
    state = _state()
    chain = ExactK4LogitsProcessor.compile(
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

    assert scores.dtype == torch.float32
    assert scores[0, 2].item() == -torch.inf
    assert scores[0, 1].item() == pytest.approx(2.0)
    assert set(chain.snapshots) == set(chain.mutable_tensors())


def test_conditional_temperature_uses_device_history():
    state = _state()
    chain = ExactK4LogitsProcessor.compile(
        [MonotonicTimeShiftLogitsProcessor(), ConditionalTemperatureLogitsWarper()],
        state=state,
        vocab_size=10,
        prompt_length=2,
    )
    scores = torch.ones((1, 10))

    assert torch.equal(
        chain._conditional_temperature(ConditionalTemperatureLogitsWarper(), scores),
        torch.full_like(scores, 0.5),
    )


def test_runtime_slot_reuses_addresses_and_separates_sampling_modes():
    holder = {}
    processors = [MonotonicTimeShiftLogitsProcessor()]
    common = dict(
        input_ids=torch.tensor([[1, 5]]),
        max_length=12,
        pad_token_id=0,
        eos_token_ids=torch.tensor([9]),
        logits_processor=processors,
        processor_signature=_processor_signature(processors),
        vocab_size=10,
    )
    sampled = _runtime_slot(holder, do_sample=True, **common)
    sampled_again = _runtime_slot(holder, do_sample=True, **common)
    greedy = _runtime_slot(holder, do_sample=False, **common)

    assert sampled_again is sampled
    assert greedy is not sampled
    assert sampled.signature[-1] is True


@pytest.mark.parametrize(
    ("cur_len", "expected"),
    ((125, True), (126, False), (185, True), (1021, False)),
)
def test_k4_never_crosses_bucket_or_max_length(cur_len, expected):
    assert _exact_k4_eligible(cur_len, 1024, 64) is expected


def test_only_k4_block_size_is_supported():
    assert _validate_block_size(4) == 4
    for invalid in (True, 1, 8, 3):
        with pytest.raises((TypeError, ValueError)):
            _validate_block_size(invalid)


def test_static_handoff_updates_token_positions_and_mask_in_place():
    inputs = _static_inputs()
    pointers = {
        name: value.data_ptr()
        for name, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }

    _update_static_model_inputs(inputs, torch.tensor([7]))

    assert inputs["decoder_input_ids"].tolist() == [[7]]
    assert inputs["cache_position"].tolist() == [3]
    assert inputs["decoder_position_ids"].tolist() == [[3]]
    assert inputs["decoder_attention_mask"][0, 0, 0, 3].item() == 0
    assert {
        name: value.data_ptr()
        for name, value in inputs.items()
        if isinstance(value, torch.Tensor)
    } == pointers


def test_snapshot_contract_includes_cache_slots_and_processor_snapshots():
    state = _state()
    processor = _processor(state)
    static = _static_inputs()
    snapshots = _snapshot_tensors(
        state,
        processor,
        static,
        cache_position_start=2,
        model_dtype=torch.float32,
    )

    assert snapshots
    assert any(tensor.shape[-2:] == (4, 3) for tensor, _ in snapshots)
    assert any(tensor.shape[0] == 4 for tensor, _ in snapshots)


def test_cache_ownership_helpers_are_dtype_generic_while_runtime_stays_fp32_only():
    state = _state()
    processor = _processor(state)
    static = _static_inputs(_cache(dtype=torch.float16))

    snapshots = _snapshot_tensors(
        state,
        processor,
        static,
        cache_position_start=2,
        model_dtype=torch.float16,
    )

    assert any(tensor.dtype == torch.float16 for tensor, _ in snapshots)


def test_capture_snapshot_restore_synchronizes_before_and_after(monkeypatch):
    order = []

    class Tensor:
        def copy_(self, snapshot):
            order.append(("copy", snapshot))

    monkeypatch.setattr(
        exact_k4_runtime.torch.cuda,
        "synchronize",
        lambda device: order.append(("sync", device)),
    )
    _restore_snapshot_tensors([(Tensor(), "snapshot")], device=torch.device("cuda"))

    assert order == [
        ("sync", torch.device("cuda")),
        ("copy", "snapshot"),
        ("sync", torch.device("cuda")),
    ]


def test_terminal_step_detection_returns_first_stop():
    assert _terminal_steps(torch.tensor([False, False, False, False])) is None
    assert _terminal_steps(torch.tensor([False, True, False, True])) == 2
    with pytest.raises(ValueError, match="invalid schema"):
        _terminal_steps(torch.zeros(3, dtype=torch.bool))


def test_terminal_rollback_restores_static_cache_processor_and_rng_offset():
    state = _state()
    processor = _processor(state)
    cache = _cache()
    source = _static_inputs(cache)
    static = {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in source.items()
    }
    state.append(torch.tensor([7]), snapshot_index=0)
    processor.snapshot(0)
    state.append(torch.tensor([9]), snapshot_index=1)
    processor.snapshot(1)
    state.append(torch.tensor([8]), snapshot_index=2)
    processor.snapshot(2)
    state.append(torch.tensor([8]), snapshot_index=3)
    processor.snapshot(3)
    cache.self_attention_cache.layers[0].keys[..., 2:6, :].fill_(1)
    cache.self_attention_cache.layers[0].values[..., 2:6, :].fill_(2)

    class Generator:
        offset = None

        def set_offset(self, value):
            self.offset = value

    generator = Generator()
    entry = _ExactK4GraphEntry(
        graph=object(),
        static_inputs=static,
        state=state,
        processor=processor,
        generator=generator,
        model_dtype=torch.float32,
        offset_increment=4,
        capture_seconds=0.1,
        peak_vram_bytes=1,
    )

    _restore_terminal_block(
        entry,
        source,
        valid_steps=2,
        block_cache_start=2,
        block_sequence_start=2,
        generator_start_offset=20,
    )

    assert state.sequence[0, :6].tolist() == [1, 5, 7, 9, 0, 0]
    assert entry.static_inputs["decoder_input_ids"].tolist() == [[9]]
    assert entry.static_inputs["cache_position"].tolist() == [4]
    assert cache.self_attention_cache.layers[0].keys[..., 4:6, :].count_nonzero() == 0
    assert cache.self_attention_cache.layers[0].values[..., 4:6, :].count_nonzero() == 0
    assert generator.offset == 28


def test_explicit_generator_and_cpu_default_are_rejected():
    with pytest.raises(ValueError, match="implicit default"):
        _default_cuda_generator(torch.Generator(), torch.device("cuda"))
    with pytest.raises(RuntimeError, match="CUDA device"):
        _default_cuda_generator(None, torch.device("cpu"))


def test_install_is_opt_in_non_nested_and_restores_engine(monkeypatch):
    from osuT5.osuT5.inference.optimized.single import engine

    original = engine.active_prefix_decode_generate
    with install_exact_k4_candidate():
        assert (
            engine.active_prefix_decode_generate
            is exact_k4_runtime.exact_k4_active_prefix_decode_generate
        )
        with pytest.raises(RuntimeError, match="cannot be nested"):
            with install_exact_k4_candidate():
                pass
    assert engine.active_prefix_decode_generate is original
    assert exact_k4_runtime._EXACT_K4_INSTALLED is False


def test_default_session_profile_omits_exact_k4_metadata():
    assert "exact_k4_candidate" not in ProductionDecodeSession().graph_profile_summary()


def test_exact_k4_session_profile_is_address_free():
    session = ProductionDecodeSession()
    session.active_state_signature = ("active",)
    session.stable_encoder_holders[("active",)] = {
        "__exact_k4_latest_stats__": {
            "exact_k4_candidate": True,
            "block_size": 4,
            "rng_policy": RNG_POLICY,
            "rng_exact": True,
            "rng_drift": None,
            "parent_backend": "ordinary_pytorch_cuda_graph_four_distinct_steps",
            "generator_offset_increment": 4,
        }
    }

    profile = session.graph_profile_summary()["exact_k4_candidate"]

    assert profile["block_size"] == 4
    assert profile["rng_exact"] is True
    assert profile["rng_drift"] is None
    assert profile["generator_offset_increment"] == 4
    assert profile["parent_backend"] == "ordinary_pytorch_cuda_graph_four_distinct_steps"
