from types import MethodType, SimpleNamespace

import pytest
import torch
from transformers.modeling_outputs import BaseModelOutput

from utils.profile_prefill_graph_scout import (
    PrefillRecord,
    bucket_prompt_length,
    capture_main_prefills,
    clone_graph_inputs,
    compare_observations,
    copy_graph_inputs,
    prefill_signature,
    Observation,
    restore_rng_state,
    snapshot_rng_state,
)


def _record(length: int = 65) -> PrefillRecord:
    return PrefillRecord(
        index=0,
        decoder_input_ids=torch.arange(length).reshape(1, length),
        decoder_attention_mask=torch.ones((1, length), dtype=torch.long),
        attention_mask=torch.ones((1, 7), dtype=torch.long),
        encoder_hidden_state=torch.zeros((1, 7, 4)),
        use_cache=True,
    )


def test_bucket_prompt_length_rounds_up_and_fails_loudly():
    assert bucket_prompt_length(1) == 64
    assert bucket_prompt_length(64) == 64
    assert bucket_prompt_length(65) == 128
    with pytest.raises(ValueError, match="positive integer"):
        bucket_prompt_length(0)


def test_prefill_signature_tracks_exact_and_bucketed_shapes():
    record = _record(65)
    assert prefill_signature(record, mode="exact")[0] == 65
    assert prefill_signature(record, mode="bucket64")[0] == 128
    with pytest.raises(ValueError, match="unsupported"):
        prefill_signature(record, mode="approximate")


def test_graph_input_clone_and_copy_include_nested_encoder_tensor():
    source = {
        "decoder_input_ids": torch.tensor([[1, 2]]),
        "encoder_outputs": BaseModelOutput(last_hidden_state=torch.ones((1, 3, 4))),
        "use_cache": True,
        "past_key_values": object(),
    }
    target = clone_graph_inputs(source)
    updated = {
        **source,
        "decoder_input_ids": torch.tensor([[4, 5]]),
        "encoder_outputs": BaseModelOutput(last_hidden_state=torch.full((1, 3, 4), 7.0)),
    }

    copied_bytes, copied_tensors = copy_graph_inputs(target, updated)

    assert copied_tensors == 2
    assert copied_bytes == 2 * 8 + 12 * 4
    assert torch.equal(target["decoder_input_ids"], updated["decoder_input_ids"])
    assert torch.equal(
        target["encoder_outputs"].last_hidden_state,
        updated["encoder_outputs"].last_hidden_state,
    )
    assert target["past_key_values"] is source["past_key_values"]


def test_capture_context_restores_prepare_method_and_only_keeps_prefill():
    class FakeModel:
        def prepare_inputs_for_generation(self, decoder_input_ids, **kwargs):
            return {"decoder_input_ids": decoder_input_ids}

    model = FakeModel()
    original = model.prepare_inputs_for_generation
    processor = SimpleNamespace(model=model)
    records = []
    encoder = BaseModelOutput(last_hidden_state=torch.zeros((1, 7, 4)))

    with capture_main_prefills(processor, records):
        model.prepare_inputs_for_generation(
            torch.ones((1, 5), dtype=torch.long),
            encoder_outputs=encoder,
            decoder_attention_mask=torch.ones((1, 5), dtype=torch.long),
        )
        model.prepare_inputs_for_generation(
            torch.ones((1, 1), dtype=torch.long),
            encoder_outputs=encoder,
            decoder_attention_mask=torch.ones((1, 1), dtype=torch.long),
        )

    assert len(records) == 1
    assert records[0].prompt_length == 5
    assert model.prepare_inputs_for_generation == original


def _observation(value: float) -> Observation:
    tensor = torch.tensor([value])
    return Observation(
        logits=tensor,
        self_keys=[tensor],
        self_values=[tensor],
        cross_keys=[tensor],
        cross_values=[tensor],
    )


def test_observation_comparison_separates_exactness_and_drift():
    exact = compare_observations(_observation(1.0), _observation(1.0))
    drifted = compare_observations(_observation(1.0), _observation(1.25))

    assert exact == {
        "exact": True,
        "finite": True,
        "max_abs_drift": 0.0,
        "shape_valid": True,
    }
    assert drifted["exact"] is False
    assert drifted["max_abs_drift"] == pytest.approx(0.25)


def test_rng_snapshot_restores_same_seed_cpu_reciprocal_behavior():
    torch.manual_seed(12345)
    snapshot = snapshot_rng_state()
    first = torch.rand(16)
    torch.rand(7)

    restore_rng_state(snapshot)
    reciprocal = torch.rand(16)

    assert torch.equal(reciprocal, first)


def test_rng_snapshot_clones_and_restores_all_cuda_states(monkeypatch):
    source = [torch.tensor([1, 2], dtype=torch.uint8), torch.tensor([3, 4], dtype=torch.uint8)]
    restored = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: source)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: restored.extend(states))

    snapshot = snapshot_rng_state()
    source[0].zero_()
    source[1].zero_()
    restore_rng_state(snapshot)

    assert len(restored) == 2
    assert torch.equal(restored[0], torch.tensor([1, 2], dtype=torch.uint8))
    assert torch.equal(restored[1], torch.tensor([3, 4], dtype=torch.uint8))


def test_rng_restore_fails_if_cuda_device_inventory_changes(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [torch.tensor([1], dtype=torch.uint8)])
    snapshot = snapshot_rng_state()
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    with pytest.raises(RuntimeError, match="state count changed"):
        restore_rng_state(snapshot)
