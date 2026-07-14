from types import MethodType, SimpleNamespace
from unittest.mock import call

import pytest
import torch
from transformers.modeling_outputs import BaseModelOutput

from utils.profile_prefill_graph_scout import (
    ACCEPTED_MAIN_DISPATCHES,
    EXPECTED_MAIN_DECODE_BUCKETS,
    PrefillRecord,
    accepted_main_profile_signature,
    bucket_prompt_length,
    capture_main_prefills,
    clone_graph_inputs,
    compare_observations,
    copy_graph_inputs,
    load_timing_pair,
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


def _timing_args():
    return SimpleNamespace(
        model_path="model-path",
        train=object(),
        device="cuda",
        max_batch_size=1,
        use_server=False,
        precision="fp32",
        attn_implementation="sdpa",
        gamemode=3,
        inference_engine="optimized",
    )


def test_timing_pair_loads_distinct_base_model_when_production_requires_it():
    args = _timing_args()
    main_binding, main_tokenizer = object(), object()
    timing_binding, timing_tokenizer = object(), object()
    calls = []

    def loader(*positional, **kwargs):
        calls.append(call(*positional, **kwargs))
        return timing_binding, timing_tokenizer

    actual = load_timing_pair(
        args,
        loader=loader,
        needs_separate=lambda value: value is args,
        main_binding=main_binding,
        main_tokenizer=main_tokenizer,
    )

    assert actual == (timing_binding, timing_tokenizer)
    assert calls == [
        call(
            "model-path",
            args.train,
            "cuda",
            max_batch_size=1,
            use_server=False,
            precision="fp32",
            attn_implementation="sdpa",
            gamemode=3,
            auto_select_gamemode_model=False,
            inference_engine="optimized",
        )
    ]


def test_timing_pair_stays_unset_when_production_shares_the_main_model():
    def unexpected_loader(*args, **kwargs):
        raise AssertionError("shared timing selection must not load another model")

    assert load_timing_pair(
        _timing_args(),
        loader=unexpected_loader,
        needs_separate=lambda args: False,
        main_binding=object(),
        main_tokenizer=object(),
    ) == (None, None)


@pytest.mark.parametrize("alias", ["model", "tokenizer"])
def test_timing_pair_rejects_aliases(alias):
    args = _timing_args()
    main_binding, main_tokenizer = object(), object()
    timing_binding = main_binding if alias == "model" else object()
    timing_tokenizer = main_tokenizer if alias == "tokenizer" else object()

    with pytest.raises(RuntimeError, match=f"aliased the main {alias}"):
        load_timing_pair(
            args,
            loader=lambda *args, **kwargs: (timing_binding, timing_tokenizer),
            needs_separate=lambda args: True,
            main_binding=main_binding,
            main_tokenizer=main_tokenizer,
        )


def _accepted_main_profile():
    generation = []
    buckets = {
        str(prefix): {
            "graph_count": 1,
            "decode_replays": index + 1,
            "capture_seconds": 0.01,
        }
        for index, prefix in enumerate(EXPECTED_MAIN_DECODE_BUCKETS)
    }
    for index in range(87):
        hits = {name: 0 for name in ACCEPTED_MAIN_DISPATCHES}
        if index == 0:
            hits.update(
                {
                    "q1_bmm_cross_attention": 12,
                    "native_q1_rope_cache_self_attention": 12,
                    "native_cross_mlp_tail": 12,
                }
            )
        generation.append(
            {
                "profile_label": "main_generation",
                "sequence_index": index,
                "precision": "fp32",
                "generated_tokens": 2,
                "generated_token_ids": [index, index + 1],
                "decoder_loop_backend": "active_prefix_cuda_graph",
                "torch_compile_enabled": False,
                "optimized_effective_config_version": "accepted-test",
                "optimized_batched_super_timing": False,
                "optimized_dispatch_mode": "accepted_batch1",
                "optimized_dispatch_policy": {
                    name: {
                        "requested": True,
                        "enabled": True,
                        "disabled_reason": None,
                    }
                    for name in ACCEPTED_MAIN_DISPATCHES
                },
                "native_cross_mlp_tail_requested": True,
                "native_cross_mlp_tail_enabled": True,
                "optimized_dispatch_capture_hits": hits,
                "decode_graph_count_after": len(EXPECTED_MAIN_DECODE_BUCKETS),
                "optimized_cuda_graphs": {
                    "graph_count": len(EXPECTED_MAIN_DECODE_BUCKETS),
                    "decode_replays": sum(
                        entry["decode_replays"] for entry in buckets.values()
                    ),
                    "capture_seconds": 0.12,
                    "buckets": buckets,
                },
            }
        )
    return {
        "schema_version": 1,
        "metadata": {
            "precision": "fp32",
            "attn_implementation": "sdpa",
            "inference_engine": "optimized",
            "use_server": False,
            "parallel": False,
        },
        "generation": generation,
    }


def test_accepted_main_profile_records_live_topology_tokens_and_832_bucket():
    signature = accepted_main_profile_signature(
        _accepted_main_profile(),
        expected_version="accepted-test",
    )

    assert signature["main_records"] == 87
    assert signature["dispatch_hits"] == {
        "q1_bmm_cross_attention": 12,
        "native_q1_self_attention": 0,
        "native_q1_rope_cache_self_attention": 12,
        "native_cross_mlp_tail": 12,
    }
    assert tuple(signature["decode_bucket_replay_counts"]) == (
        *EXPECTED_MAIN_DECODE_BUCKETS,
    )
    assert signature["decode_bucket_replay_counts"][832] > 0
    assert len(signature["token_stream_sha256"]) == 64
    assert len(signature["stopping_sha256"]) == 64


def test_accepted_main_profile_fails_on_dispatch_or_live_bucket_drift():
    payload = _accepted_main_profile()
    payload["generation"][3]["optimized_dispatch_policy"][
        "q1_bmm_cross_attention"
    ]["enabled"] = False
    with pytest.raises(RuntimeError, match="dispatch policy changed"):
        accepted_main_profile_signature(payload, expected_version="accepted-test")

    payload = _accepted_main_profile()
    del payload["generation"][-1]["optimized_cuda_graphs"]["buckets"]["832"]
    for record in payload["generation"]:
        record["optimized_cuda_graphs"]["graph_count"] -= 1
    with pytest.raises(RuntimeError, match="live decode buckets changed"):
        accepted_main_profile_signature(payload, expected_version="accepted-test")


def test_accepted_main_profile_fails_on_missing_token_or_prefill_record():
    payload = _accepted_main_profile()
    payload["generation"][0]["generated_token_ids"] = None
    with pytest.raises(RuntimeError, match="integer token IDs"):
        accepted_main_profile_signature(payload, expected_version="accepted-test")

    payload = _accepted_main_profile()
    payload["generation"].pop()
    with pytest.raises(RuntimeError, match="one record per live prefill"):
        accepted_main_profile_signature(payload, expected_version="accepted-test")
