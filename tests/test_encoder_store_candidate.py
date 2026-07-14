from types import SimpleNamespace

import pytest
import torch
from transformers.modeling_outputs import BaseModelOutput

from osuT5.osuT5.inference.optimized.scout.encoder_store import (
    BatchedAllWindowEncoderStore,
    EncoderStoreError,
    _conditioning_digest,
    _stack_conditioning,
    _window_conditioning,
    install_batched_encoder_store_candidate,
)


class _Processor:
    def generate(self, *, sequences, generation_config, profile_label):
        return self.model_generate(
            {
                "inputs": sequences[0][0].unsqueeze(0),
                "difficulty": torch.tensor([6.0]),
            },
            marker=profile_label,
        )

    def model_generate(self, model_kwargs, **generate_kwargs):
        return model_kwargs, generate_kwargs


def test_stack_conditioning_preserves_rows_and_fails_on_key_drift():
    rows = [
        {"difficulty": torch.tensor([1.0]), "mapper_idx": torch.tensor([2])},
        {"difficulty": torch.tensor([3.0]), "mapper_idx": torch.tensor([4])},
    ]
    stacked = _stack_conditioning(rows)
    assert torch.equal(stacked["difficulty"], torch.tensor([1.0, 3.0]))
    assert torch.equal(stacked["mapper_idx"], torch.tensor([2, 4]))
    assert _conditioning_digest(rows) == _conditioning_digest(rows)

    with pytest.raises(EncoderStoreError, match="keys changed"):
        _stack_conditioning([rows[0], {"difficulty": torch.tensor([3.0])}])


def test_window_conditioning_includes_exact_song_position():
    processor = SimpleNamespace(
        do_song_position_embed=True,
        miliseconds_per_sequence=1000.0,
        _get_model_cond_kwargs=lambda config: {
            "difficulty": torch.tensor([config.difficulty])
        },
    )
    config = SimpleNamespace(difficulty=6.0)
    rows = _window_conditioning(
        processor,
        config,
        torch.tensor([0, 1000]),
        4000.0,
    )

    assert torch.equal(rows[0]["song_position"], torch.tensor([[0.0, 0.25]]))
    assert torch.equal(rows[1]["song_position"], torch.tensor([[0.25, 0.5]]))


def test_inject_reuses_store_row_and_rejects_conditioning_change():
    manager = BatchedAllWindowEncoderStore(batch_size=2)
    frames = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    conditioning = [
        {"difficulty": torch.tensor([float(index)])} for index in range(3)
    ]
    store = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3)
    model = object()
    entry = SimpleNamespace(
        model=model,
        source_frames=frames,
        conditioning=conditioning,
        store=store,
    )
    processor = SimpleNamespace(model=model)
    manager._active[id(processor)] = SimpleNamespace(
        label="timing_context",
        processor_id=id(processor),
        entry=entry,
        model_generate_calls=0,
    )

    injected = manager.inject(
        processor,
        {
            "inputs": frames[0].unsqueeze(0),
            "difficulty": torch.tensor([0.0]),
        },
    )

    assert isinstance(injected["encoder_outputs"], BaseModelOutput)
    assert torch.equal(
        injected["encoder_outputs"].last_hidden_state,
        store[0:1],
    )
    with pytest.raises(EncoderStoreError, match="conditioning 'difficulty' changed"):
        manager.inject(
            processor,
            {
                "inputs": frames[1].unsqueeze(0),
                "difficulty": torch.tensor([99.0]),
            },
        )
    processor.model = object()
    with pytest.raises(EncoderStoreError, match="cross-model"):
        manager.inject(
            processor,
            {
                "inputs": frames[1].unsqueeze(0),
                "difficulty": torch.tensor([1.0]),
            },
        )


def test_compatible_entry_requires_model_source_and_conditioning_identity():
    manager = BatchedAllWindowEncoderStore(batch_size=2)
    frames = torch.zeros((2, 4))
    times = torch.tensor([0, 1])
    model = object()
    tokenizer = object()
    conditioning = [{"difficulty": torch.tensor([6.0])}] * 2
    entry = SimpleNamespace(
        model=model,
        tokenizer=tokenizer,
        source_frames=frames,
        source_frame_times=times,
        conditioning=conditioning,
    )
    manager._entries.append(entry)
    processor = SimpleNamespace(model=model, tokenizer=tokenizer)

    assert manager._compatible_entry(
        processor,
        (frames, times, 1.0),
        conditioning,
    ) is entry
    assert manager._compatible_entry(
        SimpleNamespace(model=object(), tokenizer=tokenizer),
        (frames, times, 1.0),
        conditioning,
    ) is None
    assert manager._compatible_entry(
        processor,
        (frames, times, 1.0),
        [
            {"difficulty": torch.tensor([6.0])},
            {"difficulty": torch.tensor([7.0])},
        ],
    ) is None


def test_temporary_processor_hooks_restore_and_do_not_patch_model_forward():
    original_generate = _Processor.generate
    original_model_generate = _Processor.model_generate
    decoder = torch.nn.Linear(2, 2)
    original_forward = decoder.forward
    frames = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    with install_batched_encoder_store_candidate(_Processor, batch_size=2) as manager:
        manager.begin_generation = lambda *args, **kwargs: None
        manager.inject = lambda _processor, kwargs: {
            **kwargs,
            "encoder_outputs": BaseModelOutput(
                last_hidden_state=torch.ones((1, 2, 3))
            ),
        }
        manager.end_generation = lambda *args, **kwargs: None
        processor = _Processor()
        model_kwargs, marker = processor.generate(
            sequences=(frames, torch.tensor([0, 1]), 2.0),
            generation_config=object(),
            profile_label="timing_context",
        )
        assert "encoder_outputs" in model_kwargs
        assert marker == {"marker": "timing_context"}
        assert decoder.forward == original_forward

    assert _Processor.generate is original_generate
    assert _Processor.model_generate is original_model_generate
    assert decoder.forward == original_forward


def test_store_requires_timing_then_main_and_valid_complete_passes():
    manager = BatchedAllWindowEncoderStore(batch_size=2)
    manager._entries.append(
        SimpleNamespace(
            index=0,
            labels=[],
            evidence={
                "output_store_bytes": 1,
                "batch_setup_seconds": 0.0,
                "input_copy_seconds": 0.0,
                "encoder_synchronized_seconds": 1.0,
                "storage_allocation_seconds": 0.0,
                "output_store_copy_seconds": 0.0,
                "complete_precompute_seconds": 1.0,
            },
        )
    )
    with pytest.raises(EncoderStoreError, match="timing then main"):
        manager.finalize()
    manager._labels[:] = ["timing_context", "main_generation"]
    assert manager.finalize()["labels_completed"] == [
        "timing_context",
        "main_generation",
    ]


@pytest.mark.parametrize("batch_size", [True, 0, -1, 1.5])
def test_store_rejects_invalid_batch_sizes(batch_size):
    with pytest.raises((TypeError, ValueError)):
        BatchedAllWindowEncoderStore(batch_size=batch_size)
