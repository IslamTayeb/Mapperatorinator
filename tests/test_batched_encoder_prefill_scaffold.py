"""CPU-only scaffold tests for W3 Tiger-style batched encoder prefill."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers.modeling_outputs import BaseModelOutput

from osuT5.osuT5.inference.optimized.scout.batched_encoder_prefill import (
    DEFAULT_ENCODER_BATCH_SIZE,
    PREFILL_VERSION,
    BatchedEncoderPrefillError,
    BatchedEncoderPrefillStore,
    chunk_ranges,
    conditioning_digest,
    encoder_hidden_drift,
    install_batched_encoder_prefill_candidate,
    precompute_encoder_hidden_states,
    stack_conditioning,
    window_conditioning,
)


class _MockEncoder(torch.nn.Module):
    """Deterministic CPU encoder for smoke without a full song / GPU."""

    def __init__(self, hidden_dim: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.calls: list[int] = []

    def forward(self, frames, return_dict=True, **kwargs):
        del kwargs
        self.calls.append(int(frames.shape[0]))
        # Content-only transform so B1 and B16 match (batch-size invariant).
        base = frames.to(torch.float32).mean(dim=-1, keepdim=True)
        hidden = base.unsqueeze(-1).expand(-1, frames.shape[1], self.hidden_dim)
        if return_dict:
            return SimpleNamespace(last_hidden_state=hidden)
        return (hidden,)


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


def test_default_batch_size_matches_tiger_sixteen_wide():
    assert DEFAULT_ENCODER_BATCH_SIZE == 16
    assert PREFILL_VERSION.startswith("w3-")


def test_chunk_ranges_cover_all_windows_with_tiger_width():
    assert chunk_ranges(0, 16) == []
    assert chunk_ranges(16, 16) == [(0, 16)]
    assert chunk_ranges(37, 16) == [(0, 16), (16, 32), (32, 37)]
    with pytest.raises(ValueError):
        chunk_ranges(4, 0)
    with pytest.raises(TypeError):
        chunk_ranges(4, True)


def test_stack_conditioning_preserves_rows_and_fails_on_key_drift():
    rows = [
        {"difficulty": torch.tensor([1.0]), "mapper_idx": torch.tensor([2])},
        {"difficulty": torch.tensor([3.0]), "mapper_idx": torch.tensor([4])},
    ]
    stacked = stack_conditioning(rows)
    assert torch.equal(stacked["difficulty"], torch.tensor([1.0, 3.0]))
    assert torch.equal(stacked["mapper_idx"], torch.tensor([2, 4]))
    assert conditioning_digest(rows) == conditioning_digest(rows)

    with pytest.raises(BatchedEncoderPrefillError, match="keys changed"):
        stack_conditioning([rows[0], {"difficulty": torch.tensor([3.0])}])


def test_window_conditioning_includes_exact_song_position():
    processor = SimpleNamespace(
        do_song_position_embed=True,
        miliseconds_per_sequence=1000.0,
        _get_model_cond_kwargs=lambda config: {
            "difficulty": torch.tensor([config.difficulty])
        },
    )
    config = SimpleNamespace(difficulty=6.0)
    rows = window_conditioning(
        processor,
        config,
        torch.tensor([0, 1000]),
        4000.0,
    )

    assert torch.equal(rows[0]["song_position"], torch.tensor([[0.0, 0.25]]))
    assert torch.equal(rows[1]["song_position"], torch.tensor([[0.25, 0.5]]))


def test_cpu_mock_encoder_smoke_b16_matches_b1():
    encoder = _MockEncoder(hidden_dim=3)
    frames = torch.arange(37 * 8, dtype=torch.float32).reshape(37, 8)
    conditioning = [{"difficulty": torch.tensor([float(i)])} for i in range(37)]

    b1, evidence_b1 = precompute_encoder_hidden_states(
        encoder,
        frames,
        conditioning,
        batch_size=1,
        compute_device="cpu",
        storage="cpu",
        expected_dtype=torch.float32,
    )
    encoder.calls.clear()
    b16, evidence_b16 = precompute_encoder_hidden_states(
        encoder,
        frames,
        conditioning,
        batch_size=16,
        compute_device="cpu",
        storage="cpu",
        expected_dtype=torch.float32,
    )

    assert evidence_b16["batch_size"] == 16
    assert evidence_b16["batch_count"] == 3
    assert evidence_b16["live_window_count"] == 37
    assert evidence_b16["storage"] == "cpu"
    assert encoder.calls == [16, 16, 5]
    assert torch.equal(b1, b16)
    drift = encoder_hidden_drift(b1, b16)
    assert drift["exact_window_count"] == 37
    assert drift["max_abs"] == 0.0
    assert evidence_b1["complete_precompute_seconds"] >= 0.0


def test_inject_reuses_store_row_and_moves_cpu_store_to_model_device():
    manager = BatchedEncoderPrefillStore(batch_size=2, storage="cpu")
    frames = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    conditioning = [
        {"difficulty": torch.tensor([float(index)])} for index in range(3)
    ]
    store = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3)
    model = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32)
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
    assert "inputs" in injected
    assert torch.equal(
        injected["encoder_outputs"].last_hidden_state,
        store[0:1],
    )
    with pytest.raises(BatchedEncoderPrefillError, match="conditioning 'difficulty'"):
        manager.inject(
            processor,
            {
                "inputs": frames[1].unsqueeze(0),
                "difficulty": torch.tensor([99.0]),
            },
        )


def test_compatible_entry_requires_model_source_and_conditioning_identity():
    manager = BatchedEncoderPrefillStore(batch_size=2)
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


def test_temporary_processor_hooks_restore():
    original_generate = _Processor.generate
    original_model_generate = _Processor.model_generate
    frames = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    with install_batched_encoder_prefill_candidate(_Processor, batch_size=2) as manager:
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

    assert _Processor.generate is original_generate
    assert _Processor.model_generate is original_model_generate


def test_store_requires_timing_then_main_and_valid_complete_passes():
    manager = BatchedEncoderPrefillStore(batch_size=2)
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
    with pytest.raises(BatchedEncoderPrefillError, match="timing then main"):
        manager.finalize()
    manager._labels[:] = ["timing_context", "main_generation"]
    assert manager.finalize()["labels_completed"] == [
        "timing_context",
        "main_generation",
    ]


def test_assert_processor_accepts_fp16_tip_precision():
    manager = BatchedEncoderPrefillStore(batch_size=16)
    processor = SimpleNamespace(
        parallel=False,
        precision="fp16",
        cfg_scale=1.0,
        inference_runtime=object(),
        device="cuda",
        max_batch_size=16,
        model=SimpleNamespace(
            device=torch.device("cpu"),
            dtype=torch.float16,
        ),
    )
    # fp16 must clear the precision gate; later CUDA residency fails on this host.
    with pytest.raises(BatchedEncoderPrefillError) as exc:
        manager._assert_processor(processor)
    assert "requires precision in" not in str(exc.value)


@pytest.mark.parametrize("batch_size", [True, 0, -1, 1.5])
def test_store_rejects_invalid_batch_sizes(batch_size):
    with pytest.raises((TypeError, ValueError)):
        BatchedEncoderPrefillStore(batch_size=batch_size)
