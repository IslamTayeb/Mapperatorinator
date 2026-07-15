from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.event import Event, EventType
from osuT5.osuT5.inference.processor import Processor
from osuT5.osuT5.inference.optimized.scout.prompt_context import (
    PromptContextStats,
    _ExactPromptContextSession,
    install_exact_prompt_context_preparation,
)
from osuT5.osuT5.tokenizer import ContextType, MILISECONDS_PER_STEP
from utils import run_strict_fp32_prompt_context


ROOT = Path(__file__).resolve().parents[1]


def _processor(*, center_pad_decoder: bool = False, tgt_seq_len: int = 32):
    processor = Processor.__new__(Processor)
    processor.parallel = False
    processor.inference_runtime = object()
    processor.cfg_scale = 1.0
    processor.num_beams = 1
    processor.tgt_seq_len = tgt_seq_len
    processor.center_pad_decoder = center_pad_decoder
    processor.tokenizer = SimpleNamespace(
        context_sos={ContextType.TIMING: 101, ContextType.MAP: 102},
        context_eos={ContextType.TIMING: 201, ContextType.MAP: 202},
        sos_id=1,
        pad_id=0,
    )
    return processor


def _contexts(*, input_width: int = 4, output_width: int = 3):
    in_context = [
        {
            "context_type": ContextType.TIMING,
            "tokens": torch.arange(10, 10 + input_width).unsqueeze(0),
            "add_type": True,
            "add_class": False,
        }
    ]
    out_context = [
        {
            "context_type": ContextType.MAP,
            "tokens": torch.arange(20, 20 + output_width).unsqueeze(0),
            "class": torch.tensor([[30, 31]]),
            "extra_special_tokens": torch.tensor([[32]]),
            "pre_tokens": torch.tensor([[40, 41]]),
            "add_type": True,
            "add_class": True,
        }
    ]
    return in_context, out_context


@pytest.mark.parametrize("center_pad_decoder", [False, True])
@pytest.mark.parametrize(
    ("input_width", "output_width"),
    ((0, 0), (1, 7), (12, 12), (24, 24)),
)
def test_prompt_and_decoder_cache_inputs_match_framework_exactly(
    center_pad_decoder: bool,
    input_width: int,
    output_width: int,
) -> None:
    processor = _processor(
        center_pad_decoder=center_pad_decoder,
        tgt_seq_len=32,
    )
    in_context, out_context = _contexts(
        input_width=input_width,
        output_width=output_width,
    )
    expected, expected_unconditional = Processor.get_prompts(
        processor,
        in_context,
        out_context,
    )
    expected_padded, expected_width = Processor.pad_prompts(
        processor,
        [expected, expected_unconditional],
    )
    session = _ExactPromptContextSession(processor, PromptContextStats())

    with session.install():
        actual, actual_unconditional = processor.get_prompts(in_context, out_context)
        actual_padded, actual_width = processor.pad_prompts(
            [actual, actual_unconditional]
        )

    assert actual_unconditional is None
    assert actual_width == expected_width
    assert torch.equal(actual_padded[0], expected_padded[0])
    assert torch.equal(
        actual_padded[0].ne(processor.tokenizer.pad_id),
        expected_padded[0].ne(processor.tokenizer.pad_id),
    )
    assert hashlib.sha256(actual_padded[0].numpy().tobytes()).digest() == (
        hashlib.sha256(expected_padded[0].numpy().tobytes()).digest()
    )
    assert actual_padded[0].dtype == torch.long
    assert actual_padded[0].device.type == "cpu"
    assert actual_padded[0].is_contiguous()


def test_reused_prompt_buffer_and_retry_are_exact() -> None:
    processor = _processor(tgt_seq_len=24)
    in_context, out_context = _contexts(input_width=12, output_width=12)
    expected, _ = Processor.get_prompts(processor, in_context, out_context)
    session = _ExactPromptContextSession(processor, PromptContextStats())

    first, _ = session.get_prompts(processor, in_context, out_context)
    first_pointer = first.data_ptr()
    second, _ = session.get_prompts(processor, in_context, out_context)

    assert first_pointer == second.data_ptr()
    assert torch.equal(second, expected)
    assert session.stats.prompt_retries == 6


def _context_processor():
    processor = _processor()
    processor.miliseconds_per_sequence = 1000.0
    processor.max_pre_token_len = 3
    processor.add_kiai_special_token = False
    processor.add_kiai = False
    processor.add_sv_special_token = False
    processor.add_sv = False
    processor.add_mania_sv = False
    processor.add_song_position_token = False
    processor.tokenizer.event_range = {
        EventType.TIME_SHIFT: SimpleNamespace(min_value=-50, max_value=100),
        EventType.SNAPPING: SimpleNamespace(min_value=0, max_value=16),
    }
    processor.tokenizer.event_start = {
        EventType.TIME_SHIFT: 3,
        EventType.SNAPPING: 154,
    }

    def encode(event):
        value_range = processor.tokenizer.event_range[event.type]
        if not value_range.min_value <= event.value <= value_range.max_value:
            raise ValueError("out of range")
        return (
            processor.tokenizer.event_start[event.type]
            + event.value
            - value_range.min_value
        )

    processor.tokenizer.encode = encode
    return processor


@pytest.mark.parametrize("frame_time", (0.0, 1000.0, 1510.0, 2500.0))
@pytest.mark.parametrize("add_pre_tokens", (False, True))
def test_incremental_context_encoding_matches_framework_exactly(
    frame_time: float,
    add_pre_tokens: bool,
) -> None:
    processor = _context_processor()
    events = [
        Event(EventType.TIME_SHIFT, -1000.0),
        Event(EventType.SNAPPING, 2),
        Event(EventType.TIME_SHIFT, 1010.0),
        Event(EventType.TIME_SHIFT, 1500.0),
        Event(EventType.SNAPPING, 5),
        Event(EventType.TIME_SHIFT, 4000.0),
    ]
    event_times = [-1000.0, 0.0, 1010.0, 1500.0, 1500.0, 4000.0]
    context = {
        "context_type": ContextType.MAP,
        "events": events,
        "event_times": event_times,
        "add_pre_tokens": add_pre_tokens,
    }
    expected = Processor.prepare_context_sequence(processor, context, frame_time)
    session = _ExactPromptContextSession(processor, PromptContextStats())

    actual = session.prepare_context_sequence(processor, context, frame_time)

    assert torch.equal(actual["tokens"], expected["tokens"])
    assert actual["tokens"].dtype == torch.long
    if add_pre_tokens:
        assert torch.equal(actual["pre_tokens"], expected["pre_tokens"])
    assert actual["extra_special_events"] == expected["extra_special_events"]


def test_binary_search_matches_framework_for_duplicates_and_incremental_append() -> None:
    processor = _processor()
    stats = PromptContextStats()
    session = _ExactPromptContextSession(processor, stats)
    values = [0.0, 10.0, 10.0, 20.0, 30.0]

    assert session.get_events_time_range(processor, values, 10.0, 30.0) == (
        Processor._get_events_time_range(processor, values, 10.0, 30.0)
    )
    values.extend([30.0, 40.0])
    assert session.get_events_time_range(processor, values, 20.0, 41.0) == (
        Processor._get_events_time_range(processor, values, 20.0, 41.0)
    )
    assert stats.full_order_validations == 1
    assert stats.incremental_order_validations == 1


def test_context_cache_fails_loudly_on_unowned_mutation() -> None:
    processor = _context_processor()
    context = {
        "context_type": ContextType.MAP,
        "events": [Event(EventType.TIME_SHIFT, 10.0)],
        "event_times": [10.0],
        "add_pre_tokens": False,
    }
    session = _ExactPromptContextSession(processor, PromptContextStats())
    session._encoding_state(context)
    context["events"].append(Event(EventType.TIME_SHIFT, 20.0))
    context["event_times"].append(20.0)

    with pytest.raises(RuntimeError, match="without prompt-context ownership"):
        session._encoding_state(context)


def test_instance_methods_restore_cleanly() -> None:
    processor = _processor()
    session = _ExactPromptContextSession(processor, PromptContextStats())
    patched = (
        "_get_events_time_range",
        "prepare_context_sequence",
        "get_prompts",
        "pad_prompts",
        "add_predicted_tokens_to_context",
    )
    assert all(name not in processor.__dict__ for name in patched)

    with session.install():
        assert all(name in processor.__dict__ for name in patched)

    assert all(name not in processor.__dict__ for name in patched)


def test_process_hook_is_main_only_and_restores_class_method(monkeypatch) -> None:
    processor = _processor()
    metadata = {}
    processor.profiler = SimpleNamespace(
        set_metadata=lambda **values: metadata.update(values)
    )
    in_context, out_context = _contexts()
    calls = []

    def fake_generate(instance, *args, **kwargs):
        label = kwargs.get("profile_label")
        calls.append(label)
        if label == "main_generation":
            instance._get_events_time_range([0.0, 10.0], 0.0, 11.0)
            prompts = list(instance.get_prompts(in_context, out_context))
            instance.pad_prompts(prompts)
        return label

    monkeypatch.setattr(Processor, "generate", fake_generate)
    with install_exact_prompt_context_preparation() as state:
        assert Processor.generate(processor, profile_label="timing_context") == (
            "timing_context"
        )
        assert Processor.generate(processor, profile_label="main_generation") == (
            "main_generation"
        )

    assert Processor.generate is fake_generate
    assert calls == ["timing_context", "main_generation"]
    assert state.summary()["prompt_calls"] == 1
    assert metadata["prompt_context_preparation"]["scope"] == (
        "main_generation_only"
    )


def test_sequential_profile_hashes_the_exact_decoder_prompt_and_mask() -> None:
    processor = Processor.__new__(Processor)
    processor.inference_runtime = object()
    processor.decode_session_state = None
    processor._new_decode_session_state = lambda: object()
    processor.profiler = SimpleNamespace(enabled=True, sync=lambda: None)
    processor.tokenizer = SimpleNamespace(pad_id=0)
    processor.types_first = True
    processor.lookback_time = 100.0
    processor.lookahead_time = 100.0
    processor.do_song_position_embed = False
    processor.prepare_frames = lambda frames: frames
    processor.prepare_context_sequences = lambda contexts, *args: contexts
    prompt = torch.tensor([[0, 101, 7, 1, 102]], dtype=torch.long)
    processor.get_prompts = lambda *args: (prompt, None)
    processor.pad_prompts = lambda prompts: (prompts, prompt.shape[1])
    model_inputs = {}

    def model_generate(kwargs, **unused):
        model_inputs.update(kwargs)
        return torch.tensor([[0, 101, 7, 1, 102, 8, 9]]), None

    processor.model_generate = model_generate
    processor._record_generation_stats = lambda stats: None
    processor._generated_token_ids = lambda result, original: [8, 9]
    records = []
    processor._record_generation_profile = lambda **values: records.append(values)
    processor.add_predicted_tokens_to_context = lambda *args: None
    out_context = [{"finished": False, "context_type": ContextType.MAP}]

    Processor.generate_sequential(
        processor,
        sequences=(torch.tensor([[1.0]]), torch.tensor([0.0]), 1000.0),
        in_context=[],
        out_context=out_context,
        model_kwargs={},
        req_special_tokens=[],
        verbose=False,
        profile_label="main_generation",
    )

    assert model_inputs["decoder_input_ids"] is prompt
    assert torch.equal(model_inputs["decoder_attention_mask"], prompt.ne(0))
    assert len(records) == 1
    assert records[0]["prompt_width"] == prompt.shape[1]
    assert records[0]["prompt_tokens_per_sample"] == [4]
    assert records[0]["prompt_sha256_per_sample"] == [
        hashlib.sha256(prompt.numpy().tobytes()).hexdigest()
    ]
    assert records[0]["prompt_wall_seconds"] >= 0.0


def test_unsupported_modes_and_invalid_ranges_fail_loudly() -> None:
    processor = _processor()
    processor.parallel = True
    with pytest.raises(ValueError, match="sequential"):
        _ExactPromptContextSession(processor, PromptContextStats())
    processor.parallel = False
    processor.cfg_scale = 1.5
    with pytest.raises(ValueError, match="cfg_scale=1"):
        _ExactPromptContextSession(processor, PromptContextStats())
    processor.cfg_scale = 1.0
    processor.num_beams = 2
    with pytest.raises(ValueError, match="num_beams=1"):
        _ExactPromptContextSession(processor, PromptContextStats())

    session = _ExactPromptContextSession(_processor(), PromptContextStats())
    with pytest.raises(ValueError, match="sorted"):
        session.get_events_time_range(session.processor, [0.0, 2.0, 1.0], 0, 3)
    with pytest.raises(ValueError, match="must not precede"):
        session.get_events_time_range(session.processor, [0.0, 1.0], 2, 1)


def test_runner_is_strict_fp32_only_and_has_no_hybrid_dependency(monkeypatch) -> None:
    source = (ROOT / "utils/run_strict_fp32_prompt_context.py").read_text(
        encoding="utf-8"
    )
    lowered = source.lower()
    assert "nvidia_tf32_override" in lowered
    for forbidden in ("int8", "fp16", "weight_only", "run_k1", "run_k4"):
        assert forbidden not in lowered

    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    with pytest.raises(RuntimeError, match="NVIDIA_TF32_OVERRIDE=0"):
        run_strict_fp32_prompt_context.run()

    run_strict_fp32_prompt_context._require_strict_fp32_cli(
        ["precision=fp32", "inference_engine=optimized"]
    )
    with pytest.raises(RuntimeError, match="explicit overrides"):
        run_strict_fp32_prompt_context._require_strict_fp32_cli(
            ["precision=fp32", "inference_engine=v32"]
        )
    with pytest.raises(RuntimeError, match="explicit overrides"):
        run_strict_fp32_prompt_context._require_strict_fp32_cli(
            ["inference_engine=optimized"]
        )


def test_default_processor_import_stays_cold_and_quiet() -> None:
    script = (
        "import sys; import osuT5.osuT5.inference.processor; "
        "assert 'osuT5.osuT5.inference.optimized.scout.prompt_context' "
        "not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
    )
