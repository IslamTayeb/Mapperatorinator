from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.inference.processor import Processor
from osuT5.osuT5.tokenizer import ContextType
from osuT5.osuT5.event import Event, EventType
from osuT5.osuT5.inference.optimized.scout.prompt_context import (
    PromptContextStats,
    SCOUT_VERSION,
    _ExactPromptContextSession,
    install_exact_prompt_context_preparation,
)
from utils.run_k1_int8_fp16_cross_prompt_context import (
    COMPOSITION_VERSION,
    _enrich_evidence,
)
from utils.validate_prompt_context_reciprocal import validate


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


def _contexts():
    in_context = [
        {
            "context_type": ContextType.TIMING,
            "tokens": torch.tensor([[10, 11, 12, 13]]),
            "add_type": True,
            "add_class": False,
        }
    ]
    out_context = [
        {
            "context_type": ContextType.MAP,
            "tokens": torch.tensor([[20, 21, 22]]),
            "class": torch.tensor([[30, 31]]),
            "extra_special_tokens": torch.tensor([[32]]),
            "pre_tokens": torch.tensor([[40, 41]]),
            "add_type": True,
            "add_class": True,
        }
    ]
    return in_context, out_context


@pytest.mark.parametrize("center_pad_decoder", [False, True])
def test_preallocated_prompt_matches_framework_exactly(center_pad_decoder) -> None:
    processor = _processor(center_pad_decoder=center_pad_decoder)
    in_context, out_context = _contexts()
    expected_cond, expected_uncond = Processor.get_prompts(
        processor,
        in_context,
        out_context,
    )
    expected, expected_width = Processor.pad_prompts(
        processor,
        [expected_cond, expected_uncond],
    )
    session = _ExactPromptContextSession(processor, PromptContextStats())

    with session.install():
        actual_cond, actual_uncond = processor.get_prompts(in_context, out_context)
        actual, actual_width = processor.pad_prompts([actual_cond, actual_uncond])

    assert actual_uncond is None
    assert actual_width == expected_width
    assert torch.equal(actual[0], expected[0])
    assert actual[0].is_contiguous()


def test_prompt_buffer_is_reused_and_retry_matches_framework() -> None:
    processor = _processor(tgt_seq_len=24)
    in_context, out_context = _contexts()
    in_context[0]["tokens"] = torch.arange(12).unsqueeze(0)
    out_context[0]["tokens"] = torch.arange(12, 24).unsqueeze(0)
    expected, _ = Processor.get_prompts(processor, in_context, out_context)
    session = _ExactPromptContextSession(processor, PromptContextStats())

    first, _ = session.get_prompts(processor, in_context, out_context)
    first_ptr = first.data_ptr()
    second, _ = session.get_prompts(processor, in_context, out_context)

    assert first_ptr == second.data_ptr()
    assert torch.equal(second, expected)
    assert session.stats.prompt_retries == 6


def test_binary_search_range_is_exact_and_validates_appends() -> None:
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


def test_cached_vectorized_context_encoding_matches_framework_exactly() -> None:
    processor = _processor()
    processor.miliseconds_per_sequence = 1000.0
    processor.max_pre_token_len = -1
    processor.add_kiai_special_token = False
    processor.add_kiai = False
    processor.add_sv_special_token = False
    processor.add_sv = False
    processor.add_mania_sv = False
    processor.add_song_position_token = False
    ranges = {
        EventType.TIME_SHIFT: SimpleNamespace(min_value=0, max_value=100),
        EventType.SNAPPING: SimpleNamespace(min_value=0, max_value=16),
    }
    starts = {EventType.TIME_SHIFT: 3, EventType.SNAPPING: 104}

    def encode(event):
        item = ranges[event.type]
        if not item.min_value <= event.value <= item.max_value:
            raise ValueError("out of range")
        return starts[event.type] + event.value - item.min_value

    processor.tokenizer.event_range = ranges
    processor.tokenizer.event_start = starts
    processor.tokenizer.encode = encode
    context = {
        "context_type": ContextType.MAP,
        "events": [
            Event(EventType.TIME_SHIFT, 1010),
            Event(EventType.SNAPPING, 2),
            Event(EventType.TIME_SHIFT, 1500),
        ],
        "event_times": [1010.0, 1010.0, 1500.0],
        "add_pre_tokens": False,
    }
    expected = Processor.prepare_context_sequence(processor, context, 1000.0)
    session = _ExactPromptContextSession(processor, PromptContextStats())

    actual = session.prepare_context_sequence(processor, context, 1000.0)

    assert torch.equal(actual["tokens"], expected["tokens"])
    assert actual["extra_special_events"] == expected["extra_special_events"]
    assert session.stats.cached_static_events == 1
    assert session.stats.dynamic_time_events == 2


def test_context_cache_fails_loudly_on_unowned_mutation() -> None:
    processor = _processor()
    processor.miliseconds_per_sequence = 1000.0
    processor.max_pre_token_len = -1
    processor.add_kiai_special_token = False
    processor.add_kiai = False
    processor.add_sv_special_token = False
    processor.add_sv = False
    processor.add_mania_sv = False
    processor.add_song_position_token = False
    processor.tokenizer.event_range = {
        EventType.TIME_SHIFT: SimpleNamespace(min_value=0, max_value=100)
    }
    processor.tokenizer.event_start = {EventType.TIME_SHIFT: 3}
    processor.tokenizer.encode = lambda event: 3 + event.value
    context = {
        "context_type": ContextType.MAP,
        "events": [Event(EventType.TIME_SHIFT, 10)],
        "event_times": [10.0],
        "add_pre_tokens": False,
    }
    session = _ExactPromptContextSession(processor, PromptContextStats())
    session._encoding_state(context)
    context["events"].append(Event(EventType.TIME_SHIFT, 20))
    context["event_times"].append(20.0)

    with pytest.raises(RuntimeError, match="without prompt-context ownership"):
        session._encoding_state(context)


def test_binary_search_range_fails_loudly_for_invalid_inputs() -> None:
    session = _ExactPromptContextSession(_processor(), PromptContextStats())
    with pytest.raises(ValueError, match="sorted"):
        session.get_events_time_range(session.processor, [0.0, 2.0, 1.0], 0, 3)
    with pytest.raises(ValueError, match="must not precede"):
        session.get_events_time_range(session.processor, [0.0, 1.0], 2, 1)
    with pytest.raises(TypeError, match="event_times to be a list"):
        session.get_events_time_range(session.processor, (0.0, 1.0), 0, 1)


def test_instance_methods_restore_cleanly() -> None:
    processor = _processor()
    session = _ExactPromptContextSession(processor, PromptContextStats())
    assert "get_prompts" not in processor.__dict__

    with session.install():
        assert "get_prompts" in processor.__dict__
        assert "pad_prompts" in processor.__dict__
        assert "_get_events_time_range" in processor.__dict__

    assert "get_prompts" not in processor.__dict__
    assert "pad_prompts" not in processor.__dict__
    assert "_get_events_time_range" not in processor.__dict__


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
            assert instance._get_events_time_range([0.0, 10.0], 0.0, 11.0) == (
                0,
                2,
            )
            prompts = list(instance.get_prompts(in_context, out_context))
            instance.pad_prompts(prompts)
        return label

    monkeypatch.setattr(Processor, "generate", fake_generate)
    with install_exact_prompt_context_preparation() as state:
        assert Processor.generate(processor, profile_label="timing_context") \
            == "timing_context"
        assert Processor.generate(processor, profile_label="main_generation") \
            == "main_generation"

    assert Processor.generate is fake_generate
    assert calls == ["timing_context", "main_generation"]
    assert state.summary()["prompt_calls"] == 1
    assert metadata["prompt_context_preparation"]["main_generate_calls"] == 1


def test_rejects_unsupported_runtime_modes() -> None:
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


def test_enriches_selected_runtime_evidence(tmp_path) -> None:
    path = tmp_path / "init.json"
    base = (
        "k4-split-kv-mixed-weight-shared-rope-k1-remainder-int8-mlp-"
        "fp16_packed_projections-v1"
    )
    path.write_text(json.dumps({"combined_runtime": base}), encoding="utf-8")
    summary = {
        "version": SCOUT_VERSION,
        "scope": "main_generation_only",
        "exactness_required": True,
    }

    _enrich_evidence(path, summary)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["combined_runtime"] == COMPOSITION_VERSION
    assert payload["prompt_context_incremental_control"] == base
    assert payload["prompt_context_preparation"] == summary


def test_default_processor_import_does_not_load_prompt_scout() -> None:
    script = (
        "import sys; import osuT5.osuT5.inference.processor; "
        "assert 'osuT5.osuT5.inference.optimized.scout.prompt_context' "
        "not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def _write_reciprocal_profile(tmp_path, role, *, candidate: bool) -> None:
    output = tmp_path / role
    output.mkdir()
    profile = output / "beatmap.osu.profile.json"
    prompt_wall = 0.12 if candidate else 0.43
    main_stage = 18.70 if candidate else 19.05
    complete = 27.70 if candidate else 28.05
    metadata = {}
    if candidate:
        metadata["prompt_context_preparation"] = {
            "scope": "main_generation_only",
            "exactness_required": True,
            "prompt_calls": 2,
            "range_queries": 6,
        }
    rows = []
    for index in range(2):
        rows.append(
            {
                "profile_label": "main_generation",
                "sequence_index": index,
                "frame_time_ms": float(index * 1000),
                "prompt_width": 10 + index,
                "prompt_tokens_per_sample": [10 + index],
                "prompt_sha256_per_sample": [str(index) * 64],
                "generated_token_ids": [4, 5, 6 + index],
                "generated_tokens": 4147,
                "model_elapsed_seconds": 8.9,
                "prompt_wall_seconds": prompt_wall / 2,
                "trim_lookback": index > 0,
                "trim_lookahead": index == 0,
            }
        )
    payload = {
        "metadata": metadata,
        "stages": [
            {
                "name": "validate_inputs",
                "started_at_perf_counter_seconds": 100.0,
                "finished_at_perf_counter_seconds": 100.01,
                "wall_seconds": 0.01,
            },
            {
                "name": "main_generation",
                "started_at_perf_counter_seconds": 108.0,
                "finished_at_perf_counter_seconds": 108.0 + main_stage,
                "wall_seconds": main_stage,
            },
            {
                "name": "write_osu",
                "started_at_perf_counter_seconds": 100.0 + complete - 0.01,
                "finished_at_perf_counter_seconds": 100.0 + complete,
                "wall_seconds": 0.01,
            },
        ],
        "generation": rows,
    }
    profile.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / f"{role}.profile-path.txt").write_text(
        str(profile),
        encoding="utf-8",
    )


def test_reciprocal_gate_accepts_exact_prompt_and_outer_wall_saving(tmp_path) -> None:
    for role in ("baseline_first", "baseline_second"):
        _write_reciprocal_profile(tmp_path, role, candidate=False)
    for role in ("candidate_first", "candidate_second"):
        _write_reciprocal_profile(tmp_path, role, candidate=True)

    report = validate(tmp_path)

    assert report["gates"]["promotion_pass"] is True
    assert report["parity"]["prompt_and_tokens_exact"] is True
    assert report["metrics"]["median_savings"]["prompt_wall_seconds"] == pytest.approx(
        0.31
    )


def test_reciprocal_gate_rejects_prompt_drift(tmp_path) -> None:
    for role in ("baseline_first", "baseline_second"):
        _write_reciprocal_profile(tmp_path, role, candidate=False)
    for role in ("candidate_first", "candidate_second"):
        _write_reciprocal_profile(tmp_path, role, candidate=True)
    path = tmp_path / "candidate_first" / "beatmap.osu.profile.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["generation"][0]["prompt_sha256_per_sample"] = ["f" * 64]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="not exact"):
        validate(tmp_path)
