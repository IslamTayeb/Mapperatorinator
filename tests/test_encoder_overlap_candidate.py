from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers.modeling_outputs import BaseModelOutput

from osuT5.osuT5.inference.optimized.scout import encoder_overlap
from osuT5.osuT5.inference.optimized.scout.encoder_overlap import (
    EncoderOverlapError,
    ExactMainEncoderOverlap,
    graph_manifest,
    install_encoder_material_audit,
    install_exact_main_encoder_overlap,
)


def test_overlap_stream_creation_avoids_optional_priority_range_api() -> None:
    source = Path(encoder_overlap.__file__).read_text(encoding="utf-8")

    assert "get_stream_priority_range" not in source
    assert "torch.cuda.Stream(device=device, priority=0)" in source


class _GraphState:
    def __init__(self, summary):
        self.summary = summary

    def graph_profile_summary(self):
        return self.summary


def test_graph_manifest_keeps_only_stable_address_free_shape():
    processor = SimpleNamespace(
        decode_session_state=_GraphState(
            {
                "graph_count": 2,
                "decode_replays": 50,
                "capture_seconds": 4.0,
                "buckets": {
                    "128": {
                        "graph_count": 1,
                        "decode_replays": 10,
                        "capture_seconds": 2.0,
                    },
                    "64": {
                        "graph_count": 1,
                        "decode_replays": 40,
                        "capture_seconds": 2.0,
                    },
                },
            }
        )
    )
    assert graph_manifest(processor) == {
        "graph_count": 2,
        "buckets": {"64": 1, "128": 1},
    }


@pytest.mark.parametrize(
    "summary",
    [
        {"graph_count": 0, "buckets": {}},
        {"graph_count": 2, "buckets": {"64": {"graph_count": 1}}},
        {"graph_count": 1, "buckets": {"064": {"graph_count": 1}}},
    ],
)
def test_graph_manifest_rejects_unstable_or_malformed_state(summary):
    with pytest.raises(EncoderOverlapError):
        graph_manifest(
            SimpleNamespace(decode_session_state=_GraphState(summary))
        )


def test_manifest_must_repeat_before_launch_and_must_not_change(monkeypatch):
    manager = ExactMainEncoderOverlap(stable_observations_required=2)
    processor = object()
    manager._active_label = "timing_context"
    launches = []
    aborts = []
    monkeypatch.setattr(manager, "_launch", lambda: launches.append(True))
    monkeypatch.setattr(manager, "_publish", lambda _processor: None)
    monkeypatch.setattr(manager, "abort", lambda: aborts.append(True))
    values = iter(
        [
            {"graph_count": 2, "buckets": {"64": 1, "128": 1}},
            {"graph_count": 2, "buckets": {"64": 1, "128": 1}},
            {"graph_count": 3, "buckets": {"64": 1, "128": 1, "192": 1}},
        ]
    )
    monkeypatch.setattr(encoder_overlap, "graph_manifest", lambda _processor: next(values))

    manager.after_timing_model_generate(processor)
    assert not launches
    manager.after_timing_model_generate(processor)
    assert launches == [True]
    assert manager._launch_after_timing_window == 2
    with pytest.raises(EncoderOverlapError, match="changed after overlap launch"):
        manager.after_timing_model_generate(processor)
    assert aborts == [True]


def test_guarded_manifest_growth_requires_one_barrier_per_fresh_graph(monkeypatch):
    manager = ExactMainEncoderOverlap(stable_observations_required=2)
    processor = object()
    stream = SimpleNamespace(synchronize=lambda: None)
    manager._active_label = "timing_context"
    manager._stream = stream
    enqueues = []
    monkeypatch.setattr(manager, "_enqueue_rows", enqueues.append)
    manager._launch_manifest = {"graph_count": 1, "buckets": {"64": 1}}
    manager._current_manifest = copy = manager._launch_manifest.copy()
    monkeypatch.setattr(manager, "_publish", lambda _processor: None)
    monkeypatch.setattr(
        encoder_overlap,
        "graph_manifest",
        lambda _processor: {"graph_count": 2, "buckets": {"64": 1, "128": 1}},
    )

    manager.before_timing_graph_capture()
    manager.after_timing_model_generate(processor)

    assert manager._launch_manifest == copy
    assert manager._current_manifest == {
        "graph_count": 2,
        "buckets": {"64": 1, "128": 1},
    }
    assert manager._pre_capture_barrier_count == 1
    assert manager._pending_capture_barriers == 0
    assert manager._guarded_manifest_growth[0]["capture_barriers"] == 1
    assert enqueues == [1]
    assert not manager._manifest_changed_after_launch


def test_guarded_manifest_growth_fails_if_capture_count_does_not_match(monkeypatch):
    manager = ExactMainEncoderOverlap()
    manager._active_label = "timing_context"
    manager._stream = SimpleNamespace(synchronize=lambda: None)
    manager._launch_manifest = {"graph_count": 1, "buckets": {"64": 1}}
    manager._current_manifest = manager._launch_manifest.copy()
    manager.before_timing_graph_capture()
    monkeypatch.setattr(
        encoder_overlap,
        "graph_manifest",
        lambda _processor: {"graph_count": 3, "buckets": {"64": 1, "128": 2}},
    )

    with pytest.raises(EncoderOverlapError, match="fresh capture count"):
        manager.after_timing_model_generate(object())
    assert manager._manifest_changed_after_launch


def test_registration_requires_exactly_main_then_timing():
    manager = ExactMainEncoderOverlap()
    main, timing = object(), object()
    manager.register_processor(main)
    manager.register_processor(timing)
    assert manager._main_processor is main
    assert manager._timing_processor is timing
    with pytest.raises(EncoderOverlapError, match="only main and timing"):
        manager.register_processor(object())


class _Processor:
    def __init__(self, marker):
        self.marker = marker

    def generate(self, *, sequences, generation_config, profile_label):
        return self.model_generate(
            {"inputs": sequences[0][0].unsqueeze(0)},
            marker=profile_label,
        )

    def model_generate(self, model_kwargs, **generate_kwargs):
        return model_kwargs, generate_kwargs


def test_temporary_hooks_restore_init_generate_and_model_generate():
    originals = (_Processor.__init__, _Processor.generate, _Processor.model_generate)
    frames = torch.zeros((1, 4))
    with install_exact_main_encoder_overlap(_Processor) as manager:
        manager.begin_generation = lambda *args, **kwargs: setattr(
            manager, "_active_label", kwargs["profile_label"]
        )
        manager.end_generation = lambda *args, **kwargs: setattr(
            manager, "_active_label", None
        )
        manager.inject_main = lambda _processor, kwargs: {
            **kwargs,
            "encoder_outputs": BaseModelOutput(
                last_hidden_state=torch.ones((1, 2, 3))
            ),
        }
        manager.after_timing_model_generate = lambda _processor: None
        main = _Processor("main")
        timing = _Processor("timing")
        assert manager._processors == [main, timing]
        manager._active_label = "main_generation"
        model_kwargs, marker = main.generate(
            sequences=(frames, torch.tensor([0]), 1.0),
            generation_config=object(),
            profile_label="main_generation",
        )
        assert "encoder_outputs" in model_kwargs
        assert marker == {"marker": "main_generation"}

    assert (_Processor.__init__, _Processor.generate, _Processor.model_generate) == originals


def test_material_audit_hooks_restore_init_and_generate(monkeypatch):
    originals = (_Processor.__init__, _Processor.generate)
    frames = torch.zeros((1, 4))
    with install_encoder_material_audit(_Processor) as audit:
        captured = []
        monkeypatch.setattr(
            audit,
            "capture",
            lambda processor, *, profile_label: captured.append(
                (processor.marker, profile_label)
            ),
        )
        processor = _Processor("main")
        processor.generate(
            sequences=(frames, torch.tensor([0]), 1.0),
            generation_config=object(),
            profile_label="main_generation",
        )
        assert audit._processors == [processor]
        assert captured == [("main", "main_generation")]

    assert (_Processor.__init__, _Processor.generate) == originals


@pytest.mark.parametrize("stable", [True, 0, 1, -1, 1.5])
def test_overlap_rejects_invalid_stability_threshold(stable):
    with pytest.raises(ValueError):
        ExactMainEncoderOverlap(stable_observations_required=stable)
