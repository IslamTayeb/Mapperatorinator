from __future__ import annotations

from contextlib import contextmanager
import json

import pytest

from osuT5.osuT5.inference import profiler as profiler_module
from osuT5.osuT5.inference.profiler import InferenceProfiler
from osuT5.osuT5.inference.processor import Processor
from osuT5.osuT5 import runtime_profiling


def test_generation_summary_aggregates_model_time_and_tokens(tmp_path):
    profiler = InferenceProfiler(enabled=True)
    profiler.record_generation(
        profile_label="main_generation",
        context_type="MAP",
        wall_seconds=1.2,
        model_elapsed_seconds=1.0,
        generated_tokens=100,
        generated_token_ids=[1, 2],
    )
    profiler.record_generation(
        profile_label="main_generation",
        context_type="MAP",
        wall_seconds=2.2,
        model_elapsed_seconds=2.0,
        generated_tokens=200,
        generated_token_ids=[3, 4],
    )

    summary = profiler._summary()["generation_by_label"]["main_generation"]

    assert summary == {
        "records": 2,
        "wall_seconds": 3.4000000000000004,
        "model_elapsed_seconds": 3.0,
        "generated_tokens": 300,
        "tokens_per_second": 100.0,
    }

    output = profiler.write(tmp_path / "profile.json")
    assert output is not None
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["generation"][0]["generated_token_ids"] == [1, 2]
    assert "torch_profiles" not in payload


def test_disabled_profiler_does_not_write(tmp_path):
    profiler = InferenceProfiler()
    profiler.record_generation(profile_label="main_generation")

    assert profiler.write(tmp_path / "profile.json") is None
    assert not (tmp_path / "profile.json").exists()


def test_detail_ranges_require_profile_and_restore_after_stage(monkeypatch):
    seen = []

    @contextmanager
    def fake_range(name):
        seen.append(("enter", name))
        yield
        seen.append(("exit", name))

    monkeypatch.setattr(profiler_module, "profile_range", fake_range)
    profiler = InferenceProfiler(
        enabled=True,
        detail_ranges=True,
        pass_kind="nsys_node",
    )
    with profiler.stage("main_generation"):
        pass

    assert seen == [
        ("enter", "stage.main_generation"),
        ("exit", "stage.main_generation"),
    ]
    assert profiler.metadata["authoritative_performance"] is False
    assert profiler.stages[0]["finished_at_perf_counter_seconds"] >= (
        profiler.stages[0]["started_at_perf_counter_seconds"]
    )

    with pytest.raises(ValueError, match="requires profile_inference"):
        InferenceProfiler(enabled=False, detail_ranges=True)
    with pytest.raises(ValueError, match="requires profile_detail_ranges"):
        InferenceProfiler(enabled=True, cuda_capture=True)


def test_only_untraced_control_is_authoritative():
    control = InferenceProfiler(enabled=True, pass_kind="untraced_control")
    traced = InferenceProfiler(enabled=True, pass_kind="nsys_graph")

    assert control.metadata["authoritative_performance"] is True
    assert traced.metadata["authoritative_performance"] is False
    with pytest.raises(ValueError, match="profile_pass_kind"):
        InferenceProfiler(enabled=True, pass_kind="traced-but-unclassified")


def test_nested_generation_context_cannot_disable_outer_detail_ranges():
    assert not runtime_profiling.detail_ranges_enabled()

    with runtime_profiling.detail_ranges_context(True):
        assert runtime_profiling.detail_ranges_enabled()
        with runtime_profiling.generation_profile_context(detail_ranges=False):
            assert runtime_profiling.detail_ranges_enabled()
        assert runtime_profiling.detail_ranges_enabled()

    assert not runtime_profiling.detail_ranges_enabled()


def test_cuda_capture_api_is_balanced_and_fail_loud(monkeypatch):
    calls = []

    class FakeCudart:
        @staticmethod
        def cudaProfilerStart():
            calls.append("start")

        @staticmethod
        def cudaProfilerStop():
            calls.append("stop")

    monkeypatch.setattr(profiler_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(profiler_module.torch.cuda, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(profiler_module.torch.cuda, "cudart", lambda: FakeCudart())
    monkeypatch.setattr(
        profiler_module.torch.cuda.nvtx,
        "range_push",
        lambda name: calls.append(("push", name)),
    )
    monkeypatch.setattr(
        profiler_module.torch.cuda.nvtx,
        "range_pop",
        lambda: calls.append("pop"),
    )
    profiler = InferenceProfiler(
        enabled=True,
        detail_ranges=True,
        cuda_capture=True,
        pass_kind="nsys_graph",
    )

    profiler.start_generation_capture()
    with pytest.raises(RuntimeError, match="already active"):
        profiler.start_generation_capture()
    profiler.stop_generation_capture()
    with pytest.raises(RuntimeError, match="not active"):
        profiler.stop_generation_capture()

    assert calls == [
        "sync",
        "start",
        ("push", "mapperatorinator.roi.inference_generation"),
        "sync",
        "pop",
        "stop",
    ]


def test_processor_records_specialized_dispatch_capture_hits():
    processor = Processor.__new__(Processor)
    processor.profiler = InferenceProfiler(enabled=True)
    processor.precision = "fp32"
    processor.model = object()
    processor.parallel = False
    hits = {
        "native_q1_rope_cache_self_attention": 12,
        "native_q1_self_attention": 0,
        "q1_bmm_cross_attention": 12,
        "native_cross_mlp_tail": 12,
    }

    processor._record_generation_profile(
        profile_label="main_generation",
        mode="main",
        context_type="MAP",
        wall_seconds=1.0,
        stats={
            "generated_tokens": 4,
            "elapsed_seconds": 0.5,
            "tokens_per_second": 8.0,
            "optimized_dispatch_capture_hits": hits,
        },
    )

    assert processor.profiler.generation[0]["optimized_dispatch_capture_hits"] == hits
