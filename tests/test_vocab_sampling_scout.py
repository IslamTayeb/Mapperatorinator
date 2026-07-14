from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from transformers import TopPLogitsWarper

from osuT5.osuT5.inference.optimized.single import k8_runtime
from utils import run_k4_vocab_sampling_scout
from utils.vocab_sampling_scout import (
    VocabSample,
    VocabSamplingObserver,
    benchmark_existing_tail,
    install_vocab_sampling_observer,
    summarize,
)


def _descriptor(top_p: float = 0.9):
    return (("TopPLogitsWarper", (("top_p", top_p),)),)


def _sample(*, nucleus_size: int = 2) -> VocabSample:
    pre = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    post = pre.clone()
    post[:, nucleus_size:] = -torch.inf
    return VocabSample(
        raw_logits=pre.clone(),
        pre_top_p=pre,
        post_top_p=post,
        threshold=0.25,
        selected_token=0,
        processor_descriptor=_descriptor(),
        source="k4_graph_final_step",
    )


def test_sample_requires_top_p_to_only_mask_scores() -> None:
    sample = _sample()
    sample.validate()
    sample.post_top_p[0, 0] += 1
    with pytest.raises(RuntimeError, match="unmodified score"):
        sample.validate()


def test_observer_records_bounded_graph_and_eager_samples() -> None:
    observer = VocabSamplingObserver(max_samples=1)
    parent = object()
    sample = _sample()
    buffers = {
        "raw_logits": sample.raw_logits,
        "pre_top_p": sample.pre_top_p,
        "post_top_p": sample.post_top_p,
        "threshold": torch.tensor(sample.threshold),
        "token": torch.tensor([sample.selected_token]),
        "processor_descriptor": sample.processor_descriptor,
        "physical_length": torch.tensor([8]),
        "logical_length": torch.tensor([64]),
    }
    observer.graph_buffers[id(parent)] = buffers

    observer.observe_graph_replay(parent)
    observer.observe_graph_replay(parent)

    assert len(observer.samples) == 1
    assert observer.total_graph_observations == 2
    assert observer.dropped_observations == 1
    assert observer.samples[0].nucleus_size == 2


def test_observer_excludes_final_graph_score_after_logical_eos() -> None:
    observer = VocabSamplingObserver(max_samples=1)
    parent = object()
    sample = _sample()
    observer.graph_buffers[id(parent)] = {
        "raw_logits": sample.raw_logits,
        "pre_top_p": sample.pre_top_p,
        "post_top_p": sample.post_top_p,
        "threshold": torch.tensor(sample.threshold),
        "token": torch.tensor([sample.selected_token]),
        "processor_descriptor": sample.processor_descriptor,
        "physical_length": torch.tensor([12]),
        "logical_length": torch.tensor([10]),
    }

    observer.observe_graph_replay(parent)

    assert observer.samples == []
    assert observer.wasted_graph_observations == 1


def test_patch_context_restores_all_callables() -> None:
    originals = (
        k8_runtime.K8LogitsProcessor.__call__,
        TopPLogitsWarper.__call__,
        k8_runtime._sample_counter,
        k8_runtime._capture_k8_entry,
        k8_runtime._ChildGraphSequence.replay,
    )
    observer = VocabSamplingObserver(max_samples=2)
    with install_vocab_sampling_observer(observer):
        patched = (
            k8_runtime.K8LogitsProcessor.__call__,
            TopPLogitsWarper.__call__,
            k8_runtime._sample_counter,
            k8_runtime._capture_k8_entry,
            k8_runtime._ChildGraphSequence.replay,
        )
        assert all(before is not after for before, after in zip(originals, patched))
    restored = (
        k8_runtime.K8LogitsProcessor.__call__,
        TopPLogitsWarper.__call__,
        k8_runtime._sample_counter,
        k8_runtime._capture_k8_entry,
        k8_runtime._ChildGraphSequence.replay,
    )
    assert restored == originals


def test_patch_context_restores_after_failure() -> None:
    original = k8_runtime._sample_counter
    with pytest.raises(RuntimeError, match="boom"):
        with install_vocab_sampling_observer(VocabSamplingObserver(max_samples=1)):
            raise RuntimeError("boom")
    assert k8_runtime._sample_counter is original


def test_patch_observes_real_eager_top_p_and_counter_sample() -> None:
    observer = VocabSamplingObserver(max_samples=1)
    warper = TopPLogitsWarper(top_p=0.9)
    processor = SimpleNamespace(processors=[warper])
    state = SimpleNamespace(counter_uniform=lambda: torch.tensor([0.25]))
    raw = torch.tensor([[4.0, 3.0, 2.0, 1.0]])

    with install_vocab_sampling_observer(observer):
        observer.observe_processor(processor, raw)
        post = warper(torch.zeros((1, 1), dtype=torch.long), raw)
        token = k8_runtime._sample_counter(post, state)

    assert len(observer.samples) == 1
    sample = observer.samples[0]
    assert sample.source == "eager_remainder"
    assert sample.selected_token == int(token.item())
    assert sample.nucleus_size == int(torch.isfinite(post).sum().item())
    assert sample.processor_descriptor == _descriptor()


def test_summary_computes_fixed_main_and_request_zero_cost_ceilings() -> None:
    observer = VocabSamplingObserver(max_samples=8)
    observer.samples = [_sample(nucleus_size=value) for value in (1, 2, 3, 4)]
    observer.total_graph_observations = 4
    component = {"worst_combined_ms_per_step": 0.15}

    report = summarize(
        observer,
        component,
        fixed_main_steps=8294,
        fixed_timing_steps=821,
        mixed_projection_ms_per_step=0.01,
        promotion_threshold_seconds=1.5,
    )

    ceiling = report["fixed_work_ceiling"]
    assert ceiling["ideal_main_top_p_sampling_seconds"] == pytest.approx(1.2441)
    assert ceiling["ideal_main_projection_seconds"] == pytest.approx(0.08294)
    assert ceiling["ideal_main_total_seconds"] == pytest.approx(1.32704)
    assert ceiling["ideal_complete_request_total_seconds"] == pytest.approx(1.4584)
    assert not ceiling["main_ceiling_clears_threshold"]
    assert report["decision"] == "stop_below_main_component_gate"
    assert report["distribution"]["nucleus_size_p95"] == 4


def test_component_benchmark_requires_cuda() -> None:
    if torch.cuda.is_available():
        pytest.skip("CPU fail-loud test requires a host without CUDA")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        benchmark_existing_tail([_sample()], warmup=1, iterations=1, rounds=1)


def test_runner_scopes_observer_k4_and_weight_candidate(monkeypatch, tmp_path) -> None:
    events = []

    @contextmanager
    def observer_context(observer):
        events.append("observer-enter")
        observer.samples = [_sample()]
        try:
            yield
        finally:
            events.append("observer-exit")

    @contextmanager
    def k4_context(*, block_size):
        events.append(("k4-enter", block_size))
        try:
            yield
        finally:
            events.append("k4-exit")

    def weight_run(config_name, overrides, output_init_json):
        events.append(("weight", config_name, overrides, output_init_json))

    monkeypatch.setattr(
        run_k4_vocab_sampling_scout, "install_vocab_sampling_observer", observer_context
    )
    monkeypatch.setattr(run_k4_vocab_sampling_scout, "install_k8_candidate", k4_context)
    monkeypatch.setattr(run_k4_vocab_sampling_scout, "run_weight_only", weight_run)
    monkeypatch.setattr(
        run_k4_vocab_sampling_scout,
        "benchmark_existing_tail",
        lambda *args, **kwargs: {"worst_combined_ms_per_step": 0.1},
    )
    output = tmp_path / "scout.json"
    report = run_k4_vocab_sampling_scout.run(
        "profile_salvalai",
        ["seed=12345"],
        output_init_json=tmp_path / "init.json",
        output_scout_json=output,
        max_samples=1,
        warmup=1,
        iterations=1,
        rounds=1,
        fixed_main_steps=8294,
        fixed_timing_steps=821,
        mixed_projection_ms_per_step=0.01,
        promotion_threshold_seconds=1.503,
    )

    assert events == [
        "observer-enter",
        ("k4-enter", 4),
        ("weight", "profile_salvalai", ["seed=12345"], tmp_path / "init.json"),
        "k4-exit",
        "observer-exit",
    ]
    assert output.is_file()
    assert report["production_wiring_changed"] is False
