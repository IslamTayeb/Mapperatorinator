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
    _capture_graph,
    _processor_descriptor,
    _require_recaptured_tensor,
    _representatives_by_vocab_size,
    _top_p_kwargs,
    benchmark_existing_tail,
    install_vocab_sampling_observer,
    summarize,
)


def _descriptor(top_p: float = 0.9):
    return (
        (
            "TopPLogitsWarper",
            (
                ("top_p", top_p),
                ("filter_value", -torch.inf),
                ("min_tokens_to_keep", 1),
            ),
        ),
    )


def _sample(
    *,
    nucleus_size: int = 2,
    vocab_size: int = 4,
    top_p: float = 0.9,
) -> VocabSample:
    pre = torch.arange(
        vocab_size,
        0,
        -1,
        dtype=torch.float32,
    ).reshape(1, vocab_size)
    post = pre.clone()
    post[:, nucleus_size:] = -torch.inf
    return VocabSample(
        raw_logits=pre.clone(),
        pre_top_p=pre,
        post_top_p=post,
        threshold=0.25,
        selected_token=0,
        processor_descriptor=_descriptor(top_p),
        source="k4_graph_final_step",
    )


def _component(worst_by_vocab_size: dict[int, float]) -> dict:
    return {
        "observed_vocab_sizes": sorted(worst_by_vocab_size),
        "observed_score_shapes": [
            [1, vocab_size] for vocab_size in sorted(worst_by_vocab_size)
        ],
        "representatives": [
            {
                "representative": label,
                "vocab_size": vocab_size,
                "score_shape": [1, vocab_size],
                "combined_ms_per_step_worst": worst_by_vocab_size[vocab_size],
            }
            for vocab_size in sorted(worst_by_vocab_size)
            for label in ("min", "p50", "p95", "max")
        ],
        "worst_combined_ms_per_step_by_vocab_size": {
            str(vocab_size): value
            for vocab_size, value in sorted(worst_by_vocab_size.items())
        },
        "worst_combined_ms_per_step": max(worst_by_vocab_size.values()),
    }


def _buffers(sample: VocabSample) -> dict:
    return {
        "raw_logits": sample.raw_logits,
        "pre_top_p": sample.pre_top_p,
        "post_top_p": sample.post_top_p,
        "threshold": torch.tensor(sample.threshold),
        "token": torch.tensor([sample.selected_token]),
        "processor_descriptor": sample.processor_descriptor,
    }


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


def test_observer_registers_k1_remainder_parent_as_sampling_alias() -> None:
    observer = VocabSamplingObserver(max_samples=2)
    parent = object()
    remainder_parent = object()
    sample = _sample()
    buffers = {
        **_buffers(sample),
        "physical_length": torch.tensor([8]),
        "logical_length": torch.tensor([64]),
    }
    observer.graph_buffers[id(parent)] = buffers

    observer.register_graph_alias(parent, remainder_parent)
    observer.observe_graph_replay(parent)
    observer.observe_graph_replay(remainder_parent)

    assert observer.graph_buffers[id(remainder_parent)] is buffers
    assert observer.total_graph_observations == 2
    assert [stored.source for stored in observer.samples] == [
        "k4_graph_final_step",
        "k4_graph_final_step",
    ]


def test_observer_rejects_conflicting_graph_alias() -> None:
    observer = VocabSamplingObserver(max_samples=1)
    parent = object()
    alias = object()
    observer.graph_buffers[id(parent)] = {"source": "parent"}
    observer.graph_buffers[id(alias)] = {"source": "alias"}

    with pytest.raises(RuntimeError, match="registered differently"):
        observer.register_graph_alias(parent, alias)


def test_capture_patch_registers_block_and_remainder_parent_buffers(monkeypatch) -> None:
    observer = VocabSamplingObserver(max_samples=1)
    parent = object()
    remainder_parent = object()
    sample = _sample()
    state = SimpleNamespace(
        physical_length=torch.tensor([8]),
        logical_length=torch.tensor([64]),
    )

    def fake_capture(*_args, **_kwargs):
        observer.current = {
            name: value
            for name, value in _buffers(sample).items()
            if name != "processor_descriptor"
        }
        observer.current_descriptor = sample.processor_descriptor
        return SimpleNamespace(
            parent=parent,
            remainder_parent=remainder_parent,
            state=state,
        )

    monkeypatch.setattr(k8_runtime, "_capture_k8_entry", fake_capture)
    with install_vocab_sampling_observer(observer):
        entry = k8_runtime._capture_k8_entry()

    assert entry.parent is parent
    assert observer.graph_buffers[id(parent)] is observer.graph_buffers[
        id(remainder_parent)
    ]


def test_observer_reserves_coverage_for_a_late_vocab_at_the_sample_cap() -> None:
    observer = VocabSamplingObserver(max_samples=2)
    observer._record(_buffers(_sample(nucleus_size=1)), source="eager_remainder")
    observer._record(_buffers(_sample(nucleus_size=2)), source="eager_remainder")
    observer._record(
        _buffers(_sample(nucleus_size=3, vocab_size=6)),
        source="eager_remainder",
    )

    assert {sample.vocab_size for sample in observer.samples} == {4, 6}
    assert observer.observed_vocab_counts == {4: 2, 6: 1}
    assert observer.dropped_observations == 1


def test_observer_fails_loudly_when_cap_cannot_represent_every_vocab() -> None:
    observer = VocabSamplingObserver(max_samples=1)
    observer._record(_buffers(_sample()), source="eager_remainder")

    with pytest.raises(RuntimeError, match="cannot retain every observed shape"):
        observer._record(
            _buffers(_sample(vocab_size=6)),
            source="eager_remainder",
        )


def test_observer_rebalances_bounded_samples_across_vocab_sizes() -> None:
    observer = VocabSamplingObserver(max_samples=4)
    for nucleus_size in (1, 2, 3, 4):
        observer._record(
            _buffers(_sample(nucleus_size=nucleus_size)),
            source="eager_remainder",
        )
    for nucleus_size in (2, 5):
        observer._record(
            _buffers(_sample(nucleus_size=nucleus_size, vocab_size=6)),
            source="eager_remainder",
        )

    stored_counts = {
        vocab_size: sum(sample.vocab_size == vocab_size for sample in observer.samples)
        for vocab_size in (4, 6)
    }
    assert stored_counts == {4: 2, 6: 2}


def test_observer_rejects_changed_top_p_descriptor_even_at_sample_cap() -> None:
    observer = VocabSamplingObserver(max_samples=1)
    observer._record(_buffers(_sample(top_p=0.9)), source="eager_remainder")

    with pytest.raises(RuntimeError, match="multiple TopP configurations"):
        observer._record(
            _buffers(_sample(top_p=0.95)),
            source="eager_remainder",
        )


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


def test_top_p_descriptor_preserves_every_constructor_parameter() -> None:
    warper = TopPLogitsWarper(
        top_p=0.75,
        filter_value=-123.0,
        min_tokens_to_keep=2,
    )
    descriptor = dict(_processor_descriptor(SimpleNamespace(processors=[warper]))[0][1])

    assert _top_p_kwargs(descriptor) == {
        "top_p": 0.75,
        "filter_value": -123.0,
        "min_tokens_to_keep": 2,
    }


def test_capture_graph_replays_once_before_returning_static_output(monkeypatch) -> None:
    from contextlib import contextmanager

    output = torch.tensor([-1])
    state = {"capturing": False, "replays": 0}

    class FakeGraph:
        def replay(self):
            state["replays"] += 1
            output.fill_(7)

    @contextmanager
    def fake_capture(_graph):
        state["capturing"] = True
        try:
            yield
        finally:
            state["capturing"] = False

    def callable_():
        if not state["capturing"]:
            output.fill_(3)
        return output

    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(torch.cuda, "graph", fake_capture)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    graph, actual = _capture_graph(callable_, warmup=1)

    assert isinstance(graph, FakeGraph)
    assert state["replays"] == 1
    assert actual.item() == 7


def test_recapture_mismatch_reports_quantitative_diagnostics() -> None:
    descriptor = {
        "top_p": 0.9,
        "filter_value": -torch.inf,
        "min_tokens_to_keep": 1,
    }
    with pytest.raises(
        RuntimeError,
        match=(
            r"mismatch_count=2 finite_mask_mismatch_count=1 "
            r"finite_max_abs=1.0.*min_tokens_to_keep"
        ),
    ):
        _require_recaptured_tensor(
            "TopP output",
            torch.tensor([[1.0, 2.0, -torch.inf]]),
            torch.tensor([[1.0, 3.0, 4.0]]),
            descriptor=descriptor,
        )


def test_summary_computes_fixed_main_and_request_zero_cost_ceilings() -> None:
    observer = VocabSamplingObserver(max_samples=8)
    observer.samples = [_sample(nucleus_size=value) for value in (1, 2, 3, 4)]
    observer.observed_vocab_counts = {4: 4}
    observer.total_graph_observations = 4
    component = _component({4: 0.15})

    report = summarize(
        observer,
        component,
        fixed_main_steps=8294,
        fixed_timing_steps=821,
        mixed_projection_ms_per_step=0.01,
        promotion_threshold_seconds=1.5,
    )

    ceiling = report["fixed_work_ceiling"]
    assert report["schema_version"] == 2
    assert ceiling["ideal_main_top_p_sampling_seconds"] == pytest.approx(1.2441)
    assert ceiling["ideal_main_projection_seconds"] == pytest.approx(0.08294)
    assert ceiling["ideal_main_total_seconds"] == pytest.approx(1.32704)
    assert ceiling["ideal_complete_request_total_seconds"] == pytest.approx(1.4584)
    assert not ceiling["main_ceiling_clears_threshold"]
    assert report["decision"] == "stop_below_main_component_gate"
    assert report["distribution"]["vocab_sizes"] == [4]
    assert report["distribution"]["nucleus_size_p95"] == 4


def test_representatives_are_selected_independently_for_every_vocab_size() -> None:
    samples = [
        _sample(nucleus_size=nucleus, vocab_size=vocab_size)
        for vocab_size, nuclei in ((4, (1, 2, 3, 4)), (6, (2, 3, 5, 6)))
        for nucleus in nuclei
    ]

    representatives = _representatives_by_vocab_size(samples)

    assert list(representatives) == [4, 6]
    assert set(representatives[4]) == {"min", "p50", "p95", "max"}
    assert set(representatives[6]) == {"min", "p50", "p95", "max"}
    assert {
        sample.vocab_size
        for shape_representatives in representatives.values()
        for sample in shape_representatives.values()
    } == {4, 6}


def test_summary_supports_two_vocab_sizes_and_uses_slowest_shape() -> None:
    observer = VocabSamplingObserver(max_samples=8)
    observer.samples = [
        _sample(nucleus_size=1, vocab_size=4),
        _sample(nucleus_size=3, vocab_size=4),
        _sample(nucleus_size=2, vocab_size=6),
        _sample(nucleus_size=5, vocab_size=6),
    ]
    observer.observed_vocab_counts = {4: 2, 6: 2}
    component = _component({4: 0.12, 6: 0.20})

    report = summarize(
        observer,
        component,
        fixed_main_steps=8294,
        fixed_timing_steps=821,
        mixed_projection_ms_per_step=0.01,
        promotion_threshold_seconds=1.5,
    )

    distribution = report["distribution"]
    ceiling = report["fixed_work_ceiling"]
    assert distribution["vocab_sizes"] == [4, 6]
    assert distribution["score_shapes"] == [[1, 4], [1, 6]]
    assert distribution["samples_by_vocab_size"] == {"4": 2, "6": 2}
    assert distribution["nucleus_size_by_vocab_size"] == {
        "4": {"min": 1, "p50": 3, "p95": 3, "max": 3},
        "6": {"min": 2, "p50": 5, "p95": 5, "max": 5},
    }
    assert ceiling["existing_top_p_sampling_ms_per_step"] == 0.20
    assert ceiling["existing_top_p_sampling_worst_across_shapes_ms_per_step"] == 0.20
    assert ceiling["existing_top_p_sampling_ms_per_step_by_vocab_size"] == {
        "4": 0.12,
        "6": 0.20,
    }
    assert ceiling["worst_case_vocab_sizes"] == [6]
    assert ceiling["ceiling_cost_policy"] == (
        "charge the worst observed score-shape tail to every fixed step"
    )
    assert ceiling["ideal_main_top_p_sampling_seconds"] == pytest.approx(1.6588)


def test_summary_makes_no_claim_when_one_observed_vocab_size_is_untimed() -> None:
    observer = VocabSamplingObserver(max_samples=2)
    observer.samples = [
        _sample(vocab_size=4),
        _sample(vocab_size=6),
    ]
    observer.observed_vocab_counts = {4: 1, 6: 1}

    with pytest.raises(
        RuntimeError,
        match=r"does not cover every observed vocabulary size.*\[4, 6\]",
    ):
        summarize(
            observer,
            _component({4: 0.12}),
            fixed_main_steps=8294,
            fixed_timing_steps=821,
            mixed_projection_ms_per_step=0.01,
            promotion_threshold_seconds=1.5,
        )


@pytest.mark.parametrize(
    "failure", ["duplicate", "shape", "per_vocab_worst", "global_worst"]
)
def test_summary_rejects_inconsistent_multi_vocab_component_rows(failure) -> None:
    observer = VocabSamplingObserver(max_samples=2)
    observer.samples = [_sample(vocab_size=4), _sample(vocab_size=6)]
    observer.observed_vocab_counts = {4: 1, 6: 1}
    component = _component({4: 0.12, 6: 0.20})
    if failure == "duplicate":
        component["representatives"].append(dict(component["representatives"][0]))
        expected = "repeats a vocabulary representative"
    elif failure == "shape":
        component["representatives"][0]["score_shape"] = [1, 6]
        expected = "score shape is inconsistent"
    elif failure == "per_vocab_worst":
        component["worst_combined_ms_per_step_by_vocab_size"]["4"] = 0.13
        expected = "inconsistent with representative rows"
    else:
        component["worst_combined_ms_per_step"] = 0.21
        expected = "worst timing is inconsistent"

    with pytest.raises(RuntimeError, match=expected):
        summarize(
            observer,
            component,
            fixed_main_steps=8294,
            fixed_timing_steps=821,
            mixed_projection_ms_per_step=0.01,
            promotion_threshold_seconds=1.5,
        )


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
        observer.observed_vocab_counts = {4: 1}
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
        lambda *args, **kwargs: _component({4: 0.1}),
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
