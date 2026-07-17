from __future__ import annotations

import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
    StoppingCriteriaList,
)

from osuT5.osuT5.inference.optimized.kernels.sampling_tail_cuda_graph import (
    sampling_tail_cuda_graph_candidate_context,
    sampling_tail_cuda_graph_hits,
    sampling_tail_cuda_graph_requested,
)
from osuT5.osuT5.inference.optimized.scout.device_sequence_state import (
    BatchOneSequenceState,
    BatchOneStoppingPolicy,
    baseline_glue_v2_candidate_context,
)
from osuT5.osuT5.inference.optimized.single.decode_loop import (
    _split_sampling_warpers,
)


def test_sampling_tail_context_is_cold_by_default() -> None:
    assert sampling_tail_cuda_graph_requested() is False
    assert sampling_tail_cuda_graph_hits() == 0
    with sampling_tail_cuda_graph_candidate_context():
        assert sampling_tail_cuda_graph_requested() is True
    assert sampling_tail_cuda_graph_requested() is False


def test_split_sampling_warpers_keeps_order() -> None:
    processors = LogitsProcessorList(
        [
            TemperatureLogitsWarper(0.9),
            TopPLogitsWarper(0.9),
            TemperatureLogitsWarper(1.1),
        ]
    )
    eager, sampling = _split_sampling_warpers(processors)
    assert list(eager) == []
    assert [type(p).__name__ for p in sampling] == [
        "TemperatureLogitsWarper",
        "TopPLogitsWarper",
        "TemperatureLogitsWarper",
    ]


def test_pinned_eos_matches_eager_policy() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    policy = BatchOneStoppingPolicy.from_transformers(
        StoppingCriteriaList(
            [
                MaxLengthCriteria(max_length=6),
                EosTokenCriteria(torch.tensor([7], dtype=torch.long, device=device)),
            ]
        ),
        device=device,
    )
    state = BatchOneSequenceState.allocate(
        torch.tensor([[1, 2]], dtype=torch.long, device=device),
        stopping_policy=policy,
        pinned_eos=True,
    )
    assert state.pinned_eos is not None
    _, stopped = state.append(torch.tensor([4], dtype=torch.long, device=device))
    assert stopped is False
    _, stopped = state.append(torch.tensor([7], dtype=torch.long, device=device))
    assert stopped is True


def test_baseline_glue_context_owns_decode_kwargs(monkeypatch) -> None:
    from osuT5.osuT5.inference.optimized.single import engine

    captured: dict[str, object] = {}

    def fake_generate(*args, **kwargs):
        captured.update(kwargs)
        return torch.tensor([[1, 2, 3]])

    monkeypatch.setattr(engine, "active_prefix_decode_generate", fake_generate)
    with baseline_glue_v2_candidate_context() as activation:
        engine.active_prefix_decode_generate(None)
    assert activation.calls == 1
    assert captured["preallocated_batch1_state"] is True
    assert captured["pinned_eos_flag"] is True
    assert captured["sampling_tail_cuda_graph"] is True
