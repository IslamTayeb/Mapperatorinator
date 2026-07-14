from __future__ import annotations

import pytest
import torch
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
)

from osuT5.osuT5.inference.optimized.scout.device_tail import (
    DeviceSequenceState,
    GraphCompatibleStoppingPolicy,
    fixed_block_tail,
    update_one_token_model_inputs,
)


class _IdentityProcessor:
    def __call__(self, input_ids, scores):
        del input_ids
        return scores


def _criteria(*, eos=(9,), max_length: int = 8) -> StoppingCriteriaList:
    return StoppingCriteriaList(
        [
            MaxLengthCriteria(max_length=max_length),
            EosTokenCriteria(eos_token_id=torch.tensor(eos, dtype=torch.long)),
        ]
    )


def _policy(
    state: DeviceSequenceState,
    *,
    criteria: StoppingCriteriaList | None = None,
) -> GraphCompatibleStoppingPolicy:
    return GraphCompatibleStoppingPolicy.from_transformers(
        _criteria(
            eos=tuple(int(value) for value in state.eos_token_ids.cpu().tolist()),
            max_length=state.max_length,
        )
        if criteria is None
        else criteria,
        state=state,
    )


def _logits(token: int, *, vocab: int = 11) -> torch.Tensor:
    values = torch.full((1, vocab), -100.0)
    values[:, token] = 100.0
    return values


def test_device_sequence_state_records_first_eos_and_pads_future_steps() -> None:
    state = DeviceSequenceState.allocate(
        torch.tensor([[3, 4]]),
        max_length=8,
        pad_token_id=0,
        eos_token_ids=(9, 10),
    )

    assert torch.equal(state.append(torch.tensor([5])), torch.tensor([5]))
    assert torch.equal(state.append(torch.tensor([9])), torch.tensor([9]))
    assert not bool(state.unfinished[0])
    assert int(state.logical_length[0]) == 4
    assert torch.equal(state.append(torch.tensor([7])), torch.tensor([0]))
    assert state.sequence[0, :5].tolist() == [3, 4, 5, 9, 0]
    assert int(state.logical_length[0]) == 4


def test_fixed_block_tail_uses_static_storage_and_stops_without_cat() -> None:
    state = DeviceSequenceState.allocate(
        torch.tensor([[1, 2]]),
        max_length=8,
        pad_token_id=0,
        eos_token_ids=(9,),
    )
    storage_pointer = state.sequence.data_ptr()

    generated, continues = fixed_block_tail(
        state=state,
        start_length=2,
        raw_logits_steps=[_logits(4), _logits(9), _logits(7), _logits(8)],
        logits_processor=_IdentityProcessor(),
        stopping_policy=_policy(state),
        do_sample=False,
    )

    assert state.sequence.data_ptr() == storage_pointer
    assert generated[:, 0].tolist() == [4, 9, 0, 0]
    assert continues[:, 0].tolist() == [True, False, False, False]
    assert state.sequence[0, :6].tolist() == [1, 2, 4, 9, 0, 0]
    assert int(state.logical_length[0]) == 4


def test_fixed_block_tail_records_max_length_without_eos() -> None:
    state = DeviceSequenceState.allocate(
        torch.tensor([[1, 2]]),
        max_length=6,
        pad_token_id=0,
        eos_token_ids=(9,),
    )

    generated, continues = fixed_block_tail(
        state=state,
        start_length=2,
        raw_logits_steps=[_logits(4), _logits(5), _logits(6), _logits(7)],
        logits_processor=_IdentityProcessor(),
        stopping_policy=_policy(state),
        do_sample=False,
    )

    assert generated[:, 0].tolist() == [4, 5, 6, 7]
    assert continues[:, 0].tolist() == [True, True, True, False]
    assert state.sequence.tolist() == [[1, 2, 4, 5, 6, 7]]
    assert int(state.logical_length[0]) == 6


def test_model_input_update_is_in_place() -> None:
    token = torch.tensor([[7]])
    cache_position = torch.tensor([3])
    position_ids = torch.tensor([[3]])
    inputs = {
        "decoder_input_ids": token,
        "cache_position": cache_position,
        "position_ids": position_ids,
    }
    pointers = {name: value.data_ptr() for name, value in inputs.items()}

    update_one_token_model_inputs(inputs, torch.tensor([8]))

    assert token.tolist() == [[8]]
    assert cache_position.tolist() == [4]
    assert position_ids.tolist() == [[4]]
    assert {name: value.data_ptr() for name, value in inputs.items()} == pointers


def test_fixed_tail_updates_static_model_handoff_each_step() -> None:
    state = DeviceSequenceState.allocate(
        torch.tensor([[1, 2]]),
        max_length=8,
        pad_token_id=0,
        eos_token_ids=(9,),
    )
    static_inputs = {
        "decoder_input_ids": torch.tensor([[2]]),
        "cache_position": torch.tensor([1]),
        "position_ids": torch.tensor([[1]]),
    }

    fixed_block_tail(
        state=state,
        start_length=2,
        raw_logits_steps=[_logits(4), _logits(5), _logits(6), _logits(7)],
        logits_processor=_IdentityProcessor(),
        stopping_policy=_policy(state),
        do_sample=False,
        static_model_inputs=static_inputs,
    )

    assert static_inputs["decoder_input_ids"].tolist() == [[7]]
    assert static_inputs["cache_position"].tolist() == [5]
    assert static_inputs["position_ids"].tolist() == [[5]]


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("sequence", torch.zeros((2, 4), dtype=torch.long), "shape"),
        ("unfinished", torch.ones((1,), dtype=torch.long), "dtype"),
        ("eos_token_ids", torch.empty((0,), dtype=torch.long), "non-empty"),
    ],
)
def test_state_validation_fails_loudly(field, replacement, error) -> None:
    state = DeviceSequenceState.allocate(
        torch.tensor([[1, 2]]),
        max_length=4,
        pad_token_id=0,
        eos_token_ids=(3,),
    )
    setattr(state, field, replacement)

    with pytest.raises((TypeError, ValueError), match=error):
        state.validate()


def test_fixed_block_rejects_unsupported_size_before_work() -> None:
    state = DeviceSequenceState.allocate(
        torch.tensor([[1]]),
        max_length=8,
        pad_token_id=0,
        eos_token_ids=(3,),
    )
    with pytest.raises(ValueError, match="block size"):
        fixed_block_tail(
            state=state,
            start_length=1,
            raw_logits_steps=[_logits(2), _logits(2)],
            logits_processor=_IdentityProcessor(),
            stopping_policy=_policy(state),
            do_sample=False,
        )


def test_graph_policy_matches_transformers_eos_and_max_length_semantics() -> None:
    state = DeviceSequenceState.allocate(
        torch.tensor([[1, 2]]),
        max_length=6,
        pad_token_id=0,
        eos_token_ids=(9, 10),
    )
    criteria = _criteria(eos=(9, 10), max_length=6)
    policy = _policy(state, criteria=criteria)

    for token, length in ((7, 3), (9, 4), (10, 5), (8, 6)):
        input_ids = torch.tensor([[1] * (length - 1) + [token]])
        expected = criteria(input_ids, None)
        actual = policy.stopped(
            torch.tensor([token]),
            torch.tensor([length]),
        )
        assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("criteria", "error"),
    [
        (
            StoppingCriteriaList(
                [
                    MaxLengthCriteria(max_length=8),
                    EosTokenCriteria(torch.tensor([9])),
                    MaxTimeCriteria(max_time=1.0),
                ]
            ),
            "does not support.*MaxTimeCriteria",
        ),
        (
            StoppingCriteriaList([MaxLengthCriteria(max_length=8)]),
            "exactly one",
        ),
        (
            StoppingCriteriaList(
                [
                    MaxLengthCriteria(max_length=7),
                    EosTokenCriteria(torch.tensor([9])),
                ]
            ),
            "must match.*max_length",
        ),
        (
            StoppingCriteriaList(
                [
                    MaxLengthCriteria(max_length=8),
                    EosTokenCriteria(torch.tensor([10])),
                ]
            ),
            "token ids must match",
        ),
    ],
)
def test_graph_policy_fails_loudly_on_unsupported_or_mismatched_criteria(
    criteria,
    error,
) -> None:
    state = DeviceSequenceState.allocate(
        torch.tensor([[1, 2]]),
        max_length=8,
        pad_token_id=0,
        eos_token_ids=(9,),
    )

    with pytest.raises((TypeError, ValueError), match=error):
        GraphCompatibleStoppingPolicy.from_transformers(criteria, state=state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_graph_policy_is_cuda_graph_capture_compatible() -> None:
    device = torch.device("cuda")
    state = DeviceSequenceState.allocate(
        torch.tensor([[1, 2]], device=device),
        max_length=6,
        pad_token_id=0,
        eos_token_ids=(9,),
    )
    policy = _policy(state)
    raw_logits = [_logits(token).to(device) for token in (4, 9, 7, 8)]
    templates = {
        "sequence": state.sequence.clone(),
        "physical_length": state.physical_length.clone(),
        "logical_length": state.logical_length.clone(),
        "unfinished": state.unfinished.clone(),
    }

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        generated, continues = fixed_block_tail(
            state=state,
            start_length=2,
            raw_logits_steps=raw_logits,
            logits_processor=_IdentityProcessor(),
            stopping_policy=policy,
            do_sample=False,
        )
    for name, template in templates.items():
        getattr(state, name).copy_(template)
    graph.replay()
    torch.cuda.synchronize()

    assert generated[:, 0].cpu().tolist() == [4, 9, 0, 0]
    assert continues[:, 0].cpu().tolist() == [True, False, False, False]
    assert int(state.logical_length.cpu()[0]) == 4
