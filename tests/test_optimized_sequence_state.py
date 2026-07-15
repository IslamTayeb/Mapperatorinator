from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
)

from osuT5.osuT5.inference.optimized.single.decode_loop import (
    active_prefix_decode_generate,
)
from osuT5.osuT5.inference.optimized.single.sequence_state import (
    BatchOneSequenceState,
    BatchOneStoppingPolicy,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _criteria(*, eos=(7, 9), max_length: int = 8) -> StoppingCriteriaList:
    return StoppingCriteriaList(
        [
            MaxLengthCriteria(max_length=max_length),
            EosTokenCriteria(torch.tensor(eos, dtype=torch.long)),
        ]
    )


def _policy(*, eos=(7, 9), max_length: int = 8) -> BatchOneStoppingPolicy:
    return BatchOneStoppingPolicy.from_transformers(
        _criteria(eos=eos, max_length=max_length),
        device=torch.device("cpu"),
    )


def test_preallocated_state_keeps_one_storage_and_matches_stopping() -> None:
    state = BatchOneSequenceState.allocate(
        torch.tensor([[1, 2]]),
        stopping_policy=_policy(max_length=6),
    )
    pointer = state.sequence.untyped_storage().data_ptr()

    active, stopped = state.append(torch.tensor([4]))
    assert not stopped
    assert active.tolist() == [[1, 2, 4]]
    active, stopped = state.append(torch.tensor([7]))
    assert stopped
    assert active.tolist() == [[1, 2, 4, 7]]
    assert state.sequence.untyped_storage().data_ptr() == pointer

    length_state = BatchOneSequenceState.allocate(
        torch.tensor([[1, 2]]),
        stopping_policy=_policy(eos=(9,), max_length=4),
    )
    assert not length_state.append(torch.tensor([3]))[1]
    assert length_state.append(torch.tensor([4]))[1]


@pytest.mark.parametrize(
    ("criteria", "error"),
    [
        (
            StoppingCriteriaList(
                [
                    MaxLengthCriteria(max_length=8),
                    EosTokenCriteria(torch.tensor([7])),
                    MaxTimeCriteria(max_time=1.0),
                ]
            ),
            "does not support.*MaxTimeCriteria",
        ),
        (
            StoppingCriteriaList([MaxLengthCriteria(max_length=8)]),
            "exactly one",
        ),
    ],
)
def test_stopping_policy_rejects_unsupported_transformers_semantics(
    criteria,
    error,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        BatchOneStoppingPolicy.from_transformers(
            criteria,
            device=torch.device("cpu"),
        )


class _SampleModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(is_encoder_decoder=False)
        self.prepared_storage = []

    def _get_initial_cache_position(self, cur_len, device, model_kwargs):
        del cur_len, device
        return model_kwargs

    def _has_unfinished_sequences(self, finished, synced_gpus, device):
        del synced_gpus, device
        return not bool(finished)

    def prepare_inputs_for_generation(self, input_ids, **model_kwargs):
        del model_kwargs
        self.prepared_storage.append(input_ids.untyped_storage().data_ptr())
        return {"input_ids": input_ids[:, -1:]}

    def __call__(self, input_ids, return_dict):
        del return_dict
        logits = torch.tensor(
            [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -100.0, -100.0]]],
            device=input_ids.device,
        )
        return SimpleNamespace(logits=logits)

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder,
    ):
        del outputs, is_encoder_decoder
        return model_kwargs


def _generation_config(max_length: int = 7):
    return SimpleNamespace(
        return_dict_in_generate=False,
        output_attentions=False,
        output_hidden_states=False,
        output_scores=False,
        output_logits=False,
        prefill_chunk_size=None,
        _pad_token_tensor=torch.tensor(0),
        max_length=max_length,
        do_sample=True,
    )


def _run_sample(*, candidate: bool):
    model = _SampleModel()
    torch.manual_seed(12345)
    result = active_prefix_decode_generate(
        model,
        torch.tensor([[1, 2]], dtype=torch.long),
        logits_processor=lambda ids, scores: scores,
        stopping_criteria=_criteria(eos=(7,), max_length=7),
        generation_config=_generation_config(),
        preallocated_batch1_state=candidate,
    )
    return result, torch.random.get_rng_state().clone(), model.prepared_storage


def test_preallocated_loop_matches_tokens_rng_and_removes_cat_storage_growth() -> None:
    baseline, baseline_rng, baseline_storage = _run_sample(candidate=False)
    candidate, candidate_rng, candidate_storage = _run_sample(candidate=True)

    assert torch.equal(candidate, baseline)
    assert torch.equal(candidate_rng, baseline_rng)
    assert len(set(candidate_storage)) == 1
    assert len(set(baseline_storage)) > 1


def test_preallocated_loop_does_not_call_torch_cat(monkeypatch) -> None:
    def reject_cat(*args, **kwargs):
        del args, kwargs
        raise AssertionError("optimized token append must not call torch.cat")

    monkeypatch.setattr(torch, "cat", reject_cat)
    result, _, _ = _run_sample(candidate=True)
    assert result.shape == (1, 7)


def test_preallocated_loop_rejects_batch_before_model_work() -> None:
    with pytest.raises(ValueError, match="batch_size=1"):
        active_prefix_decode_generate(
            _SampleModel(),
            torch.tensor([[1], [2]], dtype=torch.long),
            logits_processor=lambda ids, scores: scores,
            stopping_criteria=_criteria(eos=(7,), max_length=7),
            generation_config=_generation_config(),
            preallocated_batch1_state=True,
        )


def test_default_inference_import_keeps_optimized_state_cold_and_quiet() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import inference; "
                "assert 'osuT5.osuT5.inference.optimized.single.sequence_state' "
                "not in sys.modules; "
                "assert 'osuT5.osuT5.inference.optimized.single.shared_rope' "
                "not in sys.modules"
            ),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""

