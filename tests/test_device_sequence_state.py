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

from osuT5.osuT5.inference.optimized.scout.device_sequence_state import (
    BatchOneSequenceState,
    BatchOneStoppingPolicy,
    device_sequence_state_candidate_context,
)
from osuT5.osuT5.inference.optimized.single.decode_loop import (
    active_prefix_decode_generate,
)
from utils.run_device_sequence_state_candidate import (
    REPO_ROOT as RUNNER_REPO_ROOT,
    _ensure_repo_import_path,
    _parse_runner_args,
    _rng_fingerprint,
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


def test_preallocated_state_keeps_storage_and_matches_eos_and_length() -> None:
    state = BatchOneSequenceState.allocate(
        torch.tensor([[1, 2]]),
        stopping_policy=_policy(max_length=6),
    )
    pointer = state.sequence.data_ptr()

    active, stopped = state.append(torch.tensor([4]))
    assert not stopped
    assert active.tolist() == [[1, 2, 4]]
    active, stopped = state.append(torch.tensor([7]))
    assert stopped
    assert active.tolist() == [[1, 2, 4, 7]]
    assert state.sequence.data_ptr() == pointer

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
def test_stopping_policy_fails_loudly(criteria, error) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        BatchOneStoppingPolicy.from_transformers(
            criteria,
            device=torch.device("cpu"),
        )


class _SampleModel:
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        self.config = SimpleNamespace(is_encoder_decoder=False)
        self.dtype = dtype
        self.prepared_storage = []
        self.cache_marker = object()
        self.cache_observations = []

    def _get_initial_cache_position(self, cur_len, device, model_kwargs):
        del cur_len, device
        return model_kwargs

    def _has_unfinished_sequences(self, finished, synced_gpus, device):
        del synced_gpus, device
        return not bool(finished)

    def prepare_inputs_for_generation(self, input_ids, **model_kwargs):
        self.prepared_storage.append(input_ids.untyped_storage().data_ptr())
        self.cache_observations.append(model_kwargs.get("cache_marker") is self.cache_marker)
        return {"input_ids": input_ids[:, -1:]}

    def __call__(self, input_ids, return_dict):
        del return_dict
        logits = torch.tensor(
            [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -100.0, -100.0]]],
            device=input_ids.device,
            dtype=self.dtype,
        )
        return SimpleNamespace(logits=logits)

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder,
    ):
        del outputs, is_encoder_decoder
        self.cache_observations.append(model_kwargs.get("cache_marker") is self.cache_marker)
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


def _run_sample(*, candidate: bool, dtype: torch.dtype = torch.float32):
    model = _SampleModel(dtype)
    torch.manual_seed(12345)
    result = active_prefix_decode_generate(
        model,
        torch.tensor([[1, 2]], dtype=torch.long),
        logits_processor=lambda ids, scores: scores,
        stopping_criteria=_criteria(eos=(7,), max_length=7),
        generation_config=_generation_config(),
        preallocated_batch1_state=candidate,
        cache_marker=model.cache_marker,
    )
    return (
        result,
        torch.random.get_rng_state().clone(),
        model.prepared_storage,
        model.cache_observations,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_candidate_matches_default_tokens_rng_cache_and_uses_one_storage(dtype) -> None:
    baseline, baseline_rng, baseline_storage, baseline_cache = _run_sample(
        candidate=False,
        dtype=dtype,
    )
    candidate, candidate_rng, candidate_storage, candidate_cache = _run_sample(
        candidate=True,
        dtype=dtype,
    )

    assert torch.equal(candidate, baseline)
    assert torch.equal(candidate_rng, baseline_rng)
    assert candidate_cache == baseline_cache
    assert all(candidate_cache)
    assert len(set(candidate_storage)) == 1
    assert len(set(baseline_storage)) > 1


def test_candidate_does_not_call_torch_cat_for_token_append(monkeypatch) -> None:
    def reject_cat(*args, **kwargs):
        del args, kwargs
        raise AssertionError("candidate token append must not call torch.cat")

    monkeypatch.setattr(torch, "cat", reject_cat)
    result, _, _, _ = _run_sample(candidate=True)
    assert result.shape == (1, 7)


def test_candidate_rejects_batch_greater_than_one_before_model_work() -> None:
    with pytest.raises(ValueError, match="batch_size=1"):
        active_prefix_decode_generate(
            _SampleModel(),
            torch.tensor([[1], [2]], dtype=torch.long),
            logits_processor=lambda ids, scores: scores,
            stopping_criteria=_criteria(eos=(7,), max_length=7),
            generation_config=_generation_config(),
            preallocated_batch1_state=True,
        )


def test_default_decode_import_keeps_device_state_scout_cold() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "import osuT5.osuT5.inference.optimized.single.decode_loop; "
            "assert 'osuT5.osuT5.inference.optimized.scout.device_sequence_state' "
            "not in sys.modules",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_candidate_context_restores_engine_binding() -> None:
    from osuT5.osuT5.inference.optimized.single import engine

    original = engine.active_prefix_decode_generate
    with device_sequence_state_candidate_context() as activation:
        installed = engine.active_prefix_decode_generate
        assert installed is not original
        with pytest.raises(RuntimeError, match="owns preallocated"):
            installed(preallocated_batch1_state=False)
        assert activation.calls == 0
    assert engine.active_prefix_decode_generate is original


def test_candidate_runner_preserves_hydra_arguments() -> None:
    mode, manifest, remaining = _parse_runner_args(
        [
            "--mode",
            "candidate",
            "--evidence-manifest",
            "/tmp/evidence.json",
            "--",
            "--config-name",
            "profile_salvalai",
            "precision=fp16",
        ]
    )
    assert mode == "candidate"
    assert manifest == Path("/tmp/evidence.json")
    assert remaining == [
        "--config-name",
        "profile_salvalai",
        "precision=fp16",
    ]


def test_candidate_runner_adds_repo_root_for_direct_file_execution(monkeypatch) -> None:
    filtered = [entry for entry in sys.path if entry != str(RUNNER_REPO_ROOT)]
    monkeypatch.setattr(sys, "path", filtered)

    assert _ensure_repo_import_path() == RUNNER_REPO_ROOT
    assert sys.path[0] == str(RUNNER_REPO_ROOT)


def test_rng_fingerprint_tracks_all_seeded_generators() -> None:
    import random
    import numpy as np

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    first = _rng_fingerprint()
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    assert _rng_fingerprint() == first
    torch.rand(1)
    assert _rng_fingerprint()["torch_cpu_sha256"] != first["torch_cpu_sha256"]


def test_reciprocal_wrapper_is_serial_by_default_with_explicit_parallel_opt_in() -> None:
    source = (
        REPO_ROOT
        / "scripts/dcc/profile_device_sequence_state_reciprocal.sbatch"
    ).read_text(encoding="utf-8")
    assert "run_role baseline_first baseline" in source
    assert "run_role candidate_first candidate" in source
    assert "run_role candidate_second candidate" in source
    assert "run_role baseline_second baseline" in source
    assert "MAPPERATORINATOR_PRECISION:-fp32" in source
    assert "fp32) ANALYSIS_MODE=exact-fp32" in source
    assert "fp16) ANALYSIS_MODE=exact-same-precision" in source
    assert "cross_candidate_exact\"] is True" in source
    assert "dispatch_cache_topology\"][\"pass\"] is True" in source
    assert "another GPU job is running or pending" in source
    assert "MAPPERATORINATOR_ALLOW_PARALLEL_RECIPROCAL:-0" in source
    assert "MAPPERATORINATOR_ALLOW_PARALLEL_RECIPROCAL must be 0 or 1" in source
    assert "parallel_reciprocal_opt_in=$ALLOW_PARALLEL_RECIPROCAL" in source
    assert "MAPPERATORINATOR_SHARED_ROPE_BASELINE:-0" in source
    assert "utils/run_exact_rope_device_state.py" in source
    assert "shared_rope_baseline=$SHARED_ROPE_BASELINE" in source
    assert "--mode \"$kind\"" in source
    assert "rng_after_seed" in source
    assert "rng_after_inference" in source
