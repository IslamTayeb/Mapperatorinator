from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import StoppingCriteriaList
from transformers.modeling_outputs import BaseModelOutput

from osuT5.osuT5.inference.optimized.single.decode_loop import (
    _bucketed_prefix_length,
    _clone_static_graph_inputs,
    _copy_static_graph_inputs,
    _cuda_graph_signature,
    _stable_encoder_outputs,
    active_prefix_decode_generate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_fresh_python(source: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_bucket_boundaries_are_unchanged():
    assert _bucketed_prefix_length(1, 64, 2560) == 64
    assert _bucketed_prefix_length(64, 64, 2560) == 64
    assert _bucketed_prefix_length(65, 64, 2560) == 128
    assert _bucketed_prefix_length(2559, 64, 2560) == 2560
    assert _bucketed_prefix_length(3000, 64, 2560) == 2560
    with pytest.raises(ValueError, match="positive"):
        _bucketed_prefix_length(1, 0, 2560)


def test_static_graph_inputs_clone_and_copy_without_rebinding():
    source = {
        "token": torch.tensor([[1]], dtype=torch.long),
        "mask": torch.tensor([[True]], dtype=torch.bool),
        "owner": object(),
    }
    static = _clone_static_graph_inputs(source)
    token_pointer = static["token"].data_ptr()
    mask_pointer = static["mask"].data_ptr()
    assert static["owner"] is source["owner"]
    assert static["token"] is not source["token"]

    replacement = {
        "token": torch.tensor([[7]], dtype=torch.long),
        "mask": torch.tensor([[False]], dtype=torch.bool),
        "owner": source["owner"],
    }
    _copy_static_graph_inputs(static, replacement)
    assert static["token"].data_ptr() == token_pointer
    assert static["mask"].data_ptr() == mask_pointer
    assert static["token"].item() == 7
    assert static["mask"].item() is False

    with pytest.raises(RuntimeError, match="shape changed"):
        _copy_static_graph_inputs(
            static,
            {**replacement, "token": torch.ones((1, 2), dtype=torch.long)},
        )


def test_stable_encoder_output_reuses_identity_and_storage_for_same_shape():
    holder = {}
    first_source = BaseModelOutput(last_hidden_state=torch.randn(1, 4, 8))
    first = _stable_encoder_outputs(holder, first_source)
    pointer = first.last_hidden_state.data_ptr()
    second_source = BaseModelOutput(last_hidden_state=torch.randn(1, 4, 8))
    second = _stable_encoder_outputs(holder, second_source)

    assert second is first
    assert second.last_hidden_state.data_ptr() == pointer
    assert torch.equal(second.last_hidden_state, second_source.last_hidden_state)


def test_graph_signature_preserves_shape_dtype_device_and_object_identity():
    owner = object()
    inputs = {
        "token": torch.ones((1, 1), dtype=torch.long),
        "owner": owner,
    }
    signature = _cuda_graph_signature(128, inputs)
    assert signature[0] == 128
    assert ("token", (1, 1), "torch.int64", "cpu") in signature
    assert ("owner", "object", id(owner)) in signature


def test_eager_decode_never_consults_hugging_face_compile_selection():
    class StopAtLength:
        def __call__(self, input_ids, scores):
            return torch.tensor(
                [input_ids.shape[1] >= 3],
                dtype=torch.bool,
                device=input_ids.device,
            )

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(is_encoder_decoder=False)
            self._valid_auto_compile_criteria = Mock(
                side_effect=AssertionError("compile selection must not run")
            )
            self.get_compiled_call = Mock(
                side_effect=AssertionError("compiled call must not be requested")
            )
            self.forward_calls = 0

        def _get_initial_cache_position(self, cur_len, device, model_kwargs):
            return model_kwargs

        def _has_unfinished_sequences(
            self,
            this_peer_finished,
            synced_gpus,
            device,
        ):
            return not bool(this_peer_finished)

        def prepare_inputs_for_generation(self, input_ids, **model_kwargs):
            return {"input_ids": input_ids}

        def __call__(self, input_ids, return_dict):
            self.forward_calls += 1
            logits = torch.zeros(
                (input_ids.shape[0], input_ids.shape[1], 4),
                dtype=torch.float32,
                device=input_ids.device,
            )
            logits[..., 2] = 1
            return SimpleNamespace(logits=logits)

        def _update_model_kwargs_for_generation(
            self,
            outputs,
            model_kwargs,
            is_encoder_decoder,
        ):
            return model_kwargs

    model = FakeModel()
    generation_config = SimpleNamespace(
        return_dict_in_generate=False,
        output_attentions=False,
        output_hidden_states=False,
        output_scores=False,
        output_logits=False,
        prefill_chunk_size=None,
        _pad_token_tensor=torch.tensor(0, dtype=torch.long),
        max_length=3,
        do_sample=False,
    )

    output = active_prefix_decode_generate(
        model,
        torch.tensor([[1]], dtype=torch.long),
        logits_processor=lambda input_ids, logits: logits,
        stopping_criteria=StoppingCriteriaList([StopAtLength()]),
        generation_config=generation_config,
        cuda_graph_forward=False,
    )

    assert torch.equal(output, torch.tensor([[1, 2, 2]], dtype=torch.long))
    assert model.forward_calls == 2
    model._valid_auto_compile_criteria.assert_not_called()
    model.get_compiled_call.assert_not_called()


def test_fresh_server_import_does_not_load_optimized_decode_loop():
    completed = _run_fresh_python(
        """
import importlib
import sys

importlib.import_module("osuT5.osuT5.inference.server")
assert "osuT5.osuT5.inference.optimized.single.decode_loop" not in sys.modules
"""
    )
    assert completed.returncode == 0, completed.stderr
