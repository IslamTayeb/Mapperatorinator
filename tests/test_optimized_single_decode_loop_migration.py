from __future__ import annotations

import gc
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import weakref

import pytest
import torch
from transformers import StoppingCriteriaList
from transformers.modeling_outputs import BaseModelOutput

from osuT5.osuT5.inference.optimized.single.decode_loop import (
    _bucketed_prefix_length,
    _clone_static_graph_inputs,
    _copy_static_graph_inputs,
    _cuda_graph_signature,
    _new_encoder_stabilization_stats,
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


def test_stable_encoder_output_skips_identical_object_copy() -> None:
    tensor = torch.randn(1, 4, 8)
    output = BaseModelOutput(last_hidden_state=tensor)
    holder = {"encoder_outputs": output}
    stats = _new_encoder_stabilization_stats()
    version = tensor._version

    result = _stable_encoder_outputs(holder, output, profile_stats=stats)

    assert result is output
    assert tensor._version == version
    assert stats == {
        "calls": 1,
        "executed_bytes": 0,
        "executed_copies": 0,
        "holder_replacements": 0,
        "skipped_copies": 1,
        "skipped_identity_copies": 1,
        "skipped_shared_storage_copies": 0,
    }


def test_stable_encoder_output_skips_equal_storage_view_copy() -> None:
    source = torch.randn(1, 4, 8)
    same_view = source.as_strided(
        source.shape,
        source.stride(),
        source.storage_offset(),
    )
    assert same_view is not source
    assert same_view.is_set_to(source)
    holder = {
        "encoder_outputs": BaseModelOutput(last_hidden_state=same_view),
    }
    stats = _new_encoder_stabilization_stats()
    version = source._version

    result = _stable_encoder_outputs(
        holder,
        BaseModelOutput(last_hidden_state=source),
        profile_stats=stats,
    )

    assert result.last_hidden_state is same_view
    assert source._version == version
    assert stats["calls"] == 1
    assert stats["skipped_copies"] == 1
    assert stats["skipped_shared_storage_copies"] == 1
    assert stats["executed_bytes"] == 0


def test_stable_encoder_output_copies_distinct_compatible_storage() -> None:
    current_tensor = torch.zeros((1, 4, 8), dtype=torch.float32)
    source = torch.randn((1, 4, 8), dtype=torch.float32)
    current = BaseModelOutput(last_hidden_state=current_tensor)
    holder = {"encoder_outputs": current}
    stats = _new_encoder_stabilization_stats()
    pointer = current_tensor.data_ptr()

    result = _stable_encoder_outputs(
        holder,
        BaseModelOutput(last_hidden_state=source),
        profile_stats=stats,
    )

    assert result is current
    assert result.last_hidden_state.data_ptr() == pointer
    assert torch.equal(result.last_hidden_state, source)
    assert stats["calls"] == 1
    assert stats["skipped_copies"] == 0
    assert stats["executed_copies"] == 1
    assert stats["executed_bytes"] == source.numel() * source.element_size()


@pytest.mark.parametrize(
    "source",
    (
        torch.ones((1, 5, 8), dtype=torch.float32),
        torch.ones((1, 4, 8), dtype=torch.float16),
    ),
)
def test_stable_encoder_output_shape_and_dtype_mismatch_still_replace(source) -> None:
    original = BaseModelOutput(
        last_hidden_state=torch.zeros((1, 4, 8), dtype=torch.float32)
    )
    holder = {"encoder_outputs": original}
    stats = _new_encoder_stabilization_stats()

    result = _stable_encoder_outputs(
        holder,
        BaseModelOutput(last_hidden_state=source),
        profile_stats=stats,
    )

    assert result is holder["encoder_outputs"]
    assert result is not original
    assert result.last_hidden_state is not source
    assert result.last_hidden_state.shape == source.shape
    assert result.last_hidden_state.dtype == source.dtype
    assert torch.equal(result.last_hidden_state, source)
    assert stats["holder_replacements"] == 1
    assert stats["executed_copies"] == 1
    assert stats["executed_bytes"] == source.numel() * source.element_size()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_stable_encoder_output_device_mismatch_still_replaces() -> None:
    original = BaseModelOutput(last_hidden_state=torch.zeros((1, 4, 8)))
    source = torch.ones((1, 4, 8), device="cuda")
    holder = {"encoder_outputs": original}

    result = _stable_encoder_outputs(
        holder,
        BaseModelOutput(last_hidden_state=source),
    )

    assert result is not original
    assert result.last_hidden_state.device.type == "cuda"
    assert torch.equal(result.last_hidden_state, source)


def test_stable_encoder_output_clone_owns_storage_and_lifetime() -> None:
    source = torch.randn(1, 4, 8)
    source_reference = weakref.ref(source)
    encoder_outputs = BaseModelOutput(last_hidden_state=source)
    holder = {}

    result = _stable_encoder_outputs(holder, encoder_outputs)
    expected = source.clone()
    assert result.last_hidden_state is not source
    assert not result.last_hidden_state.is_set_to(source)

    del encoder_outputs
    del source
    gc.collect()

    assert source_reference() is None
    assert torch.equal(holder["encoder_outputs"].last_hidden_state, expected)


def test_stable_encoder_output_fails_loudly_on_invalid_holder_and_stats() -> None:
    source = BaseModelOutput(last_hidden_state=torch.ones(1, 4, 8))
    with pytest.raises(RuntimeError, match="non-BaseModelOutput"):
        _stable_encoder_outputs({"encoder_outputs": object()}, source)
    with pytest.raises(ValueError, match="invalid schema"):
        _stable_encoder_outputs({}, source, profile_stats={"calls": 0})


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


def test_framework_batch_prefix_context_does_not_load_native_extensions():
    completed = _run_fresh_python(
        """
import sys
from osuT5.osuT5.inference.optimized.single.runtime_context import (
    active_prefix_self_attention_context,
)

with active_prefix_self_attention_context(64):
    pass

for name in (
    "osuT5.osuT5.inference.optimized.kernels.q1_attention",
    "osuT5.osuT5.inference.optimized.kernels.cross_mlp",
    "osuT5.osuT5.inference.optimized.kernels.decoder_layer",
    "torch.utils.cpp_extension",
):
    assert name not in sys.modules, name
"""
    )
    assert completed.returncode == 0, completed.stderr


def test_fresh_decode_loop_import_keeps_native_extensions_cold_and_quiet():
    completed = _run_fresh_python(
        """
import sys
from osuT5.osuT5.inference.optimized.single import decode_loop

assert decode_loop is not None
for name in (
    "osuT5.osuT5.inference.optimized.kernels.q1_attention",
    "osuT5.osuT5.inference.optimized.kernels.cross_mlp",
    "osuT5.osuT5.inference.optimized.kernels.decoder_layer",
    "torch.utils.cpp_extension",
):
    assert name not in sys.modules, name
"""
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""


class _PerRowEosCriteria:
    eos_token_id = 2

    def __call__(self, input_ids, scores):
        del scores
        return input_ids[:, -1].eq(self.eos_token_id)


class _BatchLoopModel:
    def __init__(self, scheduled_tokens):
        self.scheduled_tokens = [torch.tensor(row) for row in scheduled_tokens]
        self.call_index = 0
        self.prepared_masks = []
        self.config = SimpleNamespace(
            is_encoder_decoder=False,
            _attn_implementation="sdpa",
        )

    def _get_initial_cache_position(self, cur_len, device, model_kwargs):
        del cur_len, device
        return model_kwargs

    def _valid_auto_compile_criteria(self, model_kwargs, generation_config):
        del model_kwargs, generation_config
        return False

    def _has_unfinished_sequences(self, finished, synced_gpus, device):
        del synced_gpus, device
        return not bool(finished)

    def prepare_inputs_for_generation(self, input_ids, **model_kwargs):
        mask = model_kwargs["decoder_attention_mask"]
        self.prepared_masks.append(mask.clone())
        return {
            "input_ids": input_ids[:, -1:],
            "decoder_attention_mask": mask,
        }

    def __call__(self, **kwargs):
        del kwargs
        tokens = self.scheduled_tokens[self.call_index]
        self.call_index += 1
        logits = torch.full((len(tokens), 1, 8), -1000.0)
        logits[torch.arange(len(tokens)), 0, tokens] = 1000.0
        return SimpleNamespace(logits=logits)

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder,
    ):
        del outputs, is_encoder_decoder
        mask = model_kwargs["decoder_attention_mask"]
        model_kwargs["decoder_attention_mask"] = torch.cat(
            (mask, torch.ones((mask.shape[0], 1), dtype=mask.dtype)),
            dim=1,
        )
        return model_kwargs


def _generation_config():
    return SimpleNamespace(
        return_dict_in_generate=False,
        output_attentions=False,
        output_hidden_states=False,
        output_scores=False,
        output_logits=False,
        prefill_chunk_size=None,
        _pad_token_tensor=torch.tensor(0),
        max_length=16,
        do_sample=False,
    )


def test_fixed_batch_loop_pads_finished_rows_and_waits_for_all_rows():
    model = _BatchLoopModel(
        (
            (2, 5, 5),
            (7, 2, 5),
            (7, 7, 2),
        )
    )
    input_ids = torch.tensor(((0, 9), (8, 9), (0, 9)))
    mask = input_ids.ne(0)

    result = active_prefix_decode_generate(
        model,
        input_ids,
        logits_processor=lambda ids, scores: scores,
        stopping_criteria=StoppingCriteriaList([_PerRowEosCriteria()]),
        generation_config=_generation_config(),
        decoder_attention_mask=mask,
    )

    assert torch.equal(
        result[:, -3:],
        torch.tensor(((2, 0, 0), (5, 2, 0), (5, 5, 2))),
    )
    assert model.call_index == 3
    assert torch.equal(model.prepared_masks[0], mask)
    assert model.prepared_masks[1].shape == (3, 3)
    assert model.prepared_masks[2].shape == (3, 4)


def test_batched_loop_rejects_missing_mask_and_wrong_stopping_shape():
    model = _BatchLoopModel(((2, 2),))
    input_ids = torch.tensor(((1,), (1,)))
    with pytest.raises(ValueError, match="requires decoder_attention_mask"):
        active_prefix_decode_generate(
            model,
            input_ids,
            logits_processor=lambda ids, scores: scores,
            stopping_criteria=StoppingCriteriaList([_PerRowEosCriteria()]),
            generation_config=_generation_config(),
        )

    class ScalarCriteria:
        eos_token_id = 2

        def __call__(self, input_ids, scores):
            del input_ids, scores
            return torch.tensor(False)

    class ScalarCriteriaList(list):
        def __call__(self, input_ids, scores):
            return self[0](input_ids, scores)

    model = _BatchLoopModel(((2, 2),))
    with pytest.raises(RuntimeError, match="one value per row"):
        active_prefix_decode_generate(
            model,
            input_ids,
            logits_processor=lambda ids, scores: scores,
            stopping_criteria=ScalarCriteriaList([ScalarCriteria()]),
            generation_config=_generation_config(),
            decoder_attention_mask=torch.ones_like(input_ids),
        )


def test_graph_signature_separates_batch_and_prefix_bucket():
    owner = object()
    b1 = _cuda_graph_signature(
        64,
        {"token": torch.ones((1, 1), dtype=torch.long), "owner": owner},
    )
    b2 = _cuda_graph_signature(
        64,
        {"token": torch.ones((2, 1), dtype=torch.long), "owner": owner},
    )
    b2_larger_prefix = _cuda_graph_signature(
        128,
        {"token": torch.ones((2, 1), dtype=torch.long), "owner": owner},
    )
    assert len({b1, b2, b2_larger_prefix}) == 3
