from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import inference
from osuT5.osuT5.inference.engine_binding import (
    InferenceEngineBinding,
)
from osuT5.osuT5.inference.generation_utils import (
    build_generation_stats,
    eos_token_ids,
)
from osuT5.osuT5.inference.optimized.single import engine as engine_module
from osuT5.osuT5.inference.optimized.single.engine import (
    ExperimentalBatchedSuperTimingRuntime,
    OPTIMIZED_PRESETS,
    OptimizedPreset,
    OptimizedSingleRuntime,
    load_optimized_single_engine,
)
from osuT5.osuT5.inference.optimized.single.state import ProductionDecodeSession
from osuT5.osuT5.inference import server
from osuT5.osuT5.event import EventType, ContextType


def _loader_kwargs(**overrides):
    values = {
        "ckpt_path": None,
        "t5_args": None,
        "device": "cuda",
        "max_batch_size": 8,
        "use_server": False,
        "precision": "fp16",
        "attn_implementation": "sdpa",
        "eval_mode": True,
        "lora_path": None,
        "gamemode": None,
        "auto_select_gamemode_model": True,
        "generation_compile": False,
    }
    values.update(overrides)
    return values


def test_runtime_metadata_exposes_two_fixed_accepted_presets():
    expected_versions = {
        "fp32": "accepted-fp32-native-cross-mlp-289-v2",
        "fp16": "accepted-fp16-all-fused-v1",
    }
    for precision, version in expected_versions.items():
        metadata = OptimizedSingleRuntime(
            OPTIMIZED_PRESETS[precision]
        ).profile_metadata()["optimized_effective_config"]
        assert metadata == {
            "version": version,
            "result_class": "documented-drift",
            "precision": precision,
            "attn_implementation": "sdpa",
            "decoder_loop_backend": "active_prefix_cuda_graph",
            "torch_compile_enabled": False,
            "batch_size": 1,
            "cfg_scale": 1.0,
            "num_beams": 1,
            "active_prefix_decode_loop": True,
            "active_prefix_decode_bucket_size": 64,
            "active_prefix_decode_cuda_graph": True,
            "active_prefix_decode_cuda_graph_warmup": 0,
            "active_prefix_decode_cuda_graph_min_decode_steps": 1,
            "stateful_monotonic_logits_processor": True,
            "q1_bmm_cross_attention": True,
            "decode_session_runtime": True,
            "decode_session_cuda_graph": True,
            "native_decode_kernels": True,
            "native_q1_self_attention": True,
            "native_q1_rope_cache_self_attention": True,
            "native_cross_mlp_tail": True,
        }


def test_experimental_batched_runtime_persists_one_private_session():
    runtime = ExperimentalBatchedSuperTimingRuntime(
        OPTIMIZED_PRESETS["fp32"],
        nominal_batch_size=4,
    )
    first = runtime.new_context_state()
    second = runtime.new_context_state()
    assert first is second
    assert runtime.profile_metadata()["public_wiring"] is False
    assert runtime.profile_metadata()["optimized_effective_config"][
        "framework_dispatch_batch_gt_1"
    ] is True


def test_precision_presets_and_bound_runtime_are_immutable():
    preset = OPTIMIZED_PRESETS["fp32"]
    runtime = OptimizedSingleRuntime(preset)

    with pytest.raises(FrozenInstanceError):
        preset.precision = "fp16"
    with pytest.raises(FrozenInstanceError):
        runtime.preset = OPTIMIZED_PRESETS["fp16"]
    with pytest.raises(TypeError):
        OPTIMIZED_PRESETS["bf16"] = preset

    forged = OptimizedPreset(
        version="forged",
        result_class="unverified",
        precision="fp32",
        torch_dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="registered precision preset"):
        OptimizedSingleRuntime(forged)


def test_runtime_overlays_structure_without_changing_sampling(monkeypatch):
    preset = OPTIMIZED_PRESETS["fp32"]
    runtime = OptimizedSingleRuntime(preset)
    captured = {}
    forwarded = {}

    def fake_generate(model, tokenizer, model_kwargs, generate_kwargs, **kwargs):
        captured.update(generate_kwargs)
        forwarded.update(kwargs)
        return "result"

    monkeypatch.setattr(engine_module, "_generate_window", fake_generate)
    result = runtime.generate_window(
        model=object(),
        tokenizer=object(),
        model_kwargs={},
        generate_kwargs={
            "do_sample": True,
            "top_p": 0.91,
            "top_k": 7,
            "temperature": 0.83,
        },
        context_state=ProductionDecodeSession(),
    )

    assert result == "result"
    assert captured["do_sample"] is True
    assert captured["top_p"] == 0.91
    assert captured["top_k"] == 7
    assert captured["temperature"] == 0.83
    assert "active_prefix_decode_bucket_size" not in captured
    assert forwarded["preset"] is preset


def test_single_loader_binds_each_immutable_precision_preset():
    for precision in ("fp32", "fp16"):
        raw_model = SimpleNamespace(
            generation_config=SimpleNamespace(disable_compile=False),
        )
        tokenizer = object()
        model_loader = Mock(return_value=(raw_model, tokenizer))

        binding, loaded_tokenizer = load_optimized_single_engine(
            model_loader=model_loader,
            loader_kwargs=_loader_kwargs(precision=precision),
        )

        assert isinstance(binding, InferenceEngineBinding)
        assert binding.raw_model is raw_model
        assert loaded_tokenizer is tokenizer
        assert model_loader.call_count == 1
        call_kwargs = model_loader.call_args.kwargs
        assert call_kwargs["generation_compile"] is True
        assert call_kwargs["precision"] == precision
        assert call_kwargs["attn_implementation"] == "sdpa"
        assert call_kwargs["use_server"] is False
        assert (
            binding.runtime.profile_metadata()["optimized_effective_config"][
                "precision"
            ]
            == precision
        )


@pytest.mark.parametrize("precision", (None, "bf16"))
def test_single_loader_rejects_unregistered_precision(precision):
    with pytest.raises(ValueError, match="precision in: fp16, fp32"):
        load_optimized_single_engine(
            model_loader=Mock(),
            loader_kwargs=_loader_kwargs(precision=precision),
        )


@pytest.mark.parametrize("precision", ("fp32", "fp16"))
def test_inference_loader_injects_raw_loader_and_returns_binding(precision):
    raw_model = SimpleNamespace(
        generation_config=SimpleNamespace(disable_compile=False),
    )
    tokenizer = object()
    with patch.object(
        inference,
        "load_model_with_server",
        return_value=(raw_model, tokenizer),
    ) as raw_loader:
        binding, loaded_tokenizer = inference.load_model_with_engine(
            ckpt_path=None,
            t5_args=None,
            device="cuda",
            inference_engine="optimized",
            precision=precision,
        )

    assert isinstance(binding, InferenceEngineBinding)
    assert binding.raw_model is raw_model
    assert loaded_tokenizer is tokenizer
    assert raw_loader.call_count == 1
    assert raw_loader.call_args.kwargs["generation_compile"] is True
    assert raw_loader.call_args.kwargs["precision"] == precision
    assert binding.runtime.preset is OPTIMIZED_PRESETS[precision]


def test_optimized_public_loader_defaults_to_fp32_preset():
    raw_model = SimpleNamespace(
        generation_config=SimpleNamespace(disable_compile=False),
    )
    with patch.object(
        inference,
        "load_model_with_server",
        return_value=(raw_model, object()),
    ) as raw_loader:
        binding, _ = inference.load_model_with_engine(
            ckpt_path=None,
            t5_args=None,
            device="cuda",
            inference_engine="optimized",
        )

    assert raw_loader.call_args.kwargs["precision"] == "fp32"
    assert binding.runtime.preset is OPTIMIZED_PRESETS["fp32"]


def test_shared_calculation_helpers_preserve_legacy_server_results():
    class FakeTokenizer:
        eos_id = 2
        pad_id = 0
        context_eos = {ContextType.MAP: 6}
        event_start = {EventType.TIME_SHIFT: 10}
        event_end = {EventType.TIME_SHIFT: 20}

    tokenizer = FakeTokenizer()
    assert eos_token_ids(
        tokenizer,
        lookback_time=20,
        lookahead_time=30,
        context_type=ContextType.MAP,
    ) == server.get_eos_token_id(
        tokenizer,
        lookback_time=20,
        lookahead_time=30,
        context_type=ContextType.MAP,
    )

    result = torch.tensor([[1, 2, 3, 0]])
    model_kwargs = {
        "decoder_input_ids": torch.tensor([[1, 2, 0, 0]]),
        "decoder_attention_mask": torch.tensor([[1, 1, 0, 0]]),
    }
    assert build_generation_stats(
        result,
        model_kwargs,
        0,
        0.5,
    ) == {
        "batch_size": 1,
        "prompt_tokens": 2,
        "prompt_tokens_per_sample": [2],
        "output_tokens": 3,
        "output_tokens_per_sample": [3],
        "generated_tokens": 1,
        "generated_tokens_per_sample": [1],
        "elapsed_seconds": 0.5,
        "tokens_per_second": 2.0,
    }
