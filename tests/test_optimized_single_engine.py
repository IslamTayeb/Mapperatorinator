from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import inference
from osuT5.osuT5.inference.engine_binding import (
    InferenceEngineBinding,
    unwrap_engine_binding,
)
from osuT5.osuT5.inference.optimized import adapter
from osuT5.osuT5.inference.optimized.single import engine as engine_module
from osuT5.osuT5.inference.optimized.single.config import (
    ACCEPTED_OPTIMIZED_SINGLE_CONFIG,
)
from osuT5.osuT5.inference.optimized.single.engine import (
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
        "precision": "fp32",
        "attn_implementation": "sdpa",
        "eval_mode": True,
        "lora_path": None,
        "gamemode": None,
        "auto_select_gamemode_model": True,
        "generation_compile": False,
        "server_allow_auto_start": True,
        "server_connect_timeout": 60.0,
        "server_request_timeout": None,
        "server_idle_timeout": 20,
        "server_batch_timeout": 0.2,
    }
    values.update(overrides)
    return values


def test_accepted_config_is_frozen_and_matches_legacy_stack():
    config = ACCEPTED_OPTIMIZED_SINGLE_CONFIG
    with pytest.raises(FrozenInstanceError):
        config.active_prefix_decode_bucket_size = 128

    assert config.generation_overrides() == {
        "precision": "fp32",
        "cfg_scale": 1.0,
        "num_beams": 1,
        "active_prefix_decode_loop": True,
        "active_prefix_decode_bucket_size": 64,
        "active_prefix_decode_cuda_graph": True,
        "active_prefix_decode_cuda_graph_warmup": 0,
        "active_prefix_decode_cuda_graph_min_decode_steps": 1,
        "stateful_monotonic_logits_processor": True,
        "q1_bmm_cross_attention": True,
        "native_q1_self_attention": True,
        "native_q1_rope_cache_self_attention": True,
        "decode_session_cuda_graph": True,
    }


def test_runtime_overlays_structure_without_changing_sampling(monkeypatch):
    runtime = OptimizedSingleRuntime()
    captured = {}

    def fake_generate(model, tokenizer, model_kwargs, generate_kwargs, **kwargs):
        captured.update(generate_kwargs)
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
    assert captured["active_prefix_decode_bucket_size"] == 64


def test_single_loader_forces_compile_fp32_sdpa_once():
    raw_model = SimpleNamespace(
        generation_config=SimpleNamespace(disable_compile=False),
    )
    tokenizer = object()
    model_loader = Mock(return_value=(raw_model, tokenizer))

    binding, loaded_tokenizer = load_optimized_single_engine(
        model_loader=model_loader,
        loader_kwargs=_loader_kwargs(),
    )

    assert isinstance(binding, InferenceEngineBinding)
    assert binding.raw_model is raw_model
    assert loaded_tokenizer is tokenizer
    assert model_loader.call_count == 1
    call_kwargs = model_loader.call_args.kwargs
    assert call_kwargs["generation_compile"] is True
    assert call_kwargs["precision"] == "fp32"
    assert call_kwargs["attn_implementation"] == "sdpa"
    assert call_kwargs["use_server"] is False


def test_engine_binding_unwrap_preserves_raw_model_identity():
    raw_model = object()
    runtime = object()
    binding = InferenceEngineBinding(raw_model=raw_model, runtime=runtime)

    assert unwrap_engine_binding(binding) == (raw_model, runtime)
    assert unwrap_engine_binding(raw_model) == (raw_model, None)


@pytest.mark.parametrize("mode", ["offline_batch", "server"])
def test_unsupported_modes_fail_before_loader_or_single_import(mode):
    loader = Mock()
    with patch.object(adapter, "import_module") as single_import:
        with pytest.raises(NotImplementedError, match="control-plane scaffolding"):
            adapter.load_optimized_engine(
                mode=mode,
                model_loader=loader,
                loader_kwargs=_loader_kwargs(),
            )

    loader.assert_not_called()
    single_import.assert_not_called()


def test_inference_loader_injects_raw_loader_and_returns_binding():
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
            optimized_inference_mode="single",
        )

    assert isinstance(binding, InferenceEngineBinding)
    assert binding.raw_model is raw_model
    assert loaded_tokenizer is tokenizer
    assert raw_loader.call_count == 1
    assert raw_loader.call_args.kwargs["generation_compile"] is True


def test_adapter_unsupported_mode_is_fresh_process_single_cold():
    repo_root = Path(__file__).resolve().parents[1]
    code = r'''\
import sys
from osuT5.osuT5.inference.optimized.adapter import load_optimized_engine

try:
    load_optimized_engine(
        mode="server",
        model_loader=lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
        loader_kwargs={},
    )
except NotImplementedError:
    pass
else:
    raise AssertionError("optimized server mode unexpectedly loaded")

assert "osuT5.osuT5.inference.optimized.single.engine" not in sys.modules
assert not any(name.startswith("osuT5.osuT5.inference.optimized.kernels") for name in sys.modules)
assert "torch.utils.cpp_extension" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_engine_shared_calculation_helpers_match_legacy_server():
    class FakeTokenizer:
        eos_id = 2
        pad_id = 0
        context_eos = {ContextType.MAP: 6}
        event_start = {EventType.TIME_SHIFT: 10}
        event_end = {EventType.TIME_SHIFT: 20}

    tokenizer = FakeTokenizer()
    assert engine_module._eos_token_ids(
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
    assert engine_module._build_generation_stats(
        result,
        model_kwargs,
        0,
        0.5,
    ) == server._build_generation_stats(result, model_kwargs, 0, 0.5)
