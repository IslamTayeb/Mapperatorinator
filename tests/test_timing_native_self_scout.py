from __future__ import annotations

from contextlib import contextmanager
import inspect
from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.event import ContextType
from osuT5.osuT5.inference.optimized.single import engine
from osuT5.osuT5.inference.optimized.single.state import ProductionDecodeSession
from osuT5.osuT5.model.custom_transformers.modeling_varwhisper import (
    VarWhisperDecoderLayer,
)


class _Model(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=dtype))

    @property
    def dtype(self) -> torch.dtype:
        return self.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.weight.device

    def generate(self, **kwargs):
        return torch.tensor([[1, 2]])


class _Session(ProductionDecodeSession):
    graph_count = 0
    graph_capture_seconds = 0.0
    graph_decode_replays = 0

    @staticmethod
    def cache_for_window(*args, **kwargs):
        return object()

    @staticmethod
    def active_prefix_decode_kwargs():
        return {}

    @staticmethod
    def graph_profile_summary():
        return {
            "graph_count": 0,
            "decode_replays": 0,
            "capture_seconds": 0.0,
            "buckets": {},
        }


def _window_inputs():
    return {
        "inputs": torch.zeros((1, 1)),
        "decoder_input_ids": torch.tensor([[1]]),
        "decoder_attention_mask": torch.tensor([[1]]),
    }


def test_initializer_is_fp32_model_owned_and_default_metadata_stays_unchanged() -> None:
    runtime = engine.OptimizedSingleRuntime(engine.OPTIMIZED_PRESETS["fp32"])
    model = _Model(torch.float32)
    before = runtime.profile_metadata()

    metadata = runtime.initialize_fp32_timing_native_self(model)

    assert metadata == {
        "version": engine.TIMING_NATIVE_SELF_SCOUT_VERSION,
        "scope": "timing-context-only",
        "precision": "fp32",
        "torch_dtype": "torch.float32",
        "result_class": "documented-drift",
        "exactness_claim": False,
        "native_q1_self_attention": True,
        "native_q1_rope_cache_self_attention": True,
        "split_kv_selector": True,
        "q1_bmm_cross_attention_retained": True,
        "native_cross_mlp_tail_retained_disabled": True,
        "approximate_weight_only_retained_disabled": True,
        "original_decoder_forward_retained": True,
        "production_selector_unchanged": True,
    }
    assert "optimized_timing_native_self_scout" not in before
    assert runtime.profile_metadata()["optimized_timing_native_self_scout"] == metadata
    assert runtime.initialize_fp32_timing_native_self(model) == metadata
    with pytest.raises(RuntimeError, match="different model"):
        runtime.initialize_fp32_timing_native_self(_Model(torch.float32))

    fp16_runtime = engine.OptimizedSingleRuntime(engine.OPTIMIZED_PRESETS["fp16"])
    with pytest.raises(TypeError, match="FP32 preset"):
        fp16_runtime.initialize_fp32_timing_native_self(_Model(torch.float16))


def test_timing_scout_enables_only_native_self_and_retains_original_decoder(
    monkeypatch,
) -> None:
    captured: list[dict[str, object]] = []

    @contextmanager
    def fake_profile_context(**kwargs):
        captured.append(kwargs)
        yield

    model = _Model(torch.float32)
    runtime = engine.OptimizedSingleRuntime(engine.OPTIMIZED_PRESETS["fp32"])
    runtime.initialize_fp32_timing_native_self(model)
    original_forward = VarWhisperDecoderLayer.forward
    monkeypatch.setattr(engine, "generation_profile_context", fake_profile_context)
    monkeypatch.setattr(engine, "_build_logits_processor_list", lambda *args, **kwargs: [])

    _, stats = runtime.generate_window(
        model=model,
        tokenizer=SimpleNamespace(pad_id=0, eos_id=2, context_eos={}),
        model_kwargs=_window_inputs(),
        generate_kwargs={"context_type": ContextType.TIMING, "pad_token_id": 0},
        context_state=_Session(),
    )

    dispatch_counts = captured[0].pop("optimized_dispatch_counts")
    assert captured[0] == {
        "q1_bmm_cross_attention": True,
        "native_q1_self_attention": True,
        "native_q1_rope_cache_self_attention": True,
        "native_cross_mlp_tail": False,
        "optimized_expected_dtype": torch.float32,
    }
    assert dispatch_counts == stats["optimized_dispatch_capture_hits"]
    assert stats["optimized_dispatch_mode"] == "fp32_timing_native_self_batch1"
    assert stats["optimized_dispatch_policy"]["timing_native_self_scout"] == {
        "requested": True,
        "enabled": True,
        "disabled_reason": None,
        "result_class": "documented-drift",
        "exactness_claim": False,
    }
    assert stats["optimized_dispatch_policy"]["native_cross_mlp_tail"]["enabled"] is False
    assert "approximate_weight_only" not in stats["optimized_dispatch_policy"]
    assert VarWhisperDecoderLayer.forward is original_forward
    assert "module.forward =" not in inspect.getsource(VarWhisperDecoderLayer.forward)


def test_timing_scout_fails_on_non_timing_use_or_mixed_runtime(monkeypatch) -> None:
    model = _Model(torch.float32)
    runtime = engine.OptimizedSingleRuntime(engine.OPTIMIZED_PRESETS["fp32"])
    runtime.initialize_fp32_timing_native_self(model)
    monkeypatch.setattr(engine, "_build_logits_processor_list", lambda *args, **kwargs: [])

    with pytest.raises(RuntimeError, match="only timing context"):
        runtime.generate_window(
            model=model,
            tokenizer=SimpleNamespace(pad_id=0, eos_id=2, context_eos={}),
            model_kwargs=_window_inputs(),
            generate_kwargs={"context_type": ContextType.MAP, "pad_token_id": 0},
            context_state=_Session(),
        )

    object.__setattr__(runtime, "_approximate_weight_only_state", object())
    with pytest.raises(RuntimeError, match="without mixed weights"):
        runtime.initialize_fp32_timing_native_self(model)
