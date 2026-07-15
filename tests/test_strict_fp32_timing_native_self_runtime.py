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


@pytest.fixture(autouse=True)
def _strict_fp32_environment(monkeypatch):
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "highest")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)


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


def _window_inputs(batch_size: int = 1):
    return {
        "inputs": torch.zeros((batch_size, 1)),
        "decoder_input_ids": torch.ones((batch_size, 1), dtype=torch.long),
        "decoder_attention_mask": torch.ones((batch_size, 1), dtype=torch.long),
    }


def _metadata() -> dict:
    return {
        "version": engine.STRICT_FP32_TIMING_NATIVE_SELF_VERSION,
        "scope": "standalone-timing-model-batch1-only",
        "precision": "fp32",
        "torch_dtype": "torch.float32",
        "tf32_disabled_required": True,
        "result_class": "exact-incremental-candidate",
        "exactness_claim": True,
        "native_q1_self_attention": True,
        "native_q1_rope_cache_self_attention": True,
        "q1_bmm_cross_attention_retained": True,
        "native_cross_mlp_tail_retained_disabled": True,
        "original_decoder_forward_retained": True,
        "accepted_kernel_implementation_reused": True,
        "owns_model_weights": False,
        "reduced_precision_weights": False,
        "reduced_precision_activations": False,
        "counter_rng": False,
        "production_selector_unchanged": True,
    }


def test_initializer_is_model_owned_fp32_exact_and_cold_by_default() -> None:
    runtime = engine.OptimizedSingleRuntime(engine.OPTIMIZED_PRESETS["fp32"])
    model = _Model(torch.float32)
    default_metadata = runtime.profile_metadata()

    assert runtime._strict_fp32_timing_native_self_owner is None
    assert "optimized_strict_fp32_timing_native_self" not in default_metadata
    assert runtime.initialize_strict_fp32_timing_native_self(model) == _metadata()
    assert runtime.initialize_strict_fp32_timing_native_self(model) == _metadata()
    assert runtime.profile_metadata()[
        "optimized_strict_fp32_timing_native_self"
    ] == _metadata()
    assert runtime.profile_metadata()["optimized_effective_config"] == (
        default_metadata["optimized_effective_config"]
    )
    with pytest.raises(RuntimeError, match="different model"):
        runtime.initialize_strict_fp32_timing_native_self(_Model(torch.float32))
    with pytest.raises(TypeError, match="FP32 preset"):
        engine.OptimizedSingleRuntime(
            engine.OPTIMIZED_PRESETS["fp16"]
        ).initialize_strict_fp32_timing_native_self(_Model(torch.float16))


def test_candidate_reuses_accepted_dispatch_and_original_decoder(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    @contextmanager
    def fake_profile_context(**kwargs):
        captured.append(kwargs)
        yield

    model = _Model(torch.float32)
    runtime = engine.OptimizedSingleRuntime(engine.OPTIMIZED_PRESETS["fp32"])
    runtime.initialize_strict_fp32_timing_native_self(model)
    original_forward = VarWhisperDecoderLayer.forward
    monkeypatch.setattr(engine, "generation_profile_context", fake_profile_context)
    monkeypatch.setattr(
        engine,
        "_build_logits_processor_list",
        lambda *args, **kwargs: [],
    )

    _, stats = runtime.generate_window(
        model=model,
        tokenizer=SimpleNamespace(pad_id=0, eos_id=2, context_eos={}),
        model_kwargs=_window_inputs(),
        generate_kwargs={"context_type": ContextType.TIMING, "pad_token_id": 0},
        context_state=_Session(),
    )

    counts = captured[0].pop("optimized_dispatch_counts")
    assert captured[0] == {
        "q1_bmm_cross_attention": True,
        "native_q1_self_attention": True,
        "native_q1_rope_cache_self_attention": True,
        "native_cross_mlp_tail": False,
        "optimized_expected_dtype": torch.float32,
    }
    assert counts == stats["optimized_dispatch_capture_hits"]
    assert stats["optimized_dispatch_mode"] == (
        "strict_fp32_timing_native_self_batch1"
    )
    assert stats["optimized_dispatch_policy"][
        "strict_fp32_timing_native_self"
    ] == {
        "requested": True,
        "enabled": True,
        "disabled_reason": None,
        "result_class": "exact-incremental-candidate",
        "exactness_claim": True,
    }
    assert stats["optimized_strict_fp32_timing_native_self"] == _metadata()
    assert stats["optimized_dispatch_policy"]["q1_bmm_cross_attention"][
        "enabled"
    ] is True
    assert stats["optimized_dispatch_policy"]["native_cross_mlp_tail"][
        "enabled"
    ] is False
    assert VarWhisperDecoderLayer.forward is original_forward
    assert "module.forward =" not in inspect.getsource(VarWhisperDecoderLayer.forward)


def test_uninitialized_timing_fallback_and_main_dispatch_are_unchanged(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    @contextmanager
    def fake_profile_context(**kwargs):
        captured.append(kwargs)
        yield

    model = _Model(torch.float32)
    runtime = engine.OptimizedSingleRuntime(engine.OPTIMIZED_PRESETS["fp32"])
    monkeypatch.setattr(engine, "generation_profile_context", fake_profile_context)
    monkeypatch.setattr(
        engine,
        "_build_logits_processor_list",
        lambda *args, **kwargs: [],
    )
    _, timing = runtime.generate_window(
        model=model,
        tokenizer=SimpleNamespace(pad_id=0, eos_id=2, context_eos={}),
        model_kwargs=_window_inputs(),
        generate_kwargs={"context_type": ContextType.TIMING, "pad_token_id": 0},
        context_state=_Session(),
    )
    _, main = runtime.generate_window(
        model=model,
        tokenizer=SimpleNamespace(pad_id=0, eos_id=2, context_eos={}),
        model_kwargs=_window_inputs(),
        generate_kwargs={"context_type": ContextType.MAP, "pad_token_id": 0},
        context_state=_Session(),
    )

    assert captured[0]["native_q1_self_attention"] is False
    assert captured[0]["native_q1_rope_cache_self_attention"] is False
    assert captured[0]["q1_bmm_cross_attention"] is True
    assert captured[1]["native_q1_self_attention"] is True
    assert captured[1]["native_q1_rope_cache_self_attention"] is True
    assert captured[1]["q1_bmm_cross_attention"] is True
    assert timing["optimized_dispatch_mode"] == "accepted_batch1"
    assert main["optimized_dispatch_mode"] == "accepted_batch1"
    assert "optimized_strict_fp32_timing_native_self" not in timing
    assert "optimized_strict_fp32_timing_native_self" not in main


def test_candidate_fails_loudly_outside_owned_timing_batch1(monkeypatch) -> None:
    model = _Model(torch.float32)
    runtime = engine.OptimizedSingleRuntime(engine.OPTIMIZED_PRESETS["fp32"])
    runtime.initialize_strict_fp32_timing_native_self(model)
    monkeypatch.setattr(
        engine,
        "_build_logits_processor_list",
        lambda *args, **kwargs: [],
    )
    common = {
        "model": model,
        "tokenizer": SimpleNamespace(pad_id=0, eos_id=2, context_eos={}),
        "context_state": _Session(),
    }
    with pytest.raises(RuntimeError, match="only timing context"):
        runtime.generate_window(
            **common,
            model_kwargs=_window_inputs(),
            generate_kwargs={"context_type": ContextType.MAP, "pad_token_id": 0},
        )
    with pytest.raises(RuntimeError, match="different model"):
        engine._generate_window(
            _Model(torch.float32),
            common["tokenizer"],
            _window_inputs(),
            {"context_type": ContextType.TIMING, "pad_token_id": 0},
            context_state=_Session(),
            preset=engine.OPTIMIZED_PRESETS["fp32"],
            strict_fp32_timing_native_self_owner=model,
        )
