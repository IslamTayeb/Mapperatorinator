from __future__ import annotations

import inspect
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.inference import runtime_dispatch
from osuT5.osuT5.inference.optimized.single import engine
from osuT5.osuT5.event import ContextType
from osuT5.osuT5.model.custom_transformers.modeling_varwhisper import (
    VarWhisperDecoderLayer,
)


def test_decoder_hook_context_restores_nested_and_exception_paths() -> None:
    empty = runtime_dispatch.decoder_layer_runtime_hooks()
    first = runtime_dispatch.DecoderLayerRuntimeHooks(
        cross_mlp_tail_forward=lambda **kwargs: ("first", kwargs),
    )
    second = runtime_dispatch.DecoderLayerRuntimeHooks(
        cross_mlp_tail_forward=lambda **kwargs: ("second", kwargs),
    )

    with runtime_dispatch.decoder_layer_runtime_hooks_context(first):
        assert runtime_dispatch.decoder_layer_runtime_hooks() is first
        with runtime_dispatch.decoder_layer_runtime_hooks_context(second):
            assert runtime_dispatch.decoder_layer_runtime_hooks() is second
        assert runtime_dispatch.decoder_layer_runtime_hooks() is first
        with pytest.raises(RuntimeError, match="boom"):
            with runtime_dispatch.decoder_layer_runtime_hooks_context(second):
                raise RuntimeError("boom")
        assert runtime_dispatch.decoder_layer_runtime_hooks() is first
    assert runtime_dispatch.decoder_layer_runtime_hooks() is empty


def test_decoder_context_preloads_before_installing_hook(monkeypatch) -> None:
    from osuT5.osuT5.inference.optimized.kernels import decoder_layer
    from osuT5.osuT5.inference.optimized.single.runtime_context import (
        decoder_layer_runtime_context,
    )

    events: list[str] = []
    monkeypatch.setattr(
        decoder_layer,
        "preload_native_decoder_layer",
        lambda: events.append("preload"),
    )
    with decoder_layer_runtime_context(native_cross_mlp_tail=True):
        events.append("entered")
        assert (
            runtime_dispatch.decoder_layer_runtime_hooks().cross_mlp_tail_forward
            is not None
        )
    assert events == ["preload", "entered"]
    assert (
        runtime_dispatch.decoder_layer_runtime_hooks().cross_mlp_tail_forward
        is None
    )


def test_decoder_layer_keeps_original_forward_and_uses_narrow_hook() -> None:
    forward = VarWhisperDecoderLayer.forward
    source = inspect.getsource(forward)

    with runtime_dispatch.decoder_layer_runtime_hooks_context(
        runtime_dispatch.DecoderLayerRuntimeHooks(
            cross_mlp_tail_forward=lambda **kwargs: None,
        )
    ):
        assert VarWhisperDecoderLayer.forward is forward

    assert "decoder_layer_runtime_hooks().cross_mlp_tail_forward" in source
    assert "module.forward =" not in source


def test_cpu_inputs_fall_back_before_native_kernel_call(monkeypatch) -> None:
    from osuT5.osuT5.inference.optimized.kernels import cross_mlp

    def must_not_run(*args, **kwargs):
        raise AssertionError("native kernel must not run")

    monkeypatch.setattr(cross_mlp, "native_one_token_rmsnorm_linear", must_not_run)
    module = SimpleNamespace(
        training=False,
        activation_dropout=0.0,
        dropout=0.0,
    )
    output = cross_mlp.native_cross_mlp_tail_forward(
        module=module,
        hidden_states=torch.zeros(1, 1, 4),
        encoder_hidden_states=torch.zeros(1, 2, 4),
        past_key_value=object(),
        self_attn_outputs=(torch.zeros(1, 1, 4),),
        output_attentions=False,
        cu_seqlens=None,
        encoder_cu_seqlens=None,
        layer_name="decoder.layer0",
    )
    assert output is None


def test_native_decoder_extension_is_cold_singleton_and_uses_device_guard(
    monkeypatch,
) -> None:
    from osuT5.osuT5.inference.optimized.kernels import decoder_layer

    calls = []
    extension = object()

    def fake_load_inline(**kwargs):
        calls.append(kwargs)
        return extension

    monkeypatch.setattr(decoder_layer, "_NATIVE_DECODER_LAYER", None)
    monkeypatch.setattr(decoder_layer, "load_inline", fake_load_inline)
    assert decoder_layer.preload_native_decoder_layer() is extension
    assert decoder_layer.preload_native_decoder_layer() is extension
    assert len(calls) == 1
    assert calls[0]["name"] == "mapperatorinator_native_decoder_layer"
    assert calls[0]["functions"] == [
        "one_token_mlp_residual",
        "one_token_rmsnorm_linear",
        "one_token_linear_residual",
    ]
    source = calls[0]["cuda_sources"]
    assert "#include <c10/cuda/CUDAGuard.h>" in source
    assert source.count("c10::cuda::CUDAGuard") == 3


def test_native_cross_mlp_is_the_single_accepted_optimized_preset() -> None:
    raw_model = SimpleNamespace(
        generation_config=SimpleNamespace(disable_compile=False),
    )
    loader = lambda **kwargs: (raw_model, object())
    kwargs = {
        "precision": "fp32",
        "attn_implementation": "sdpa",
        "use_server": False,
    }

    accepted, _ = engine.load_optimized_single_engine(
        model_loader=loader,
        loader_kwargs=kwargs,
    )
    assert type(accepted.runtime) is engine.OptimizedSingleRuntime
    assert accepted.runtime.profile_metadata()["optimized_effective_config"][
        "native_cross_mlp_tail"
    ] is True
    assert (
        accepted.runtime.profile_metadata()["optimized_result_class"]
        == "documented-drift"
    )


def test_accepted_tail_is_disabled_only_for_timing_context() -> None:
    assert engine._native_cross_mlp_tail_enabled(
        context_type=ContextType.MAP,
    )
    assert engine._native_cross_mlp_tail_enabled(
        context_type=None,
    )
    assert not engine._native_cross_mlp_tail_enabled(
        context_type=ContextType.TIMING,
    )


def test_ordinary_generate_window_installs_promoted_dispatch(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    @contextmanager
    def fake_profile_context(**kwargs):
        captured.append(kwargs)
        yield

    class FakeSession:
        @staticmethod
        def cache_for_window(*args, **kwargs):
            return object()

        @staticmethod
        def active_prefix_decode_kwargs():
            return {}

    model = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        generate=lambda **kwargs: torch.tensor([[1, 2]]),
    )
    tokenizer = SimpleNamespace(pad_id=0, eos_id=2, context_eos={})
    model_kwargs = {
        "inputs": torch.zeros(1, 1),
        "decoder_input_ids": torch.tensor([[1]]),
        "decoder_attention_mask": torch.tensor([[1]]),
    }
    monkeypatch.setattr(engine, "generation_profile_context", fake_profile_context)
    monkeypatch.setattr(engine, "_build_logits_processor_list", lambda *args, **kwargs: [])

    _, main_stats = engine._generate_window(
        model,
        tokenizer,
        model_kwargs,
        {"context_type": ContextType.MAP, "pad_token_id": 0},
        context_state=FakeSession(),
    )
    _, timing_stats = engine._generate_window(
        model,
        tokenizer,
        model_kwargs,
        {"context_type": ContextType.TIMING, "pad_token_id": 0},
        context_state=FakeSession(),
    )

    assert captured[0] == {
        "q1_bmm_cross_attention": True,
        "native_q1_self_attention": True,
        "native_q1_rope_cache_self_attention": True,
        "native_cross_mlp_tail": True,
    }
    assert captured[1] == {
        "q1_bmm_cross_attention": True,
        "native_q1_self_attention": False,
        "native_q1_rope_cache_self_attention": False,
        "native_cross_mlp_tail": False,
    }
    assert main_stats["native_cross_mlp_tail_enabled"] is True
    assert timing_stats["native_cross_mlp_tail_enabled"] is False
    assert timing_stats["native_cross_mlp_tail_disabled_reason"] == "timing_context"


def test_native_tail_preserves_accepted_cross_bmm_operation_order(monkeypatch) -> None:
    from osuT5.osuT5.inference.optimized.kernels import cross_mlp

    events: list[str] = []

    class FakeTensor:
        device = SimpleNamespace(type="cuda")
        dtype = torch.float32
        shape = (1, 1, 4)

        def view(self, *shape):
            self.shape = shape
            return self

        def transpose(self, *args):
            return self

        def reshape(self, *shape):
            self.shape = shape
            return self

        def __mul__(self, other):
            return self

    class FakeCache:
        pass

    hidden = FakeTensor()
    cache_tensor = FakeTensor()
    cache_layer = SimpleNamespace(
        is_initialized=True,
        keys=cache_tensor,
        values=cache_tensor,
    )
    cache = FakeCache()
    cache.is_updated = {0: True}
    cache.cross_attention_cache = SimpleNamespace(layers=[cache_layer])
    weight = object()
    module = SimpleNamespace(
        training=False,
        activation_dropout=0.0,
        dropout=0.0,
        cross_attn=SimpleNamespace(
            layer_idx=0,
            num_heads=2,
            head_dim=2,
            Wq=SimpleNamespace(weight=weight, bias=None),
            Wo=SimpleNamespace(weight=weight, bias=None),
        ),
        cross_attn_layer_norm=SimpleNamespace(weight=weight, eps=1e-5),
        final_layer_norm=SimpleNamespace(weight=weight, eps=1e-5),
        fc1=SimpleNamespace(weight=weight, bias=None),
        fc2=SimpleNamespace(weight=weight, bias=None),
    )

    def rmsnorm_linear(*args, **kwargs):
        events.append("cross_rmsnorm_q")
        return FakeTensor()

    def bmm(*args, **kwargs):
        events.append("bmm")
        return FakeTensor()

    def softmax(*args, **kwargs):
        events.append("softmax")
        return FakeTensor()

    def linear_residual(*args, **kwargs):
        events.append("cross_out_residual")
        return FakeTensor()

    def mlp_residual(*args, **kwargs):
        events.append("mlp")
        return FakeTensor()

    monkeypatch.setattr(cross_mlp, "EncoderDecoderCache", FakeCache)
    monkeypatch.setattr(cross_mlp, "native_one_token_rmsnorm_linear", rmsnorm_linear)
    monkeypatch.setattr(cross_mlp, "native_one_token_linear_residual", linear_residual)
    monkeypatch.setattr(cross_mlp, "native_one_token_mlp_residual", mlp_residual)
    monkeypatch.setattr(cross_mlp.torch, "bmm", bmm)
    monkeypatch.setattr(cross_mlp.torch, "softmax", softmax)

    result = cross_mlp.native_cross_mlp_tail_forward(
        module=module,
        hidden_states=hidden,
        encoder_hidden_states=FakeTensor(),
        past_key_value=cache,
        self_attn_outputs=(hidden,),
        output_attentions=False,
        cu_seqlens=None,
        encoder_cu_seqlens=None,
        layer_name="decoder.layer0",
    )

    assert result is not None
    assert events == [
        "cross_rmsnorm_q",
        "bmm",
        "softmax",
        "bmm",
        "cross_out_residual",
        "mlp",
    ]
