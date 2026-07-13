from __future__ import annotations

import inspect
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


def test_scout_runtime_is_explicit_and_default_preset_is_unchanged(
    monkeypatch,
) -> None:
    raw_model = SimpleNamespace(
        generation_config=SimpleNamespace(disable_compile=False),
    )
    loader = lambda **kwargs: (raw_model, object())
    kwargs = {
        "precision": "fp32",
        "attn_implementation": "sdpa",
        "use_server": False,
    }

    monkeypatch.delenv(engine.NATIVE_CROSS_MLP_SCOUT_ENV, raising=False)
    accepted, _ = engine.load_optimized_single_engine(
        model_loader=loader,
        loader_kwargs=kwargs,
    )
    assert type(accepted.runtime) is engine.OptimizedSingleRuntime
    assert "native_cross_mlp_tail" not in accepted.runtime.profile_metadata()[
        "optimized_effective_config"
    ]

    monkeypatch.setenv(engine.NATIVE_CROSS_MLP_SCOUT_ENV, "1")
    candidate, _ = engine.load_optimized_single_engine(
        model_loader=loader,
        loader_kwargs=kwargs,
    )
    assert type(candidate.runtime) is engine.NativeCrossMlpScoutRuntime
    assert candidate.runtime.profile_metadata()["optimized_effective_config"][
        "native_cross_mlp_tail"
    ] is True
    assert (
        candidate.runtime.profile_metadata()["optimized_result_class"]
        == "candidate-unverified"
    )

    monkeypatch.setenv(engine.NATIVE_CROSS_MLP_SCOUT_ENV, "invalid")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        engine.load_optimized_single_engine(
            model_loader=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("invalid selector must fail before model load")
            ),
            loader_kwargs=kwargs,
        )


def test_scout_tail_is_disabled_only_for_timing_context() -> None:
    assert engine._native_cross_mlp_tail_enabled(
        requested=True,
        context_type=ContextType.MAP,
    )
    assert engine._native_cross_mlp_tail_enabled(
        requested=True,
        context_type=None,
    )
    assert not engine._native_cross_mlp_tail_enabled(
        requested=True,
        context_type=ContextType.TIMING,
    )
    assert not engine._native_cross_mlp_tail_enabled(
        requested=False,
        context_type=ContextType.MAP,
    )
