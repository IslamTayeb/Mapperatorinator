from __future__ import annotations

import inspect
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from transformers.activations import GELUActivation

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
    assert source.count("AT_DISPATCH_FLOATING_TYPES_AND_HALF") == 3
    assert "template<typename scalar_t, int BLOCK_SIZE>" in source
    assert source.count("template<typename scalar_t, int OUTPUTS_PER_BLOCK>") == 4
    assert "input must be fp32 or fp16" in source
    assert "dtype must match input" in source
    assert "static_cast<float>(input[idx])" in source
    assert "static_cast<scalar_t>(exact_gelu(sum))" in source
    assert "float local_sum = 0.0f" in source
    assert "float local_dot = 0.0f" in source
    assert "float sum = 0.0f" in source


def test_native_decoder_wrappers_preserve_fp16_storage_and_argument_order(
    monkeypatch,
) -> None:
    from osuT5.osuT5.inference.optimized.kernels import decoder_layer

    calls = []

    class _Extension:
        def one_token_mlp_residual(self, *args):
            calls.append(("mlp", args))
            return args[0]

        def one_token_rmsnorm_linear(self, *args):
            calls.append(("rmsnorm_linear", args))
            return args[0]

        def one_token_linear_residual(self, *args):
            calls.append(("linear_residual", args))
            return args[0]

    monkeypatch.setattr(decoder_layer, "_NATIVE_DECODER_LAYER", _Extension())
    hidden = torch.zeros((1, 1, 4), dtype=torch.float16)
    norm = torch.ones(4, dtype=torch.float16)
    square = torch.eye(4, dtype=torch.float16)
    bias = torch.zeros(4, dtype=torch.float16)

    assert decoder_layer.native_one_token_rmsnorm_linear(
        hidden,
        norm,
        square,
        bias,
        eps=1e-5,
        outputs_per_block=8,
    ).dtype == torch.float16
    kind, args = calls.pop(0)
    assert kind == "rmsnorm_linear"
    assert all(
        actual is expected
        for actual, expected in zip(args[:4], (hidden, norm, square, bias), strict=True)
    )
    assert args[4:] == (1e-5, 8)

    assert decoder_layer.native_one_token_linear_residual(
        hidden,
        hidden,
        square,
        bias,
        outputs_per_block=8,
    ).dtype == torch.float16
    kind, args = calls.pop(0)
    assert kind == "linear_residual"
    assert all(
        actual is expected
        for actual, expected in zip(args[:4], (hidden, hidden, square, bias), strict=True)
    )
    assert args[4:] == (8,)

    assert decoder_layer.native_one_token_mlp_residual(
        hidden,
        norm,
        square,
        bias,
        square,
        bias,
        eps=1e-5,
        outputs_per_block=8,
    ).dtype == torch.float16
    kind, args = calls.pop(0)
    assert kind == "mlp"
    assert all(
        actual is expected
        for actual, expected in zip(
            args[:6],
            (hidden, norm, square, bias, square, bias),
            strict=True,
        )
    )
    assert args[6:] == (1e-5, 8)


def test_native_cross_mlp_is_enabled_in_both_accepted_presets() -> None:
    for precision in ("fp32", "fp16"):
        raw_model = SimpleNamespace(
            generation_config=SimpleNamespace(disable_compile=True),
        )
        loader = lambda **kwargs: (raw_model, object())
        kwargs = {
            "precision": precision,
            "attn_implementation": "sdpa",
            "use_server": False,
        }

        candidate, _ = engine.load_optimized_single_engine(
            model_loader=loader,
            loader_kwargs=kwargs,
        )
        assert type(candidate.runtime) is engine.OptimizedSingleRuntime
        assert candidate.runtime.profile_metadata()["optimized_effective_config"][
            "native_cross_mlp_tail"
        ] is True
        assert (
            candidate.runtime.profile_metadata()["optimized_result_class"]
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


def test_ordinary_generate_window_installs_shared_specialized_dispatch(monkeypatch) -> None:
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

        @staticmethod
        def graph_profile_summary():
            return {
                "graph_count": 0,
                "decode_replays": 0,
                "capture_seconds": 0.0,
                "buckets": {},
            }

    model = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float16,
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
        preset=engine.OPTIMIZED_PRESETS["fp16"],
    )
    _, timing_stats = engine._generate_window(
        model,
        tokenizer,
        model_kwargs,
        {"context_type": ContextType.TIMING, "pad_token_id": 0},
        context_state=FakeSession(),
        preset=engine.OPTIMIZED_PRESETS["fp16"],
    )

    main_dispatch_counts = captured[0].pop("optimized_dispatch_counts")
    timing_dispatch_counts = captured[1].pop("optimized_dispatch_counts")
    assert captured[0] == {
        "q1_bmm_cross_attention": True,
        "native_q1_self_attention": True,
        "native_q1_rope_cache_self_attention": True,
        "native_cross_mlp_tail": True,
        "optimized_expected_dtype": torch.float16,
    }
    assert captured[1] == {
        "q1_bmm_cross_attention": True,
        "native_q1_self_attention": False,
        "native_q1_rope_cache_self_attention": False,
        "native_cross_mlp_tail": False,
        "optimized_expected_dtype": torch.float16,
    }
    assert main_stats["native_cross_mlp_tail_enabled"] is True
    assert timing_stats["native_cross_mlp_tail_enabled"] is False
    assert timing_stats["native_cross_mlp_tail_disabled_reason"] == "timing_context"
    assert main_dispatch_counts == main_stats["optimized_dispatch_capture_hits"]
    assert timing_dispatch_counts == timing_stats["optimized_dispatch_capture_hits"]


def test_batched_super_timing_window_deliberately_disables_all_native_dispatch(
    monkeypatch,
) -> None:
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
        dtype=torch.float16,
        generate=lambda **kwargs: torch.tensor([[1, 2], [1, 2]]),
    )
    tokenizer = SimpleNamespace(pad_id=0, eos_id=2, context_eos={})
    model_kwargs = {
        "inputs": torch.zeros(2, 1),
        "decoder_input_ids": torch.tensor([[1], [1]]),
        "decoder_attention_mask": torch.tensor([[1], [1]]),
    }
    monkeypatch.setattr(engine, "generation_profile_context", fake_profile_context)
    monkeypatch.setattr(
        engine,
        "_build_logits_processor_list",
        lambda *args, **kwargs: [],
    )

    _, stats = engine._generate_window(
        model,
        tokenizer,
        model_kwargs,
        {"context_type": ContextType.MAP, "pad_token_id": 0},
        context_state=FakeSession(),
        preset=engine.OPTIMIZED_PRESETS["fp16"],
        allow_batched_decode=True,
    )

    dispatch_counts = captured[0].pop("optimized_dispatch_counts")
    assert captured[0] == {
        "q1_bmm_cross_attention": False,
        "native_q1_self_attention": False,
        "native_q1_rope_cache_self_attention": False,
        "native_cross_mlp_tail": False,
        "optimized_expected_dtype": torch.float16,
    }
    assert dispatch_counts == stats["optimized_dispatch_capture_hits"] == {
        "native_q1_rope_cache_self_attention": 0,
        "native_q1_self_attention": 0,
        "q1_bmm_cross_attention": 0,
        "native_cross_mlp_tail": 0,
    }
    assert stats["optimized_dispatch_mode"] == "framework_batch"
    assert stats["optimized_batched_super_timing"] is True
    assert {
        row["disabled_reason"]
        for row in stats["optimized_dispatch_policy"].values()
    } == {"batch_gt_1"}


def test_nominal_batched_policy_keeps_a_one_row_tail_on_framework(monkeypatch):
    captured = []

    @contextmanager
    def fake_profile_context(**kwargs):
        captured.append(kwargs)
        yield

    context_state = SimpleNamespace(
        cache_for_window=lambda *args, **kwargs: object(),
        active_prefix_decode_kwargs=lambda: {},
    )
    model = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        generate=lambda **kwargs: torch.tensor([[1, 2]]),
    )
    monkeypatch.setattr(engine, "generation_profile_context", fake_profile_context)
    monkeypatch.setattr(
        engine,
        "_build_logits_processor_list",
        lambda *args, **kwargs: [],
    )

    _, stats = engine._generate_window(
        model,
        SimpleNamespace(pad_id=0, eos_id=2, context_eos={}),
        {
            "inputs": torch.zeros(1, 1),
            "decoder_input_ids": torch.tensor([[1]]),
            "decoder_attention_mask": torch.tensor([[1]]),
        },
        {"context_type": ContextType.MAP, "pad_token_id": 0},
        context_state=context_state,
        preset=engine.OPTIMIZED_PRESETS["fp32"],
        allow_batched_decode=True,
        specialized_dispatch_batch_size=4,
    )

    dispatch_counts = captured[0].pop("optimized_dispatch_counts")
    assert all(value is False for value in captured[0].values() if isinstance(value, bool))
    assert all(value == 0 for value in dispatch_counts.values())
    assert stats["optimized_dispatch_mode"] == "framework_batch"
    assert {
        row["disabled_reason"]
        for row in stats["optimized_dispatch_policy"].values()
    } == {"nominal_batched_policy"}
    assert (
        stats["native_cross_mlp_tail_disabled_reason"]
        == "nominal_batched_policy"
    )


@pytest.mark.parametrize(
    ("precision", "wrong_dtype"),
    (("fp32", torch.float16), ("fp16", torch.float32)),
)
def test_optimized_runtime_rejects_model_dtype_that_disagrees_with_metadata(
    precision,
    wrong_dtype,
) -> None:
    model = SimpleNamespace(device=torch.device("cpu"), dtype=wrong_dtype)

    with pytest.raises(TypeError, match="runtime loaded model dtype"):
        engine._generate_window(
            model,
            SimpleNamespace(),
            {},
            {},
            context_state=SimpleNamespace(),
            preset=engine.OPTIMIZED_PRESETS[precision],
        )


@pytest.mark.parametrize(
    ("preset_precision", "model_dtype", "requested_precision"),
    (
        ("fp32", torch.float32, "fp16"),
        ("fp16", torch.float16, "fp32"),
    ),
)
def test_bound_runtime_rejects_generation_time_precision_switches(
    preset_precision,
    model_dtype,
    requested_precision,
) -> None:
    model = SimpleNamespace(device=torch.device("cpu"), dtype=model_dtype)

    with pytest.raises(ValueError, match="configuration changed after validation"):
        engine._generate_window(
            model,
            SimpleNamespace(),
            {"inputs": torch.zeros((1, 1), dtype=model_dtype)},
            {"precision": requested_precision},
            context_state=SimpleNamespace(),
            preset=engine.OPTIMIZED_PRESETS[preset_precision],
        )


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
        activation_fn=GELUActivation(),
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

    def cross_bmm(*args, **kwargs):
        events.extend(("bmm", "softmax", "bmm"))
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
    monkeypatch.setattr(cross_mlp, "_q1_bmm_cross_attention", cross_bmm)
    monkeypatch.setattr(
        cross_mlp,
        "_validate_cross_mlp_tensors",
        lambda *args: (cache_tensor, cache_tensor),
    )

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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_native_tail_tensor_validation_accepts_one_shared_storage_dtype(dtype) -> None:
    from osuT5.osuT5.inference.optimized.kernels import cross_mlp

    hidden = torch.zeros((1, 1, 4), dtype=dtype)
    weight = torch.zeros((4, 4), dtype=dtype)
    bias = torch.zeros(4, dtype=dtype)
    module = SimpleNamespace(
        cross_attn_layer_norm=SimpleNamespace(weight=bias),
        cross_attn=SimpleNamespace(
            Wq=SimpleNamespace(weight=weight, bias=bias),
            Wo=SimpleNamespace(weight=weight, bias=bias),
        ),
        final_layer_norm=SimpleNamespace(weight=bias),
        fc1=SimpleNamespace(weight=weight, bias=bias),
        fc2=SimpleNamespace(weight=weight, bias=bias),
    )
    cache_layer = SimpleNamespace(
        keys=torch.zeros((1, 2, 3, 2), dtype=dtype),
        values=torch.zeros((1, 2, 3, 2), dtype=dtype),
    )

    keys, values = cross_mlp._validate_cross_mlp_tensors(
        module,
        hidden,
        cache_layer,
    )

    assert keys.dtype == dtype
    assert values.dtype == dtype


def test_native_tail_tensor_validation_rejects_mixed_storage_dtype() -> None:
    from osuT5.osuT5.inference.optimized.kernels import cross_mlp

    hidden = torch.zeros((1, 1, 4), dtype=torch.float16)
    half_weight = torch.zeros((4, 4), dtype=torch.float16)
    half_bias = torch.zeros(4, dtype=torch.float16)
    module = SimpleNamespace(
        cross_attn_layer_norm=SimpleNamespace(weight=half_bias),
        cross_attn=SimpleNamespace(
            Wq=SimpleNamespace(weight=half_weight.float(), bias=half_bias),
            Wo=SimpleNamespace(weight=half_weight, bias=half_bias),
        ),
        final_layer_norm=SimpleNamespace(weight=half_bias),
        fc1=SimpleNamespace(weight=half_weight, bias=half_bias),
        fc2=SimpleNamespace(weight=half_weight, bias=half_bias),
    )
    cache_layer = SimpleNamespace(
        keys=torch.zeros((1, 2, 3, 2), dtype=torch.float16),
        values=torch.zeros((1, 2, 3, 2), dtype=torch.float16),
    )

    with pytest.raises(TypeError, match="cross query weight dtype"):
        cross_mlp._validate_cross_mlp_tensors(module, hidden, cache_layer)
