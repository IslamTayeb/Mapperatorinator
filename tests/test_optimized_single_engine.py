from __future__ import annotations

from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from functools import partial
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
    OPTIMIZED_PRESETS,
    SUPER_TIMING_MIN_CONFIGURED_BATCH_SIZE,
    VALIDATED_SUPER_TIMING_ACTUAL_BATCH_SIZES,
    OptimizedPreset,
    OptimizedSingleRuntime,
    OptimizedSuperTimingRuntime,
    _exact_batch_partition,
    build_profiling_batched_super_timing_runtime,
    load_optimized_single_engine,
)
from osuT5.osuT5.inference.processor import Processor
from osuT5.osuT5.inference.profiler import InferenceProfiler
from osuT5.osuT5.inference.super_timing_generator import (
    _configure_optimized_super_timing_runtime,
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
        "fp32": "accepted-fp32-native-cross-mlp-289-v3",
        "fp16": "accepted-fp16-all-fused-v2",
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
            "native_q1_rope_cache_split_kv": precision == "fp32",
            "native_q1_rope_cache_split_kv_split_count": (
                8 if precision == "fp32" else None
            ),
            "native_q1_rope_cache_split_kv_prefix_buckets": (
                tuple(range(192, 833, 64)) if precision == "fp32" else ()
            ),
            "native_cross_mlp_tail": True,
        }


def test_profiling_batched_runtime_persists_one_private_session():
    runtime = build_profiling_batched_super_timing_runtime(
        "fp32",
        nominal_batch_size=4,
    )
    assert isinstance(runtime, OptimizedSuperTimingRuntime)
    assert runtime.preset is OPTIMIZED_PRESETS["fp32"]
    first = runtime.new_context_state()
    second = runtime.new_context_state()
    assert first is second
    metadata = runtime.super_timing_profile_metadata()
    assert metadata["optimized_super_timing_public_wiring"] is False
    assert metadata["optimized_super_timing_config"][
        "framework_dispatch_batch_gt_1"
    ] is True
    assert metadata["optimized_super_timing_config"][
        "validated_actual_batch_sizes"
    ] is None
    assert runtime.plan_batches(num_samples=11, max_batch_size=4) == (4, 4, 3)


@pytest.mark.parametrize(
    ("precision", "maximum"),
    (
        ("fp32", 1),
        ("fp32", 3),
        ("fp32", 15),
        ("fp32", 32),
        ("fp16", 16),
        ("fp16", 32),
    ),
)
def test_single_runtime_preserves_configured_super_timing_maximum(
    precision,
    maximum,
):
    original = OptimizedSingleRuntime(OPTIMIZED_PRESETS[precision])
    timing = original.for_super_timing(max_batch_size=maximum)

    assert isinstance(timing, OptimizedSuperTimingRuntime)
    assert timing is not original
    assert timing.configured_max_batch_size == maximum
    assert timing.public_wiring is True
    assert timing.new_context_state() is timing.new_context_state()
    assert original.new_context_state() is not original.new_context_state()
    with pytest.raises(FrozenInstanceError):
        timing.configured_max_batch_size = 1


def test_super_timing_validated_batches_are_immutable_and_precision_specific():
    assert VALIDATED_SUPER_TIMING_ACTUAL_BATCH_SIZES == {
        "fp32": (1, 2, 3, 4, 8, 10, 11),
        "fp16": (10, 11),
    }
    assert SUPER_TIMING_MIN_CONFIGURED_BATCH_SIZE == {"fp32": 1, "fp16": 16}
    with pytest.raises(TypeError):
        VALIDATED_SUPER_TIMING_ACTUAL_BATCH_SIZES["fp32"] = (32,)
    with pytest.raises(ValueError, match="max_batch_size>=16"):
        OptimizedSingleRuntime(OPTIMIZED_PRESETS["fp16"]).for_super_timing(
            max_batch_size=15
        )


@pytest.mark.parametrize(
    ("total", "allowed", "expected"),
    (
        (1, (1, 2, 3, 4, 8, 10, 11), (1,)),
        (5, (1, 2, 3, 4, 8, 10, 11), (4, 1)),
        (12, (1, 2, 3, 4, 8, 10, 11), (11, 1)),
        (19, (1, 2, 3, 4, 8, 10, 11), (11, 8)),
        (11, (1, 2, 3, 4), (4, 4, 3)),
        (20, (10, 11), (10, 10)),
        (21, (10, 11), (11, 10)),
        (22, (10, 11), (11, 11)),
        (19, (10, 11), None),
    ),
)
def test_exact_batch_partition_uses_only_actual_validated_shapes(
    total,
    allowed,
    expected,
):
    assert _exact_batch_partition(total, allowed_shapes=allowed) == expected


def test_public_runtime_plans_only_validated_actual_graph_shapes():
    fp32 = OptimizedSingleRuntime(OPTIMIZED_PRESETS["fp32"]).for_super_timing(
        max_batch_size=16
    )
    assert fp32.plan_batches(num_samples=19, max_batch_size=16) == (11, 8)

    fp32_small = OptimizedSingleRuntime(
        OPTIMIZED_PRESETS["fp32"]
    ).for_super_timing(max_batch_size=4)
    assert fp32_small.plan_batches(num_samples=11, max_batch_size=4) == (4, 4, 3)

    fp16 = OptimizedSingleRuntime(OPTIMIZED_PRESETS["fp16"]).for_super_timing(
        max_batch_size=16
    )
    assert fp16.plan_batches(num_samples=21, max_batch_size=16) == (11, 10)
    with pytest.raises(ValueError, match="cannot partition 19 windows"):
        fp16.plan_batches(num_samples=19, max_batch_size=16)


def test_generator_configures_request_local_runtime_without_mutating_original():
    original = OptimizedSingleRuntime(OPTIMIZED_PRESETS["fp32"])
    profiler = InferenceProfiler(enabled=True)
    profiler.set_metadata(
        optimized_effective_config={"ordinary": "unchanged"},
        optimized_runtime_owner="ordinary-runtime",
    )
    processor = SimpleNamespace(
        inference_runtime=original,
        do_sample=False,
        num_beams=1,
        cfg_scale=1.0,
        decode_session_state=object(),
        max_batch_size=99,
        profiler=profiler,
    )

    timing = _configure_optimized_super_timing_runtime(
        processor,
        max_batch_size=12,
    )

    assert isinstance(timing, OptimizedSuperTimingRuntime)
    assert timing.configured_max_batch_size == 12
    assert processor.inference_runtime is timing
    assert processor.inference_runtime is not original
    assert processor.decode_session_state is None
    assert processor.max_batch_size == 12
    assert profiler.metadata["optimized_super_timing_public_wiring"] is True
    assert profiler.metadata["optimized_super_timing_config"][
        "validated_actual_batch_sizes"
    ] == [1, 2, 3, 4, 8, 10, 11]
    assert profiler.metadata["optimized_effective_config"] == {
        "ordinary": "unchanged"
    }
    assert profiler.metadata["optimized_runtime_owner"] == "ordinary-runtime"
    assert original.new_context_state() is not original.new_context_state()


def test_generator_leaves_v32_runtime_and_legacy_timer_policy_untouched():
    processor = SimpleNamespace(
        inference_runtime=None,
        do_sample=False,
        num_beams=2,
        cfg_scale=1.5,
    )

    assert (
        _configure_optimized_super_timing_runtime(
            processor,
            max_batch_size=8,
        )
        is None
    )
    assert processor.inference_runtime is None


@pytest.mark.parametrize(
    "overrides",
    (
        {"do_sample": True},
        {"num_beams": 2},
        {"cfg_scale": 1.1},
    ),
)
def test_generator_rejects_non_greedy_optimized_super_timing(overrides):
    values = {
        "inference_runtime": OptimizedSingleRuntime(OPTIMIZED_PRESETS["fp32"]),
        "do_sample": False,
        "num_beams": 1,
        "cfg_scale": 1.0,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match="greedy decoding"):
        _configure_optimized_super_timing_runtime(
            SimpleNamespace(**values),
            max_batch_size=16,
        )


def test_production_super_timing_runtime_rejects_unvalidated_batch():
    with pytest.raises(ValueError, match="configured max batch size >= 16"):
        OptimizedSuperTimingRuntime(
            OPTIMIZED_PRESETS["fp16"],
            configured_max_batch_size=4,
            public_wiring=True,
        )

    with pytest.raises(TypeError, match="configured max batch size must be an integer"):
        OptimizedSuperTimingRuntime(
            OPTIMIZED_PRESETS["fp32"],
            configured_max_batch_size=True,
            public_wiring=False,
        )


def test_existing_profiling_runtime_is_not_replaced_by_generator():
    runtime = OptimizedSuperTimingRuntime(
        OPTIMIZED_PRESETS["fp32"],
        configured_max_batch_size=4,
        public_wiring=False,
    )
    processor = SimpleNamespace(
        inference_runtime=runtime,
        do_sample=False,
        num_beams=1,
        cfg_scale=1.0,
        decode_session_state=object(),
        max_batch_size=16,
        profiler=InferenceProfiler(enabled=False),
    )

    configured = _configure_optimized_super_timing_runtime(
        processor,
        max_batch_size=16,
    )

    assert configured is runtime
    assert processor.inference_runtime is runtime
    assert processor.max_batch_size == 4
    assert runtime.super_timing_profile_metadata()[
        "optimized_super_timing_config"
    ]["batch_planner"] == "profiling_fixed_chunks_with_arbitrary_tail"


def test_generator_rejects_reusing_already_public_super_timing_runtime():
    runtime = OptimizedSingleRuntime(OPTIMIZED_PRESETS["fp32"]).for_super_timing(
        max_batch_size=16
    )
    processor = SimpleNamespace(
        inference_runtime=runtime,
        do_sample=False,
        num_beams=1,
        cfg_scale=1.0,
    )

    with pytest.raises(RuntimeError, match="must not reuse an already-public"):
        _configure_optimized_super_timing_runtime(
            processor,
            max_batch_size=16,
        )


def test_processor_batch_plan_uses_runtime_planner_and_fails_loudly():
    planner = Mock(return_value=(4, 3))
    processor = object.__new__(Processor)
    processor.inference_runtime = SimpleNamespace(plan_batches=planner)

    assert processor._generation_batch_plan(
        num_samples=7,
        max_batch_size=4,
    ) == (4, 3)
    planner.assert_called_once_with(num_samples=7, max_batch_size=4)

    processor.inference_runtime.plan_batches = Mock(return_value=(4, 4))
    with pytest.raises(ValueError, match="does not cover every input"):
        processor._generation_batch_plan(num_samples=7, max_batch_size=4)

    processor.inference_runtime.plan_batches = Mock(return_value=(5, 2))
    with pytest.raises(ValueError, match="invalid actual batch size"):
        processor._generation_batch_plan(num_samples=7, max_batch_size=4)


def test_processor_legacy_batch_plan_remains_fixed_chunks_with_tail():
    processor = object.__new__(Processor)
    processor.inference_runtime = None

    assert processor._generation_batch_plan(
        num_samples=11,
        max_batch_size=4,
    ) == (4, 4, 3)


def test_batched_inference_executes_the_runtime_actual_shape_plan():
    processor = object.__new__(Processor)
    processor.inference_runtime = OptimizedSingleRuntime(
        OPTIMIZED_PRESETS["fp32"]
    ).for_super_timing(max_batch_size=16)
    processor.max_batch_size = 16
    processor.num_beams = 1
    processor.cfg_scale = 1.0
    processor.tokenizer = SimpleNamespace(pad_id=0)
    processor.profiler = InferenceProfiler(enabled=False)
    processor.last_generation_stats = None
    processor.super_timing_iteration = 0
    processor.stack_prompts = lambda cond, uncond: (
        torch.arange(19, dtype=torch.long).reshape(19, 1) + 1,
        None,
        1,
    )
    actual_batches = []

    def generate_func(kwargs):
        actual_batches.append(int(kwargs["inputs"].shape[0]))
        return kwargs["decoder_input_ids"]

    results = list(
        processor._batched_inference(
            generate_func,
            [torch.ones((1, 1), dtype=torch.long)] * 19,
            [torch.ones((1, 1), dtype=torch.long)] * 19,
            torch.ones((19, 2)),
            [{"feature": torch.ones((1, 1))} for _ in range(19)],
            verbose=False,
        )
    )

    assert actual_batches == [11, 8]
    assert [int(result.shape[0]) for result, _ in results] == [11, 8]


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
            generation_config=SimpleNamespace(disable_compile=True),
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
        assert call_kwargs["generation_compile"] is False
        assert call_kwargs["precision"] == precision
        assert call_kwargs["attn_implementation"] == "sdpa"
        assert call_kwargs["use_server"] is False
        assert (
            binding.runtime.profile_metadata()["optimized_effective_config"][
                "precision"
            ]
            == precision
        )


@pytest.mark.parametrize("disable_compile", (False, None, 1))
def test_single_loader_rejects_loader_that_ignores_compile_disable(
    disable_compile,
):
    raw_model = SimpleNamespace(
        generation_config=SimpleNamespace(disable_compile=disable_compile),
    )
    model_loader = Mock(return_value=(raw_model, object()))

    with pytest.raises(
        RuntimeError,
        match=r"requires generation_config\.disable_compile to be boolean True",
    ):
        load_optimized_single_engine(
            model_loader=model_loader,
            loader_kwargs=_loader_kwargs(),
        )

    assert model_loader.call_args.kwargs["generation_compile"] is False


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
        generation_config=SimpleNamespace(disable_compile=True),
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
    assert raw_loader.call_args.kwargs["generation_compile"] is False
    assert raw_loader.call_args.kwargs["precision"] == precision
    assert binding.runtime.preset is OPTIMIZED_PRESETS[precision]


def test_optimized_public_loader_defaults_to_fp32_preset():
    raw_model = SimpleNamespace(
        generation_config=SimpleNamespace(disable_compile=True),
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


def test_generate_window_preserves_custom_generate_dispatch(monkeypatch):
    captured = {}

    class FakeModel:
        dtype = torch.float32
        device = torch.device("cpu")

        def generate(self, **kwargs):
            captured.update(kwargs)
            return torch.tensor([[1, 2]], dtype=torch.long)

    class FakeContextState:
        def cache_for_window(self, model, **kwargs):
            return object()

        def active_prefix_decode_kwargs(self):
            return {}

        def graph_profile_summary(self):
            return {"graphs": []}

    monkeypatch.setattr(
        engine_module,
        "_build_logits_processor_list",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        engine_module,
        "eos_token_ids",
        lambda *args, **kwargs: [2],
    )
    monkeypatch.setattr(
        engine_module,
        "build_generation_stats",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        engine_module,
        "generation_profile_context",
        lambda **kwargs: nullcontext(),
    )

    result, stats = engine_module._generate_window(
        FakeModel(),
        object(),
        {"inputs": torch.tensor([[1]], dtype=torch.long)},
        {"precision": "fp32"},
        context_state=FakeContextState(),
        preset=OPTIMIZED_PRESETS["fp32"],
    )

    assert torch.equal(result, torch.tensor([[1, 2]], dtype=torch.long))
    assert stats["torch_compile_enabled"] is False
    assert stats["optimized_cuda_graphs"] == {"graphs": []}
    custom_generate = captured["custom_generate"]
    assert isinstance(custom_generate, partial)
    assert custom_generate.func is engine_module.active_prefix_decode_generate


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
