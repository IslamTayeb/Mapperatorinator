from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.event import ContextType
from osuT5.osuT5.inference.engine_binding import InferenceEngineBinding
from osuT5.osuT5.inference.optimized.single import engine
from osuT5.osuT5.inference.optimized.single import stage_precision_hybrid as hybrid


class _Model(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.register_parameter("marker", torch.nn.Parameter(torch.zeros((), dtype=dtype)))

    @property
    def dtype(self):
        return self.marker.dtype


class _WeightState:
    def __init__(self, owner):
        self.owner = owner

    def validate_owner(self, model):
        if model is not self.owner:
            raise RuntimeError("wrong owner")

    @staticmethod
    def metadata():
        return {
            "version": "approximate-fp16-weights-fp32-state-v2",
            "exactness_claim": False,
        }


def _binding(precision: str, *, mixed: bool = False):
    model = _Model(engine.OPTIMIZED_PRESETS[precision].torch_dtype)
    runtime = engine.OptimizedSingleRuntime(engine.OPTIMIZED_PRESETS[precision])
    if mixed:
        object.__setattr__(runtime, "_approximate_weight_only_state", _WeightState(model))
    return InferenceEngineBinding(model, runtime)


def test_coordinator_requires_main_then_distinct_timing_and_main_only_weights():
    coordinator = hybrid.StagePrecisionHybridCoordinator()
    with pytest.raises(RuntimeError, match="timing model loaded before main"):
        coordinator.bind_timing(_binding("fp16"))

    main = _binding("fp32", mixed=True)
    main_hybrid = coordinator.bind_main(main)
    timing = _binding("fp16")
    timing_hybrid = coordinator.bind_timing(timing)

    assert main_hybrid.raw_model is main.raw_model
    assert timing_hybrid.raw_model is timing.raw_model
    metadata = main_hybrid.runtime.profile_metadata()[
        "optimized_stage_precision_hybrid"
    ]
    assert metadata["timing"]["precision"] == "fp16"
    assert metadata["timing"]["mixed_weight_state"] is False
    assert metadata["main"]["precision"] == "fp32"
    assert metadata["main"]["mixed_weight_state"] is True
    assert metadata["models_distinct"] is True

    with pytest.raises(RuntimeError, match="loaded more than once"):
        coordinator.bind_timing(_binding("fp16"))


def test_timing_runtime_bridges_public_fp32_to_specialized_fp16(monkeypatch):
    coordinator = hybrid.StagePrecisionHybridCoordinator()
    coordinator.bind_main(_binding("fp32", mixed=True))
    timing_binding = coordinator.bind_timing(_binding("fp16"))
    captured = {}

    def fake_generate(model, tokenizer, model_kwargs, generate_kwargs, **kwargs):
        captured.update(
            model=model,
            generate_kwargs=generate_kwargs,
            kwargs=kwargs,
        )
        return "tokens", {"precision": "fp16"}

    monkeypatch.setattr(hybrid, "_generate_window", fake_generate)

    @contextmanager
    def fake_stage_block_size(block_size):
        assert block_size == 1
        yield

    monkeypatch.setattr(hybrid, "k8_block_size_context", fake_stage_block_size)
    result, stats = timing_binding.runtime.generate_window(
        model=timing_binding.raw_model,
        tokenizer=object(),
        model_kwargs={},
        generate_kwargs={
            "precision": "fp32",
            "context_type": ContextType.TIMING,
        },
        context_state=engine.ProductionDecodeSession(),
    )

    assert result == "tokens"
    assert captured["generate_kwargs"]["precision"] == "fp16"
    assert captured["kwargs"]["preset"] is engine.OPTIMIZED_PRESETS["fp16"]
    assert captured["kwargs"]["allow_specialized_timing_dispatch"] is True
    assert stats["optimized_stage_precision_hybrid_role"] == "timing_fp16"

    with pytest.raises(RuntimeError, match="only timing context"):
        timing_binding.runtime.generate_window(
            model=timing_binding.raw_model,
            tokenizer=object(),
            model_kwargs={},
            generate_kwargs={"precision": "fp32", "context_type": ContextType.MAP},
            context_state=engine.ProductionDecodeSession(),
        )


def test_generate_window_opt_in_enables_all_dtype_aware_timing_dispatch(monkeypatch):
    captured = []

    @contextmanager
    def fake_context(**kwargs):
        captured.append(kwargs)
        yield

    class _Session:
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
            return {"graph_count": 0, "decode_replays": 0, "buckets": {}}

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
    monkeypatch.setattr(engine, "generation_profile_context", fake_context)
    monkeypatch.setattr(engine, "_build_logits_processor_list", lambda *args, **kwargs: [])

    _, stats = engine._generate_window(
        model,
        tokenizer,
        model_kwargs,
        {"context_type": ContextType.TIMING, "pad_token_id": 0},
        context_state=_Session(),
        preset=engine.OPTIMIZED_PRESETS["fp16"],
        allow_specialized_timing_dispatch=True,
    )

    dispatch = captured[0]
    assert dispatch["q1_bmm_cross_attention"] is True
    assert dispatch["native_q1_self_attention"] is True
    assert dispatch["native_q1_rope_cache_self_attention"] is True
    assert dispatch["native_cross_mlp_tail"] is True
    assert dispatch["optimized_expected_dtype"] is torch.float16
    assert stats["optimized_specialized_timing_dispatch"] == {
        "requested": True,
        "enabled": True,
        "result_class": "documented-drift",
        "exactness_claim": False,
    }
