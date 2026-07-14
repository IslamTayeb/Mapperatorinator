from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.event import ContextType
from osuT5.osuT5.inference.engine_binding import InferenceEngineBinding
from osuT5.osuT5.inference.optimized.single import engine
from osuT5.osuT5.inference.optimized.single import timing_precision_matrix as matrix


class _Model(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros((), dtype=dtype))

    @property
    def dtype(self):
        return self.marker.dtype


class _WeightState:
    int8_mlp_enabled = False

    def __init__(self, owner):
        self.owner = owner

    def validate_owner(self, model):
        if model is not self.owner:
            raise RuntimeError("wrong packed-weight owner")

    @staticmethod
    def metadata():
        return {"version": "test-packed-weights", "cross_mode": "fp16"}


def _binding(precision: str) -> InferenceEngineBinding:
    return InferenceEngineBinding(
        raw_model=_Model(engine.OPTIMIZED_PRESETS[precision].torch_dtype),
        runtime=engine.OptimizedSingleRuntime(engine.OPTIMIZED_PRESETS[precision]),
    )


def test_full_fp16_binding_is_separate_documented_drift() -> None:
    binding = _binding("fp16")

    wrapped, metadata = matrix.TimingPrecisionMatrixRuntime.from_binding(
        binding,
        mode=matrix.FULL_FP16,
    )

    assert wrapped.raw_model is binding.raw_model
    assert wrapped.runtime.preset is engine.OPTIMIZED_PRESETS["fp16"]
    assert metadata["mode"] == matrix.FULL_FP16
    assert metadata["full_fp16_model_storage"] is True
    assert metadata["fp16_packed_weights"] is False
    assert metadata["native_cross_mlp_tail"] is True
    assert metadata["exactness_claim"] is False


def test_fp16_weight_binding_keeps_fp32_state(monkeypatch) -> None:
    binding = _binding("fp32")
    state = _WeightState(binding.raw_model)

    def initialize(self, model, *, mode):
        assert model is binding.raw_model
        assert mode == "fp16_packed_projections"
        object.__setattr__(self, "_approximate_weight_only_state", state)
        return state.metadata()

    monkeypatch.setattr(
        engine.OptimizedSingleRuntime,
        "initialize_approximate_weight_only_cross",
        initialize,
    )
    wrapped, metadata = matrix.TimingPrecisionMatrixRuntime.from_binding(
        binding,
        mode=matrix.FP16_WEIGHTS_FP32_STATE,
    )

    assert wrapped.runtime.preset is engine.OPTIMIZED_PRESETS["fp32"]
    assert metadata["full_fp16_model_storage"] is False
    assert metadata["fp16_packed_weights"] is True
    assert metadata["fp32_activations_caches_norm_attention_reductions_logits"] is True
    assert metadata["weight_initialization"] == state.metadata()


@pytest.mark.parametrize(
    ("mode", "precision", "specialized", "timing_weights"),
    (
        (matrix.FULL_FP16, "fp16", True, False),
        (matrix.FP16_WEIGHTS_FP32_STATE, "fp32", False, True),
    ),
)
def test_timing_bridge_uses_k1_and_keeps_public_request_fp32(
    monkeypatch,
    mode,
    precision,
    specialized,
    timing_weights,
) -> None:
    binding = _binding(precision)
    state = _WeightState(binding.raw_model) if timing_weights else None
    if state is not None:
        object.__setattr__(binding.runtime, "_approximate_weight_only_state", state)
    owner = matrix.TimingPrecisionMatrixRuntime(
        mode=mode,
        model=binding.raw_model,
        runtime=binding.runtime,
    )
    captured = {}

    def fake_generate(model, tokenizer, model_kwargs, generate_kwargs, **kwargs):
        captured.update(generate_kwargs=generate_kwargs, kwargs=kwargs)
        return "tokens", {}

    @contextmanager
    def fake_block(block_size):
        assert block_size == 1
        yield

    monkeypatch.setattr(matrix, "_generate_window", fake_generate)
    monkeypatch.setattr(matrix, "k8_block_size_context", fake_block)

    result, stats = owner.generate_window(
        model=binding.raw_model,
        tokenizer=object(),
        model_kwargs={},
        generate_kwargs={
            "precision": "fp32",
            "context_type": ContextType.TIMING,
        },
        context_state=engine.ProductionDecodeSession(),
    )

    assert result == "tokens"
    assert captured["generate_kwargs"]["precision"] == precision
    assert captured["kwargs"]["allow_specialized_timing_dispatch"] is specialized
    assert captured["kwargs"]["allow_timing_approximate_weight_only"] is timing_weights
    assert captured["kwargs"]["approximate_weight_only_state"] is state
    assert stats["optimized_timing_precision_matrix"]["mode"] == mode


def test_timing_bridge_fails_on_main_context_and_wrong_owner() -> None:
    binding = _binding("fp16")
    owner = matrix.TimingPrecisionMatrixRuntime(
        mode=matrix.FULL_FP16,
        model=binding.raw_model,
        runtime=binding.runtime,
    )
    with pytest.raises(RuntimeError, match="only timing context"):
        owner.generate_window(
            model=binding.raw_model,
            tokenizer=object(),
            model_kwargs={},
            generate_kwargs={"precision": "fp32", "context_type": ContextType.MAP},
            context_state=engine.ProductionDecodeSession(),
        )
    with pytest.raises(RuntimeError, match="wrong model"):
        owner.generate_window(
            model=_Model(torch.float16),
            tokenizer=object(),
            model_kwargs={},
            generate_kwargs={"precision": "fp32", "context_type": ContextType.TIMING},
            context_state=engine.ProductionDecodeSession(),
        )


def test_generate_window_full_fp16_opt_in_enables_specialized_topology(monkeypatch) -> None:
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
    monkeypatch.setattr(engine, "generation_profile_context", fake_context)
    monkeypatch.setattr(engine, "_build_logits_processor_list", lambda *args, **kwargs: [])
    _, stats = engine._generate_window(
        model,
        SimpleNamespace(pad_id=0, eos_id=2, context_eos={}),
        {
            "inputs": torch.zeros(1, 1),
            "decoder_input_ids": torch.tensor([[1]]),
            "decoder_attention_mask": torch.tensor([[1]]),
        },
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
    assert stats["optimized_specialized_timing_dispatch"]["enabled"] is True


def test_default_engine_dispatch_flags_remain_off() -> None:
    signature = __import__("inspect").signature(engine._generate_window)
    assert signature.parameters["allow_specialized_timing_dispatch"].default is False
    assert signature.parameters["allow_timing_approximate_weight_only"].default is False
