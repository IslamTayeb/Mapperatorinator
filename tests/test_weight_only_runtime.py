from __future__ import annotations

import inspect
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.inference import runtime_dispatch
from osuT5.osuT5.inference.optimized.kernels import weight_only_runtime
from osuT5.osuT5.inference.optimized.single import engine
from osuT5.osuT5.model.custom_transformers.modeling_varwhisper import (
    VarWhisperDecoderLayer,
)


def _state(owner) -> weight_only_runtime.ApproximateWeightOnlyState:
    projection = torch.nn.Linear(4, 8, bias=False)
    norm = torch.nn.RMSNorm(4)
    packed = SimpleNamespace(
        weight=torch.zeros((8, 4), dtype=torch.float16),
        bias=None,
        source_weight_bytes=128,
        packed_weight_bytes=64,
    )
    return weight_only_runtime.ApproximateWeightOnlyState(
        owner_ref=lambda: owner,
        layer_packs={},
        final_norm=norm,
        final_projection=projection,
        final_projection_pack=packed,
        extension_init_seconds=3.0,
        pack_seconds=0.5,
        allocated_bytes_delta=64,
        reserved_bytes_delta=128,
        retained_fp32_source_weight_bytes=256,
        packed_weight_bytes=128,
    )


def test_weight_only_context_installs_all_narrow_hooks_and_restores() -> None:
    owner = torch.nn.Module()
    state = _state(owner)
    before = runtime_dispatch.decoder_layer_runtime_hooks()

    with weight_only_runtime.approximate_weight_only_runtime_context(state):
        hooks = runtime_dispatch.decoder_layer_runtime_hooks()
        assert hooks.self_attention_block_forward is not None
        assert hooks.cross_mlp_tail_forward is not None
        assert hooks.decoder_final_norm_forward is not None
        assert hooks.output_projection_forward is not None

    assert runtime_dispatch.decoder_layer_runtime_hooks() is before


def test_original_decoder_forward_remains_installed() -> None:
    forward = VarWhisperDecoderLayer.forward
    source = inspect.getsource(forward)
    with weight_only_runtime.approximate_weight_only_runtime_context(
        _state(torch.nn.Module())
    ):
        assert VarWhisperDecoderLayer.forward is forward
    assert "self_attention_block_forward" in source
    assert "module.forward =" not in source


def test_state_owner_and_metadata_fail_loudly() -> None:
    owner = torch.nn.Module()
    state = _state(owner)
    state.validate_owner(owner)
    with pytest.raises(RuntimeError, match="different model"):
        state.validate_owner(torch.nn.Module())

    metadata = state.metadata()
    assert metadata["result_class"] == "documented-drift"
    assert metadata["exactness_claim"] is False
    assert metadata["fp32_activations_caches_reductions_logits"] is True
    assert metadata["outputs_per_block"] == {
        "self_qkv": 4,
        "self_output": 4,
        "mlp_fc1": 2,
        "mlp_fc2": 4,
        "final_logits": 2,
    }


def test_runtime_initialization_is_idempotent_and_model_owned(monkeypatch) -> None:
    owner = torch.nn.Module()
    state = _state(owner)
    calls = []

    monkeypatch.setattr(
        weight_only_runtime.ApproximateWeightOnlyState,
        "initialize",
        classmethod(lambda cls, model: calls.append(model) or state),
    )
    runtime = engine.OptimizedSingleRuntime(engine.OPTIMIZED_PRESETS["fp32"])

    first = runtime.initialize_approximate_weight_only(owner)
    second = runtime.initialize_approximate_weight_only(owner)

    assert first == second == state.metadata()
    assert calls == [owner]
    with pytest.raises(RuntimeError, match="different model"):
        runtime.initialize_approximate_weight_only(torch.nn.Module())


def test_fp16_runtime_rejects_weight_only_initialization() -> None:
    runtime = engine.OptimizedSingleRuntime(engine.OPTIMIZED_PRESETS["fp16"])
    with pytest.raises(TypeError, match="requires the FP32 preset"):
        runtime.initialize_approximate_weight_only(torch.nn.Module())


def test_profile_metadata_is_unchanged_until_explicit_initialization(monkeypatch) -> None:
    runtime = engine.OptimizedSingleRuntime(engine.OPTIMIZED_PRESETS["fp32"])
    before = runtime.profile_metadata()
    assert "optimized_approximate_weight_only" not in before

    owner = torch.nn.Module()
    state = _state(owner)
    monkeypatch.setattr(
        weight_only_runtime.ApproximateWeightOnlyState,
        "initialize",
        classmethod(lambda cls, model: state),
    )
    runtime.initialize_approximate_weight_only(owner)
    after = runtime.profile_metadata()
    assert after["optimized_approximate_weight_only"]["exactness_claim"] is False


def test_default_engine_import_does_not_import_weight_only_runtime() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import osuT5.osuT5.inference.optimized.single.engine; "
                "assert 'osuT5.osuT5.inference.optimized.kernels.weight_only' "
                "not in sys.modules; "
                "assert 'osuT5.osuT5.inference.optimized.kernels.weight_only_runtime' "
                "not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_generation_context_rejects_conflicting_decoder_candidates() -> None:
    from osuT5.osuT5 import runtime_profiling

    with pytest.raises(ValueError, match="mutually exclusive"):
        with runtime_profiling.generation_profile_context(
            native_cross_mlp_tail=True,
            approximate_weight_only_state=_state(torch.nn.Module()),
        ):
            pass


def test_final_norm_fusion_rejects_hidden_state_output() -> None:
    owner = torch.nn.Module()
    state = _state(owner)
    with pytest.raises(RuntimeError, match="does not support hidden-state"):
        weight_only_runtime._defer_final_norm(
            state=state,
            module=state.final_norm,
            hidden_states=torch.zeros((1, 1, 4)),
            output_hidden_states=True,
        )


def test_final_norm_fusion_rejects_cpu_storage() -> None:
    state = _state(torch.nn.Module())
    with pytest.raises(RuntimeError, match="CUDA tensor"):
        weight_only_runtime._defer_final_norm(
            state=state,
            module=state.final_norm,
            hidden_states=torch.zeros((1, 1, 4)),
            output_hidden_states=False,
        )


def test_final_norm_and_projection_fall_back_for_prefill_shapes() -> None:
    state = _state(torch.nn.Module())
    prefill = torch.zeros((1, 4, 4))

    assert weight_only_runtime._defer_final_norm(
        state=state,
        module=state.final_norm,
        hidden_states=prefill,
        output_hidden_states=False,
    ) is None
    assert weight_only_runtime._final_projection_forward(
        state=state,
        module=state.final_projection,
        hidden_states=prefill,
        dispatch_counts=None,
    ) is None


def test_default_generation_profile_kwargs_do_not_expose_candidate(monkeypatch) -> None:
    captured = {}

    def fake_profile_context(**kwargs):
        captured.update(kwargs)
        return nullcontext()

    class FakeSession:
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
            return {}

    model = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        generate=lambda **kwargs: torch.tensor([[1, 2]]),
    )
    monkeypatch.setattr(engine, "generation_profile_context", fake_profile_context)
    monkeypatch.setattr(engine, "_build_logits_processor_list", lambda *args, **kwargs: [])

    engine._generate_window(
        model,
        SimpleNamespace(pad_id=0, eos_id=2, context_eos={}),
        {
            "inputs": torch.zeros((1, 1)),
            "decoder_input_ids": torch.tensor([[1]]),
            "decoder_attention_mask": torch.tensor([[1]]),
        },
        {"pad_token_id": 0},
        context_state=FakeSession(),
        preset=engine.OPTIMIZED_PRESETS["fp32"],
    )

    assert "approximate_weight_only_state" not in captured


def test_explicit_candidate_replaces_only_main_specialized_hooks(monkeypatch) -> None:
    captured = []

    def fake_profile_context(**kwargs):
        captured.append(kwargs)
        return nullcontext()

    class FakeSession:
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
            return {}

    model = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        generate=lambda **kwargs: torch.tensor([[1, 2]]),
    )
    state = _state(model)
    monkeypatch.setattr(engine, "generation_profile_context", fake_profile_context)
    monkeypatch.setattr(engine, "_build_logits_processor_list", lambda *args, **kwargs: [])
    common_model_kwargs = {
        "inputs": torch.zeros((1, 1)),
        "decoder_input_ids": torch.tensor([[1]]),
        "decoder_attention_mask": torch.tensor([[1]]),
    }

    _, main_stats = engine._generate_window(
        model,
        SimpleNamespace(pad_id=0, eos_id=2, context_eos={}),
        common_model_kwargs,
        {"pad_token_id": 0},
        context_state=FakeSession(),
        preset=engine.OPTIMIZED_PRESETS["fp32"],
        approximate_weight_only_state=state,
    )
    _, timing_stats = engine._generate_window(
        model,
        SimpleNamespace(pad_id=0, eos_id=2, context_eos={}),
        common_model_kwargs,
        {"pad_token_id": 0, "context_type": "timing"},
        context_state=FakeSession(),
        preset=engine.OPTIMIZED_PRESETS["fp32"],
        approximate_weight_only_state=state,
    )

    assert captured[0]["approximate_weight_only_state"] is state
    assert captured[0]["native_q1_self_attention"] is False
    assert captured[0]["native_q1_rope_cache_self_attention"] is False
    assert captured[0]["native_cross_mlp_tail"] is False
    assert "approximate_weight_only_state" not in captured[1]
    assert main_stats["optimized_dispatch_mode"] == "approximate_weight_only_batch1"
    assert main_stats["approximate_weight_only"]["exactness_claim"] is False
    assert timing_stats["optimized_dispatch_mode"] == "accepted_batch1"
    assert timing_stats["optimized_dispatch_policy"]["approximate_weight_only"] == {
        "requested": True,
        "enabled": False,
        "disabled_reason": "timing_context",
        "result_class": "documented-drift",
        "exactness_claim": False,
    }
