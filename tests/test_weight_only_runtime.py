from __future__ import annotations

import inspect
import subprocess
import sys
import weakref
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


def _state(
    owner,
    *,
    int8_mlp: bool = False,
    dp4a_self_qkv: bool = False,
    cross_mode: str = weight_only_runtime.CROSS_ACCEPTED,
) -> weight_only_runtime.ApproximateWeightOnlyState:
    projection = torch.nn.Linear(4, 8, bias=False)
    norm = torch.nn.RMSNorm(4)
    packed = SimpleNamespace(
        weight=torch.zeros((8, 4), dtype=torch.float16),
        bias=None,
        source_weight_bytes=128,
        packed_weight_bytes=64,
    )
    return weight_only_runtime.ApproximateWeightOnlyState(
        owner_ref=weakref.ref(owner),
        layer_packs={},
        final_norm=norm,
        final_projection=projection,
        final_projection_pack=packed,
        extension_init_seconds=3.0,
        extension_allocated_bytes_delta=16,
        extension_reserved_bytes_delta=32,
        pack_seconds=0.5,
        allocated_bytes_delta=64,
        reserved_bytes_delta=128,
        retained_fp32_source_weight_bytes=256,
        packed_weight_bytes=128,
        cross_mode=cross_mode,
        cross_packs=None,
        int8_mlp_packs={} if int8_mlp else None,
        int8_mlp_extension_init_seconds=1.5 if int8_mlp else 0.0,
        int8_mlp_pack_seconds=0.25 if int8_mlp else 0.0,
        int8_mlp_packed_weight_bytes=48 if int8_mlp else 0,
        dp4a_self_qkv_packs={} if dp4a_self_qkv else None,
        dp4a_extension_init_seconds=2.5 if dp4a_self_qkv else 0.0,
        dp4a_pack_seconds=0.125 if dp4a_self_qkv else 0.0,
        dp4a_packed_weight_bytes=96 if dp4a_self_qkv else 0,
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


def test_no_initialization_leaves_default_decoder_hooks_neutral() -> None:
    hooks = runtime_dispatch.decoder_layer_runtime_hooks()

    assert hooks == runtime_dispatch.DecoderLayerRuntimeHooks()
    assert hooks.self_attention_block_forward is None
    assert hooks.cross_mlp_tail_forward is None
    assert hooks.decoder_final_norm_forward is None
    assert hooks.output_projection_forward is None


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
    assert metadata["version"] == "approximate-fp16-weights-fp32-state-v2"
    assert metadata["fp16_weight_regions"] == [
        "self_qkv",
        "self_output",
        "mlp_fc1",
        "mlp_fc2",
        "final_logits",
    ]
    assert metadata["fp32_selected_decode_matrix_regions"] == [
        "cross_query",
        "cross_output",
    ]
    assert metadata["cross_candidate"] == {
        "mode": "accepted",
        "scope": "main-model-only",
        "attention_accumulation": "fp32",
        "production_selector_unchanged": True,
        "result_class": "control",
        "exactness_claim": False,
        "accepted_q1_bmm": True,
    }
    assert "fp32_weight_regions" not in metadata
    assert metadata["outputs_per_block"] == {
        "self_qkv": 4,
        "self_output": 4,
        "mlp_fc1": 2,
        "mlp_fc2": 4,
        "final_logits": 2,
    }
    assert metadata["extension_allocated_bytes_delta"] == 16
    assert metadata["extension_reserved_bytes_delta"] == 32
    assert metadata["weight_pack_allocated_bytes_delta"] == 64
    assert metadata["weight_pack_reserved_bytes_delta"] == 128


def test_weight_owned_self_attention_records_effective_split_variant() -> None:
    counts = {
        "weight_only_self_attention_block": 0,
        "native_q1_rope_cache_self_attention": 0,
    }

    for variant, prefix in (
        ("accepted", 128),
        ("split_kv_8", 192),
        ("split_kv_8", 640),
        ("split_kv_8", 640),
    ):
        weight_only_runtime._record_weight_only_self_attention_dispatch(
            counts,
            variant=variant,
            prefix_length=prefix,
        )

    assert counts == {
        "weight_only_self_attention_block": 4,
        "native_q1_rope_cache_self_attention": 4,
        "native_q1_rope_cache_self_attention_split_kv_8": 3,
        "native_q1_rope_cache_self_attention_split_kv_8_prefix_192": 1,
        "native_q1_rope_cache_self_attention_split_kv_8_prefix_640": 2,
    }
    before = dict(counts)
    with pytest.raises(RuntimeError, match="unknown q1 variant"):
        weight_only_runtime._record_weight_only_self_attention_dispatch(
            counts,
            variant="unknown",
            prefix_length=640,
        )
    assert counts == before


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


def test_int8_mlp_initialization_is_explicit_idempotent_and_declared(monkeypatch) -> None:
    owner = torch.nn.Module()
    state = _state(owner, int8_mlp=True)
    calls = []

    def initialize(cls, model, *, int8_mlp=False):
        calls.append((model, int8_mlp))
        return state

    monkeypatch.setattr(
        weight_only_runtime.ApproximateWeightOnlyState,
        "initialize",
        classmethod(initialize),
    )
    runtime = engine.OptimizedSingleRuntime(engine.OPTIMIZED_PRESETS["fp32"])

    first = runtime.initialize_approximate_int8_mlp_weight_only(owner)
    second = runtime.initialize_approximate_int8_mlp_weight_only(owner)

    assert first == second == state.metadata()
    assert calls == [(owner, True)]
    overlay = first["int8_mlp_overlay"]
    assert overlay["scope"] == "main-model-decoder-mlp-only"
    assert overlay["replaces_effective_regions"] == ["mlp_fc1", "mlp_fc2"]
    assert overlay["retained_unused_fp16_regions"] == ["mlp_fc1", "mlp_fc2"]
    assert overlay["packed_weight_bytes"] == 48


def test_int8_mlp_initialization_rejects_preexisting_plain_mixed_state(
    monkeypatch,
) -> None:
    owner = torch.nn.Module()
    state = _state(owner)
    monkeypatch.setattr(
        weight_only_runtime.ApproximateWeightOnlyState,
        "initialize",
        classmethod(lambda cls, model: state),
    )
    runtime = engine.OptimizedSingleRuntime(engine.OPTIMIZED_PRESETS["fp32"])
    runtime.initialize_approximate_weight_only(owner)

    with pytest.raises(RuntimeError, match="without the INT8 MLP overlay"):
        runtime.initialize_approximate_int8_mlp_weight_only(owner)


def test_repeated_initialize_and_context_teardown_reuse_one_owned_state(
    monkeypatch,
) -> None:
    owner = torch.nn.Module()
    state = _state(owner)
    calls = []
    monkeypatch.setattr(
        weight_only_runtime.ApproximateWeightOnlyState,
        "initialize",
        classmethod(lambda cls, model: calls.append(model) or state),
    )
    runtime = engine.OptimizedSingleRuntime(engine.OPTIMIZED_PRESETS["fp32"])
    default_hooks = runtime_dispatch.decoder_layer_runtime_hooks()
    pack_identity = id(state.final_projection_pack)

    for _ in range(3):
        assert runtime.initialize_approximate_weight_only(owner) == state.metadata()
        assert runtime._approximate_weight_only_state is state
        assert id(state.final_projection_pack) == pack_identity
        with weight_only_runtime.approximate_weight_only_runtime_context(state):
            installed = runtime_dispatch.decoder_layer_runtime_hooks()
            assert installed is not default_hooks
            assert installed.self_attention_block_forward.keywords["state"] is state
            assert installed.cross_mlp_tail_forward.keywords["state"] is state
        assert runtime_dispatch.decoder_layer_runtime_hooks() is default_hooks

    assert calls == [owner]


def test_fp16_runtime_rejects_weight_only_initialization() -> None:
    runtime = engine.OptimizedSingleRuntime(engine.OPTIMIZED_PRESETS["fp16"])
    with pytest.raises(TypeError, match="requires the FP32 preset"):
        runtime.initialize_approximate_weight_only(torch.nn.Module())
    with pytest.raises(TypeError, match="requires the FP32 preset"):
        runtime.initialize_approximate_int8_mlp_weight_only(torch.nn.Module())
    with pytest.raises(TypeError, match="requires the FP32 optimized preset"):
        runtime.initialize_approximate_int8_mlp_weight_only_cross(
            torch.nn.Module(),
            mode=weight_only_runtime.CROSS_FP16_PACKED,
        )


def test_cross_candidate_initialization_is_explicit_mode_owned(monkeypatch) -> None:
    owner = torch.nn.Module()
    state = _state(owner)
    object.__setattr__(state, "cross_mode", weight_only_runtime.CROSS_FP16_PACKED)
    calls = []

    def initialize(cls, model, *, cross_mode="accepted"):
        calls.append((model, cross_mode))
        return state

    monkeypatch.setattr(
        weight_only_runtime.ApproximateWeightOnlyState,
        "initialize",
        classmethod(initialize),
    )
    runtime = engine.OptimizedSingleRuntime(engine.OPTIMIZED_PRESETS["fp32"])

    metadata = runtime.initialize_approximate_weight_only_cross(
        owner,
        mode=weight_only_runtime.CROSS_FP16_PACKED,
    )

    assert metadata["cross_candidate"]["mode"] == "fp16_packed_projections"
    assert metadata["cross_candidate"]["projection_delta_only"] is True
    assert metadata["cross_candidate"]["accepted_q1_bmm"] is True
    assert metadata["cross_candidate"]["incremental_exactness_required"] is True
    assert calls == [(owner, "fp16_packed_projections")]
    with pytest.raises(ValueError, match="must be fp16_packed"):
        engine.OptimizedSingleRuntime(
            engine.OPTIMIZED_PRESETS["fp32"]
        ).initialize_approximate_weight_only_cross(owner, mode="split8_attention")


def test_cross_int8_composition_initialization_is_owned_and_idempotent(
    monkeypatch,
) -> None:
    owner = torch.nn.Module()
    state = _state(
        owner,
        int8_mlp=True,
        cross_mode=weight_only_runtime.CROSS_FP16_PACKED,
    )
    calls = []

    def initialize(cls, model, *, cross_mode="accepted", int8_mlp=False):
        calls.append((model, cross_mode, int8_mlp))
        return state

    monkeypatch.setattr(
        weight_only_runtime.ApproximateWeightOnlyState,
        "initialize",
        classmethod(initialize),
    )
    runtime = engine.OptimizedSingleRuntime(engine.OPTIMIZED_PRESETS["fp32"])

    first = runtime.initialize_approximate_int8_mlp_weight_only_cross(
        owner,
        mode=weight_only_runtime.CROSS_FP16_PACKED,
    )
    second = runtime.initialize_approximate_int8_mlp_weight_only_cross(
        owner,
        mode=weight_only_runtime.CROSS_FP16_PACKED,
    )

    assert first == second == state.metadata()
    assert calls == [(owner, weight_only_runtime.CROSS_FP16_PACKED, True)]
    assert first["int8_mlp_overlay"]["dispatch_counter"] == "int8_weight_mlp_tail"
    assert first["cross_candidate"]["mode"] == weight_only_runtime.CROSS_FP16_PACKED
    with pytest.raises(ValueError, match="must be fp16_packed"):
        engine.OptimizedSingleRuntime(
            engine.OPTIMIZED_PRESETS["fp32"]
        ).initialize_approximate_int8_mlp_weight_only_cross(
            owner,
            mode="split8_attention",
        )


def test_dp4a_self_qkv_composition_initialization_is_owned_and_declared(
    monkeypatch,
) -> None:
    owner = torch.nn.Module()
    state = _state(
        owner,
        int8_mlp=True,
        dp4a_self_qkv=True,
        cross_mode=weight_only_runtime.CROSS_FP16_PACKED,
    )
    calls = []

    def initialize(
        cls,
        model,
        *,
        cross_mode="accepted",
        int8_mlp=False,
        dp4a_self_qkv=False,
    ):
        calls.append((model, cross_mode, int8_mlp, dp4a_self_qkv))
        return state

    monkeypatch.setattr(
        weight_only_runtime.ApproximateWeightOnlyState,
        "initialize",
        classmethod(initialize),
    )
    runtime = engine.OptimizedSingleRuntime(engine.OPTIMIZED_PRESETS["fp32"])

    first = runtime.initialize_approximate_int8_mlp_weight_only_cross_dp4a_self_qkv(
        owner,
        mode=weight_only_runtime.CROSS_FP16_PACKED,
    )
    second = runtime.initialize_approximate_int8_mlp_weight_only_cross_dp4a_self_qkv(
        owner,
        mode=weight_only_runtime.CROSS_FP16_PACKED,
    )

    assert first == second == state.metadata()
    assert calls == [
        (owner, weight_only_runtime.CROSS_FP16_PACKED, True, True)
    ]
    overlay = first["dp4a_self_qkv_overlay"]
    assert overlay["dispatch_counter"] == "dp4a_self_qkv_projection"
    assert overlay["persistent_ctas"] == 68
    assert overlay["packed_weight_bytes"] == 96
    with pytest.raises(ValueError, match="requires fp16_packed"):
        engine.OptimizedSingleRuntime(
            engine.OPTIMIZED_PRESETS["fp32"]
        ).initialize_approximate_int8_mlp_weight_only_cross_dp4a_self_qkv(
            owner,
            mode="accepted",
        )


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
                "not in sys.modules; "
                "assert 'osuT5.osuT5.inference.optimized.scout.int8_mlp' "
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

    class FakeModel(SimpleNamespace):
        pass

    model = FakeModel(
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
    assert main_stats["optimized_dispatch_policy"][
        "native_q1_rope_cache_self_attention"
    ] == {
        "requested": True,
        "enabled": False,
        "disabled_reason": "approximate_weight_only",
    }
    assert main_stats["optimized_dispatch_policy"][
        "effective_native_q1_rope_cache_self_attention"
    ] == {
        "requested": True,
        "enabled": True,
        "owner": "approximate_weight_only",
        "kernel": "native_q1_rope_cache_attention",
        "standard_attention_hook_enabled": False,
        "split_kv_selector_enabled": True,
        "disabled_reason": None,
    }
    assert timing_stats["optimized_dispatch_mode"] == "accepted_batch1"
    assert timing_stats["optimized_dispatch_policy"][
        "effective_native_q1_rope_cache_self_attention"
    ] == {
        "requested": False,
        "enabled": False,
        "owner": None,
        "kernel": None,
        "standard_attention_hook_enabled": False,
        "split_kv_selector_enabled": False,
        "disabled_reason": "timing_context",
    }
    assert timing_stats["optimized_dispatch_policy"]["approximate_weight_only"] == {
        "requested": True,
        "enabled": False,
        "disabled_reason": "timing_context",
        "result_class": "documented-drift",
        "exactness_claim": False,
    }
