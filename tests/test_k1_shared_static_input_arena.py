from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.inference.optimized.single.decode_loop import (
    _SharedStaticInputArena,
    _resolve_shared_static_input_arena,
    active_prefix_decode_generate,
)
from osuT5.osuT5.inference.optimized.single.state import ProductionDecodeSession


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_fresh_python(source: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _inputs(dtype: torch.dtype, value: float, *, owner=None):
    return {
        "hidden_states": torch.full((1, 1, 8), value, dtype=dtype),
        "input_ids": torch.tensor([[int(value)]], dtype=torch.long),
        "attention_mask": torch.tensor([[True, False]], dtype=torch.bool),
        "owner": owner,
    }


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
def test_arena_is_dtype_generic_exact_and_address_stable(dtype):
    owner = object()
    arena = _SharedStaticInputArena.create(_inputs(dtype, 1, owner=owner))
    pointers = arena.tensor_addresses
    owned_bytes = sum(
        value.numel() * value.element_size()
        for value in arena.static_inputs.values()
        if isinstance(value, torch.Tensor)
    )

    for value in range(2, 18):
        source = _inputs(dtype, value, owner=owner)
        arena.refresh(source)
        assert torch.equal(
            arena.static_inputs["hidden_states"], source["hidden_states"]
        )
        assert torch.equal(arena.static_inputs["input_ids"], source["input_ids"])
        assert torch.equal(
            arena.static_inputs["attention_mask"], source["attention_mask"]
        )
        assert arena.tensor_addresses == pointers

    arena.register_graph()
    arena.register_graph()
    arena.register_graph()
    summary = arena.profile_summary()
    assert summary == {
        "enabled": True,
        "graph_entries": 3,
        "refreshes": 16,
        "refresh_copy_calls": 48,
        "refresh_copy_bytes": 16 * owned_bytes,
        "owned_tensor_bytes": owned_bytes,
        "avoided_duplicate_graph_input_bytes": 2 * owned_bytes,
    }


def test_arena_fails_loudly_on_signature_or_owner_change():
    owner = object()
    arena = _SharedStaticInputArena.create(
        _inputs(torch.float32, 1, owner=owner)
    )
    with pytest.raises(RuntimeError, match="signature changed"):
        arena.refresh(_inputs(torch.float16, 2, owner=owner))
    with pytest.raises(RuntimeError, match="signature changed"):
        arena.refresh(_inputs(torch.float32, 2, owner=object()))

    arena.static_inputs["hidden_states"] = arena.static_inputs[
        "hidden_states"
    ].clone()
    with pytest.raises(RuntimeError, match="tensor address changed"):
        arena.validate_tensor_addresses()


def test_graph_cache_resolves_one_owner_and_rejects_competing_arenas():
    owner = object()
    first = _SharedStaticInputArena.create(
        _inputs(torch.float32, 1, owner=owner)
    )
    signature = first.signature
    graph_cache = {
        ("candidate", 64): {"_shared_static_input_arena": first},
        ("candidate", 128): {"_shared_static_input_arena": first},
    }
    assert _resolve_shared_static_input_arena(graph_cache, signature) is first

    second = _SharedStaticInputArena.create(
        _inputs(torch.float32, 2, owner=owner)
    )
    graph_cache[("candidate", 192)] = {"_shared_static_input_arena": second}
    with pytest.raises(RuntimeError, match="multiple shared"):
        _resolve_shared_static_input_arena(graph_cache, signature)


def test_session_profile_deduplicates_shared_arena_ownership():
    arena = _SharedStaticInputArena.create(
        _inputs(torch.float16, 1, owner=object())
    )
    arena.register_graph()
    arena.register_graph()
    arena.refreshes = 7
    arena.refresh_copy_calls = 21
    arena.refresh_copy_bytes = 1234
    session = ProductionDecodeSession()
    for prefix in (64, 128):
        session.graph_cache[("shared", prefix)] = {
            "active_prefix_length": prefix,
            "capture_seconds": 0.1,
            "decode_replays": 5,
            "_shared_static_input_arena": arena,
        }

    summary = session.graph_profile_summary()
    candidate = summary["shared_static_input_arena"]
    assert candidate["enabled"] is True
    assert candidate["arena_count"] == 1
    assert candidate["graph_entries"] == 2
    assert candidate["refreshes"] == 7
    assert candidate["refresh_copy_calls"] == 21
    assert candidate["refresh_copy_bytes"] == 1234
    assert candidate["owned_tensor_bytes"] == arena.owned_tensor_bytes
    assert (
        candidate["avoided_duplicate_graph_input_bytes"]
        == arena.owned_tensor_bytes
    )


def test_candidate_context_only_patches_k1_dispatch_and_restores_cleanly():
    from osuT5.osuT5.inference.optimized.single import engine
    from osuT5.osuT5.inference.optimized.single.shared_static_input_arena import (
        install_shared_static_input_arena_candidate,
    )
    from osuT5.osuT5.model.custom_transformers.modeling_varwhisper import (
        VarWhisperDecoderLayer,
    )

    original_generate = engine.active_prefix_decode_generate
    original_decoder_forward = VarWhisperDecoderLayer.forward
    with install_shared_static_input_arena_candidate():
        candidate_generate = engine.active_prefix_decode_generate
        assert candidate_generate is not original_generate
        assert VarWhisperDecoderLayer.forward is original_decoder_forward
        with pytest.raises(RuntimeError, match="cannot be nested"):
            with install_shared_static_input_arena_candidate():
                pass

    assert engine.active_prefix_decode_generate is original_generate
    assert VarWhisperDecoderLayer.forward is original_decoder_forward


def test_candidate_context_forces_opt_in_flag_and_rejects_override(monkeypatch):
    from osuT5.osuT5.inference.optimized.single import engine
    from osuT5.osuT5.inference.optimized.single.shared_static_input_arena import (
        install_shared_static_input_arena_candidate,
    )

    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "candidate-result"

    monkeypatch.setattr(engine, "active_prefix_decode_generate", original)
    with install_shared_static_input_arena_candidate():
        assert engine.active_prefix_decode_generate(1, marker=2) == "candidate-result"
        assert calls == [((1,), {"marker": 2, "shared_static_input_arena": True})]
        with pytest.raises(ValueError, match="requires its opt-in flag"):
            engine.active_prefix_decode_generate(
                shared_static_input_arena=False
            )
    assert engine.active_prefix_decode_generate is original


def test_candidate_context_restores_dispatch_after_body_failure():
    from osuT5.osuT5.inference.optimized.single import engine
    from osuT5.osuT5.inference.optimized.single.shared_static_input_arena import (
        install_shared_static_input_arena_candidate,
    )

    original = engine.active_prefix_decode_generate
    with pytest.raises(RuntimeError, match="body failed"):
        with install_shared_static_input_arena_candidate():
            raise RuntimeError("body failed")
    assert engine.active_prefix_decode_generate is original


def test_decode_loop_rejects_arena_without_graph_forward():
    config = SimpleNamespace(
        return_dict_in_generate=False,
        output_attentions=False,
        output_hidden_states=False,
        output_scores=False,
        output_logits=False,
        prefill_chunk_size=None,
        _pad_token_tensor=torch.tensor(0),
        max_length=4,
        do_sample=False,
    )
    with pytest.raises(ValueError, match="requires CUDA graph forward"):
        active_prefix_decode_generate(
            None,
            torch.tensor([[1]], dtype=torch.long),
            logits_processor=[],
            stopping_criteria=[],
            generation_config=config,
            shared_static_input_arena=True,
        )


def test_candidate_and_native_extensions_stay_cold_by_default():
    completed = _run_fresh_python(
        """
import sys
from osuT5.osuT5.inference.optimized.single import engine

assert "osuT5.osuT5.inference.optimized.single.shared_static_input_arena" not in sys.modules
for name in (
    "osuT5.osuT5.inference.optimized.kernels.q1_attention",
    "osuT5.osuT5.inference.optimized.kernels.cross_mlp",
    "torch.utils.cpp_extension",
):
    assert name not in sys.modules, name
assert engine.active_prefix_decode_generate.__name__ == "active_prefix_decode_generate"
"""
    )
    assert completed.returncode == 0, completed.stderr


def test_candidate_module_import_alone_does_not_import_engine_or_native_code():
    completed = _run_fresh_python(
        """
import sys
import osuT5.osuT5.inference.optimized.single.shared_static_input_arena

for name in (
    "osuT5.osuT5.inference.optimized.single.engine",
    "osuT5.osuT5.inference.optimized.kernels.q1_attention",
    "osuT5.osuT5.inference.optimized.kernels.cross_mlp",
    "torch.utils.cpp_extension",
):
    assert name not in sys.modules, name
"""
    )
    assert completed.returncode == 0, completed.stderr
