from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.inference.runtime_dispatch import (
    AttentionRuntimeHooks,
    attention_runtime_hooks,
    attention_runtime_hooks_context,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    REPO_ROOT
    / "osuT5/osuT5/model/custom_transformers/modeling_varwhisper.py"
)


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


def test_fresh_default_v32_import_is_optimized_and_native_cold():
    completed = _run_fresh_python(
        """
import importlib
import sys

for name in (
    "inference",
    "osuT5.osuT5.inference.server",
    "osuT5.osuT5.model.custom_transformers.modeling_varwhisper",
):
    importlib.import_module(name)

for name in sys.modules:
    assert not name.startswith("osuT5.osuT5.inference.optimized"), name
    assert not name.startswith("osuT5.osuT5.inference.native"), name
    assert not name.startswith("torch.utils.cpp_extension"), name
"""
    )
    assert completed.returncode == 0, completed.stderr


def test_legacy_native_shims_have_no_optimized_or_cpp_extension_side_effect():
    completed = _run_fresh_python(
        """
import importlib
import sys

for name in (
    "osuT5.osuT5.inference.native_q1_attention",
):
    importlib.import_module(name)
assert "osuT5.osuT5.inference.optimized.kernels.q1_attention" not in sys.modules
assert "torch.utils.cpp_extension" not in sys.modules
"""
    )
    assert completed.returncode == 0, completed.stderr


def test_q1_kernel_import_does_not_build_and_loader_is_singleton(monkeypatch):
    from osuT5.osuT5.inference.optimized.kernels import q1_attention

    calls = []
    extension = SimpleNamespace()

    def fake_load_inline(**kwargs):
        calls.append(kwargs)
        return extension

    monkeypatch.setattr(q1_attention, "_NATIVE_Q1_ATTENTION", None)
    monkeypatch.setattr(q1_attention, "load_inline", fake_load_inline)
    assert q1_attention._NATIVE_Q1_ATTENTION is None
    assert q1_attention.preload_native_q1_attention() is extension
    assert q1_attention.preload_native_q1_attention() is extension
    assert len(calls) == 1
    assert calls[0]["name"] == "mapperatorinator_q1_attention"
    assert calls[0]["functions"] == [
        "q1_attention",
        "q1_rope_cache_attention",
    ]
    assert calls[0]["extra_cuda_cflags"] == ["-O3"]


def test_q1_wrappers_preserve_mask_conversion_and_argument_order(monkeypatch):
    from osuT5.osuT5.inference.optimized.kernels import q1_attention

    calls = []

    class _Extension:
        def q1_attention(self, *args):
            calls.append(("q1", args))
            return args[0]

        def q1_rope_cache_attention(self, *args):
            calls.append(("rope", args))
            return args[0]

    monkeypatch.setattr(q1_attention, "_NATIVE_Q1_ATTENTION", _Extension())
    query = torch.randn(1, 2, 1, 4)
    key = torch.randn(1, 2, 3, 4)
    value = torch.randn(1, 2, 3, 4)
    mask = torch.arange(6, dtype=torch.float64).reshape(2, 3).transpose(0, 1)
    q1_attention.native_q1_attention(query, key, value, mask)

    kind, args = calls.pop(0)
    assert kind == "q1"
    assert args[0] is query
    assert args[1] is key
    assert args[2] is value
    converted_mask = args[3]
    assert converted_mask.dtype == torch.float32
    assert converted_mask.is_contiguous()
    assert converted_mask.shape == (6,)

    qkv = torch.randn(1, 1, 3, 2, 4)
    cache_keys = torch.randn(1, 2, 8, 4)
    cache_values = torch.randn(1, 2, 8, 4)
    cos = torch.randn(1, 4)
    sin = torch.randn(1, 4)
    cache_position = torch.tensor([3])
    q1_attention.native_q1_rope_cache_attention(
        qkv,
        cache_keys,
        cache_values,
        cos,
        sin,
        cache_position,
        mask,
        7,
    )
    kind, args = calls.pop(0)
    assert kind == "rope"
    assert args[0] is qkv
    assert args[1] is cache_keys
    assert args[2] is cache_values
    assert args[3] is cos
    assert args[4] is sin
    assert args[5] is cache_position
    assert args[6].dtype == torch.float32
    assert args[6].is_contiguous()
    assert args[7] == 7


def test_attention_runtime_hooks_restore_for_nested_and_exception_paths():
    empty = attention_runtime_hooks()
    first = AttentionRuntimeHooks(sdpa_attention_forward=lambda: "first")
    second = AttentionRuntimeHooks(sdpa_attention_forward=lambda: "second")

    with attention_runtime_hooks_context(first):
        assert attention_runtime_hooks() is first
        with attention_runtime_hooks_context(second):
            assert attention_runtime_hooks() is second
        assert attention_runtime_hooks() is first
        with pytest.raises(RuntimeError, match="boom"):
            with attention_runtime_hooks_context(second):
                raise RuntimeError("boom")
        assert attention_runtime_hooks() is first
    assert attention_runtime_hooks() is empty


def test_q1_bmm_only_context_imports_dispatch_without_native_extension():
    completed = _run_fresh_python(
        """
import sys
from osuT5.osuT5.inference.optimized.single.runtime_context import attention_runtime_context

with attention_runtime_context(q1_bmm_cross_attention=True):
    assert "osuT5.osuT5.inference.optimized.kernels.dispatch" in sys.modules
assert "osuT5.osuT5.inference.optimized.kernels.q1_attention" not in sys.modules
assert "torch.utils.cpp_extension" not in sys.modules
"""
    )
    assert completed.returncode == 0, completed.stderr


def test_native_generation_context_preloads_then_installs_hooks(monkeypatch):
    from osuT5.osuT5.inference.optimized.kernels import q1_attention
    from osuT5.osuT5.inference.optimized.single.runtime_context import attention_runtime_context

    events = []
    monkeypatch.setattr(
        q1_attention,
        "preload_native_q1_attention",
        lambda: events.append("preload"),
    )
    empty = attention_runtime_hooks()
    with attention_runtime_context(native_q1_self_attention=True):
        events.append("entered")
        hooks = attention_runtime_hooks()
        assert hooks.sdpa_attention_forward is not None
        assert hooks.q1_rope_cache_self_attention_forward is None
    assert events == ["preload", "entered"]
    assert attention_runtime_hooks() is empty


def test_q1_bmm_context_installs_only_sdpa_hook():
    from osuT5.osuT5.inference.optimized.single.runtime_context import attention_runtime_context

    with attention_runtime_context(q1_bmm_cross_attention=True):
        hooks = attention_runtime_hooks()
        assert hooks.sdpa_attention_forward is not None
        assert hooks.q1_rope_cache_self_attention_forward is None


def test_nested_default_generation_context_temporarily_disables_attention_hooks():
    from osuT5.osuT5.inference.optimized.single.runtime_context import attention_runtime_context

    with attention_runtime_context(q1_bmm_cross_attention=True):
        outer_hooks = attention_runtime_hooks()
        assert outer_hooks.sdpa_attention_forward is not None
        with attention_runtime_context():
            inner_hooks = attention_runtime_hooks()
            assert inner_hooks.sdpa_attention_forward is None
            assert inner_hooks.q1_rope_cache_self_attention_forward is None
        assert attention_runtime_hooks() is outer_hooks


def test_active_prefix_context_owns_state_and_preserves_other_hooks():
    from osuT5.osuT5.inference.optimized.single.runtime_context import (
        active_prefix_self_attention_context,
        active_prefix_self_attention_length,
        attention_runtime_context,
    )

    assert active_prefix_self_attention_length() is None
    with attention_runtime_context(q1_bmm_cross_attention=True):
        q1_hook = attention_runtime_hooks().sdpa_attention_forward
        with active_prefix_self_attention_context(64):
            hooks = attention_runtime_hooks()
            assert active_prefix_self_attention_length() == 64
            assert hooks.sdpa_attention_inputs is not None
            assert hooks.sdpa_attention_forward is q1_hook
            prefix_hook = hooks.sdpa_attention_inputs
            with attention_runtime_context(
                q1_bmm_cross_attention=True,
                native_q1_self_attention=False,
                native_q1_rope_cache_self_attention=False,
            ):
                nested_hooks = attention_runtime_hooks()
                assert nested_hooks.sdpa_attention_inputs is prefix_hook
                assert nested_hooks.sdpa_attention_forward is not None
            with active_prefix_self_attention_context(None):
                assert active_prefix_self_attention_length() is None
                assert attention_runtime_hooks().sdpa_attention_inputs is None
            assert active_prefix_self_attention_length() == 64
            assert attention_runtime_hooks().sdpa_attention_inputs is not None
        assert active_prefix_self_attention_length() is None
        assert attention_runtime_hooks().sdpa_attention_inputs is None
        assert attention_runtime_hooks().sdpa_attention_forward is q1_hook


def test_fused_generation_context_installs_only_fused_hook(monkeypatch):
    from osuT5.osuT5.inference.optimized.kernels import q1_attention
    from osuT5.osuT5.inference.optimized.single.runtime_context import attention_runtime_context

    monkeypatch.setattr(q1_attention, "preload_native_q1_attention", lambda: None)
    with attention_runtime_context(native_q1_rope_cache_self_attention=True):
        hooks = attention_runtime_hooks()
        assert hooks.sdpa_attention_forward is None
        assert hooks.q1_rope_cache_self_attention_forward is not None


def test_model_hot_path_has_no_optimized_or_native_imports():
    tree = ast.parse(MODEL_PATH.read_text(encoding="utf-8"))
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    rendered = [ast.unparse(node) for node in imports]
    assert not any("inference.optimized" in item for item in rendered)
    assert not any("inference.native" in item for item in rendered)
    assert not any("cpp_extension" in item for item in rendered)
    source = MODEL_PATH.read_text(encoding="utf-8")
    assert "torch.bmm" not in source
    assert "native_q1_rope_cache_self_attention_enabled" not in source
    assert "native_q1_self_attention_enabled" not in source
    assert "q1_bmm_cross_attention_enabled" not in source
    assert "active_prefix_self_attention_length" not in source
    assert "key[:, :, :prefix_length" not in source
