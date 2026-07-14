from __future__ import annotations

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


def test_fresh_optimized_engine_import_keeps_native_extensions_cold():
    completed = _run_fresh_python(
        """
import importlib
import sys

importlib.import_module("osuT5.osuT5.inference.optimized.single.engine")

for name in sys.modules:
    assert not name.endswith(".optimized.kernels.q1_attention"), name
    assert not name.endswith(".optimized.kernels.cross_mlp"), name
    assert not name.endswith(".optimized.kernels.decoder_layer"), name
    assert not name.startswith("torch.utils.cpp_extension"), name
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
        "split_kv_q1_rope_cache_attention",
    ]
    assert calls[0]["extra_cuda_cflags"] == ["-O3"]
    source = calls[0]["cuda_sources"]
    assert "#include <cuda_fp16.h>" in source
    assert "#include <c10/cuda/CUDAGuard.h>" in source
    assert source.count("c10::cuda::CUDAGuard") == 3
    assert source.count("__global__ void q1_attention_kernel(") == 1
    assert source.count("__global__ void q1_rope_cache_attention_kernel(") == 1
    assert source.count("__global__ void split_kv_prepare_rope_cache_kernel(") == 1
    assert source.count("__global__ void split_kv_partial_kernel(") == 1
    assert source.count("__global__ void split_kv_merge_kernel(") == 1
    assert "q1_attention_half_kernel" not in source
    assert "q1_rope_cache_attention_half_kernel" not in source
    assert "q1_attention_kernel<float, block_size>" in source
    assert "q1_attention_kernel<__half, block_size>" in source
    assert "q1_rope_cache_attention_kernel<float, block_size>" in source
    assert "q1_rope_cache_attention_kernel<__half, block_size>" in source
    assert "q.scalar_type() == torch::kFloat32 || q.scalar_type() == torch::kFloat16" in source
    assert "qkv.scalar_type() == torch::kFloat32 || qkv.scalar_type() == torch::kFloat16" in source
    float_traits_start = source.index("struct Q1ScalarTraits<float>")
    half_traits_start = source.index("struct Q1ScalarTraits<__half>")
    q1_body_start = source.index("__global__ void q1_attention_kernel(")
    q1_wrapper_start = source.index("torch::Tensor q1_attention(")
    fused_body_start = source.index("__global__ void q1_rope_cache_attention_kernel(")
    fused_wrapper_start = source.index("torch::Tensor q1_rope_cache_attention(")
    float_traits = source[float_traits_start:half_traits_start]
    half_traits = source[half_traits_start:q1_body_start]
    q1_body = source[q1_body_start:q1_wrapper_start]
    fused_body = source[fused_body_start:fused_wrapper_start]

    assert float_traits.count("return value;") == 2
    assert "__half2float" not in float_traits
    assert "__float2half_rn" not in float_traits
    assert "return __half2float(value);" in half_traits
    assert "return __float2half_rn(value);" in half_traits
    assert source.count("__half2float") == 1
    assert source.count("__float2half_rn") == 1
    for body in (q1_body, fused_body):
        assert "using Traits = Q1ScalarTraits<scalar_t>;" in body
        assert "Traits::load" in body
        assert "Traits::store" in body
        assert "__half2float" not in body
        assert "__float2half_rn" not in body
        assert "float score = 0.0f" in body
        assert "float denom = 0.0f" in body
        assert "float numer = 0.0f" in body
        assert "extern __shared__ float shared_mem[]" in body
    assert source.count('asm("trap;")') == 2
    assert "split-KV q1 requires fp32 or fp16 storage" in source
    assert "split_kv_prepare_rope_cache_kernel<scalar_t, block_size>" in source
    assert "split_kv_partial_kernel<scalar_t, block_size>" in source
    assert "split_kv_merge_kernel<scalar_t, block_size>" in source
    assert "LAUNCH_SPLIT_KV(float, FLOAT_PTR)" in source
    assert "LAUNCH_SPLIT_KV(__half, HALF_PTR)" in source
    assert "auto fp32_options = qkv.options().dtype(torch::kFloat32);" in source
    assert "Traits::load(cache_keys[" in source
    assert "Traits::load(cache_values[" in source
    assert "Traits::store(numerator / global_denom)" in source
    assert "constexpr int split_count = 8;" in source
    assert "properties.major == 7 && properties.minor == 5" in source


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_q1_wrappers_preserve_dtype_mask_conversion_and_argument_order(
    monkeypatch,
    dtype,
):
    from osuT5.osuT5.inference.optimized.kernels import q1_attention

    calls = []

    class _Extension:
        def q1_attention(self, *args):
            calls.append(("q1", args))
            return args[0]

        def q1_rope_cache_attention(self, *args):
            calls.append(("rope", args))
            return args[0]

        def split_kv_q1_rope_cache_attention(self, *args):
            calls.append(("split_rope", args))
            return args[0]

    monkeypatch.setattr(q1_attention, "_NATIVE_Q1_ATTENTION", _Extension())
    monkeypatch.setattr(q1_attention, "_validate_q1_inputs", lambda *args: 6)
    monkeypatch.setattr(q1_attention, "_validate_rope_cache_inputs", lambda *args: None)
    query = torch.randn(1, 2, 1, 4, dtype=dtype)
    key = torch.randn(1, 2, 6, 4, dtype=dtype)
    value = torch.randn(1, 2, 6, 4, dtype=dtype)
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
    assert args[0].dtype == args[1].dtype == args[2].dtype == dtype

    qkv = torch.randn(1, 1, 3, 2, 4, dtype=dtype)
    cache_keys = torch.randn(1, 2, 8, 4, dtype=dtype)
    cache_values = torch.randn(1, 2, 8, 4, dtype=dtype)
    cos = torch.randn(1, 1, 4, dtype=dtype)
    sin = torch.randn(1, 1, 4, dtype=dtype)
    cache_position = torch.tensor([3])
    rope_mask = torch.arange(7, dtype=torch.float64)
    q1_attention.native_q1_rope_cache_attention(
        qkv,
        cache_keys,
        cache_values,
        cos,
        sin,
        cache_position,
        rope_mask,
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
    assert all(args[index].dtype == dtype for index in range(5))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_supported_sm75_live_prefix_routes_split8_and_other_cases_fall_back(
    monkeypatch,
    dtype,
):
    from osuT5.osuT5.inference.optimized.kernels import q1_attention

    qkv = torch.zeros((1, 1, 3, 2, 4), dtype=dtype)
    calls = []

    class _Extension:
        def q1_rope_cache_attention(self, *args):
            calls.append(("accepted", args))
            return args[0]

        def split_kv_q1_rope_cache_attention(self, *args):
            calls.append(("split8", args))
            return args[0]

    monkeypatch.setattr(q1_attention, "_NATIVE_Q1_ATTENTION", _Extension())
    monkeypatch.setattr(q1_attention, "_validate_rope_cache_inputs", lambda *args: None)
    monkeypatch.setattr(q1_attention, "_native_mask", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        q1_attention,
        "native_q1_rope_cache_attention_variant",
        lambda qkv, prefix: "split_kv_8" if prefix == 640 else "accepted",
    )
    arguments = (
        qkv,
        torch.zeros((1, 2, 832, 4)),
        torch.zeros((1, 2, 832, 4)),
        torch.zeros((1, 1, 4)),
        torch.zeros((1, 1, 4)),
        torch.tensor([1]),
        None,
    )

    q1_attention.native_q1_rope_cache_attention(*arguments, 640)
    q1_attention.native_q1_rope_cache_attention(*arguments, 128)

    assert [kind for kind, _ in calls] == ["split8", "accepted"]
    assert calls[0][1][-1] == 640
    assert calls[1][1][-1] == 128


def test_split_kv_policy_is_supported_dtype_sm75_and_live_prefix_only():
    from osuT5.osuT5.inference.optimized.kernels import q1_attention

    assert q1_attention.native_q1_rope_cache_attention_variant(
        torch.zeros((1, 1, 3, 2, 4)), 640
    ) == "accepted"
    assert tuple(sorted(q1_attention._SPLIT_KV_Q1_PREFIXES)) == tuple(
        range(192, 833, 64)
    )
    assert q1_attention._SPLIT_KV_Q1_SPLITS == 8
    assert q1_attention._split_kv_q1_eligible(
        dtype=torch.float32,
        device_type="cuda",
        active_prefix_length=640,
        capability=(7, 5),
    )
    assert q1_attention._split_kv_q1_eligible(
        dtype=torch.float16,
        device_type="cuda",
        active_prefix_length=640,
        capability=(7, 5),
    )
    for overrides in (
        {"dtype": torch.bfloat16},
        {"device_type": "cpu"},
        {"active_prefix_length": 128},
        {"active_prefix_length": 896},
        {"capability": (8, 0)},
    ):
        arguments = {
            "dtype": torch.float32,
            "device_type": "cuda",
            "active_prefix_length": 640,
            "capability": (7, 5),
        }
        arguments.update(overrides)
        assert not q1_attention._split_kv_q1_eligible(**arguments)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_explicit_split_and_accepted_component_wrappers_share_validation(
    monkeypatch,
    dtype,
):
    from osuT5.osuT5.inference.optimized.kernels import q1_attention

    qkv = torch.zeros((1, 1, 3, 2, 4), dtype=dtype)
    arguments = (
        qkv,
        torch.zeros((1, 2, 832, 4), dtype=dtype),
        torch.zeros((1, 2, 832, 4), dtype=dtype),
        torch.zeros((1, 1, 4), dtype=dtype),
        torch.zeros((1, 1, 4), dtype=dtype),
        torch.tensor([1]),
        None,
        640,
    )
    calls = []

    class _Extension:
        def q1_rope_cache_attention(self, *args):
            calls.append(("accepted", args))
            return args[0]

        def split_kv_q1_rope_cache_attention(self, *args):
            calls.append(("split", args))
            return args[0]

    monkeypatch.setattr(q1_attention, "_NATIVE_Q1_ATTENTION", _Extension())
    monkeypatch.setattr(
        q1_attention,
        "_validate_rope_cache_inputs",
        lambda *args: None,
    )
    monkeypatch.setattr(q1_attention, "_native_mask", lambda *args, **kwargs: None)
    monkeypatch.setattr(q1_attention, "_device_capability", lambda device: (7, 5))

    accepted = q1_attention.accepted_q1_rope_cache_attention(*arguments)
    split = q1_attention.split_kv_q1_rope_cache_attention(*arguments)

    assert accepted is qkv
    assert split is qkv
    assert [name for name, _ in calls] == ["accepted", "split"]
    assert calls[1][1][-1] == 640


def test_explicit_split_wrapper_rejects_unsupported_contract_before_build(
    monkeypatch,
):
    from osuT5.osuT5.inference.optimized.kernels import q1_attention

    qkv = torch.zeros((1, 1, 3, 2, 4), dtype=torch.float16)
    arguments = (
        qkv,
        torch.zeros((1, 2, 832, 4), dtype=torch.float16),
        torch.zeros((1, 2, 832, 4), dtype=torch.float16),
        torch.zeros((1, 1, 4), dtype=torch.float16),
        torch.zeros((1, 1, 4), dtype=torch.float16),
        torch.tensor([1]),
        None,
        640,
    )
    monkeypatch.setattr(
        q1_attention,
        "_validate_rope_cache_inputs",
        lambda *args: None,
    )
    monkeypatch.setattr(q1_attention, "_native_mask", lambda *args, **kwargs: None)
    monkeypatch.setattr(q1_attention, "_device_capability", lambda device: (8, 0))
    monkeypatch.setattr(
        q1_attention,
        "_load_native_q1_attention",
        lambda: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    with pytest.raises(RuntimeError, match="requires FP32 or FP16 SM75"):
        q1_attention.split_kv_q1_rope_cache_attention(*arguments)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_q1_supported_cpu_inputs_fail_before_extension_build(monkeypatch, dtype):
    from osuT5.osuT5.inference.optimized.kernels import q1_attention

    monkeypatch.setattr(
        q1_attention,
        "_load_native_q1_attention",
        lambda: (_ for _ in ()).throw(AssertionError("must not build")),
    )
    query = torch.zeros((1, 2, 1, 4), dtype=dtype)
    key = torch.zeros((1, 2, 3, 4), dtype=dtype)
    value = torch.zeros_like(key)

    with pytest.raises(RuntimeError, match="CUDA tensors"):
        q1_attention.native_q1_attention(query, key, value, None)


def test_q1_rejects_unsupported_and_mixed_dtypes_before_extension_build(monkeypatch):
    from osuT5.osuT5.inference.optimized.kernels import q1_attention

    monkeypatch.setattr(
        q1_attention,
        "_load_native_q1_attention",
        lambda: (_ for _ in ()).throw(AssertionError("must not build")),
    )
    key = torch.zeros((1, 2, 3, 4), dtype=torch.float16)
    value = torch.zeros_like(key)
    with pytest.raises(TypeError, match="float32 or float16"):
        q1_attention.native_q1_attention(key.double()[..., :1, :], key, value, None)

    query = torch.zeros((1, 2, 1, 4), dtype=torch.float16)
    with pytest.raises(TypeError, match="key must have dtype"):
        q1_attention.native_q1_attention(query, key.float(), value, None)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rope_cache_supported_cpu_inputs_fail_before_extension_build(
    monkeypatch,
    dtype,
):
    from osuT5.osuT5.inference.optimized.kernels import q1_attention

    monkeypatch.setattr(
        q1_attention,
        "_load_native_q1_attention",
        lambda: (_ for _ in ()).throw(AssertionError("must not build")),
    )
    qkv = torch.zeros((1, 1, 3, 2, 4), dtype=dtype)
    keys = torch.zeros((1, 2, 8, 4), dtype=dtype)
    values = torch.zeros_like(keys)
    cos = torch.zeros((1, 1, 4), dtype=dtype)
    sin = torch.zeros_like(cos)

    with pytest.raises(RuntimeError, match="CUDA tensors"):
        q1_attention.native_q1_rope_cache_attention(
            qkv,
            keys,
            values,
            cos,
            sin,
            torch.tensor([1]),
            None,
            4,
        )

    with pytest.raises(TypeError, match="cache_keys must have dtype"):
        q1_attention.native_q1_rope_cache_attention(
            qkv,
            keys.float() if dtype == torch.float16 else keys.half(),
            values,
            cos,
            sin,
            torch.tensor([1]),
            None,
            4,
        )


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
        assert hooks.sdpa_attention_forward.keywords["expected_dtype"] == torch.float32
        assert hooks.q1_rope_cache_self_attention_forward is None
    assert events == ["preload", "entered"]
    assert attention_runtime_hooks() is empty


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
        assert (
            hooks.q1_rope_cache_self_attention_forward.keywords[
                "native_q1_rope_cache_attention_variant"
            ]
            is q1_attention.native_q1_rope_cache_attention_variant
        )


def test_fp16_generation_context_uses_the_same_dispatch_policy(monkeypatch):
    from osuT5.osuT5.inference.optimized.kernels import q1_attention
    from osuT5.osuT5.inference.optimized.single.runtime_context import (
        attention_runtime_context,
    )

    monkeypatch.setattr(q1_attention, "preload_native_q1_attention", lambda: None)
    with attention_runtime_context(
        q1_bmm_cross_attention=True,
        native_q1_self_attention=True,
        native_q1_rope_cache_self_attention=True,
        expected_dtype=torch.float16,
    ):
        hooks = attention_runtime_hooks()
        assert hooks.sdpa_attention_forward.keywords["expected_dtype"] == torch.float16
        assert (
            hooks.q1_rope_cache_self_attention_forward.keywords["expected_dtype"]
            == torch.float16
        )


def test_generation_context_rejects_unsupported_dtype() -> None:
    from osuT5.osuT5.inference.optimized.single.runtime_context import (
        attention_runtime_context,
    )

    with pytest.raises(TypeError, match="only float32 or float16"):
        with attention_runtime_context(expected_dtype=torch.bfloat16):
            pass
