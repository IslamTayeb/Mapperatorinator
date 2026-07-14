from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.inference.optimized.scout import weight_only


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_production_import_does_not_import_weight_only_scout() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib, sys; "
                "importlib.import_module('osuT5.osuT5.inference.optimized.single.engine'); "
                "assert 'osuT5.osuT5.inference.optimized.scout.weight_only' "
                "not in sys.modules; "
                "assert 'torch.utils.cpp_extension' not in sys.modules"
            ),
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_extension_is_cold_singleton_and_exposes_all_mixed_weight_kernels(
    monkeypatch,
) -> None:
    calls = []
    extension = object()

    def fake_load_inline(**kwargs):
        calls.append(kwargs)
        return extension

    monkeypatch.setattr(weight_only, "_WEIGHT_ONLY_EXTENSION", None)
    monkeypatch.setattr(weight_only, "_load_inline", fake_load_inline)

    assert weight_only.preload_weight_only_extension() is extension
    assert weight_only.preload_weight_only_extension() is extension
    assert len(calls) == 1
    assert calls[0]["name"] == "mapperatorinator_weight_only_fp16_scout"
    assert calls[0]["functions"] == [
        "weight_only_linear",
        "weight_only_rmsnorm_linear",
        "weight_only_linear_residual",
        "weight_only_mlp_residual",
    ]


def test_cuda_source_keeps_fp32_state_and_uses_vectorized_fp16_weight_reads() -> None:
    source = weight_only._CUDA_SOURCE

    assert "const __half2* weight2" in source
    assert "__half22float2" in source
    assert "const float* __restrict__ input" in source
    assert "float* __restrict__ output" in source
    assert "float local_dot = 0.0f" in source
    assert "warp_sum" in source
    assert "input activations must use FP32 storage" in source
    assert "matrix weights must use FP16 storage" in source
    assert "bias must retain FP32 storage" in source
    assert "norm weight must remain contiguous FP32" in source
    assert "half2 weight loads" in source
    assert "wmma" not in source.lower()
    assert "c10::cuda::CUDAGuard" in source


def test_cpu_activation_fails_before_extension_build(monkeypatch) -> None:
    packed = weight_only.PackedLinear(
        weight=torch.zeros((4, 4), dtype=torch.float16),
        bias=None,
        source_weight_bytes=64,
    )
    monkeypatch.setattr(
        weight_only,
        "_load_weight_only_extension",
        lambda: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    with pytest.raises(RuntimeError, match="CUDA tensor"):
        weight_only.weight_only_linear(torch.zeros((1, 1, 4)), packed)


def test_wrong_storage_dtype_fails_loudly_before_extension_build(monkeypatch) -> None:
    monkeypatch.setattr(
        weight_only,
        "_load_weight_only_extension",
        lambda: (_ for _ in ()).throw(AssertionError("must not build")),
    )
    monkeypatch.setattr(
        weight_only,
        "_require_fp32_activation",
        lambda *args, **kwargs: None,
    )
    packed = weight_only.PackedLinear(
        weight=torch.zeros((4, 4), dtype=torch.float32),
        bias=None,
        source_weight_bytes=64,
    )

    with pytest.raises(TypeError, match="FP16 storage"):
        weight_only.weight_only_linear(torch.zeros((1, 1, 4)), packed)


def test_wrappers_preserve_argument_order_and_fp32_output(monkeypatch) -> None:
    calls = []

    class Extension:
        def weight_only_linear(self, *args):
            calls.append(("linear", args))
            return args[0]

        def weight_only_rmsnorm_linear(self, *args):
            calls.append(("rmsnorm", args))
            return args[0]

        def weight_only_linear_residual(self, *args):
            calls.append(("residual", args))
            return args[0]

        def weight_only_mlp_residual(self, *args):
            calls.append(("mlp", args))
            return args[0]

    monkeypatch.setattr(weight_only, "_WEIGHT_ONLY_EXTENSION", Extension())
    monkeypatch.setattr(
        weight_only,
        "_require_fp32_activation",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        weight_only,
        "_require_fp16_weight",
        lambda *args, **kwargs: None,
    )
    hidden = torch.zeros((1, 1, 4), dtype=torch.float32)
    norm = torch.ones(4, dtype=torch.float32)
    first = weight_only.PackedLinear(
        weight=torch.ones((8, 4), dtype=torch.float16),
        bias=torch.zeros(8, dtype=torch.float32),
        source_weight_bytes=128,
    )
    second = weight_only.PackedLinear(
        weight=torch.ones((4, 8), dtype=torch.float16),
        bias=torch.zeros(4, dtype=torch.float32),
        source_weight_bytes=128,
    )

    assert weight_only.weight_only_linear(hidden, first).dtype == torch.float32
    assert calls.pop(0) == (
        "linear",
        (hidden, first.weight, first.bias, 8),
    )
    assert weight_only.weight_only_rmsnorm_linear(
        hidden,
        norm,
        first,
        eps=1e-5,
    ).dtype == torch.float32
    assert calls.pop(0) == (
        "rmsnorm",
        (hidden, norm, first.weight, first.bias, 1e-5, 8),
    )
    assert weight_only.weight_only_linear_residual(
        hidden,
        hidden,
        second,
    ).dtype == torch.float32
    assert calls.pop(0) == (
        "residual",
        (hidden, hidden, second.weight, second.bias, 8),
    )
    assert weight_only.weight_only_mlp_residual(
        hidden,
        norm,
        first,
        second,
        eps=1e-5,
    ).dtype == torch.float32
    assert calls.pop(0) == (
        "mlp",
        (
            hidden,
            norm,
            first.weight,
            first.bias,
            second.weight,
            second.bias,
            1e-5,
            8,
        ),
    )


def test_decoder_pack_reports_replacement_and_scout_memory_separately() -> None:
    def packed(rows: int, columns: int) -> weight_only.PackedLinear:
        tensor = torch.zeros((rows, columns), dtype=torch.float16)
        return weight_only.PackedLinear(
            weight=tensor,
            bias=None,
            source_weight_bytes=tensor.numel() * 4,
        )

    pack = weight_only.DecoderWeightPack(
        self_qkv=packed(12, 4),
        self_out=packed(4, 4),
        cross_q=packed(4, 4),
        cross_out=packed(4, 4),
        fc1=packed(16, 4),
        fc2=packed(4, 16),
    )

    report = pack.memory_report()
    assert report["packed_weight_bytes"] * 2 == report["source_weight_bytes"]
    assert report["projected_replacement_saving_bytes"] == report["packed_weight_bytes"]
    assert report["scout_resident_increment_bytes"] == report["packed_weight_bytes"]


def test_pack_requires_fp32_cuda_source_without_silently_converting_bias(
    monkeypatch,
) -> None:
    checked = []
    monkeypatch.setattr(
        weight_only,
        "_require_fp32_activation",
        lambda value, *, name: checked.append((name, value.dtype)),
    )
    module = SimpleNamespace(
        weight=torch.ones((3, 4), dtype=torch.float32),
        bias=torch.zeros(3, dtype=torch.float32),
    )

    packed = weight_only.PackedLinear.from_module(module)

    assert packed.weight.dtype == torch.float16
    assert packed.bias is module.bias
    assert packed.source_weight_bytes == 3 * 4 * 4
    assert checked == [
        ("source weight", torch.float32),
        ("source bias", torch.float32),
    ]
