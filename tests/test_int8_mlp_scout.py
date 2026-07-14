from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.inference.optimized.scout import int8_mlp


def test_extension_is_cold_singleton_with_explicit_manual_binding(monkeypatch) -> None:
    calls = []
    extension = object()

    def fake_load_inline(**kwargs):
        calls.append(kwargs)
        return extension

    monkeypatch.setattr(int8_mlp, "_INT8_MLP_EXTENSION", None)
    monkeypatch.setattr(int8_mlp, "_load_inline", fake_load_inline)

    assert int8_mlp.preload_int8_mlp_extension() is extension
    assert int8_mlp.preload_int8_mlp_extension() is extension
    assert len(calls) == 1
    assert calls[0]["name"] == "mapperatorinator_int8_mlp_scout_v1"
    assert "functions" not in calls[0]
    assert "torch::Tensor int8_weight_mlp_residual(" in calls[0]["cpp_sources"]
    assert "PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)" in calls[0]["cpp_sources"]
    assert '&int8_weight_mlp_residual' in calls[0]["cpp_sources"]


def test_cuda_source_uses_int8_storage_and_fp32_state_accumulation() -> None:
    source = int8_mlp._CUDA_SOURCE

    assert "const char4* weight4" in source
    assert "const int8_t* __restrict__ weight" in source
    assert "float sum = 0.0f" in source
    assert "sum *= scale[out]" in source
    assert "const float* __restrict__ input" in source
    assert "float* __restrict__ output" in source
    assert "input activations must remain contiguous FP32" in source
    assert "weight must be contiguous INT8" in source
    assert "scale must remain contiguous FP32" in source
    assert "norm weight must remain contiguous FP32" in source
    assert "c10::cuda::CUDAGuard" in source
    assert "wmma" not in source.lower()


def test_symmetric_per_row_quantization_handles_zero_rows_and_bounds() -> None:
    weight = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [-2.0, -1.0, 1.0, 2.0],
        ],
        dtype=torch.float32,
    )

    quantized, scale = int8_mlp.symmetric_int8_per_row(weight)

    assert quantized.dtype == torch.int8
    assert scale.dtype == torch.float32
    assert torch.equal(quantized[0], torch.zeros(4, dtype=torch.int8))
    assert scale[0].item() == 1.0
    assert quantized[1].min().item() == -127
    assert quantized[1].max().item() == 127
    restored = quantized[1].float() * scale[1]
    assert torch.allclose(restored, weight[1], atol=float(scale[1]) / 2 + 1e-7)


def test_quantization_rejects_wrong_dtype_shape_and_vector_width() -> None:
    with pytest.raises(TypeError, match="FP32"):
        int8_mlp.symmetric_int8_per_row(torch.ones((2, 4), dtype=torch.float16))
    with pytest.raises(ValueError, match="two-dimensional"):
        int8_mlp.symmetric_int8_per_row(torch.ones(4))
    with pytest.raises(ValueError, match="divisible by four"):
        int8_mlp.symmetric_int8_per_row(torch.ones((2, 3)))


def test_cpu_input_fails_before_extension_build(monkeypatch) -> None:
    packed = int8_mlp.Int8PackedLinear(
        weight=torch.zeros((4, 4), dtype=torch.int8),
        scale=torch.ones(4),
        bias=None,
        source_weight_bytes=64,
    )
    pack = int8_mlp.Int8MlpPack(fc1=packed, fc2=packed)
    monkeypatch.setattr(
        int8_mlp,
        "_load_int8_mlp_extension",
        lambda: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    with pytest.raises(RuntimeError, match="CUDA tensor"):
        int8_mlp.int8_weight_mlp_residual(
            torch.zeros((1, 1, 4)),
            torch.ones(4),
            pack,
            eps=1e-5,
        )


def test_wrapper_preserves_pack_argument_order_and_fp32_output(monkeypatch) -> None:
    calls = []

    class Extension:
        @staticmethod
        def int8_weight_mlp_residual(*args):
            calls.append(args)
            return args[0]

    monkeypatch.setattr(int8_mlp, "_INT8_MLP_EXTENSION", Extension())
    monkeypatch.setattr(int8_mlp, "_require_fp32_cuda", lambda *args, **kwargs: None)
    monkeypatch.setattr(int8_mlp, "_require_int8_pack", lambda *args, **kwargs: None)
    hidden = torch.zeros((1, 1, 4), dtype=torch.float32)
    norm = torch.ones(4, dtype=torch.float32)
    fc1 = int8_mlp.Int8PackedLinear(
        weight=torch.ones((8, 4), dtype=torch.int8),
        scale=torch.ones(8),
        bias=torch.zeros(8),
        source_weight_bytes=128,
    )
    fc2 = int8_mlp.Int8PackedLinear(
        weight=torch.ones((4, 8), dtype=torch.int8),
        scale=torch.ones(4),
        bias=torch.zeros(4),
        source_weight_bytes=128,
    )
    pack = int8_mlp.Int8MlpPack(fc1=fc1, fc2=fc2)

    output = int8_mlp.int8_weight_mlp_residual(
        hidden,
        norm,
        pack,
        eps=1e-5,
        fc1_outputs_per_block=2,
        fc2_outputs_per_block=4,
    )

    assert output.dtype == torch.float32
    assert calls == [
        (
            hidden,
            norm,
            fc1.weight,
            fc1.scale,
            fc1.bias,
            fc2.weight,
            fc2.scale,
            fc2.bias,
            1e-5,
            2,
            4,
        )
    ]


def test_pack_memory_separates_int8_weights_fp32_scales_and_sources() -> None:
    def linear(rows: int, columns: int) -> int8_mlp.Int8PackedLinear:
        return int8_mlp.Int8PackedLinear(
            weight=torch.zeros((rows, columns), dtype=torch.int8),
            scale=torch.ones(rows, dtype=torch.float32),
            bias=None,
            source_weight_bytes=rows * columns * 4,
        )

    pack = int8_mlp.Int8MlpPack(fc1=linear(8, 4), fc2=linear(4, 8))
    report = pack.memory_report()

    expected_packed = 8 * 4 + 8 * 4 + 4 * 8 + 4 * 4
    assert report["packed_weight_bytes"] == expected_packed
    assert report["retained_fp32_source_weight_bytes"] == (8 * 4 + 4 * 8) * 4
    assert report["resident_increment_bytes"] == expected_packed


def test_module_pack_keeps_bias_fp32_and_quantizes_only_weights(monkeypatch) -> None:
    checked = []
    monkeypatch.setattr(
        int8_mlp,
        "_require_fp32_cuda",
        lambda value, *, name: checked.append((name, value.dtype)) or value,
    )
    module = SimpleNamespace(
        weight=torch.tensor(
            [[-1.0, -0.5, 0.5, 1.0], [0.0, 0.25, 0.5, 0.75]],
            dtype=torch.float32,
        ),
        bias=torch.zeros(2, dtype=torch.float32),
    )

    packed = int8_mlp.Int8PackedLinear.from_module(module)

    assert packed.weight.dtype == torch.int8
    assert packed.scale.dtype == torch.float32
    assert packed.bias is module.bias
    assert checked == [
        ("source weight", torch.float32),
        ("source bias", torch.float32),
    ]
