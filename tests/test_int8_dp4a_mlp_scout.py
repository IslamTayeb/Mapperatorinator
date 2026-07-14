from __future__ import annotations

import pytest
import torch

from osuT5.osuT5.inference.optimized.scout import int8_dp4a_mlp
from osuT5.osuT5.inference.optimized.scout.int8_mlp import (
    Int8MlpPack,
    Int8PackedLinear,
)


def _linear(rows: int, columns: int) -> Int8PackedLinear:
    return Int8PackedLinear(
        weight=torch.ones((rows, columns), dtype=torch.int8),
        scale=torch.ones(rows, dtype=torch.float32),
        bias=torch.zeros(rows, dtype=torch.float32),
        source_weight_bytes=rows * columns * 4,
    )


def _pack() -> Int8MlpPack:
    return Int8MlpPack(fc1=_linear(3072, 768), fc2=_linear(768, 3072))


def test_extension_is_cold_singleton_and_sm75_only(monkeypatch) -> None:
    calls = []
    extension = object()

    def fake_load_inline(**kwargs):
        calls.append(kwargs)
        return extension

    monkeypatch.setattr(int8_dp4a_mlp, "_DP4A_MLP_EXTENSION", None)
    monkeypatch.setattr(int8_dp4a_mlp, "_load_inline", fake_load_inline)

    assert int8_dp4a_mlp.preload_dp4a_mlp_extension() is extension
    assert int8_dp4a_mlp.preload_dp4a_mlp_extension() is extension
    assert len(calls) == 1
    assert calls[0]["name"] == "mapperatorinator_sm75_dp4a_mlp_scout_v1"
    assert "functions" not in calls[0]
    assert "-gencode=arch=compute_75,code=sm_75" in calls[0]["extra_cuda_cflags"]


def test_cuda_source_is_two_stage_signed_dp4a_with_fp32_edges() -> None:
    source = int8_dp4a_mlp._CUDA_SOURCE

    assert "rmsnorm_quant_dp4a_fc1_gelu_kernel" in source
    assert "quant_dp4a_fc2_residual_kernel" in source
    assert source.count("__dp4a") == 2
    assert "static_assert(HIDDEN_DIM % 4 == 0" in source
    assert "static_assert(INTERMEDIATE_DIM % 4 == 0" in source
    assert "int accumulator = 0" in source
    assert "int8_t quantized[HIDDEN_DIM]" in source
    assert "int8_t quantized[INTERMEDIATE_DIM]" in source
    assert "const float* __restrict__ residual" in source
    assert "output[out] = value + residual[out]" in source
    assert "float* __restrict__ output" in source
    assert "wmma" not in source.lower()
    assert "cooperative_groups" not in source


def test_cpu_inputs_fail_before_extension_build(monkeypatch) -> None:
    monkeypatch.setattr(
        int8_dp4a_mlp,
        "_load_dp4a_mlp_extension",
        lambda: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    with pytest.raises(RuntimeError, match="CUDA tensor"):
        int8_dp4a_mlp.dp4a_mlp_residual(
            torch.zeros((1, 1, 768), dtype=torch.float32),
            torch.ones(768, dtype=torch.float32),
            _pack(),
            eps=1e-5,
        )


@pytest.mark.parametrize(
    ("warps", "ctas", "message"),
    ((2, 32, "active_warps"), (8, 0, "persistent_ctas"), (8, 69, "persistent_ctas")),
)
def test_launch_shape_fails_loudly_before_extension(
    monkeypatch, warps: int, ctas: int, message: str
) -> None:
    hidden = torch.empty((1, 1, 768), dtype=torch.float32)
    norm = torch.ones(768, dtype=torch.float32)
    monkeypatch.setattr(
        int8_dp4a_mlp,
        "_require_fp32_cuda",
        lambda value, *, name: value,
    )
    monkeypatch.setattr(int8_dp4a_mlp, "_validate_linear", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match=message):
        int8_dp4a_mlp._validate_inputs(
            hidden,
            norm,
            _pack(),
            active_warps=warps,
            persistent_ctas=ctas,
        )


def test_fixed_shapes_are_enforced(monkeypatch) -> None:
    hidden = torch.empty((1, 1, 768), dtype=torch.float32)
    norm = torch.ones(768, dtype=torch.float32)
    pack = Int8MlpPack(fc1=_linear(3071, 768), fc2=_linear(768, 3072))
    monkeypatch.setattr(
        int8_dp4a_mlp,
        "_require_fp32_cuda",
        lambda value, *, name: value,
    )

    with pytest.raises(ValueError, match="fc1 weight must have shape"):
        int8_dp4a_mlp._validate_inputs(
            hidden,
            norm,
            pack,
            active_warps=8,
            persistent_ctas=68,
        )


def test_pack_validation_rejects_minus_128_and_bad_scales(monkeypatch) -> None:
    pack = _pack()
    monkeypatch.setattr(int8_dp4a_mlp, "_validate_linear", lambda *args, **kwargs: None)
    pack.fc1.weight[0, 0] = -128
    with pytest.raises(ValueError, match="must not contain -128"):
        int8_dp4a_mlp.validate_dp4a_mlp_pack(pack)

    pack = _pack()
    pack.fc2.scale[0] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        int8_dp4a_mlp.validate_dp4a_mlp_pack(pack)


def test_wrapper_argument_order_and_output_contract(monkeypatch) -> None:
    calls = []
    hidden = torch.zeros((1, 1, 768), dtype=torch.float32)
    norm = torch.ones(768, dtype=torch.float32)
    pack = _pack()
    values = (
        hidden.clone(),
        torch.zeros((1, 1, 3072), dtype=torch.float32),
        torch.zeros(768, dtype=torch.int8),
        torch.ones(1, dtype=torch.float32),
        torch.zeros(3072, dtype=torch.int8),
        torch.ones(1, dtype=torch.float32),
    )

    class Extension:
        @staticmethod
        def dp4a_mlp_residual(*args):
            calls.append(args)
            return values

    monkeypatch.setattr(
        int8_dp4a_mlp,
        "_validate_inputs",
        lambda *args, **kwargs: (hidden, norm, pack),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    monkeypatch.setattr(int8_dp4a_mlp, "_DP4A_MLP_EXTENSION", Extension())

    observed = int8_dp4a_mlp.dp4a_mlp_residual_with_state(
        hidden,
        norm,
        pack,
        eps=1e-5,
        active_warps=4,
        persistent_ctas=48,
    )

    assert observed == values
    assert calls == [
        (
            hidden,
            norm,
            pack.fc1.weight,
            pack.fc1.scale,
            pack.fc1.bias,
            pack.fc2.weight,
            pack.fc2.scale,
            pack.fc2.bias,
            1e-5,
            4,
            48,
        )
    ]
