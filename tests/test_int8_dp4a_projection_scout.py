from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.inference.optimized.scout import int8_dp4a_projection as scout
from osuT5.osuT5.inference.optimized.scout.int8_mlp import Int8PackedLinear


def _pack(*, outputs: int = 768) -> Int8PackedLinear:
    return Int8PackedLinear(
        weight=torch.zeros((outputs, 768), dtype=torch.int8),
        scale=torch.ones(outputs, dtype=torch.float32),
        bias=torch.zeros(outputs, dtype=torch.float32),
        source_weight_bytes=outputs * 768 * 4,
    )


def test_import_is_cold_and_preload_is_singleton(monkeypatch) -> None:
    calls = []
    extension = object()

    def fake_load_inline(**kwargs):
        calls.append(kwargs)
        return extension

    monkeypatch.setattr(scout, "_DP4A_PROJECTION_EXTENSION", None)
    monkeypatch.setattr(scout, "_load_inline", fake_load_inline)

    assert scout.preload_dp4a_projection_extension() is extension
    assert scout.preload_dp4a_projection_extension() is extension
    assert len(calls) == 1
    assert calls[0]["name"] == "mapperatorinator_sm75_dp4a_projection_scout_v1"
    assert "-gencode=arch=compute_75,code=sm_75" in calls[0]["extra_cuda_cflags"]


def test_cuda_source_is_true_signed_dp4a_and_one_persistent_kernel() -> None:
    source = scout._CUDA_SOURCE

    assert "__dp4a" in source
    assert "int accumulator = 0" in source
    assert "int8_t shared_quantized[HIDDEN_DIM]" in source
    assert "__shared__ __align__(4)" in source
    assert "persistent_rmsnorm_signed_dp4a_kernel" in source
    assert "gridDim.x * ACTIVE_WARPS" in source
    assert "value < -127 ? -127" in source
    assert "rmsnorm_dynamic_quantize_kernel<<<" not in source
    assert "signed_dp4a_linear_kernel<<<" not in source
    assert "#include <vector>" in scout._CPP_SOURCE


def test_cpu_input_fails_before_extension_build(monkeypatch) -> None:
    monkeypatch.setattr(
        scout,
        "_load_dp4a_projection_extension",
        lambda: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    with pytest.raises(RuntimeError, match="CUDA tensor"):
        scout.dp4a_rmsnorm_linear(
            torch.zeros((1, 1, 768), dtype=torch.float32),
            torch.ones(768, dtype=torch.float32),
            _pack(),
            eps=1e-5,
        )


@pytest.mark.parametrize("eps", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_eps_fails_before_device_or_build(eps, monkeypatch) -> None:
    monkeypatch.setattr(
        scout,
        "_load_dp4a_projection_extension",
        lambda: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    with pytest.raises(ValueError, match="eps must be finite and positive"):
        scout.dp4a_rmsnorm_linear(
            torch.zeros((1, 1, 768), dtype=torch.float32),
            torch.ones(768, dtype=torch.float32),
            _pack(),
            eps=eps,
        )


def test_pack_content_validation_fails_loudly() -> None:
    valid = _pack(outputs=2304)
    report = scout.validate_dp4a_pack(valid)
    assert report == {
        "scale_min": 1.0,
        "scale_max": 1.0,
        "weight_contains_minus_128": False,
    }

    minus_128 = _pack()
    minus_128.weight[0, 0] = -128
    with pytest.raises(ValueError, match="must not contain -128"):
        scout.validate_dp4a_pack(minus_128)

    bad_scale = _pack()
    bad_scale.scale[0] = 0
    with pytest.raises(ValueError, match="finite and positive"):
        scout.validate_dp4a_pack(bad_scale)

    bad_bias = _pack()
    bad_bias.bias[0] = float("nan")
    with pytest.raises(ValueError, match="bias must be finite"):
        scout.validate_dp4a_pack(bad_bias)


def test_launch_shape_and_supported_projection_shapes_fail_loudly(monkeypatch) -> None:
    input = torch.zeros((1, 1, 768), dtype=torch.float32)
    norm = torch.ones(768, dtype=torch.float32)
    pack = _pack()
    monkeypatch.setattr(scout, "_require_fp32_cuda", lambda value, **_: value)

    with pytest.raises(ValueError, match="active_warps must be 4 or 8"):
        scout._validate_inputs(input, norm, pack, active_warps=2, persistent_ctas=1)
    with pytest.raises(ValueError, match="SM75 output coverage"):
        scout._validate_inputs(input, norm, pack, active_warps=8, persistent_ctas=69)

    wrong = _pack(outputs=769)
    with pytest.raises(ValueError, match="cross-Q.*self-QKV"):
        scout._validate_inputs(input, norm, wrong, active_warps=8, persistent_ctas=1)


def test_wrapper_preserves_owned_output_state_contract(monkeypatch) -> None:
    input = torch.zeros((1, 1, 768), dtype=torch.float32)
    norm = torch.ones(768, dtype=torch.float32)
    pack = _pack(outputs=2304)
    output = torch.ones((1, 1, 2304), dtype=torch.float32)
    quantized = torch.ones(768, dtype=torch.int8)
    scale = torch.tensor([0.25], dtype=torch.float32)
    calls = []
    extension = SimpleNamespace(
        dp4a_rmsnorm_linear=lambda *args: calls.append(args)
        or [output, quantized, scale]
    )
    monkeypatch.setattr(
        scout,
        "_validate_inputs",
        lambda *args, **kwargs: (input, norm, pack),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    monkeypatch.setattr(scout, "_load_dp4a_projection_extension", lambda: extension)

    actual = scout.dp4a_rmsnorm_linear_with_state(
        input,
        norm,
        pack,
        eps=1e-5,
        active_warps=8,
        persistent_ctas=32,
    )

    assert actual == (output, quantized, scale)
    assert calls[0][-2:] == (8, 32)


def test_default_optimized_package_does_not_import_dp4a_scout() -> None:
    source = (
        scout.__file__.replace("int8_dp4a_projection.py", "../__init__.py")
    )
    with open(source, encoding="utf-8") as handle:
        assert "int8_dp4a_projection" not in handle.read()
