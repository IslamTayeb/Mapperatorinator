from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.inference.optimized.scout import cutlass_w8a8_projection as scout
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

    monkeypatch.setattr(scout, "_W8A8_PROJECTION_EXTENSION", None)
    monkeypatch.setattr(scout, "_load_inline", fake_load_inline)

    assert scout.preload_w8a8_projection_extension() is extension
    assert scout.preload_w8a8_projection_extension() is extension
    assert len(calls) == 1
    assert calls[0]["name"] == "mapperatorinator_sm75_w8a8_projection_scout_v1"
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
        "_load_w8a8_projection_extension",
        lambda: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    with pytest.raises(RuntimeError, match="CUDA tensor"):
        scout.w8a8_rmsnorm_linear(
            torch.zeros((1, 1, 768), dtype=torch.float32),
            torch.ones(768, dtype=torch.float32),
            _pack(),
            eps=1e-5,
        )


@pytest.mark.parametrize("eps", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_eps_fails_before_device_or_build(eps, monkeypatch) -> None:
    monkeypatch.setattr(
        scout,
        "_load_w8a8_projection_extension",
        lambda: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    with pytest.raises(ValueError, match="eps must be finite and positive"):
        scout.w8a8_rmsnorm_linear(
            torch.zeros((1, 1, 768), dtype=torch.float32),
            torch.ones(768, dtype=torch.float32),
            _pack(),
            eps=eps,
        )


def test_pack_content_validation_fails_loudly() -> None:
    valid = _pack(outputs=2304)
    report = scout.validate_w8a8_pack(valid)
    assert report == {
        "scale_min": 1.0,
        "scale_max": 1.0,
        "weight_contains_minus_128": False,
    }

    minus_128 = _pack()
    minus_128.weight[0, 0] = -128
    with pytest.raises(ValueError, match="must not contain -128"):
        scout.validate_w8a8_pack(minus_128)

    bad_scale = _pack()
    bad_scale.scale[0] = 0
    with pytest.raises(ValueError, match="finite and positive"):
        scout.validate_w8a8_pack(bad_scale)

    bad_bias = _pack()
    bad_bias.bias[0] = float("nan")
    with pytest.raises(ValueError, match="bias must be finite"):
        scout.validate_w8a8_pack(bad_bias)


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
        # CUDA extension keeps the signed-DP4A entry symbol for the fallback.
        dp4a_rmsnorm_linear=lambda *args: calls.append(args)
        or [output, quantized, scale]
    )
    monkeypatch.setattr(
        scout,
        "_validate_inputs",
        lambda *args, **kwargs: (input, norm, pack),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    monkeypatch.setattr(scout, "_load_w8a8_projection_extension", lambda: extension)

    actual = scout.w8a8_rmsnorm_linear_with_state(
        input,
        norm,
        pack,
        eps=1e-5,
        active_warps=8,
        persistent_ctas=32,
    )

    assert actual == (output, quantized, scale)
    assert calls[0][-2:] == (8, 32)


def test_default_optimized_package_does_not_import_w8a8_scout() -> None:
    source = (
        scout.__file__.replace("cutlass_w8a8_projection.py", "../__init__.py")
    )
    with open(source, encoding="utf-8") as handle:
        assert "cutlass_w8a8_projection" not in handle.read()


def test_cutlass_probe_reports_missing_tree(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MAPPERATORINATOR_CUTLASS_HOME", raising=False)
    probe = scout.probe_cutlass_availability(cutlass_home=str(tmp_path / "missing"))
    assert probe["cutlass_available"] is False
    assert probe["backend"] == "sm75_signed_dp4a_w8a8_fallback"
    assert "MAPPERATORINATOR_CUTLASS_HOME" in probe["revisit_condition"]


def test_cutlass_probe_detects_marker_but_select_keeps_fallback(tmp_path) -> None:
    root = tmp_path / "cutlass"
    marker = root / "include/cutlass/gemm/device/gemm.h"
    marker.parent.mkdir(parents=True)
    marker.write_text("// stub\n")
    probe = scout.probe_cutlass_availability(cutlass_home=str(root))
    assert probe["cutlass_available"] is True
    selected = scout.select_w8a8_backend(cutlass_home=str(root))
    assert selected["cutlass_available"] is True
    assert selected["cutlass_gemm_wired"] is False
    assert selected["backend"] == "sm75_signed_dp4a_w8a8_fallback"
    assert "not wired" in selected["note"]
