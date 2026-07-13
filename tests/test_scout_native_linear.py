import pytest
import torch

from osuT5.osuT5.inference.optimized.scout import native_linear


def _cpu_linear_inputs(dtype: torch.dtype = torch.float16):
    input = torch.zeros((1, 1, 4), dtype=dtype)
    norm_weight = torch.ones(4, dtype=dtype)
    projection = torch.ones((6, 4), dtype=dtype)
    square_weight = torch.ones((4, 4), dtype=dtype)
    return input, norm_weight, projection, square_weight


def test_preload_is_explicit_and_singleton(monkeypatch) -> None:
    calls = []
    extension = object()

    def fake_load_inline(**kwargs):
        calls.append(kwargs)
        return extension

    monkeypatch.setattr(native_linear, "_SCOUT_NATIVE_LINEAR", None)
    monkeypatch.setattr(native_linear, "_load_inline", fake_load_inline)

    assert native_linear._SCOUT_NATIVE_LINEAR is None
    assert native_linear.preload_scout_native_linear() is extension
    assert native_linear.preload_scout_native_linear() is extension
    assert len(calls) == 1
    assert calls[0]["functions"] == [
        "scout_one_token_rmsnorm_linear",
        "scout_one_token_linear_residual",
    ]


def test_cuda_source_uses_fp16_dispatch_and_fp32_accumulation() -> None:
    source = native_linear._CUDA_SOURCE

    assert source.count("AT_DISPATCH_FLOATING_TYPES_AND_HALF") == 2
    assert "std::isfinite(eps)" in source
    assert "float local_sq = 0.0f;" in source
    assert "float local_dot = 0.0f;" in source
    assert "float sum = 0.0f;" in source
    assert "static_cast<float>(input[idx])" in source
    assert "static_cast<float>(norm_weight[idx])" in source
    assert "static_cast<float>(row[idx])" in source
    assert "static_cast<float>(residual[out_idx])" in source
    assert source.count("static_cast<scalar_t>(result)") == 2
    assert "input.options()" in source


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_valid_cpu_contracts_fail_before_extension_build(monkeypatch, dtype) -> None:
    input, norm_weight, projection, square_weight = _cpu_linear_inputs(dtype)

    def unexpected_build():
        raise AssertionError("validation must run before extension compilation")

    monkeypatch.setattr(native_linear, "_load_scout_native_linear", unexpected_build)

    with pytest.raises(ValueError, match="CUDA tensors"):
        native_linear.scout_one_token_rmsnorm_linear(
            input,
            norm_weight,
            projection,
            eps=1e-6,
            outputs_per_block=4,
        )
    with pytest.raises(ValueError, match="CUDA tensors"):
        native_linear.scout_one_token_linear_residual(
            input,
            input,
            square_weight,
            outputs_per_block=4,
        )


def test_rmsnorm_linear_validation_is_strict() -> None:
    input, norm_weight, projection, _ = _cpu_linear_inputs()

    with pytest.raises(ValueError, match="output_dim elements"):
        native_linear._validate_rmsnorm_linear(
            input, norm_weight, projection, torch.zeros(5), 1e-6, 4
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        native_linear._validate_rmsnorm_linear(
            input, norm_weight, projection, None, float("nan"), 4
        )
    with pytest.raises(ValueError, match="2, 4, or 8"):
        native_linear._validate_rmsnorm_linear(
            input, norm_weight, projection, None, 1e-6, 16
        )


def test_linear_residual_validation_is_strict() -> None:
    input, _, _, square_weight = _cpu_linear_inputs()

    with pytest.raises(ValueError, match="residual shape"):
        native_linear._validate_linear_residual(
            input, input[..., :3], square_weight, None, 4
        )
    with pytest.raises(ValueError, match="hidden_dim elements"):
        native_linear._validate_linear_residual(
            input, input, square_weight, torch.zeros(3), 4
        )
    with pytest.raises(TypeError, match="weight dtype"):
        native_linear._validate_linear_residual(
            input, input, square_weight.float(), None, 4
        )
