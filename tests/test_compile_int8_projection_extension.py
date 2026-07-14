from types import SimpleNamespace

import pytest

from utils import compile_int8_mlp_extension as compile_util


def test_cpu_compile_smoke_requires_all_projection_bindings(monkeypatch, tmp_path) -> None:
    module = tmp_path / "extension.so"
    module.write_bytes(b"compiled")
    extension = SimpleNamespace(
        __file__=str(module),
        int8_weight_mlp_residual=lambda: None,
        int8_weight_rmsnorm_linear=lambda: None,
        int8_weight_linear_residual=lambda: None,
    )
    monkeypatch.setattr(compile_util.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(compile_util, "preload_int8_mlp_extension", lambda: extension)

    result = compile_util.compile_extension()

    assert result["pass"] is True
    assert result["exported_bindings"] == [
        "int8_weight_mlp_residual",
        "int8_weight_rmsnorm_linear",
        "int8_weight_linear_residual",
    ]


def test_cpu_compile_smoke_fails_on_missing_projection_binding(monkeypatch, tmp_path) -> None:
    module = tmp_path / "extension.so"
    module.write_bytes(b"compiled")
    extension = SimpleNamespace(
        __file__=str(module),
        int8_weight_mlp_residual=lambda: None,
        int8_weight_rmsnorm_linear=lambda: None,
    )
    monkeypatch.setattr(compile_util.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(compile_util, "preload_int8_mlp_extension", lambda: extension)

    with pytest.raises(RuntimeError, match="int8_weight_linear_residual"):
        compile_util.compile_extension()
