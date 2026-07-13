from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.inference.optimized.scout import q1_attention


def _cpu_q1(dtype: torch.dtype = torch.float16):
    query = torch.zeros((1, 2, 1, 4), dtype=dtype)
    key = torch.zeros((1, 2, 3, 4), dtype=dtype)
    value = torch.zeros((1, 2, 3, 4), dtype=dtype)
    return query, key, value


def test_native_extension_import_is_cold_and_preload_is_singleton(monkeypatch) -> None:
    calls = []
    extension = object()

    def fake_load_inline(**kwargs):
        calls.append(kwargs)
        return extension

    monkeypatch.setattr(q1_attention, "_NATIVE_Q1_ATTENTION", None)
    monkeypatch.setattr(q1_attention, "_load_inline", fake_load_inline)

    assert q1_attention.preload_native_q1_attention() is extension
    assert q1_attention.preload_native_q1_attention() is extension
    assert len(calls) == 1
    assert calls[0]["name"] == "mapperatorinator_native_prefix_q1_attention_scout"
    assert calls[0]["functions"] == ["q1_attention", "q1_rope_cache_attention"]
    source = calls[0]["cuda_sources"]
    assert "__half2float" in source
    assert "__float2half_rn" in source
    assert "float score = 0.0f" in source
    assert "float numerator = 0.0f" in source


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_cpu_q1_is_rejected_before_extension_build(monkeypatch, dtype) -> None:
    query, key, value = _cpu_q1(dtype)
    monkeypatch.setattr(
        q1_attention,
        "_load_native_q1_attention",
        lambda: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    with pytest.raises(RuntimeError, match="CUDA tensor"):
        q1_attention.native_q1_attention(query, key, value, None)


def test_q1_wrapper_flattens_mask_to_fp32(monkeypatch) -> None:
    query, key, value = _cpu_q1()
    mask = torch.zeros((1, 1, 1, 3), dtype=torch.float16)
    calls = []
    extension = SimpleNamespace(
        q1_attention=lambda *args: calls.append(args) or torch.ones_like(query)
    )
    monkeypatch.setattr(q1_attention, "_validate_q1", lambda *args: (query.dtype, 2, 3, 4))
    monkeypatch.setattr(q1_attention, "_load_native_q1_attention", lambda: extension)

    output = q1_attention.native_q1_attention(query, key, value, mask)

    assert output.dtype == query.dtype
    assert calls[0][-1].shape == (3,)
    assert calls[0][-1].dtype == torch.float32


def test_rope_cache_validation_rejects_bad_position_before_build(monkeypatch) -> None:
    qkv = torch.zeros((1, 1, 3, 2, 4), dtype=torch.float16)
    keys = torch.zeros((1, 2, 8, 4), dtype=torch.float16)
    values = torch.zeros_like(keys)
    cos = torch.zeros((1, 1, 4), dtype=torch.float16)
    sin = torch.zeros_like(cos)
    monkeypatch.setattr(q1_attention, "_require_supported_tensor", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="scalar int64"):
        q1_attention._validate_rope_cache_q1(
            qkv,
            keys,
            values,
            cos,
            sin,
            torch.tensor([1], dtype=torch.int32),
            4,
        )
