from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.inference.optimized.scout import split_kv_q1


def _cpu_inputs(dtype: torch.dtype = torch.float32):
    qkv = torch.zeros((1, 1, 3, 6, 64), dtype=dtype)
    keys = torch.zeros((1, 6, 128, 64), dtype=dtype)
    values = torch.zeros_like(keys)
    cos = torch.ones((1, 1, 64), dtype=dtype)
    sin = torch.zeros_like(cos)
    position = torch.tensor([127], dtype=torch.long)
    return qkv, keys, values, cos, sin, position


def test_import_is_cold_and_extension_preload_is_singleton(monkeypatch) -> None:
    calls = []
    extension = object()
    monkeypatch.setattr(split_kv_q1, "_SPLIT_KV_EXTENSION", None)
    monkeypatch.setattr(
        split_kv_q1,
        "_load_inline",
        lambda **kwargs: calls.append(kwargs) or extension,
    )

    assert split_kv_q1.preload_split_kv_q1() is extension
    assert split_kv_q1.preload_split_kv_q1() is extension
    assert len(calls) == 1
    assert calls[0]["name"] == "mapperatorinator_sm75_split_kv_q1_scout"
    assert calls[0]["functions"] == ["split_kv_q1_rope_cache_attention"]
    assert "code=sm_75" in calls[0]["extra_cuda_cflags"][-1]


def test_source_has_separate_cache_owner_partial_and_online_merge() -> None:
    source = split_kv_q1._CUDA_SOURCE

    assert "prepare_rope_cache_kernel" in source
    assert "split_partial_kernel" in source
    assert "merge_partials_kernel" in source
    assert "dim3(heads, split_count)" in source
    assert "partial_max[index] - global_max" in source
    assert "partial_numer[index * head_dim + dim]" in source
    assert "properties.major == 7 && properties.minor == 5" in source
    assert "cache_position[0]" in source
    assert "float numerator = 0.0f" in source


@pytest.mark.parametrize("split_count", [0, 3, 16, True])
def test_unsupported_splits_fail_before_extension_build(monkeypatch, split_count) -> None:
    inputs = _cpu_inputs()
    monkeypatch.setattr(split_kv_q1, "_validate_rope_cache_inputs", lambda *args: None)
    monkeypatch.setattr(
        split_kv_q1,
        "_load_split_kv_extension",
        lambda: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    with pytest.raises(ValueError, match="split_count"):
        split_kv_q1.split_kv_q1_rope_cache_attention(
            *inputs, None, 128, split_count
        )


def test_non_sm75_fails_before_extension_build(monkeypatch) -> None:
    inputs = _cpu_inputs()
    monkeypatch.setattr(split_kv_q1, "_validate_rope_cache_inputs", lambda *args: None)
    monkeypatch.setattr(split_kv_q1, "_native_mask", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    monkeypatch.setattr(
        split_kv_q1,
        "_load_split_kv_extension",
        lambda: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    with pytest.raises(RuntimeError, match="requires SM75"):
        split_kv_q1.split_kv_q1_rope_cache_attention(*inputs, None, 128, 2)


def test_wrapper_passes_validated_mask_and_split(monkeypatch) -> None:
    inputs = _cpu_inputs()
    calls = []
    expected = torch.ones((1, 6, 1, 64))
    extension = SimpleNamespace(
        split_kv_q1_rope_cache_attention=lambda *args: calls.append(args) or expected
    )
    mask = torch.zeros(128)
    monkeypatch.setattr(split_kv_q1, "_validate_rope_cache_inputs", lambda *args: None)
    monkeypatch.setattr(split_kv_q1, "_native_mask", lambda *args, **kwargs: mask)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    monkeypatch.setattr(split_kv_q1, "_load_split_kv_extension", lambda: extension)

    output = split_kv_q1.split_kv_q1_rope_cache_attention(*inputs, mask, 128, 8)

    assert output is expected
    assert calls[0][-3] is mask
    assert calls[0][-2:] == (128, 8)
