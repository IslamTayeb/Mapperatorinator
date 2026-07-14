from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.inference.optimized.scout import cross_attention


def _cpu_inputs(*, kv_dtype: torch.dtype = torch.float32):
    query = torch.zeros((1, 12, 1, 64), dtype=torch.float32)
    keys = torch.zeros((1, 12, 1024, 64), dtype=kv_dtype)
    values = torch.zeros_like(keys)
    return query, keys, values


def test_import_is_cold_and_preload_is_singleton(monkeypatch) -> None:
    calls = []
    extension = object()

    def fake_load_inline(**kwargs):
        calls.append(kwargs)
        return extension

    monkeypatch.setattr(cross_attention, "_CROSS_ATTENTION_EXTENSION", None)
    monkeypatch.setattr(cross_attention, "_load_inline", fake_load_inline)

    assert cross_attention.preload_cross_attention_scout() is extension
    assert cross_attention.preload_cross_attention_scout() is extension
    assert len(calls) == 1
    assert calls[0]["name"] == "mapperatorinator_cross_attention_split_kv_scout"
    assert calls[0]["functions"] == [
        "cross_attention_one_pass",
        "cross_attention_split",
    ]
    assert calls[0]["cpp_sources"] == cross_attention._CPP_SOURCE


def test_cpp_source_declares_every_generated_binding() -> None:
    source = cross_attention._CPP_SOURCE

    assert "torch::Tensor cross_attention_one_pass(" in source
    assert "torch::Tensor cross_attention_split(" in source
    assert "int64_t splits" in source


def test_source_has_genuine_split_grid_and_online_softmax_merge() -> None:
    source = cross_attention._CUDA_SOURCE

    assert "kHeads * splits" in source
    assert "split_online_softmax_partials" in source
    assert "merge_online_softmax_partials" in source
    assert "partial_maxima" in source
    assert "partial_denominators" in source
    assert "expf(partial_maxima[index] - global_maximum)" in source
    assert "one_pass_online_softmax" in source
    assert "float* __restrict__ output" in source
    assert "const kv_t* __restrict__ keys" in source
    assert "const kv_t* __restrict__ values" in source
    assert "#include <c10/cuda/CUDAGuard.h>" in source
    assert "c10::cuda::CUDAGuard" in source


@pytest.mark.parametrize("kv_dtype", [torch.float32, torch.float16])
def test_cpu_inputs_fail_before_extension_build(monkeypatch, kv_dtype) -> None:
    query, keys, values = _cpu_inputs(kv_dtype=kv_dtype)
    monkeypatch.setattr(
        cross_attention,
        "_load_cross_attention_extension",
        lambda: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    with pytest.raises(RuntimeError, match="query must be a CUDA tensor"):
        cross_attention.cross_attention_split(query, keys, values, 8)


def test_split_and_shape_validation_fail_loudly(monkeypatch) -> None:
    query, keys, values = _cpu_inputs()
    monkeypatch.setattr(
        cross_attention,
        "_validate_inputs",
        lambda *args: None,
    )

    with pytest.raises(TypeError, match="integer"):
        cross_attention.cross_attention_split(query, keys, values, True)
    with pytest.raises(ValueError, match="2, 4, or 8"):
        cross_attention.cross_attention_split(query, keys, values, 1)

    monkeypatch.undo()
    bad_query = torch.zeros((1, 12, 2, 64))
    with pytest.raises(ValueError, match=r"\[1,12,1,64\]"):
        cross_attention.cross_attention_one_pass(bad_query, keys, values)


def test_wrapper_dispatches_fp16_kv_without_converting_fp32_output(monkeypatch) -> None:
    query, keys, values = _cpu_inputs(kv_dtype=torch.float16)
    expected = torch.ones_like(query)
    calls = []
    extension = SimpleNamespace(
        cross_attention_split=lambda *args: calls.append(args) or expected,
    )
    monkeypatch.setattr(cross_attention, "_validate_inputs", lambda *args: None)
    monkeypatch.setattr(
        cross_attention,
        "_load_cross_attention_extension",
        lambda: extension,
    )

    output = cross_attention.cross_attention_split(query, keys, values, 8)

    assert output is expected
    assert output.dtype == torch.float32
    assert calls == [(query, keys, values, 8)]


def test_static_validation_rejects_wrong_dtype_shape_and_storage_pair() -> None:
    query, keys, values = _cpu_inputs()
    with pytest.raises(RuntimeError, match="CUDA tensor"):
        cross_attention._validate_inputs(query, keys, values)

    with pytest.raises(TypeError, match="query must use FP32"):
        cross_attention._validate_inputs(query.half(), keys, values)
    with pytest.raises(TypeError, match="storage dtypes must match"):
        cross_attention._validate_inputs(query, keys.half(), values)
    with pytest.raises(ValueError, match=r"\[1,12,1,64\]"):
        cross_attention._validate_inputs(
            torch.zeros((1, 12, 2, 64)),
            keys,
            values,
        )
    with pytest.raises(ValueError, match=r"\[1,12,1024,64\]"):
        cross_attention._validate_inputs(
            query,
            torch.zeros((1, 12, 512, 64)),
            torch.zeros((1, 12, 512, 64)),
        )
