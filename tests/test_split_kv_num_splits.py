"""Unit coverage for split-KV num_splits env override."""

from __future__ import annotations

import pytest

from osuT5.osuT5.inference.optimized.kernels import q1_attention


def test_default_split_count_is_eight(monkeypatch) -> None:
    monkeypatch.delenv(q1_attention._SPLIT_KV_SPLITS_ENV, raising=False)
    assert q1_attention.configured_split_kv_q1_splits() == 8


@pytest.mark.parametrize("value", [4, 6, 12, 16])
def test_env_override_accepts_sweep_values(monkeypatch, value: int) -> None:
    monkeypatch.setenv(q1_attention._SPLIT_KV_SPLITS_ENV, str(value))
    assert q1_attention.configured_split_kv_q1_splits() == value


def test_env_override_rejects_unknown(monkeypatch) -> None:
    monkeypatch.setenv(q1_attention._SPLIT_KV_SPLITS_ENV, "10")
    with pytest.raises(ValueError, match="must be one of"):
        q1_attention.configured_split_kv_q1_splits()


def test_cuda_source_accepts_runtime_split_count() -> None:
    source = q1_attention.__file__
    text = open(source, encoding="utf-8").read()
    assert "int64_t split_count_arg" in text
    assert "mapperatorinator_q1_attention_splitkv_runtime" in text
    assert "split_count_arg == 4 || split_count_arg == 6" in text
