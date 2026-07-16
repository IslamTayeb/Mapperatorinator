from __future__ import annotations

import pytest

from osuT5.osuT5.inference.optimized.kernels.q1_attention import (
    flashdecode_split_count,
    native_q1_rope_cache_attention_flashdecode_variant,
)


def test_flashdecode_split_count_steps_up_on_long_prefixes() -> None:
    assert flashdecode_split_count(192) == 8
    assert flashdecode_split_count(447) == 8
    assert flashdecode_split_count(448) == 16
    assert flashdecode_split_count(640) == 16
    assert flashdecode_split_count(703) == 16
    assert flashdecode_split_count(704) == 32
    assert flashdecode_split_count(832) == 32


def test_flashdecode_split_count_rejects_invalid_prefix() -> None:
    with pytest.raises(ValueError, match="positive int"):
        flashdecode_split_count(0)
    with pytest.raises(ValueError, match="positive int"):
        flashdecode_split_count(True)  # type: ignore[arg-type]


def test_flashdecode_variant_requires_split_kv_eligibility(monkeypatch) -> None:
    from osuT5.osuT5.inference.optimized.kernels import q1_attention

    class _FakeQkv:
        is_cuda = True
        dtype = object()
        device = type("D", (), {"type": "cuda"})()

    monkeypatch.setattr(
        q1_attention,
        "native_q1_rope_cache_attention_variant",
        lambda _qkv, prefix: (
            "split_kv_8" if prefix in q1_attention._SPLIT_KV_Q1_PREFIXES else "accepted"
        ),
    )
    qkv = _FakeQkv()
    assert (
        native_q1_rope_cache_attention_flashdecode_variant(qkv, 640)
        == "split_kv_flashdecode"
    )
    assert native_q1_rope_cache_attention_flashdecode_variant(qkv, 64) == "accepted"
