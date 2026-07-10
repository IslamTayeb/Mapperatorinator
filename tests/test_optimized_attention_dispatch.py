from __future__ import annotations

from types import SimpleNamespace

import torch

from osuT5.osuT5.inference.optimized.kernels import dispatch
from osuT5.osuT5.inference.optimized.kernels.dispatch import (
    q1_rope_cache_self_attention_forward,
    sdpa_q1_attention_forward,
)


def test_q1_bmm_dispatch_preserves_exact_calculation_order():
    generator = torch.Generator().manual_seed(12345)
    query = torch.randn((1, 4, 1, 8), generator=generator)
    key = torch.randn((1, 4, 16, 8), generator=generator)
    value = torch.randn((1, 4, 16, 8), generator=generator)
    module = SimpleNamespace(is_cross_attention=True, training=False)

    result = sdpa_q1_attention_forward(
        module=module,
        query=query,
        key=key,
        value=value,
        bs=1,
        dim=32,
        attention_mask=None,
        q1_bmm_cross_attention=True,
        native_q1_self_attention=False,
        native_q1_attention=None,
    )

    q = query.reshape(4, 1, 8)
    k = key.reshape(4, -1, 8)
    v = value.reshape(4, -1, 8)
    scores = torch.bmm(q, k.transpose(1, 2)) * (8 ** -0.5)
    expected = torch.bmm(torch.softmax(scores, dim=-1), v).view(1, 4, 1, 8)
    expected = expected.transpose(1, 2).contiguous().view(1, -1, 32)

    assert result is not None
    assert torch.equal(result[0], expected)


def test_dispatch_returns_none_outside_exact_q1_contract():
    module = SimpleNamespace(is_cross_attention=True, training=False)
    query = torch.randn((2, 4, 1, 8))
    key = torch.randn((2, 4, 16, 8))
    value = torch.randn((2, 4, 16, 8))

    assert sdpa_q1_attention_forward(
        module=module,
        query=query,
        key=key,
        value=value,
        bs=2,
        dim=32,
        attention_mask=None,
        q1_bmm_cross_attention=True,
        native_q1_self_attention=False,
        native_q1_attention=None,
    ) is None


def test_native_self_dispatch_preserves_wrapper_and_layout():
    generator = torch.Generator().manual_seed(23456)
    query = torch.randn((1, 4, 1, 8), generator=generator)
    key = torch.randn((1, 4, 16, 8), generator=generator)
    value = torch.randn((1, 4, 16, 8), generator=generator)
    module = SimpleNamespace(is_cross_attention=False, training=False)
    calls = []

    def native(q, k, v, mask):
        calls.append((q, k, v, mask))
        return torch.randn((1, 4, 1, 8), generator=generator)

    result = sdpa_q1_attention_forward(
        module=module,
        query=query,
        key=key,
        value=value,
        bs=1,
        dim=32,
        attention_mask=None,
        q1_bmm_cross_attention=False,
        native_q1_self_attention=True,
        native_q1_attention=native,
    )

    assert calls == [(query, key, value, None)]
    assert result is not None
    assert result[0].shape == (1, 1, 32)


def test_fused_rope_cache_dispatch_preserves_sequence_aliases_and_mask(monkeypatch):
    events = []
    hidden_states = torch.arange(8, dtype=torch.float32).view(1, 1, 8)
    position_ids = torch.tensor([[6]])
    cache_position = torch.tensor([6])
    attention_mask = torch.zeros((1, 1, 1, 12), dtype=torch.float32)
    sliding_window_mask = torch.ones((1, 1, 1, 12), dtype=torch.float32)

    class FakeStaticCache:
        def __init__(self):
            self.layers = [
                SimpleNamespace(
                    is_initialized=True,
                    keys=torch.zeros((1, 2, 12, 4)),
                    values=torch.ones((1, 2, 12, 4)),
                )
            ]

    class Projection:
        def __call__(self, inputs):
            events.append(("projection", inputs))
            return torch.arange(24, dtype=torch.float32).view(1, 1, 24)

    class Rotary:
        def __call__(self, qkv, *, position_ids):
            events.append(("rotary", qkv, position_ids))
            return torch.full((1, 4), 2.0), torch.full((1, 4), 3.0)

    module = SimpleNamespace(
        training=False,
        Wqkv=Projection(),
        rotary_emb=Rotary(),
        num_heads=2,
        head_dim=4,
        layer_idx=0,
        local_attention=(128, 128),
        all_head_size=8,
    )
    cache = FakeStaticCache()
    kernel_output = torch.arange(8, dtype=torch.float32).view(1, 2, 1, 4)

    def native(qkv, keys, values, cos, sin, position, mask, prefix_length):
        events.append(
            (
                "native",
                qkv,
                keys,
                values,
                cos,
                sin,
                position,
                mask,
                prefix_length,
            )
        )
        return kernel_output

    monkeypatch.setattr(dispatch, "StaticCache", FakeStaticCache)
    monkeypatch.setattr(
        dispatch,
        "active_prefix_self_attention_length",
        lambda: 7,
    )

    result = q1_rope_cache_self_attention_forward(
        module=module,
        hidden_states=hidden_states,
        bs=1,
        is_varlen=False,
        past_key_value=cache,
        cache_position=cache_position,
        position_ids=position_ids,
        profile_ranges=False,
        range_prefix="decoder.layer0.self",
        attention_mask=attention_mask,
        sliding_window_mask=sliding_window_mask,
        native_q1_rope_cache_attention=native,
    )

    assert [event[0] for event in events] == ["projection", "rotary", "native"]
    qkv = events[1][1]
    native_event = events[2]
    assert events[0][1] is hidden_states
    assert native_event[1] is qkv
    assert native_event[2] is cache.layers[0].keys
    assert native_event[3] is cache.layers[0].values
    assert native_event[6] is cache_position
    assert native_event[7].shape[-1] == 7
    assert torch.equal(native_event[7], sliding_window_mask[..., :7])
    assert native_event[8] == 7
    assert result is not None
    expected = kernel_output.transpose(1, 2).contiguous().view(1, 1, 8)
    assert torch.equal(result[0], expected)


def test_fused_rope_cache_dispatch_rejects_without_side_effects(monkeypatch):
    calls = []
    monkeypatch.setattr(
        dispatch,
        "active_prefix_self_attention_length",
        lambda: None,
    )
    module = SimpleNamespace(training=False)

    result = q1_rope_cache_self_attention_forward(
        module=module,
        hidden_states=torch.zeros((1, 1, 8), dtype=torch.float32),
        bs=1,
        is_varlen=False,
        past_key_value=object(),
        cache_position=torch.tensor([0]),
        position_ids=torch.tensor([[0]]),
        profile_ranges=False,
        range_prefix="decoder.layer0.self",
        attention_mask=None,
        sliding_window_mask=None,
        native_q1_rope_cache_attention=lambda *args: calls.append(args),
    )

    assert result is None
    assert calls == []
