from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.inference.optimized.scout.native_prefix import (
    fp32_rms_norm,
    framework_prefix,
    framework_q1_attention,
    specialized_prefix_attention_context,
)


class _Linear:
    def __init__(self, weight: torch.Tensor):
        self.weight = weight
        self.bias = None


def _layer(dtype: torch.dtype):
    hidden = 4
    heads = 2
    head_dim = 2
    identity = torch.eye(hidden, dtype=dtype)
    self_attn = SimpleNamespace(
        training=False,
        layer_idx=0,
        num_heads=heads,
        head_dim=head_dim,
        all_head_size=hidden,
        Wqkv=_Linear(torch.cat((identity, identity, identity))),
        Wo=_Linear(identity.clone()),
        rotary_emb=lambda qkv, position_ids: (
            torch.ones((1, 1, head_dim), dtype=dtype),
            torch.zeros((1, 1, head_dim), dtype=dtype),
        ),
    )
    cross_attn = SimpleNamespace(
        training=False,
        layer_idx=0,
        num_heads=heads,
        head_dim=head_dim,
        all_head_size=hidden,
        Wq=_Linear(identity.clone()),
        Wo=_Linear(identity.clone()),
    )
    return SimpleNamespace(
        training=False,
        self_attn=self_attn,
        cross_attn=cross_attn,
        self_attn_layer_norm=torch.nn.RMSNorm(hidden, dtype=dtype),
        cross_attn_layer_norm=torch.nn.RMSNorm(hidden, dtype=dtype),
        final_layer_norm=torch.nn.RMSNorm(hidden, dtype=dtype),
        fc1=_Linear(identity.clone()),
        fc2=_Linear(identity.clone()),
        activation_fn=lambda value: value,
        activation_dropout=0.0,
        dropout=0.0,
    )


def _cache(dtype: torch.dtype):
    self_keys = torch.zeros((1, 2, 4, 2), dtype=dtype)
    self_values = torch.zeros_like(self_keys)
    cross_keys = torch.arange(12, dtype=dtype).reshape(1, 2, 3, 2) / 10
    cross_values = cross_keys + 1
    return SimpleNamespace(
        self_attention_cache=SimpleNamespace(
            layers=[SimpleNamespace(is_initialized=True, keys=self_keys, values=self_values)]
        ),
        cross_attention_cache=SimpleNamespace(
            layers=[SimpleNamespace(is_initialized=True, keys=cross_keys, values=cross_values)]
        ),
        is_updated={0: True},
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_framework_prefix_writes_only_active_slot_and_preserves_dtype(dtype) -> None:
    module = _layer(dtype)
    cache = _cache(dtype)
    self_layer = cache.self_attention_cache.layers[0]
    cross_before = cache.cross_attention_cache.layers[0].values.clone()
    future_before = self_layer.keys[..., 2:, :].clone()

    output = framework_prefix(
        module,
        hidden_states=torch.tensor([[[1.0, -2.0, 0.5, 3.0]]], dtype=dtype),
        attention_mask=torch.zeros((1, 1, 1, 4), dtype=dtype),
        encoder_hidden_states=torch.ones((1, 3, 4), dtype=dtype),
        past_key_value=cache,
        cache_position=torch.tensor([1]),
        position_ids=torch.tensor([[1]]),
        active_prefix_length=2,
    )

    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(self_layer.keys[..., 1:2, :]) > 0
    assert torch.equal(self_layer.keys[..., 2:, :], future_before)
    assert torch.equal(cache.cross_attention_cache.layers[0].values, cross_before)


def test_fp32_rms_norm_reduction_is_finite_for_fp16_storage() -> None:
    hidden = torch.tensor([[[300.0, -250.0, 100.0, 50.0]]], dtype=torch.float16)
    output = fp32_rms_norm(hidden, torch.ones(4, dtype=torch.float16), None)

    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()


def test_framework_attention_rejects_mixed_storage_dtypes() -> None:
    query = torch.zeros((1, 2, 1, 4), dtype=torch.float16)
    key = torch.zeros((1, 2, 3, 4), dtype=torch.float32)
    value = torch.zeros_like(key)

    with pytest.raises(TypeError, match="dtypes must match"):
        framework_q1_attention(query, key, value, None)


def test_native_prefix_maps_native_self_to_framework_bmm_cross(monkeypatch) -> None:
    import importlib

    native_prefix_module = importlib.import_module(
        "osuT5.osuT5.inference.optimized.scout.native_prefix"
    )
    captured = {}

    def fake_prefix(module, **kwargs):
        captured.update(kwargs)
        return module

    monkeypatch.setattr(native_prefix_module, "_prefix", fake_prefix)
    sentinel = object()
    result = native_prefix_module.native_prefix(
        sentinel,
        hidden_states=object(),
        attention_mask=None,
        encoder_hidden_states=object(),
        past_key_value=object(),
        cache_position=object(),
        position_ids=object(),
        active_prefix_length=64,
    )

    assert result is sentinel
    assert (
        captured["self_attention_fn"]
        is native_prefix_module.native_q1_rope_cache_attention
    )
    assert captured["cross_attention_fn"] is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_specialized_policy_selects_dtype_kernels_and_restores_hooks(
    monkeypatch,
    dtype,
) -> None:
    from osuT5.osuT5.inference import runtime_dispatch
    from osuT5.osuT5.inference.optimized.kernels import q1_attention as production_q1
    from osuT5.osuT5.inference.optimized.scout import q1_attention as scout_q1

    selected = production_q1 if dtype == torch.float32 else scout_q1
    preloads = []
    monkeypatch.setattr(
        selected,
        "preload_native_q1_attention",
        lambda: preloads.append(dtype),
    )
    def marker(**kwargs):
        return kwargs

    outer = runtime_dispatch.AttentionRuntimeHooks(sdpa_attention_inputs=marker)

    with runtime_dispatch.attention_runtime_hooks_context(outer):
        before = runtime_dispatch.attention_runtime_hooks()
        with specialized_prefix_attention_context(dtype=dtype, native_self=True):
            hooks = runtime_dispatch.attention_runtime_hooks()
            assert hooks is not before
            assert hooks.sdpa_attention_inputs is marker
            assert hooks.sdpa_attention_forward is not None
            assert hooks.q1_rope_cache_self_attention_forward is not None
            assert hooks.sdpa_attention_forward.keywords["expected_dtype"] == dtype
            assert (
                hooks.sdpa_attention_forward.keywords["native_q1_attention"]
                is selected.native_q1_attention
            )
            assert (
                hooks.q1_rope_cache_self_attention_forward.keywords[
                    "native_q1_rope_cache_attention"
                ]
                is selected.native_q1_rope_cache_attention
            )
            assert (
                hooks.q1_rope_cache_self_attention_forward.keywords[
                    "expected_dtype"
                ]
                == dtype
            )
        assert runtime_dispatch.attention_runtime_hooks() is before

    assert preloads == [dtype]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_specialized_framework_policy_keeps_only_bmm_cross_hook(dtype) -> None:
    from osuT5.osuT5.inference import runtime_dispatch

    before = runtime_dispatch.attention_runtime_hooks()
    with specialized_prefix_attention_context(dtype=dtype, native_self=False):
        hooks = runtime_dispatch.attention_runtime_hooks()
        assert hooks.sdpa_attention_forward is not None
        assert hooks.sdpa_attention_forward.keywords["expected_dtype"] == dtype
        assert hooks.sdpa_attention_forward.keywords["q1_bmm_cross_attention"] is True
        assert hooks.sdpa_attention_forward.keywords["native_q1_self_attention"] is False
        assert hooks.sdpa_attention_forward.keywords["native_q1_attention"] is None
        assert hooks.q1_rope_cache_self_attention_forward is None
    assert runtime_dispatch.attention_runtime_hooks() is before
