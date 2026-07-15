from __future__ import annotations

import pytest
import torch

from osuT5.osuT5.inference.optimized.single.shared_rope import (
    build_shared_decoder_rope_plan,
    shared_decoder_rope,
    shared_decoder_rope_forward_context,
)
from osuT5.osuT5.model.custom_transformers.modeling_varwhisper import (
    VarWhisperDecoderLayer,
)


class _Config:
    def __init__(self, theta: float = 10_000.0):
        self.theta = theta

    def to_dict(self):
        return {
            "rope_theta": self.theta,
            "rope_scaling": None,
            "hidden_size": 4,
            "max_position_embeddings": 64,
        }


def _rope_init(*args, **kwargs):
    del args, kwargs
    return None


class _Rotary(torch.nn.Module):
    def __init__(self, *, theta: float = 10_000.0, rope_type: str = "default"):
        super().__init__()
        self.config = _Config(theta)
        self.rope_type = rope_type
        self.max_seq_len_cached = 64
        self.original_max_seq_len = 64
        self.rope_init_fn = _rope_init
        self.register_buffer("inv_freq", torch.tensor([1.0, 0.5]))
        self.original_inv_freq = self.inv_freq
        self.attention_scaling = 1.0
        self.calls = 0

    def forward(self, x, position_ids):
        self.calls += 1
        phase = position_ids.to(dtype=torch.float32).unsqueeze(-1) * self.inv_freq
        emb = torch.cat((phase, phase), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class _SelfAttention(torch.nn.Module):
    def __init__(self, index: int, *, theta: float, rope_type: str):
        super().__init__()
        self.layer_idx = index
        self.rotary_emb = _Rotary(theta=theta, rope_type=rope_type)


def _decoder_layer(index: int, *, theta: float = 10_000.0, rope_type="default"):
    layer = VarWhisperDecoderLayer.__new__(VarWhisperDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.self_attn = _SelfAttention(
        index,
        theta=theta,
        rope_type=rope_type,
    )
    return layer


class _Model(torch.nn.Module):
    def __init__(self, *, thetas=(10_000.0, 10_000.0, 10_000.0)):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [_decoder_layer(index, theta=theta) for index, theta in enumerate(thetas)]
        )

    def baseline(self, x, position_ids):
        return [
            layer.self_attn.rotary_emb(x, position_ids)
            for layer in self.layers
        ]

    def shared(self, plan, x, position_ids, counts):
        with shared_decoder_rope_forward_context(plan):
            return [
                shared_decoder_rope(
                    layer.self_attn.rotary_emb,
                    x,
                    position_ids,
                    dispatch_counts=counts,
                )
                for layer in self.layers
            ]


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
def test_shared_rope_is_exact_for_both_production_storage_dtypes(dtype) -> None:
    baseline = _Model()
    candidate = _Model()
    candidate.load_state_dict(baseline.state_dict())
    x = torch.tensor([[[0.25, -0.5, 0.75, 1.0]]], dtype=dtype)
    position_ids = torch.tensor([[7]])
    expected = baseline.baseline(x, position_ids)
    counts = {
        "shared_decoder_rope_compute": 0,
        "shared_decoder_rope_reuse": 0,
    }

    original_model_forward = candidate.forward
    original_layer_forwards = [layer.forward for layer in candidate.layers]
    actual = candidate.shared(
        build_shared_decoder_rope_plan(candidate),
        x,
        position_ids,
        counts,
    )

    assert all(
        torch.equal(actual_value, expected_value)
        for pair in zip(actual, expected, strict=True)
        for actual_value, expected_value in zip(pair[0], pair[1], strict=True)
    )
    assert [layer.self_attn.rotary_emb.calls for layer in candidate.layers] == [
        1,
        0,
        0,
    ]
    assert counts == {
        "shared_decoder_rope_compute": 1,
        "shared_decoder_rope_reuse": 2,
    }
    assert "forward" not in candidate.__dict__
    assert candidate.forward.__func__ is original_model_forward.__func__
    assert [layer.forward for layer in candidate.layers] == original_layer_forwards
    assert all("forward" not in layer.__dict__ for layer in candidate.layers)


def test_plan_groups_only_exact_equivalent_rope_and_is_request_local() -> None:
    first = _Model(thetas=(10_000.0, 20_000.0, 10_000.0))
    second = _Model()
    first_plan = build_shared_decoder_rope_plan(first)
    second_plan = build_shared_decoder_rope_plan(second)

    assert first_plan.group_count == 2
    assert first_plan.members[0].group_id == first_plan.members[2].group_id
    assert first_plan.members[0].group_id != first_plan.members[1].group_id
    assert first_plan.model_id != second_plan.model_id
    x = torch.ones(1, 1, 4)
    position_ids = torch.tensor([[3]])
    with shared_decoder_rope_forward_context(first_plan):
        with pytest.raises(RuntimeError, match="cannot nest"):
            with shared_decoder_rope_forward_context(second_plan):
                pass
        for layer in first.layers:
            shared_decoder_rope(
                layer.self_attn.rotary_emb,
                x,
                position_ids,
            )


def test_shared_rope_rejects_mismatched_inputs_and_unplanned_modules() -> None:
    model = _Model()
    plan = build_shared_decoder_rope_plan(model)
    position = torch.tensor([[3]])
    with pytest.raises(RuntimeError, match="inputs differ"):
        with shared_decoder_rope_forward_context(plan):
            shared_decoder_rope(
                model.layers[0].self_attn.rotary_emb,
                torch.ones(1, 1, 4),
                position,
            )
            shared_decoder_rope(
                model.layers[1].self_attn.rotary_emb,
                torch.ones(1, 1, 1, 4),
                position,
            )

    with shared_decoder_rope_forward_context(plan):
        with pytest.raises(RuntimeError, match="unplanned"):
            shared_decoder_rope(_Rotary(), torch.ones(1, 1, 4), position)
        for layer in model.layers:
            shared_decoder_rope(
                layer.self_attn.rotary_emb,
                torch.ones(1, 1, 4),
                position,
            )


def test_shared_rope_rejects_unsupported_state_before_execution() -> None:
    model = _Model()
    model.layers[0].self_attn.rotary_emb.rope_type = "dynamic"
    with pytest.raises(ValueError, match="only static default"):
        build_shared_decoder_rope_plan(model)

    model = _Model()
    model.layers[0].forward = lambda *args, **kwargs: None
    with pytest.raises(RuntimeError, match="original forward"):
        build_shared_decoder_rope_plan(model)

    model = _Model()
    plan = build_shared_decoder_rope_plan(model)
    with pytest.raises(TypeError, match="floating point"):
        with shared_decoder_rope_forward_context(plan):
            shared_decoder_rope(
                model.layers[0].self_attn.rotary_emb,
                torch.ones(1, 1, 4, dtype=torch.long),
                torch.tensor([[3]]),
            )

    with pytest.raises(TypeError, match="position_ids must use torch.long"):
        with shared_decoder_rope_forward_context(plan):
            shared_decoder_rope(
                model.layers[0].self_attn.rotary_emb,
                torch.ones(1, 1, 4),
                torch.tensor([[3]], dtype=torch.int32),
            )

    with pytest.raises(ValueError, match="tensors must use one device"):
        with shared_decoder_rope_forward_context(plan):
            shared_decoder_rope(
                model.layers[0].self_attn.rotary_emb,
                torch.ones(1, 1, 4, device="meta"),
                torch.tensor([[3]], device="meta"),
            )


def test_context_allows_the_existing_nonfused_self_attention_fallback() -> None:
    model = _Model()
    plan = build_shared_decoder_rope_plan(model)
    original_fallback = model.layers[1].self_attn.rotary_emb.forward
    with shared_decoder_rope_forward_context(plan):
        shared_decoder_rope(
            model.layers[0].self_attn.rotary_emb,
            torch.ones(1, 1, 4),
            torch.tensor([[3]]),
        )
        fallback_output = model.layers[1].self_attn.rotary_emb(
            torch.ones(1, 1, 4),
            torch.tensor([[3]]),
        )

    assert all(torch.isfinite(value).all() for value in fallback_output)
    assert (
        model.layers[1].self_attn.rotary_emb.forward.__func__
        is original_fallback.__func__
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
def test_shared_rope_runtime_hook_restores_cleanly(monkeypatch, dtype) -> None:
    from osuT5.osuT5.inference import runtime_dispatch
    from osuT5.osuT5.inference.optimized.kernels import q1_attention
    from osuT5.osuT5.inference.optimized.single.runtime_context import (
        attention_runtime_context,
    )

    monkeypatch.setattr(q1_attention, "preload_native_q1_attention", lambda: None)
    original = runtime_dispatch.attention_runtime_hooks()
    with attention_runtime_context(
        q1_bmm_cross_attention=True,
        native_q1_self_attention=True,
        native_q1_rope_cache_self_attention=True,
        share_decoder_rope=True,
        expected_dtype=dtype,
    ):
        hook = runtime_dispatch.attention_runtime_hooks().q1_rope_cache_self_attention_forward
        assert hook is not None
        assert hook.keywords["share_decoder_rope"] is True
        assert hook.keywords["expected_dtype"] == dtype

    assert runtime_dispatch.attention_runtime_hooks() == original
