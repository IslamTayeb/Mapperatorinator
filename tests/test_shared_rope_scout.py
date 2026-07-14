from __future__ import annotations

import pytest
import torch

from osuT5.osuT5.inference.optimized.scout.shared_rope import (
    SharedRopeStats,
    build_shared_rope_plan,
    shared_decoder_rope_context,
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


def _rope_init(*args, **kwargs):
    del args, kwargs
    return None


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
    def __init__(self, *, thetas=(10_000.0, 10_000.0, 10_000.0), clone_position=False):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [_decoder_layer(index, theta=theta) for index, theta in enumerate(thetas)]
        )
        self.clone_position = clone_position
        self.register_buffer("cache", torch.zeros(len(self.layers), 1, 1, 4))

    def forward(self, x, position_ids):
        value = x
        for index, layer in enumerate(self.layers):
            layer_position = (
                position_ids.clone()
                if self.clone_position and index > 0
                else position_ids
            )
            cos, sin = layer.self_attn.rotary_emb(value, layer_position)
            value = value + cos + sin
            self.cache[index].copy_(value)
        return value


def _calls(model):
    return [layer.self_attn.rotary_emb.calls for layer in model.layers]


def test_shared_rope_is_exact_and_computes_once_per_group_per_forward():
    baseline = _Model()
    candidate = _Model()
    candidate.load_state_dict(baseline.state_dict())
    x = torch.tensor([[[0.25, -0.5, 0.75, 1.0]]])
    position_ids = torch.tensor([[7]])

    expected = baseline(x, position_ids)
    expected_cache = baseline.cache.clone()
    stats = SharedRopeStats()
    with shared_decoder_rope_context(candidate, stats=stats):
        actual = candidate(x, position_ids)
        first_cache = candidate.cache.clone()
        second = candidate(x, position_ids)

    assert torch.equal(actual, expected)
    assert torch.equal(second, expected)
    assert torch.equal(first_cache, expected_cache)
    assert torch.equal(candidate.cache, expected_cache)
    assert _calls(baseline) == [1, 1, 1]
    assert _calls(candidate) == [2, 0, 0]
    assert stats.as_dict() == {
        "module_count": 3,
        "group_count": 1,
        "forwards": 2,
        "computes": 2,
        "reuses": 4,
        "eliminated_per_forward": 2,
        "expected_computes": 2,
        "expected_reuses": 4,
        "group_computes": {
            next(iter(stats.group_computes)): 2,
        },
        "member_names": list(stats.member_names),
        "group_members": {
            next(iter(stats.group_members)): list(next(iter(stats.group_members.values()))),
        },
    }


def test_exact_parameter_groups_are_validated_before_install():
    model = _Model(thetas=(10_000.0, 20_000.0, 10_000.0))
    plan = build_shared_rope_plan(model)

    assert plan.module_count == 3
    assert plan.group_count == 2
    members = [member.group_id for member in plan.members]
    assert members[0] == members[2]
    assert members[0] != members[1]


def test_forward_cache_resets_and_mismatched_position_storage_fails_loudly():
    model = _Model(clone_position=True)
    stats = SharedRopeStats()

    with shared_decoder_rope_context(model, stats=stats):
        with pytest.raises(RuntimeError, match="received mismatched inputs"):
            model(torch.ones(1, 1, 4), torch.tensor([[3]]))


def test_context_restores_instance_state_and_default_forward_ownership():
    model = _Model()
    model_forward = model.forward
    rope_forwards = [layer.self_attn.rotary_emb.forward for layer in model.layers]
    layer_forwards = [layer.forward for layer in model.layers]
    assert "forward" not in model.__dict__
    assert all("forward" not in layer.self_attn.rotary_emb.__dict__ for layer in model.layers)

    with shared_decoder_rope_context(model):
        assert "forward" in model.__dict__
        assert all("forward" in layer.self_attn.rotary_emb.__dict__ for layer in model.layers)
        assert [layer.forward for layer in model.layers] == layer_forwards

    assert "forward" not in model.__dict__
    assert all("forward" not in layer.self_attn.rotary_emb.__dict__ for layer in model.layers)
    assert model.forward.__func__ is model_forward.__func__
    assert all(
        layer.self_attn.rotary_emb.forward.__func__ is original.__func__
        for layer, original in zip(model.layers, rope_forwards, strict=True)
    )
    assert [layer.forward for layer in model.layers] == layer_forwards


def test_nested_install_original_decoder_patch_and_dynamic_rope_are_rejected():
    model = _Model()
    with shared_decoder_rope_context(model):
        with pytest.raises(RuntimeError, match="already active"):
            with shared_decoder_rope_context(model):
                pass

    layer = model.layers[0]
    layer.forward = lambda *args, **kwargs: None
    with pytest.raises(RuntimeError, match="original decoder forward"):
        build_shared_rope_plan(model)
    delattr(layer, "forward")

    model.layers[0].self_attn.rotary_emb.rope_type = "dynamic"
    with pytest.raises(ValueError, match="only static default RoPE"):
        build_shared_rope_plan(model)


def test_default_import_does_not_change_runtime_or_model_state():
    from osuT5.osuT5.inference import runtime_dispatch
    from osuT5.osuT5.inference.optimized.scout import shared_rope

    model = _Model()
    assert shared_rope._ACTIVE_MODELS == set()
    assert runtime_dispatch.attention_runtime_hooks() == runtime_dispatch.AttentionRuntimeHooks()
    assert "forward" not in model.__dict__


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph smoke requires CUDA")
def test_shared_rope_capture_records_one_compute_and_replays_without_python():
    model = _Model().cuda().eval()
    x = torch.ones(1, 1, 4, device="cuda")
    position_ids = torch.tensor([[3]], device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            model(x, position_ids)
    torch.cuda.current_stream().wait_stream(stream)

    stats = SharedRopeStats()
    graph = torch.cuda.CUDAGraph()
    with shared_decoder_rope_context(model, stats=stats), torch.cuda.graph(graph):
        output = model(x, position_ids)
    calls_after_capture = _calls(model)
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()

    assert stats.forwards == 1
    assert stats.computes == 1
    assert stats.reuses == 2
    assert _calls(model) == calls_after_capture
    assert torch.isfinite(output).all()
