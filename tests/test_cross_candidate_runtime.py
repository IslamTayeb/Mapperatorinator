from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers.activations import GELUActivation

from osuT5.osuT5.inference.optimized.kernels import weight_only_runtime


ROOT = Path(__file__).resolve().parents[1]


class _FakeEncoderDecoderCache:
    def __init__(self) -> None:
        layer = SimpleNamespace(
            is_initialized=True,
            keys=torch.zeros((1, 12, 1024, 64)),
            values=torch.zeros((1, 12, 1024, 64)),
        )
        self.cross_attention_cache = SimpleNamespace(layers=[layer])
        self.is_updated = {0: True}


def _module() -> SimpleNamespace:
    linear = SimpleNamespace(
        weight=torch.zeros((768, 768)),
        bias=None,
    )
    return SimpleNamespace(
        training=False,
        activation_dropout=0.0,
        dropout=0.0,
        activation_fn=GELUActivation(),
        cross_attn=SimpleNamespace(
            layer_idx=0,
            num_heads=12,
            head_dim=64,
            all_head_size=768,
            Wq=linear,
            Wo=linear,
        ),
        cross_attn_layer_norm=SimpleNamespace(weight=torch.ones(768), eps=1e-5),
        final_layer_norm=SimpleNamespace(weight=torch.ones(768), eps=1e-5),
    )


def _exercise(
    monkeypatch,
    mode: str,
    *,
    int8_mlp: bool = False,
) -> tuple[dict[str, int], list[str]]:
    calls: list[str] = []
    module = _module()
    cross_packs = (object(), object())
    state = SimpleNamespace(
        cross_mode=mode,
        int8_mlp_enabled=int8_mlp,
        pack_for_layer=lambda layer: SimpleNamespace(fc1=object(), fc2=object()),
        cross_pack_for_layer=lambda layer: cross_packs,
        int8_mlp_pack_for_layer=lambda layer: object(),
    )
    monkeypatch.setattr(weight_only_runtime, "EncoderDecoderCache", _FakeEncoderDecoderCache)
    monkeypatch.setattr(weight_only_runtime, "_require_fp32_cuda", lambda value, **kwargs: value)

    def native_rms(*args, **kwargs):
        calls.append("native_q")
        return torch.zeros((1, 1, 768))

    def packed_rms(*args, **kwargs):
        calls.append("packed_q")
        assert args[2] is cross_packs[0]
        return torch.zeros((1, 1, 768))

    def accepted_bmm(*args, **kwargs):
        calls.append("accepted_bmm")
        assert kwargs == {"expected_dtype": torch.float32}
        return torch.zeros((1, 12, 1, 64))

    def native_out(*args, **kwargs):
        calls.append("native_out")
        return torch.zeros((1, 1, 768))

    def packed_out(*args, **kwargs):
        calls.append("packed_out")
        assert args[2] is cross_packs[1]
        return torch.zeros((1, 1, 768))

    monkeypatch.setattr(weight_only_runtime, "native_one_token_rmsnorm_linear", native_rms)
    monkeypatch.setattr(weight_only_runtime, "weight_only_rmsnorm_linear", packed_rms)
    monkeypatch.setattr(weight_only_runtime, "_q1_bmm_cross_attention", accepted_bmm)
    monkeypatch.setattr(weight_only_runtime, "native_one_token_linear_residual", native_out)
    monkeypatch.setattr(weight_only_runtime, "weight_only_linear_residual", packed_out)
    monkeypatch.setattr(
        weight_only_runtime,
        "weight_only_mlp_residual",
        lambda hidden, *args, **kwargs: calls.append("mlp") or hidden,
    )
    from osuT5.osuT5.inference.optimized.scout import int8_mlp as int8_mlp_module

    monkeypatch.setattr(
        int8_mlp_module,
        "int8_weight_mlp_residual",
        lambda hidden, *args, **kwargs: calls.append("int8_mlp") or hidden,
    )
    counts = {
        "weight_only_mlp_tail": 0,
        "q1_bmm_cross_attention": 0,
    }
    if int8_mlp:
        counts["int8_weight_mlp_tail"] = 0
    output = weight_only_runtime._cross_mlp_tail_forward(
        state=state,
        module=module,
        hidden_states=torch.zeros((1, 1, 768)),
        encoder_hidden_states=torch.zeros((1, 1024, 768)),
        past_key_value=_FakeEncoderDecoderCache(),
        self_attn_outputs=(torch.zeros((1, 1, 768)),),
        output_attentions=False,
        cu_seqlens=None,
        encoder_cu_seqlens=None,
        layer_name="decoder.layer0",
        dispatch_counts=counts,
    )
    assert output is not None and tuple(output[0].shape) == (1, 1, 768)
    return counts, calls


def test_fp16_packed_cross_changes_only_projection_kernels(monkeypatch) -> None:
    counts, calls = _exercise(monkeypatch, weight_only_runtime.CROSS_FP16_PACKED)

    assert calls == ["packed_q", "accepted_bmm", "packed_out", "mlp"]
    assert counts == {
        "weight_only_mlp_tail": 1,
        "q1_bmm_cross_attention": 1,
        "fp16_packed_cross_projection_candidate": 1,
    }


def test_fp16_packed_cross_preserves_the_int8_mlp_overlay(monkeypatch) -> None:
    counts, calls = _exercise(
        monkeypatch,
        weight_only_runtime.CROSS_FP16_PACKED,
        int8_mlp=True,
    )

    assert calls == ["packed_q", "accepted_bmm", "packed_out", "int8_mlp"]
    assert counts["weight_only_mlp_tail"] == 1
    assert counts["int8_weight_mlp_tail"] == 1
    assert counts["q1_bmm_cross_attention"] == 1


def test_selected_composition_has_no_split8_cross_runtime_surface() -> None:
    for path in (
        ROOT / "osuT5/osuT5/inference/optimized/kernels/weight_only_runtime.py",
        ROOT / "osuT5/osuT5/inference/optimized/single/engine.py",
        ROOT / "utils/run_k4_shared_rope_cross_candidate.py",
        ROOT / "scripts/dcc/verify_weight_only_full_song_reciprocal.sbatch",
    ):
        source = path.read_text(encoding="utf-8")
        assert "cross_attention_split" not in source
        assert "CROSS_SPLIT8" not in source
        assert "split8_attention" not in source
    assert not (
        ROOT / "osuT5/osuT5/inference/optimized/scout/cross_attention.py"
    ).exists()
    assert not (ROOT / "utils/run_k4_shared_rope_split8_cross.py").exists()


@pytest.mark.parametrize("mode", ["bad", "split8_attention", "", None])
def test_state_rejects_unsupported_cross_modes_before_cuda(mode) -> None:
    with pytest.raises(ValueError, match="cross candidate mode"):
        weight_only_runtime.ApproximateWeightOnlyState.initialize(
            torch.nn.Module(),
            cross_mode=mode,
        )
