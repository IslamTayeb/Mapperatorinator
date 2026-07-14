from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from osuT5.osuT5.inference.optimized.scout.int8_mlp import (
    Int8MlpPack,
    Int8PackedLinear,
)
from utils.profile_dp4a_mlp_component import (
    CONFIGS,
    DECODER_LAYERS,
    FIXED_MAIN_STEPS,
    _dynamic_quantization_reference,
    _integer_linear_reference,
    _live_entries,
    _quantized_reference,
    summarize_component,
)


def _bucket(*, baseline: float, candidate: float, count: int = 100) -> dict:
    candidates = {config: {"ms_per_call": candidate} for config in CONFIGS}
    checks = {
        "baseline_finite": True,
        "baseline_repeat_bitwise_equal": True,
        "baseline_memory_stable": True,
        "selected_weight_only_self_dispatch_observed": True,
        "selected_native_rope_cache_self_dispatch_observed": True,
        "selected_q1_bmm_cross_dispatch_observed": True,
        "selected_weight_only_mlp_dispatch_observed": True,
        "selected_int8_dispatch_observed": True,
        "selected_fp16_cross_dispatch_observed": True,
        "selected_weight_only_final_projection_observed": True,
        "rejected_native_cross_dispatch_absent": True,
        "nonfused_native_self_fallback_absent": True,
    }
    for config in CONFIGS:
        checks.update(
            {
                f"{config}_finite": True,
                f"{config}_repeat_bitwise_equal": True,
                f"{config}_memory_stable": True,
                f"{config}_quantized_reference": True,
            }
        )
        if config != next(iter(CONFIGS)):
            checks[f"{config}_cross_config_bitwise_equal"] = True
    return {
        "decode_replays": count,
        "checks": checks,
        "l2_hot": {
            "baseline_ms_per_call": baseline,
            "candidates": candidates,
        },
        "cache_cold": {
            "baseline_ms_per_call": baseline * 1.1,
            "candidates": {
                config: {"ms_per_call": candidate * 1.1} for config in CONFIGS
            },
        },
    }


def test_summary_normalizes_sentinel_mix_to_fixed_8294_and_gates_hot_saving() -> None:
    report = summarize_component(
        {
            "128": _bucket(baseline=0.0165, candidate=0.0125, count=3000),
            "576": _bucket(baseline=0.0168, candidate=0.0127, count=1000),
            "640": _bucket(baseline=0.0170, candidate=0.0129, count=500),
        },
        total_replays=7597,
    )

    measured = 4500
    expected_baseline = (
        DECODER_LAYERS * (3000 * 0.0165 + 1000 * 0.0168 + 500 * 0.0170)
        / 1000
        * FIXED_MAIN_STEPS
        / measured
    )
    expected_candidate = (
        DECODER_LAYERS * (3000 * 0.0125 + 1000 * 0.0127 + 500 * 0.0129)
        / 1000
        * FIXED_MAIN_STEPS
        / measured
    )
    hot = report["modes"]["l2_hot"]
    assert hot["fixed_8294_baseline_seconds"] == pytest.approx(expected_baseline)
    assert hot["fixed_8294_candidate_seconds"] == pytest.approx(expected_candidate)
    assert hot["fixed_8294_saving_seconds"] == pytest.approx(
        expected_baseline - expected_candidate
    )
    assert report["l2_hot_saving_pass"]
    assert report["cache_cold_saving_pass"]
    assert report["component_pass"]


def test_summary_stops_on_subthreshold_hot_or_cold_regression_or_invariant() -> None:
    slow = summarize_component(
        {"128": _bucket(baseline=0.0165, candidate=0.0145)},
        total_replays=100,
    )
    assert not slow["l2_hot_saving_pass"]
    assert not slow["component_pass"]

    cold = _bucket(baseline=0.0165, candidate=0.0125)
    cold["cache_cold"]["candidates"] = {
        config: {"ms_per_call": 0.019} for config in CONFIGS
    }
    cold_report = summarize_component({"128": cold}, total_replays=100)
    assert cold_report["l2_hot_saving_pass"]
    assert not cold_report["cache_cold_saving_pass"]
    assert not cold_report["component_pass"]

    bad = _bucket(baseline=0.0165, candidate=0.0125)
    bad["checks"]["w8_c68_quantized_reference"] = False
    bad_report = summarize_component({"128": bad}, total_replays=100)
    assert bad_report["invariant_failures"] == {
        "128": ["w8_c68_quantized_reference"]
    }
    assert not bad_report["component_pass"]


def test_live_manifest_accepts_new_positive_buckets_but_requires_sentinels() -> None:
    def entry(prefix: int, count: int = 1) -> dict:
        return {
            "active_prefix_length": prefix,
            "graph": object(),
            "outputs": object(),
            "static_inputs": {},
            "decode_replays": count,
        }

    entries = _live_entries(
        {
            (128,): entry(128, 10),
            (576,): entry(576, 20),
            (640,): entry(640, 30),
            (832,): entry(832, 2),
        }
    )
    assert tuple(entries) == (128, 576, 640, 832)

    with pytest.raises(RuntimeError, match="missing sentinel"):
        _live_entries({(128,): entry(128), (640,): entry(640)})
    with pytest.raises(RuntimeError, match="must be positive"):
        _live_entries(
            {
                (128,): entry(128),
                (576,): entry(576, 0),
                (640,): entry(640),
            }
        )


def test_integer_dp4a_reference_uses_observed_quantized_activation() -> None:
    pack = SimpleNamespace(
        weight=torch.tensor([[1, -2, 3, -4], [-5, 6, -7, 8]], dtype=torch.int8),
        scale=torch.tensor([0.5, 0.25]),
        bias=torch.tensor([1.0, -1.0]),
    )
    quantized = torch.tensor([2, 3, -4, 5], dtype=torch.int8)
    activation_scale = torch.tensor([0.125])

    actual = _integer_linear_reference(quantized, activation_scale, pack)
    integer = pack.weight.float() @ quantized.float()
    expected = integer * activation_scale * pack.scale + pack.bias

    assert torch.equal(actual, expected)


def test_dynamic_quantization_reference_checks_scale_rounding_and_minus_128() -> None:
    values = torch.tensor([[-2.0, -0.5, 0.0, 1.0, 2.0]], dtype=torch.float32)
    scale = torch.tensor([2.0 / 127.0])
    quantized = torch.round(values / scale).clamp(-127, 127).to(torch.int8)

    valid = _dynamic_quantization_reference(values, quantized, scale)
    assert valid["pass"]
    assert not valid["contains_minus_128"]

    bad_scale = _dynamic_quantization_reference(values, quantized, scale * 2)
    assert not bad_scale["pass"]
    bad_quantized = quantized.clone()
    bad_quantized[0, 0] = -128
    assert not _dynamic_quantization_reference(values, bad_quantized, scale)["pass"]


def test_two_stage_quantized_reference_preserves_residual_and_zero_gelu() -> None:
    def linear(rows: int, columns: int) -> Int8PackedLinear:
        return Int8PackedLinear(
            weight=torch.zeros((rows, columns), dtype=torch.int8),
            scale=torch.ones(rows),
            bias=torch.zeros(rows),
            source_weight_bytes=rows * columns * 4,
        )

    pack = Int8MlpPack(fc1=linear(3072, 768), fc2=linear(768, 3072))
    residual = torch.arange(768, dtype=torch.float32).view(1, 1, 768) / 100
    state = (
        residual.clone(),
        torch.zeros((1, 1, 3072)),
        torch.zeros(768, dtype=torch.int8),
        torch.ones(1),
        torch.zeros(3072, dtype=torch.int8),
        torch.ones(1),
    )

    reference, evidence = _quantized_reference(
        residual,
        torch.ones(768),
        pack,
        state,
        eps=1e-5,
    )

    assert torch.equal(reference, residual)
    assert evidence["fc1_kernel_vs_quantized_reference"]["max_abs"] == 0.0
    assert evidence["fc2_kernel_vs_quantized_reference"]["max_abs"] == 0.0
    assert evidence["fc1_dynamic_quantization"]["pass"] is False
    assert evidence["fc2_dynamic_quantization"]["pass"] is True


def test_dcc_wrapper_pins_clean_pushed_2080_ti_worktree() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/dcc/profile_dp4a_mlp_component.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --gres=gpu:2080:1" in source
    assert "NVIDIA GeForce RTX 2080 Ti" in source
    assert "MAPPERATORINATOR_COMMIT" in source
    assert "MAPPERATORINATOR_BRANCH" in source
    assert 'status --porcelain' in source
    assert 'rev-parse "$REMOTE_REF"' in source
    assert "Run this wrapper inside a Slurm allocation" in source
    assert "running-jobs.txt" in source
    assert "another GPU job is already running" in source
    assert "slurm-job.txt" in source
    assert "--bucket-mode sentinel" in source
    assert "--cold-iters 100" in source
    assert "profile_inference=true" in source
