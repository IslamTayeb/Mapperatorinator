from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.inference.optimized.speculative.target_span_gpu import (
    GATE_NAME,
    TargetSpanGateConfig,
    compare_cache_prefixes,
    hash_cache,
    load_and_validate_prior_gate,
    zero_static_self_cache_suffix,
)


def _fake_cache():
    self_layer = SimpleNamespace(
        is_initialized=True,
        keys=torch.zeros((1, 1, 8, 2), dtype=torch.float32),
        values=torch.zeros((1, 1, 8, 2), dtype=torch.float32),
    )
    cross_layer = SimpleNamespace(
        is_initialized=True,
        keys=torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2),
        values=torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2) + 1,
    )
    return SimpleNamespace(
        self_attention_cache=SimpleNamespace(layers=[self_layer]),
        cross_attention_cache=SimpleNamespace(layers=[cross_layer]),
        is_updated={0: True},
    )


def test_k2_is_the_only_ungated_target_span_shape():
    assert load_and_validate_prior_gate(2, None) is None
    with pytest.raises(ValueError, match="K=4 is gated on a passing K=2 report"):
        load_and_validate_prior_gate(4, None)
    with pytest.raises(ValueError, match="K=8 is gated on a passing K=4 report"):
        load_and_validate_prior_gate(8, None)


def test_k4_requires_the_immediately_previous_passing_gate(tmp_path):
    report_path = tmp_path / "k2.json"
    report_path.write_text(
        json.dumps({"gate": GATE_NAME, "speculation_k": 2, "pass": True}),
        encoding="utf-8",
    )

    assert load_and_validate_prior_gate(4, report_path)["pass"] is True

    report_path.write_text(
        json.dumps({"gate": GATE_NAME, "speculation_k": 2, "pass": False}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires the K=2 report to pass"):
        load_and_validate_prior_gate(4, report_path)


def test_verifier_only_suffix_zeroing_restores_exact_prefill_hash():
    cache = _fake_cache()
    prefill_hash = hash_cache(cache)
    cache.self_attention_cache.layers[0].keys[:, :, 3:5, :].fill_(4.0)
    cache.self_attention_cache.layers[0].values[:, :, 3:5, :].fill_(5.0)

    assert hash_cache(cache) != prefill_hash

    zero_static_self_cache_suffix(cache, start_position=3, end_position=5)

    assert hash_cache(cache) == prefill_hash


def test_cache_prefix_comparison_reports_bitwise_hash_and_allclose_separately():
    reference = _fake_cache()
    candidate = _fake_cache()
    candidate.self_attention_cache.layers[0].keys[:, :, 2, 0] = 1e-5

    assert hash_cache(reference, self_prefix_length=3) != hash_cache(candidate, self_prefix_length=3)
    comparison = compare_cache_prefixes(
        reference,
        candidate,
        self_prefix_length=3,
        atol=1e-4,
        rtol=1e-4,
    )
    assert comparison["allclose"] is True
    assert comparison["max_abs"] == pytest.approx(1e-5)
    # Exact-output admits this internal FP32 drift; the unequal hash is stronger
    # bitwise-calculation-exact diagnostic evidence, not a rejection by itself.


@pytest.mark.parametrize("speculation_k", [2, 4, 8])
def test_target_span_gate_config_accepts_only_campaign_shapes(speculation_k):
    assert TargetSpanGateConfig(speculation_k=speculation_k).speculation_k == speculation_k


def test_rollback_oracle_rejects_out_of_bounds_intervals():
    cache = _fake_cache()
    with pytest.raises(ValueError, match="exceeds"):
        zero_static_self_cache_suffix(cache, start_position=7, end_position=9)
