from __future__ import annotations

import json

import pytest

from osuT5.osuT5.inference.optimized.kernels import q1_attention
from osuT5.osuT5.inference.optimized.kernels import weight_only_runtime
from utils import run_k4_shared_rope_coalesced_split_kv as candidate


def _base_payload():
    return {
        "combined_runtime": "k4-split-kv-mixed-weight-shared-rope-v1",
        "result_class": "documented-drift",
        "exactness_claim": False,
    }


def test_runner_patches_only_weight_owned_self_attention_and_restores(
    monkeypatch,
    tmp_path,
) -> None:
    original_attention = weight_only_runtime.native_q1_rope_cache_attention
    original_variant = weight_only_runtime.native_q1_rope_cache_attention_variant
    observed = []

    def fake_run(config_name, overrides, output):
        observed.append(
            (
                config_name,
                list(overrides),
                weight_only_runtime.native_q1_rope_cache_attention,
                weight_only_runtime.native_q1_rope_cache_attention_variant,
            )
        )
        output.write_text(json.dumps(_base_payload()), encoding="utf-8")

    monkeypatch.setattr(candidate, "run_shared_stack", fake_run)
    output = tmp_path / "init.json"
    candidate.run("profile_salvalai", ["seed=12345"], output)

    assert observed == [
        (
            "profile_salvalai",
            ["seed=12345"],
            q1_attention.native_q1_rope_cache_attention_coalesced,
            q1_attention.native_q1_rope_cache_attention_coalesced_variant,
        )
    ]
    assert weight_only_runtime.native_q1_rope_cache_attention is original_attention
    assert weight_only_runtime.native_q1_rope_cache_attention_variant is original_variant
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["combined_runtime"] == candidate.COMPOSITION_VERSION
    assert payload["coalesced_split_kv"]["scope"] == (
        "main-model-mixed-weight-self-attention-only"
    )


def test_runner_restores_aliases_on_failure(monkeypatch, tmp_path) -> None:
    original_attention = weight_only_runtime.native_q1_rope_cache_attention
    original_variant = weight_only_runtime.native_q1_rope_cache_attention_variant
    monkeypatch.setattr(
        candidate,
        "run_shared_stack",
        lambda *args: (_ for _ in ()).throw(RuntimeError("failed run")),
    )

    with pytest.raises(RuntimeError, match="failed run"):
        candidate.run("profile_salvalai", [], tmp_path / "init.json")

    assert weight_only_runtime.native_q1_rope_cache_attention is original_attention
    assert weight_only_runtime.native_q1_rope_cache_attention_variant is original_variant


def test_initialization_evidence_rejects_the_wrong_base(tmp_path) -> None:
    output = tmp_path / "init.json"
    output.write_text(json.dumps({"combined_runtime": "wrong"}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="exact shared-stack base"):
        candidate._enrich_initialization_evidence(output)
