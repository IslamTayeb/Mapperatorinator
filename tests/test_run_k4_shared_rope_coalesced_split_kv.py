from __future__ import annotations

import json

import pytest

from osuT5.osuT5.inference.optimized.kernels import q1_attention
from osuT5.osuT5.inference.optimized.kernels import weight_only_runtime
from osuT5.osuT5.inference.optimized.kernels.weight_only_runtime import (
    CROSS_ACCEPTED,
    CROSS_FP16_PACKED,
    CROSS_SPLIT8,
)
from utils import run_k4_shared_rope_coalesced_split_kv as candidate


def _base_payload(cross_mode=CROSS_ACCEPTED):
    return {
        "combined_runtime": candidate._base_composition_version(cross_mode),
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

    monkeypatch.setattr(candidate, "run_k1_int8", fake_run)
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
    assert payload["coalesced_split_kv"]["cross_mode"] == CROSS_ACCEPTED
    assert payload["coalesced_split_kv"]["incremental_exactness_claim"] is False


def test_runner_restores_aliases_on_failure(monkeypatch, tmp_path) -> None:
    original_attention = weight_only_runtime.native_q1_rope_cache_attention
    original_variant = weight_only_runtime.native_q1_rope_cache_attention_variant
    monkeypatch.setattr(
        candidate,
        "run_k1_int8",
        lambda *args: (_ for _ in ()).throw(RuntimeError("failed run")),
    )

    with pytest.raises(RuntimeError, match="failed run"):
        candidate.run("profile_salvalai", [], tmp_path / "init.json")

    assert weight_only_runtime.native_q1_rope_cache_attention is original_attention
    assert weight_only_runtime.native_q1_rope_cache_attention_variant is original_variant


def test_initialization_evidence_rejects_the_wrong_base(tmp_path) -> None:
    output = tmp_path / "init.json"
    output.write_text(json.dumps({"combined_runtime": "wrong"}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="requested base topology"):
        candidate._enrich_initialization_evidence(
            output,
            cross_mode=CROSS_ACCEPTED,
        )


@pytest.mark.parametrize("cross_mode", [CROSS_FP16_PACKED, CROSS_SPLIT8])
def test_runner_composes_exactly_one_cross_mode_with_coalesced_self(
    monkeypatch,
    tmp_path,
    cross_mode,
) -> None:
    calls = []

    def fake_cross(config_name, overrides, output, *, mode):
        calls.append((config_name, list(overrides), mode))
        output.write_text(json.dumps(_base_payload(mode)), encoding="utf-8")

    monkeypatch.setattr(candidate, "run_cross_candidate", fake_cross)
    output = tmp_path / "init.json"
    candidate.run(
        "profile_salvalai",
        ["seed=12345"],
        output,
        cross_mode=cross_mode,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert calls == [("profile_salvalai", ["seed=12345"], cross_mode)]
    assert payload["coalesced_split_kv"]["cross_mode"] == cross_mode
    assert payload["combined_runtime"] == candidate._composition_version(cross_mode)


def test_runner_rejects_unknown_cross_mode_before_patching(tmp_path) -> None:
    with pytest.raises(ValueError, match="cross_mode"):
        candidate.run(
            "profile_salvalai",
            [],
            tmp_path / "init.json",
            cross_mode="both",
        )
