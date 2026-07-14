from __future__ import annotations

import copy
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from utils.profile_int8_projection_component import (
    FIXED_MAIN_SAVING_GATE_SECONDS,
    MEASURED_INT8_VS_FP16_MLP_SPEEDUP,
    REQUIRED_DISPATCHES,
    _retained_real_tensor_context,
    _validate_retained_composition_manifest,
    summarize_component,
)


EXPECTED_REGIONS = (
    "self_norm_qkv",
    "self_out_residual",
    "cross_norm_q",
    "cross_out_residual",
    "final_norm_logits",
)
EXPECTED_BASELINE = {
    "self_norm_qkv": "fp16_weight",
    "self_out_residual": "fp16_weight",
    "cross_norm_q": "fp32_weight",
    "cross_out_residual": "fp32_weight",
    "final_norm_logits": "fp16_weight",
}
OUTPUTS_PER_BLOCK = (2, 4, 8)


def _buckets(*, baseline_ms: float = 0.3, candidate_ms: float = 0.1):
    candidates = {
        str(opb): {
            "ms_per_call": candidate_ms + opb / 100_000,
            "finite": True,
            "repeat_deterministic": True,
            "memory_stable": True,
        }
        for opb in OUTPUTS_PER_BLOCK
    }
    return {
        "128": {
            "decode_replays": 100,
            "regions": {
                region: {
                    "baseline_ms_per_call": baseline_ms,
                    "baseline_finite": True,
                    "baseline_repeat_deterministic": True,
                    "baseline_memory_stable": True,
                    "candidates": copy.deepcopy(candidates),
                }
                for region in EXPECTED_REGIONS
            },
        }
    }


def _weight_state() -> dict:
    return {
        "version": "approximate-fp16-weights-fp32-state-v2",
        "result_class": "documented-drift",
        "exactness_claim": False,
        "fp32_activations_caches_reductions_logits": True,
        "fp16_weight_regions": [
            "self_qkv",
            "self_output",
            "mlp_fc1",
            "mlp_fc2",
            "final_logits",
        ],
        "fp32_selected_decode_matrix_regions": ["cross_query", "cross_output"],
        "outputs_per_block": {
            "self_qkv": 4,
            "self_output": 4,
            "mlp_fc1": 2,
            "mlp_fc2": 4,
            "final_logits": 2,
        },
        "int8_mlp_overlay": {
            "version": "per-row-symmetric-int8-mlp-v1",
            "result_class": "documented-drift",
            "exactness_claim": False,
            "scope": "main-model-decoder-mlp-only",
            "composition_control": "k4-split-kv-mixed-weight-shared-rope-v1",
            "replaces_effective_regions": ["mlp_fc1", "mlp_fc2"],
            "retained_unused_fp16_regions": ["mlp_fc1", "mlp_fc2"],
            "fp32_activations_norm_bias_reductions_residual_outputs": True,
            "quantization": "symmetric-per-output-row",
            "dispatch_counter": "int8_weight_mlp_tail",
        },
    }


def _manifest() -> dict:
    dispatch_policy = {
        "q1_bmm_cross_attention": {"enabled": True, "disabled_reason": None},
        "native_q1_self_attention": {
            "enabled": False,
            "disabled_reason": "approximate_weight_only",
        },
        "native_q1_rope_cache_self_attention": {
            "enabled": False,
            "disabled_reason": "approximate_weight_only",
        },
        "effective_native_q1_rope_cache_self_attention": {
            "enabled": True,
            "owner": "approximate_weight_only",
            "kernel": "native_q1_rope_cache_attention",
            "standard_attention_hook_enabled": False,
            "split_kv_selector_enabled": True,
        },
        "native_cross_mlp_tail": {
            "enabled": False,
            "disabled_reason": "approximate_weight_only",
        },
    }
    manifest = {
        "composition_version": (
            "k4-split-kv-mixed-weight-shared-rope-k1-remainder-int8-mlp-v1"
        ),
        "scouted_regions": list(EXPECTED_REGIONS),
        "baseline_by_region": dict(EXPECTED_BASELINE),
        "excluded_already_int8_regions": ["mlp_fc1", "mlp_fc2"],
        "weight_state": _weight_state(),
        "shared_rope": {
            "version": "shared-decoder-rope-v1",
            "scope": "main-model-only",
            "incremental_exactness_claim": True,
            "original_decoder_forward_required": True,
            "stats": {
                "module_count": 12,
                "group_count": 1,
                "forwards": 2,
                "computes": 2,
                "expected_computes": 2,
                "reuses": 22,
                "expected_reuses": 22,
            },
        },
        "main_generation_records": [
            {
                "generated_tokens": 12,
                "optimized_dispatch_mode": "approximate_weight_only_batch1",
                "optimized_dispatch_policy": dispatch_policy,
                "optimized_dispatch_capture_hits": {
                    "weight_only_self_attention_block": 24,
                    "native_q1_rope_cache_self_attention": 24,
                    "q1_bmm_cross_attention": 24,
                    "weight_only_mlp_tail": 24,
                    "int8_weight_mlp_tail": 24,
                    "weight_only_final_projection": 2,
                    "native_q1_rope_cache_self_attention_split_kv_8": 12,
                    "native_q1_rope_cache_self_attention_split_kv_8_prefix_640": 12,
                },
                "optimized_cuda_graphs": {
                    "k8_candidate": {
                        "block_size": 4,
                        "prefill_steps": 1,
                        "eligible_steps": 8,
                        "block_replays": 2,
                        "remainder_steps": 6,
                        "remainder_backend": "cuda_graph",
                        "remainder_graph_replays": 6,
                        "eager_remainder_steps": 0,
                        "physical_steps": 15,
                        "logical_steps": 12,
                        "wasted_steps": 3,
                        "rng_policy": "counter_request_seed_window_prompt_v2",
                        "rng_exact": False,
                        "rng_early_eos_isolation": True,
                        "capture_state_restore_synchronized": True,
                    }
                },
            }
        ],
    }
    cached_record = copy.deepcopy(manifest["main_generation_records"][0])
    cached_record["generated_tokens"] = 5
    cached_record["optimized_dispatch_capture_hits"] = {
        name: 0 for name in cached_record["optimized_dispatch_capture_hits"]
    }
    cached_record["optimized_cuda_graphs"]["k8_candidate"].update(
        prefill_steps=1,
        eligible_steps=4,
        block_replays=1,
        remainder_steps=0,
        remainder_graph_replays=0,
        physical_steps=5,
        logical_steps=5,
        wasted_steps=0,
    )
    manifest["main_generation_records"].append(cached_record)
    return manifest


def test_summary_uses_literal_incremental_baselines_and_excludes_mlp() -> None:
    report = summarize_component(_buckets(), total_replays=200)

    assert tuple(report["regions"]) == EXPECTED_REGIONS
    assert not {"mlp_fc1", "mlp_fc2"} & set(report["regions"])
    assert {
        region: entry["current_baseline"]
        for region, entry in report["regions"].items()
    } == EXPECTED_BASELINE
    assert report["coverage_fraction"] == pytest.approx(0.5)
    assert report["selective_fixed_work_main_saving_seconds"] > 1.9
    assert report["saving_gate_seconds"] == FIXED_MAIN_SAVING_GATE_SECONDS
    assert report["sizing_pass"]
    assert report["promotion_pass"]


def test_modeled_ceiling_is_derived_from_live_measured_baselines() -> None:
    buckets = _buckets(baseline_ms=0.3)
    baselines = {
        "self_norm_qkv": 0.11,
        "self_out_residual": 0.17,
        "cross_norm_q": 0.23,
        "cross_out_residual": 0.29,
        "final_norm_logits": 0.31,
    }
    for region, value in baselines.items():
        buckets["128"]["regions"][region]["baseline_ms_per_call"] = value

    report = summarize_component(buckets, total_replays=200)
    measured_seconds = (
        12 * 100 * sum(value for name, value in baselines.items() if name != "final_norm_logits")
        + 100 * baselines["final_norm_logits"]
    ) / 1000
    fixed_seconds = measured_seconds * 2
    expected_ceiling = fixed_seconds * (1 - 1 / MEASURED_INT8_VS_FP16_MLP_SPEEDUP)

    assert report["current_projection_family_seconds"] == pytest.approx(fixed_seconds)
    assert report["live_modeled_realistic_ceiling_seconds"] == pytest.approx(
        expected_ceiling
    )
    buckets["128"]["regions"]["cross_norm_q"]["baseline_ms_per_call"] *= 2
    changed = summarize_component(buckets, total_replays=200)
    assert changed["live_modeled_realistic_ceiling_seconds"] > expected_ceiling


def test_retained_manifest_accepts_exact_live_composition() -> None:
    report = _validate_retained_composition_manifest(_manifest())

    assert report["pass"]
    assert report["k1"] == {
        "records": 2,
        "remainder_steps": 6,
        "remainder_graph_replays": 6,
        "eager_remainder_steps": 0,
        "physical_steps": 20,
        "logical_steps": 17,
        "k4_padding_steps": 3,
    }
    assert report["dispatch_totals"]["int8_weight_mlp_tail"] == 24
    assert report["excluded_already_int8_regions"] == ["mlp_fc1", "mlp_fc2"]


def test_real_tensor_capture_context_installs_and_restores_retained_hooks(
    monkeypatch,
) -> None:
    from osuT5.osuT5.inference.optimized.scout import shared_rope
    from osuT5.osuT5 import runtime_profiling

    events: list[str] = []

    @contextmanager
    def fake_rope(model, *, stats):
        del model
        events.append("rope_enter")
        stats.forwards = 1
        try:
            yield stats
        finally:
            events.append("rope_exit")

    @contextmanager
    def fake_generation(**kwargs):
        events.append("runtime_enter")
        counts = kwargs["optimized_dispatch_counts"]
        counts.update({name: 1 for name in REQUIRED_DISPATCHES})
        try:
            yield
        finally:
            events.append("runtime_exit")

    monkeypatch.setattr(shared_rope, "shared_decoder_rope_context", fake_rope)
    monkeypatch.setattr(runtime_profiling, "generation_profile_context", fake_generation)
    state = SimpleNamespace(validate_owner=lambda model: events.append("validate_owner"))

    with _retained_real_tensor_context(torch.nn.Module(), state, prefix=128):
        events.append("capture")

    assert events == [
        "validate_owner",
        "rope_enter",
        "runtime_enter",
        "capture",
        "runtime_exit",
        "rope_exit",
    ]

    events.clear()
    with pytest.raises(RuntimeError, match="capture failed"):
        with _retained_real_tensor_context(torch.nn.Module(), state, prefix=128):
            raise RuntimeError("capture failed")
    assert events[-2:] == ["runtime_exit", "rope_exit"]


def _set_excess_k4_padding(value: dict) -> None:
    record = value["main_generation_records"][0]
    record["generated_tokens"] = 11
    record["optimized_cuda_graphs"]["k8_candidate"].update(
        physical_steps=15,
        logical_steps=11,
        wasted_steps=4,
    )


def _set_k1_only_padding(value: dict) -> None:
    record = value["main_generation_records"][0]
    record["generated_tokens"] = 4
    record["optimized_cuda_graphs"]["k8_candidate"].update(
        prefill_steps=1,
        eligible_steps=0,
        block_replays=0,
        remainder_steps=4,
        remainder_graph_replays=4,
        physical_steps=5,
        logical_steps=4,
        wasted_steps=1,
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value.update(composition_version="stale"),
            "composition version",
        ),
        (
            lambda value: value["baseline_by_region"].update(cross_norm_q="fp16_weight"),
            "baseline dispatch map",
        ),
        (
            lambda value: value.update(excluded_already_int8_regions=["mlp_fc1"]),
            "exclude the retained INT8 MLP",
        ),
        (
            lambda value: value["weight_state"]["int8_mlp_overlay"].update(
                replaces_effective_regions=["mlp_fc1"]
            ),
            "replaces_effective_regions",
        ),
        (
            lambda value: value["shared_rope"]["stats"].update(reuses=21),
            "reuse accounting",
        ),
        (
            lambda value: value["main_generation_records"][0]["optimized_cuda_graphs"][
                "k8_candidate"
            ].update(remainder_backend="eager"),
            "captured K1",
        ),
        (
            lambda value: value["main_generation_records"][0]["optimized_cuda_graphs"][
                "k8_candidate"
            ].update(remainder_graph_replays=5),
            "own every remainder step",
        ),
        (
            lambda value: value["main_generation_records"][0]["optimized_cuda_graphs"][
                "k8_candidate"
            ].update(eager_remainder_steps=1),
            "fell back to eager",
        ),
        (
            _set_excess_k4_padding,
            "padding exceeds one partial block",
        ),
        (
            _set_k1_only_padding,
            "padding occurred without a K4 block replay",
        ),
        (
            lambda value: value["main_generation_records"][0][
                "optimized_dispatch_capture_hits"
            ].update(int8_weight_mlp_tail=23),
            "dispatch ownership",
        ),
    ],
)
def test_retained_manifest_fails_loudly_on_composition_drift(mutate, message) -> None:
    manifest = _manifest()
    mutate(manifest)

    with pytest.raises(RuntimeError, match=message):
        _validate_retained_composition_manifest(manifest)


def test_summary_selectively_retains_only_positive_incremental_regions() -> None:
    buckets = _buckets(baseline_ms=0.1, candidate_ms=0.08)
    for candidate in buckets["128"]["regions"]["final_norm_logits"]["candidates"].values():
        candidate["ms_per_call"] = 0.12

    report = summarize_component(buckets, total_replays=100)

    assert not report["regions"]["final_norm_logits"]["retained_by_selective_policy"]
    assert report["regions"]["final_norm_logits"]["fixed_work_saving_seconds"] < 0
    assert report["selective_fixed_work_main_saving_seconds"] > 0


def test_summary_fails_loudly_when_every_configuration_is_invalid() -> None:
    buckets = _buckets()
    for candidate in buckets["128"]["regions"]["cross_norm_q"]["candidates"].values():
        candidate["finite"] = False

    with pytest.raises(RuntimeError, match="no valid INT8 configuration for cross_norm_q"):
        summarize_component(buckets, total_replays=100)


def test_summary_rejects_invalid_replay_counts() -> None:
    with pytest.raises(ValueError, match="bucket evidence"):
        summarize_component({}, total_replays=1)
    with pytest.raises(ValueError, match="replay counts"):
        summarize_component(_buckets(), total_replays=99)
