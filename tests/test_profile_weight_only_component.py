from __future__ import annotations

import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from utils.profile_weight_only_component import (
    BASELINE_MAIN_SECONDS,
    BASELINE_MAIN_TOKENS,
    DECODER_LAYERS,
    EXPECTED_DECODE_REPLAYS,
    EXPECTED_PREFILLS,
    OUTPUTS_PER_BLOCK_CHOICES,
    REGIONS,
    _candidate_configurations,
    _select_weighted_outputs_per_block,
    _selective_weight_only_prefix,
    _validate_live_workload,
    summarize_weight_only_component,
)


def _candidate(
    *,
    ms: float,
    drift: float = 0.1,
    stable: bool = True,
    finite: bool = True,
) -> dict:
    return {
        "ms_per_call": ms,
        "saving_ms_per_call": 0.1 - ms,
        "graph_capture_seconds": 0.02,
        "absolute_peak_vram_bytes": 120,
        "memory_stable": stable,
        "finite": finite,
        "output_max_abs_drift": drift,
    }


def _sweep(
    *,
    region: str,
    candidate_ms: dict[str, float] | None = None,
    drift: float = 0.1,
) -> dict:
    configurations = _candidate_configurations(region)
    candidate_ms = candidate_ms or {
        key: 0.05 + index * 0.001
        for index, key in enumerate(configurations)
    }
    return {
        "accepted": {
            "ms_per_call": 0.1,
            "graph_capture_seconds": 0.01,
            "absolute_peak_vram_bytes": 100,
            "memory_stable": True,
            "finite": True,
        },
        "candidates": {
            key: _candidate(ms=candidate_ms[key], drift=drift)
            for key in configurations
        },
        "rounds": [],
    }


def _selected(
    *,
    accepted: float = 0.1,
    candidate: float = 0.05,
    drift: float = 0.1,
    stable: bool = True,
    finite: bool = True,
    region: str = "self_norm_qkv",
) -> dict:
    candidate_key, configuration = next(iter(_candidate_configurations(region).items()))
    return {
        "selected_candidate_key": candidate_key,
        "selected_configuration": configuration,
        "accepted_ms_per_call": accepted,
        "candidate_ms_per_call": candidate,
        "saving_ms_per_call": accepted - candidate,
        "accepted_graph_capture_seconds": 0.01,
        "candidate_graph_capture_seconds": 0.02,
        "accepted_absolute_peak_vram_bytes": 100,
        "candidate_absolute_peak_vram_bytes": 120,
        "accepted_memory_stable": True,
        "candidate_memory_stable": stable,
        "accepted_finite": True,
        "candidate_finite": finite,
        "output_max_abs_drift": drift,
        "opb_sweep": _sweep(region=region, drift=drift),
    }


def _bucket(*, count: int, accepted: float = 0.5, candidate: float = 0.2) -> dict:
    return {
        "decode_replays": count,
        "regions": {name: _sweep(region=name) for name in REGIONS},
        "regions_selected": {
            name: _selected(region=name) for name in REGIONS
        },
        "selected_final_norm_logits": _selected(region="final_norm_logits"),
        "prefix_total": {
            "accepted_ms_per_call": accepted,
            "candidate_ms_per_call": candidate,
            "saving_ms_per_call": accepted - candidate,
            "accepted_graph_capture_seconds": 0.1,
            "candidate_graph_capture_seconds": 0.2,
            "accepted_absolute_peak_vram_bytes": 180,
            "candidate_absolute_peak_vram_bytes": 200,
            "accepted_memory_stable": True,
            "candidate_memory_stable": True,
            "candidate_finite": True,
            "candidate_cache_behavior_pass": True,
            "output_max_abs_drift": 0.25,
            "cache_key_slot_max_abs_drift": 0.01,
            "cache_value_slot_max_abs_drift": 0.02,
            "candidate_checks": {"pass": True},
            "rounds": [],
        },
    }


def _live_run(*, generated_tokens: int, counts: dict[int, int]) -> dict:
    graph_cache = {
        (prefix,): {
            "active_prefix_length": prefix,
            "graph": object(),
            "outputs": object(),
            "static_inputs": {},
            "decode_replays": count,
        }
        for prefix, count in counts.items()
    }
    return {
        "processor": SimpleNamespace(
            last_generation_stats={"generated_tokens": generated_tokens}
        ),
        "session": SimpleNamespace(graph_cache=graph_cache),
    }


def test_live_workload_uses_dynamic_bucket_counts_but_requires_fixed_work() -> None:
    counts = {128: 2_000, 832: EXPECTED_DECODE_REPLAYS - 2_000}
    entries, manifest = _validate_live_workload(
        _live_run(generated_tokens=BASELINE_MAIN_TOKENS, counts=counts)
    )

    assert tuple(entries) == (128, 832)
    assert manifest == {
        "generated_tokens": BASELINE_MAIN_TOKENS,
        "decode_replays": EXPECTED_DECODE_REPLAYS,
        "prefills": EXPECTED_PREFILLS,
        "bucket_counts": {"128": 2_000, "832": EXPECTED_DECODE_REPLAYS - 2_000},
    }


@pytest.mark.parametrize(
    ("generated_tokens", "counts", "message"),
    [
        (BASELINE_MAIN_TOKENS - 1, {128: EXPECTED_DECODE_REPLAYS}, "token count"),
        (
            BASELINE_MAIN_TOKENS,
            {128: EXPECTED_DECODE_REPLAYS - 1},
            "decode workload",
        ),
    ],
)
def test_live_workload_fails_loudly_when_fixed_work_changes(
    generated_tokens: int,
    counts: dict[int, int],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _validate_live_workload(
            _live_run(generated_tokens=generated_tokens, counts=counts)
        )


def test_weighted_opb_selection_is_global_across_live_buckets() -> None:
    first = _bucket(count=9)
    second = _bucket(count=1)
    for region in REGIONS:
        configurations = _candidate_configurations(region)
        preferred = next(iter(configurations))
        first_ms = {key: 0.08 for key in configurations}
        second_ms = {key: 0.08 for key in configurations}
        first_ms[preferred] = 0.02
        second_ms[preferred] = 0.2
        first["regions"][region] = _sweep(region=region, candidate_ms=first_ms)
        second["regions"][region] = _sweep(region=region, candidate_ms=second_ms)

    selection = _select_weighted_outputs_per_block({"128": first, "832": second})

    assert all(
        value["selected_candidate_key"]
        == next(iter(_candidate_configurations(region)))
        for region, value in selection.items()
    )
    assert selection["cross_norm_q"]["selective_combined_dispatch"] == "accepted_fp32"
    assert selection["mlp"]["selective_combined_dispatch"] == "fp16_weight"


def test_summary_uses_combined_prefix_not_sum_of_isolated_region_savings() -> None:
    buckets = {"128": _bucket(count=1_000), "640": _bucket(count=500)}

    report = summarize_weight_only_component(buckets, total_replays=1_500)

    expected = 1_500 * DECODER_LAYERS * (0.5 - 0.2) / 1_000
    expected += 1_500 * (0.1 - 0.05) / 1_000
    assert report["weighted_fixed_work_saving_seconds"] == expected
    assert report["coverage_fraction"] == 1.0
    assert report["sizing_pass"]
    assert report["structural_runtime_pass"]
    assert report["numeric_drift"]["selected_combined"]["maximum_observed_abs"] == 0.25
    assert report["numeric_drift"]["exactness_claim"] is False
    assert report["projected_main_seconds"] == BASELINE_MAIN_SECONDS - expected
    assert report["projected_main_tps_at_8294_tokens"] == (
        BASELINE_MAIN_TOKENS / (BASELINE_MAIN_SECONDS - expected)
    )


def test_numeric_drift_is_separate_from_structural_runtime_pass() -> None:
    bucket = _bucket(count=100)
    bucket["regions_selected"]["mlp"]["output_max_abs_drift"] = 4.25

    report = summarize_weight_only_component({"128": bucket}, total_replays=100)

    assert report["numeric_drift"]["diagnostic_maximum_region_output_abs"] == 4.25
    assert report["numeric_drift"]["selected_combined"]["maximum_observed_abs"] == 0.25
    assert report["structural_runtime_pass"]
    assert "correctness_pass" not in report


def test_summary_rejects_nonfinite_memory_and_cache_behavior_failures() -> None:
    bucket = _bucket(count=100)
    bucket["regions_selected"]["mlp"]["candidate_finite"] = False
    bucket["regions_selected"]["self_norm_qkv"]["candidate_memory_stable"] = False
    bucket["prefix_total"]["candidate_cache_behavior_pass"] = False

    report = summarize_weight_only_component({"128": bucket}, total_replays=100)

    assert not report["structural_runtime_pass"]
    assert report["structural_runtime_failures"] == {
        "128": [
            "self_norm_qkv:memory_unstable",
            "mlp:nonfinite",
            "prefix_total:cache_behavior_failed",
        ]
    }


def test_summary_requires_every_live_bucket() -> None:
    with pytest.raises(ValueError, match="all live replay buckets"):
        summarize_weight_only_component({"128": _bucket(count=100)}, total_replays=101)


def test_summary_fails_loudly_on_empty_or_invalid_counts() -> None:
    for buckets, total in (({}, 1), ({"128": _bucket(count=1)}, 0)):
        with pytest.raises(ValueError):
            summarize_weight_only_component(buckets, total_replays=total)

    invalid = _bucket(count=0)
    with pytest.raises(ValueError, match="non-positive replay count"):
        summarize_weight_only_component({"128": invalid}, total_replays=1)


def test_projection_and_setup_evidence_are_finite() -> None:
    report = summarize_weight_only_component(
        {"128": _bucket(count=1_000)}, total_replays=1_000
    )
    assert math.isfinite(report["projected_main_tps_at_8294_tokens"])
    assert report["setup"]["selected_candidate_graph_recapture_seconds"] > 0
    assert report["vram"]["selected_candidate_absolute_peak_bytes"] == 200


def test_selective_prefix_keeps_accepted_cross_query_output_and_bmm() -> None:
    source = inspect.getsource(_selective_weight_only_prefix)

    assert "native_one_token_rmsnorm_linear" in source
    assert "framework_q1_attention" in source
    assert "native_one_token_linear_residual" in source
    assert "pack.cross_q" not in source
    assert "pack.cross_out" not in source
    assert "weight_only_mlp_residual" in source


def test_dcc_wrapper_requires_clean_exact_branch_and_profiles_all_buckets() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/dcc/profile_weight_only_component.sbatch").read_text()

    assert "MAPPERATORINATOR_REPO:?" in source
    assert "MAPPERATORINATOR_COMMIT:?" in source
    assert "MAPPERATORINATOR_BRANCH:?" in source
    assert 'git -C "$REPO" status --porcelain' in source
    assert 'git -C "$REPO" rev-parse HEAD' in source
    assert 'git -C "$REPO" branch --show-current' in source
    assert "NVIDIA GeForce RTX 2080 Ti" in source
    assert 'RUN_ROOT="$WORK/runs/' in source
    assert "--bucket-mode all" in source
    assert "--warmup 100" in source
    assert "--iters 1000" in source
    assert "sbatch" not in source
