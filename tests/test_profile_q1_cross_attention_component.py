from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest
import torch

from osuT5.osuT5.inference.optimized.scout.hybrid_qk_cross import (
    hybrid_qk_fp32_value_attention,
)
from utils.profile_q1_cross_attention_component import (
    COMPONENT_SAVING_GATE_SECONDS,
    RELAXED_MAX_ABS_DRIFT,
    SELECTED_MAIN_SECONDS,
    SENTINEL_PREFIXES,
    SESSION_LABELS,
    summarize_hybrid_component,
)


CURRENT = "selected_packed_wq_wo_fp32_kv"
HYBRID = "hybrid_packed_wq_fp16_k_fp32_v_fp32_wo"


def test_live_decode_session_labels_match_processor_profile_contract() -> None:
    assert SESSION_LABELS == ("timing_context", "main_generation")


def _buckets(*, current_ms: float = 0.5, hybrid_ms: float = 0.4, drift: float = 0.001):
    return {
        prefix: {
            "variants": {
                CURRENT: {"ms_per_decode_step": current_ms, "checks_pass": True},
                HYBRID: {
                    "ms_per_decode_step": hybrid_ms,
                    "max_abs_drift_vs_selected": drift,
                    "max_abs_drift_vs_fp32": drift * 2,
                    "checks_pass": True,
                },
            }
        }
        for prefix in SENTINEL_PREFIXES
    }


def test_summary_weights_live_sentinels_and_assigns_zero_delta_elsewhere() -> None:
    main_counts = {128: 1000, 576: 1000, 640: 1000, 832: 5000}
    timing_counts = {128: 100, 576: 100, 640: 100, 832: 500}
    report = summarize_hybrid_component(
        _buckets(hybrid_ms=0.39), main_counts=main_counts, timing_counts=timing_counts
    )

    assert report["main_saving_seconds"] == pytest.approx(0.33)
    assert report["timing_saving_seconds"] == pytest.approx(0.033)
    assert report["raw_main_replay_saving_seconds"] == pytest.approx(0.33)
    assert report["projected_fixed_work_seconds"] == pytest.approx(
        SELECTED_MAIN_SECONDS - 0.33
    )
    assert report["unmeasured_main_buckets"] == [832]
    assert report["unmeasured_bucket_delta_policy"] == "zero_delta"
    assert report["sizing_pass"]
    assert report["component_retention_pass"]
    assert not report["production_promotion_pass"]

    charged = summarize_hybrid_component(
        _buckets(hybrid_ms=0.39),
        main_counts=main_counts,
        timing_counts={64: 500, 128: 100},
        main_key_pack_setup_seconds=0.02,
    )
    assert charged["timing_saving_seconds"] == pytest.approx(0.011)
    assert charged["main_saving_seconds"] == pytest.approx(0.31)
    assert charged["sizing_pass"]
    assert charged["per_bucket"]["576"]["timing_replays"] == 0
    assert charged["per_bucket"]["640"]["timing_replays"] == 0


def test_summary_rejects_partial_coverage_invalid_counts_and_excess_drift() -> None:
    counts = {128: 1000, 576: 1000, 640: 1000}
    partial = _buckets()
    partial.pop(576)
    with pytest.raises(ValueError, match="sentinel"):
        summarize_hybrid_component(partial, main_counts=counts, timing_counts=counts)

    with pytest.raises(ValueError, match="positive"):
        summarize_hybrid_component(
            _buckets(), main_counts={**counts, 128: 0}, timing_counts=counts
        )

    report = summarize_hybrid_component(
        _buckets(drift=RELAXED_MAX_ABS_DRIFT + 1e-6),
        main_counts=counts,
        timing_counts=counts,
    )
    assert not report["checks_pass"]
    assert not report["component_retention_pass"]


def test_summary_requires_realistic_point_three_second_main_ceiling() -> None:
    counts = {128: 1000, 576: 1000, 640: 1000}
    report = summarize_hybrid_component(
        _buckets(hybrid_ms=0.401), main_counts=counts, timing_counts=counts
    )

    assert report["main_saving_seconds"] < COMPONENT_SAVING_GATE_SECONDS
    assert not report["sizing_pass"]


def test_hybrid_qk_keeps_value_reduction_and_output_fp32() -> None:
    query = torch.tensor([[[[1.0, -2.0]], [[0.5, 3.0]]]], dtype=torch.float32)
    key_fp32 = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[-1.0, 0.5], [0.5, 1.0], [2.0, -1.0]]]],
        dtype=torch.float32,
    )
    key = key_fp32.half().contiguous()
    value = (key_fp32 + 2.0).contiguous()

    output = hybrid_qk_fp32_value_attention(query.contiguous(), key, value)
    q = query.reshape(2, 1, 2).half().float()
    k = key.reshape(2, 3, 2).float()
    v = value.reshape(2, 3, 2)
    expected = torch.bmm(
        torch.softmax(torch.bmm(q, k.transpose(1, 2)) * (2**-0.5), dim=-1),
        v,
    ).view(1, 2, 1, 2)

    assert output.dtype == torch.float32
    assert torch.equal(output, expected)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("query_dtype", "key_dtype", "value_dtype", "message"),
    [
        (torch.float16, torch.float16, torch.float32, "query activation"),
        (torch.float32, torch.float32, torch.float32, "key cache"),
        (torch.float32, torch.float16, torch.float16, "value cache"),
    ],
)
def test_hybrid_qk_rejects_wrong_storage_dtypes(
    query_dtype, key_dtype, value_dtype, message
) -> None:
    query = torch.zeros((1, 2, 1, 4), dtype=query_dtype)
    key = torch.zeros((1, 2, 3, 4), dtype=key_dtype)
    value = torch.zeros((1, 2, 3, 4), dtype=value_dtype)

    with pytest.raises(TypeError, match=message):
        hybrid_qk_fp32_value_attention(query, key, value)


def test_hybrid_qk_rejects_shape_and_contiguity_errors() -> None:
    query = torch.zeros((1, 2, 1, 4), dtype=torch.float32)
    key = torch.zeros((1, 2, 3, 4), dtype=torch.float16)
    value = torch.zeros((1, 2, 3, 4), dtype=torch.float32)
    with pytest.raises(ValueError, match="matching 4D"):
        hybrid_qk_fp32_value_attention(query, key[:, :, :2], value)
    noncontiguous_key = torch.zeros((1, 2, 4, 3), dtype=torch.float16).transpose(-1, -2)
    noncontiguous_value = torch.zeros((1, 2, 4, 3), dtype=torch.float32).transpose(-1, -2)
    with pytest.raises(ValueError, match="contiguous"):
        hybrid_qk_fp32_value_attention(query, noncontiguous_key, noncontiguous_value)


def test_hybrid_scout_is_not_imported_by_default_optimized_package() -> None:
    module_name = "osuT5.osuT5.inference.optimized.scout.hybrid_qk_cross"
    sys.modules.pop(module_name, None)
    importlib.import_module("osuT5.osuT5.inference.optimized")

    assert module_name not in sys.modules


def test_dcc_wrapper_is_exact_commit_sm75_and_json_text_only() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts/dcc/profile_q1_cross_attention_component.sbatch"
    ).read_text(encoding="utf-8")

    assert "MAPPERATORINATOR_COMMIT" in source
    assert "MAPPERATORINATOR_BRANCH" in source
    assert '"$REMOTE/$BRANCH"' in source
    assert "NVIDIA GeForce RTX 2080 Ti" in source
    assert '"7.5"' in source
    assert "--warmup 100" in source
    assert "--iters 1000" in source
    assert "hybrid-qk-cross-component" in source
    assert "component.json" in source
    assert "component.txt" in source
    assert "--capture-prefix" not in source
    assert ".md" not in source
    assert ".html" not in source
    assert ".png" not in source
