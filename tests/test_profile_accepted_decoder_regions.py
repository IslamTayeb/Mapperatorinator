from types import SimpleNamespace

import pytest

from utils.profile_accepted_decoder_regions import (
    REGIONS,
    REQUIRED_HEADROOM_SECONDS,
    SCHEMA_VERSION,
    _accepted_context,
    _validate_report,
    aggregate_region_times,
    classify_detail_range,
    summarize_weighted_ceiling,
)
from utils.profile_native_prefix_dtype_scout import SENTINEL_BUCKETS


def _bucket(
    *,
    region_ms: float = 0.1,
    event_ms: float = 1.0,
    production_graph_ms: float = 0.5,
):
    regions = {name: region_ms for name in REGIONS}
    scale = production_graph_ms / event_ms
    assigned = sum(regions.values())
    return {
        "production_graph_ms_per_call": production_graph_ms,
        "cuda_event_ms_per_call": {
            "mean": event_ms,
            "minimum": event_ms - 0.1,
            "maximum": event_ms + 0.1,
        },
        "regions_ms_per_call": regions,
        "production_calibrated_regions_ms_per_call": {
            name: value * scale for name, value in regions.items()
        },
        "eager_to_production_graph_scale": scale,
        "assigned_region_ms_per_call": assigned,
        "unattributed_ms_per_call": max(0.0, event_ms - assigned),
        "over_attributed_ms_per_call": max(0.0, assigned - event_ms),
    }


def _report(*, region_ms: float = 0.1):
    buckets = {str(prefix): _bucket(region_ms=region_ms) for prefix in SENTINEL_BUCKETS}
    counts = {prefix: 1 for prefix in SENTINEL_BUCKETS}
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {"measured_buckets": list(SENTINEL_BUCKETS)},
        "buckets": buckets,
        "weighted": summarize_weighted_ceiling(buckets, live_counts=counts),
    }


def test_accepted_context_restores_all_specialized_fp32_dispatch(monkeypatch):
    from osuT5.osuT5 import runtime_profiling

    sentinel = object()
    calls = []

    def fake_context(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(runtime_profiling, "generation_profile_context", fake_context)

    assert _accepted_context(640, detail_ranges=True) is sentinel
    assert calls == [
        {
            "detail_ranges": True,
            "active_prefix_self_attention_length": 640,
            "q1_bmm_cross_attention": True,
            "native_q1_self_attention": True,
            "native_q1_rope_cache_self_attention": True,
        }
    ]


@pytest.mark.parametrize(
    ("marker", "region"),
    [
        ("decoder.layer0.self_attn_norm", "self_norm_qkv"),
        ("attention.layer11.self.qkv_proj", "self_norm_qkv"),
        ("attention.layer0.self.rope", "fused_self_attention"),
        ("attention.layer0.self.cache_update", "fused_self_attention"),
        ("attention.layer0.self.sdpa", "fused_self_attention"),
        (
            "attention.layer0.self.rope_cache_native_q1",
            "fused_self_attention",
        ),
        ("attention.layer1.self.out_proj", "self_out_residual"),
        ("decoder.layer2.cross_attn_norm", "cross_norm_q"),
        ("attention.layer2.cross.q_proj", "cross_norm_q"),
        ("attention.layer2.cross.sdpa", "q1_bmm_cross"),
        (
            "attention.layer2.cross.out_proj",
            "cross_out_residual",
        ),
        ("decoder.layer2.self.residual", "self_out_residual"),
        ("decoder.layer2.cross.residual", "cross_out_residual"),
        ("decoder.layer3.mlp_norm", "mlp"),
        ("decoder.layer3.mlp.fc1", "mlp"),
        ("decoder.layer3.mlp.activation", "mlp"),
        ("decoder.layer3.mlp.activation_dropout", "mlp"),
        ("decoder.layer3.mlp.fc2", "mlp"),
        ("decoder.layer3.mlp.output_dropout_residual", "mlp"),
        ("decoder.final_norm", "final_norm_logits"),
        ("decoder.output_projection", "final_norm_logits"),
        ("mapperatorinator.decoder.layer0.self_attn_norm", "self_norm_qkv"),
    ],
)
def test_classifies_only_non_overlapping_detail_ranges(marker, region):
    assert classify_detail_range(marker) == region


@pytest.mark.parametrize(
    "marker",
    [
        "decoder.layer0.self_attn",
        "decoder.layer0.cross_attn",
        "attention.layer0.cross.cache_reuse",
        "encoder.final_norm",
        "aten::linear",
    ],
)
def test_excludes_outer_and_unassigned_ranges(marker):
    assert classify_detail_range(marker) is None


def test_aggregates_device_time_with_cuda_compatibility_fallback():
    events = []
    for index, region in enumerate(REGIONS, start=1):
        marker = {
            "self_norm_qkv": "mapperatorinator.decoder.layer0.self_attn_norm",
            "fused_self_attention": "mapperatorinator.attention.layer0.self.rope",
            "self_out_residual": "mapperatorinator.attention.layer0.self.out_proj",
            "cross_norm_q": "mapperatorinator.decoder.layer0.cross_attn_norm",
            "q1_bmm_cross": "mapperatorinator.attention.layer0.cross.sdpa",
            "cross_out_residual": "mapperatorinator.attention.layer0.cross.out_proj",
            "mlp": "mapperatorinator.decoder.layer0.mlp.fc1",
            "final_norm_logits": "mapperatorinator.decoder.final_norm",
        }[region]
        if index % 2:
            event = SimpleNamespace(key=marker, device_time_total=index * 2_000.0)
        else:
            event = SimpleNamespace(key=marker, cuda_time_total=index * 2_000.0)
        events.append(event)
    events.append(SimpleNamespace(key="aten::linear", device_time_total=99_000.0))

    regions, raw = aggregate_region_times(events, iterations=2)

    assert tuple(regions) == REGIONS
    assert regions == pytest.approx(
        {region: float(index) for index, region in enumerate(REGIONS, start=1)}
    )
    assert "aten::linear" not in raw


def test_aggregation_fails_when_a_required_region_is_missing_or_invalid():
    events = [
        SimpleNamespace(
            key="mapperatorinator.decoder.layer0.self_attn_norm",
            device_time_total=1.0,
        )
    ]
    with pytest.raises(RuntimeError, match="did not record positive device time"):
        aggregate_region_times(events, iterations=1)
    with pytest.raises(ValueError, match="iterations must be positive"):
        aggregate_region_times(events, iterations=0)


def test_weighted_ceiling_uses_live_sentinel_counts_and_declared_threshold():
    buckets = {
        str(prefix): _bucket(
            region_ms=0.1 * index,
            event_ms=2.0,
            production_graph_ms=0.5,
        )
        for index, prefix in enumerate(SENTINEL_BUCKETS, start=1)
    }
    counts = {128: 2, 576: 3, 640: 5, 832: 99}

    weighted = summarize_weighted_ceiling(buckets, live_counts=counts)

    per_region = (2 * 0.025 + 3 * 0.05 + 5 * 0.075) / 1_000.0
    assert weighted["regions_seconds"] == pytest.approx(
        {region: per_region for region in REGIONS}
    )
    assert weighted["optimistic_removable_seconds"] == pytest.approx(
        len(REGIONS) * per_region
    )
    assert weighted["measured_production_graph_seconds"] == pytest.approx(
        (2 + 3 + 5) * 0.5 / 1_000
    )
    assert weighted["required_headroom_seconds"] == REQUIRED_HEADROOM_SECONDS
    assert weighted["unmeasured_buckets_assumed_seconds"] == 0.0
    assert weighted["clears_required_headroom"] is False
    assert weighted["largest_region_seconds"] == pytest.approx(per_region)
    assert all(
        not decision["clears_1_412s"]
        for decision in weighted["region_decisions"].values()
    )


def test_report_validation_fails_loudly_on_bad_decision_or_region_schema():
    report = _report(region_ms=1_000.0)
    _validate_report(report)

    report["weighted"]["clears_required_headroom"] = False
    with pytest.raises(ValueError, match="decision is inconsistent"):
        _validate_report(report)

    report = _report()
    report["buckets"]["128"]["regions_ms_per_call"].pop("mlp")
    with pytest.raises(ValueError, match="exact region order"):
        _validate_report(report)
