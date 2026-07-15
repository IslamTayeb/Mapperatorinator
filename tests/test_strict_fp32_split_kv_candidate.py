from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from osuT5.osuT5.inference.optimized.single.engine import (
    OPTIMIZED_PRESETS,
    SPLIT_KV_Q1_PREFIX_BUCKETS,
    SPLIT_KV_Q1_SPLIT_COUNT,
    _optimized_config_metadata,
)
from utils.analyze_strict_fp32_candidate import (
    CandidateGateError,
    _declared_policy_delta,
    _load_dispatch_delta_spec,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "scripts/dcc/strict_fp32_split_kv_dispatch_delta.json"
PREFIXES = tuple(range(192, 833, 64))
CONFIG_PATHS = (
    "preset.optimized_effective_config.native_q1_rope_cache_split_kv",
    "preset.optimized_effective_config.native_q1_rope_cache_split_kv_prefix_buckets",
    "preset.optimized_effective_config.native_q1_rope_cache_split_kv_split_count",
)
DISPATCH_PATHS = (
    "records.*.optimized_dispatch_capture_hits."
    "native_q1_rope_cache_self_attention_split_kv_8",
    *(
        "records.*.optimized_dispatch_capture_hits."
        f"native_q1_rope_cache_self_attention_split_kv_8_prefix_{prefix}"
        for prefix in PREFIXES
    ),
)
EXPECTED_PATHS = sorted((*CONFIG_PATHS, *DISPATCH_PATHS))


def _baseline_contract() -> dict:
    return {
        "preset": {
            "optimized_effective_config_version": (
                "accepted-fp32-native-cross-mlp-289-v3"
            ),
            "optimized_effective_config": {
                "precision": "fp32",
                "native_q1_rope_cache_self_attention": True,
                "q1_bmm_cross_attention": True,
            },
        },
        "dispatch_and_graph": [
            {
                "profile_label": "main_generation",
                "optimized_dispatch_capture_hits": {
                    "native_q1_rope_cache_self_attention": 144,
                    "q1_bmm_cross_attention": 144,
                },
            }
        ],
    }


def _candidate_contract() -> dict:
    contract = copy.deepcopy(_baseline_contract())
    config = contract["preset"]["optimized_effective_config"]
    config.update(
        {
            "native_q1_rope_cache_split_kv": True,
            "native_q1_rope_cache_split_kv_prefix_buckets": list(PREFIXES),
            "native_q1_rope_cache_split_kv_split_count": 8,
        }
    )
    hits = contract["dispatch_and_graph"][0][
        "optimized_dispatch_capture_hits"
    ]
    hits["native_q1_rope_cache_self_attention_split_kv_8"] = 132
    for prefix in PREFIXES:
        hits[
            f"native_q1_rope_cache_self_attention_split_kv_8_prefix_{prefix}"
        ] = 12
    return contract


def test_checked_in_allowlist_is_exactly_the_split_kv_policy_surface() -> None:
    raw = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

    assert raw == {
        "schema_version": "mapperatorinator.dispatch-delta-allowlist.v1",
        "allowed_paths": EXPECTED_PATHS,
        "required_paths": EXPECTED_PATHS,
    }
    parsed = _load_dispatch_delta_spec(SPEC_PATH)
    assert parsed["allowed_paths"] == EXPECTED_PATHS
    assert parsed["required_paths"] == EXPECTED_PATHS
    assert parsed["source_sha256"]


def test_allowlist_accepts_all_and_only_split_kv_metadata_deltas() -> None:
    spec = _load_dispatch_delta_spec(SPEC_PATH)
    result = _declared_policy_delta(
        _baseline_contract(),
        _candidate_contract(),
        spec,
    )

    assert result["pass"] is True
    assert result["observed_paths"] == EXPECTED_PATHS

    unrelated = _candidate_contract()
    unrelated["dispatch_and_graph"][0]["optimized_dispatch_capture_hits"][
        "q1_bmm_cross_attention"
    ] = 143
    with pytest.raises(CandidateGateError, match="undeclared=.*q1_bmm_cross_attention"):
        _declared_policy_delta(_baseline_contract(), unrelated, spec)

    incomplete = _candidate_contract()
    del incomplete["dispatch_and_graph"][0]["optimized_dispatch_capture_hits"][
        "native_q1_rope_cache_self_attention_split_kv_8_prefix_832"
    ]
    with pytest.raises(CandidateGateError, match="missing_required=.*prefix_832"):
        _declared_policy_delta(_baseline_contract(), incomplete, spec)


def test_fp32_runtime_metadata_matches_the_checked_split_kv_contract() -> None:
    config = _optimized_config_metadata(OPTIMIZED_PRESETS["fp32"])

    assert SPLIT_KV_Q1_SPLIT_COUNT == 8
    assert SPLIT_KV_Q1_PREFIX_BUCKETS == PREFIXES
    assert config["native_q1_rope_cache_split_kv"] is True
    assert config["native_q1_rope_cache_split_kv_split_count"] == 8
    assert config["native_q1_rope_cache_split_kv_prefix_buckets"] == PREFIXES
