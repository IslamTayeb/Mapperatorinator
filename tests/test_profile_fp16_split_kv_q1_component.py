from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

import utils.profile_fp16_split_kv_q1_component as profile


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/dcc/profile_fp16_split_kv_q1_component.sbatch"


def _graph_entry(prefix: int, count: int) -> dict:
    return {
        "active_prefix_length": prefix,
        "decode_replays": count,
        "graph": object(),
        "outputs": object(),
        "static_inputs": {},
    }


def _accepted(ms: float = 0.2) -> dict:
    return {
        "ms_per_call": ms,
        "memory_stable": True,
        "cache_verifier": {"pass": True},
    }


def _split(ms: float = 0.1, drift: float = 0.0) -> dict:
    return {
        "ms_per_call": ms,
        "memory_stable": True,
        "cache_verifier": {"pass": True},
        "drift": {
            "output_max_abs": drift,
            "cache_key_slot_max_abs": 0.0,
            "cache_value_slot_max_abs": 0.0,
        },
    }


def test_fp16_args_fail_loudly_on_precision_or_runtime_changes() -> None:
    args = SimpleNamespace(
        inference_engine="optimized",
        precision="fp16",
        attn_implementation="sdpa",
        device="cuda",
        use_server=False,
        parallel=False,
        cfg_scale=1.0,
        num_beams=1,
    )
    profile.validate_fp16_args(args)
    args.precision = "fp32"
    with pytest.raises(ValueError, match="precision"):
        profile.validate_fp16_args(args)


def test_live_graph_cache_uses_dynamic_counts_and_requires_sentinels() -> None:
    cache = {
        (128,): _graph_entry(128, 3),
        (576,): _graph_entry(576, 4),
        (640,): _graph_entry(640, 5),
        (832,): _graph_entry(832, 2),
    }
    entries = profile.validate_live_graph_cache(cache)
    assert list(entries) == [128, 576, 640, 832]
    assert entries[832]["decode_replays"] == 2

    cache.pop((576,))
    with pytest.raises(ValueError, match="missing sentinel"):
        profile.validate_live_graph_cache(cache)


def test_summary_weights_live_counts_and_keeps_128_accepted_fallback() -> None:
    buckets = {
        "128": {
            "selector": "accepted_fallback",
            "accepted_fp16": _accepted(0.1),
        },
        "576": {
            "selector": "split_kv_8",
            "accepted_fp16": _accepted(0.2),
            "split_kv_8_fp16": _split(0.1),
        },
        "640": {
            "selector": "split_kv_8",
            "accepted_fp16": _accepted(0.3),
            "split_kv_8_fp16": _split(0.15),
        },
    }
    report = profile.summarize(
        buckets,
        live_counts={128: 10, 576: 20, 640: 30, 832: 40},
    )
    assert report["selected"] == {
        "128": "accepted_fallback",
        "576": "split_kv_8_fp16",
        "640": "split_kv_8_fp16",
    }
    assert report["coverage_fraction"] == pytest.approx(0.6)
    assert report["projected_main_saving_seconds"] == pytest.approx(0.078)
    assert report["invariants_pass"]
    assert report["correctness_pass"]
    assert report["sizing_pass"]
    assert report["promotion_pass"]
    assert profile.report_exit_code(report) == 0


def test_summary_rejects_drift_even_when_candidate_is_faster() -> None:
    buckets = {
        "128": {
            "selector": "accepted_fallback",
            "accepted_fp16": _accepted(),
        },
        "576": {
            "selector": "split_kv_8",
            "accepted_fp16": _accepted(),
            "split_kv_8_fp16": _split(drift=profile.MAX_ABS_DRIFT + 1e-6),
        },
        "640": {
            "selector": "split_kv_8",
            "accepted_fp16": _accepted(),
            "split_kv_8_fp16": _split(),
        },
    }
    report = profile.summarize(
        buckets,
        live_counts={128: 1, 576: 1, 640: 1},
    )
    assert report["candidate_invariant_failures"]["576"] == [
        f"output_max_abs>{profile.MAX_ABS_DRIFT}"
    ]
    assert report["performance_fallbacks"] == {}
    assert not report["invariants_pass"]
    assert not report["correctness_pass"]
    # Sizing is an independent performance signal from the still-valid 640
    # bucket; the invariant gate remains authoritative and forces exit 1.
    assert report["sizing_pass"]
    assert not report["promotion_pass"]
    assert profile.report_exit_code(report) == 1


def test_summary_treats_not_faster_as_valid_performance_only_fallback() -> None:
    buckets = {
        "128": {
            "selector": "accepted_fallback",
            "accepted_fp16": _accepted(0.1),
        },
        "576": {
            "selector": "split_kv_8",
            "accepted_fp16": _accepted(0.2),
            "split_kv_8_fp16": _split(0.21),
        },
        "640": {
            "selector": "split_kv_8",
            "accepted_fp16": _accepted(0.3),
            "split_kv_8_fp16": _split(0.3),
        },
    }

    report = profile.summarize(
        buckets,
        live_counts={128: 10, 576: 20, 640: 30},
    )

    assert report["candidate_invariant_failures"] == {}
    assert report["performance_fallbacks"] == {
        "576": ["not_faster"],
        "640": ["not_faster"],
    }
    assert report["selected"] == {
        "128": "accepted_fallback",
        "576": "accepted_fallback",
        "640": "accepted_fallback",
    }
    assert report["invariants_pass"]
    assert report["correctness_pass"]
    assert not report["sizing_pass"]
    assert not report["promotion_pass"]
    assert profile.report_exit_code(report) == 3


def test_summary_can_promote_a_valid_per_bucket_speed_selector() -> None:
    buckets = {
        "128": {
            "selector": "accepted_fallback",
            "accepted_fp16": _accepted(0.1),
        },
        "576": {
            "selector": "split_kv_8",
            "accepted_fp16": _accepted(0.2),
            "split_kv_8_fp16": _split(0.21),
        },
        "640": {
            "selector": "split_kv_8",
            "accepted_fp16": _accepted(0.3),
            "split_kv_8_fp16": _split(0.15),
        },
    }

    report = profile.summarize(
        buckets,
        live_counts={128: 10, 576: 20, 640: 30},
    )

    assert report["performance_fallbacks"] == {"576": ["not_faster"]}
    assert report["selected"]["576"] == "accepted_fallback"
    assert report["selected"]["640"] == "split_kv_8_fp16"
    assert report["invariants_pass"]
    assert report["sizing_pass"]
    assert report["promotion_pass"]
    assert profile.report_exit_code(report) == 0


def test_report_exit_code_rejects_inconsistent_promotion_decision() -> None:
    with pytest.raises(ValueError, match="decision disagrees"):
        profile.report_exit_code(
            {
                "invariants_pass": True,
                "sizing_pass": True,
                "promotion_pass": False,
            }
        )


def test_summary_fails_loudly_when_accepted_control_is_invalid() -> None:
    accepted = _accepted()
    accepted["cache_verifier"] = {"pass": False}
    buckets = {
        "128": {
            "selector": "accepted_fallback",
            "accepted_fp16": accepted,
        },
        "576": {
            "selector": "split_kv_8",
            "accepted_fp16": _accepted(),
            "split_kv_8_fp16": _split(),
        },
        "640": {
            "selector": "split_kv_8",
            "accepted_fp16": _accepted(),
            "split_kv_8_fp16": _split(),
        },
    }
    with pytest.raises(ValueError, match="accepted cache verifier failed"):
        profile.summarize(
            buckets,
            live_counts={128: 1, 576: 1, 640: 1},
        )


def test_reciprocal_timing_alternates_order_and_restores(monkeypatch) -> None:
    timings = {"accepted_fp16": 0.2, "split_kv_8_fp16": 0.1}
    restores = []

    monkeypatch.setattr(
        profile,
        "_time_graph",
        lambda graph, *, warmup, iters: {
            "ms_per_call": timings[graph],
            "memory_stable": True,
            "allocated_before_bytes": 1,
            "allocated_after_bytes": 1,
        },
    )
    medians, rows, memory = profile.reciprocal_timings(
        {name: name for name in timings},
        restore=lambda: restores.append(True),
        warmup=2,
        iters=3,
        cycles=4,
    )
    assert medians == timings
    assert len(rows) == len(restores) == 8
    assert rows[0]["order"] == ["accepted_fp16", "split_kv_8_fp16"]
    assert rows[2]["order"] == ["split_kv_8_fp16", "accepted_fp16"]
    assert memory == {"accepted_fp16": True, "split_kv_8_fp16": True}


def test_dcc_wrapper_is_fp16_sentinel_only_and_fail_loud() -> None:
    completed = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    source = WRAPPER.read_text(encoding="utf-8")
    for required in (
        "MAPPERATORINATOR_REPO:?",
        "MAPPERATORINATOR_COMMIT:?",
        "MAPPERATORINATOR_BRANCH:?",
        'rev-parse "$REMOTE/$BRANCH"',
        "MAPPERATORINATOR_REMOTE:-origin",
        "NVIDIA GeForce RTX 2080 Ti",
        "precision=fp16",
        "profile_fp16_split_kv_q1_component.py",
        '[[ ! -s "$RUN_ROOT/component.json" ]]',
    ):
        assert required in source
    assert "sbatch" not in source
