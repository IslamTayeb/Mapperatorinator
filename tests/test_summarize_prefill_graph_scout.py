import copy

import pytest

from utils.summarize_prefill_graph_scout import (
    EXPECTED_PREFILLS,
    VARIANTS,
    summarize,
)


def _record(index: int, *, replay_ms: float = 10.0, copy_ms: float = 0.1):
    return {
        "index": index,
        "replay_ms": replay_ms,
        "copy_ms": copy_ms,
        "repeat_deterministic": True,
        "same_precision_graph_exact": True,
        "fp32_reference_exact": True,
        "finite": True,
        "cache_valid": True,
        "fp32_max_abs_drift": 0.0,
    }


def _payload():
    eager = [{"index": index, "execution_ms": 30.0} for index in range(EXPECTED_PREFILLS)]
    return {
        "schema_version": 1,
        "metadata": {
            "baseline_main_seconds": 30.0,
            "baseline_request_seconds": 43.0,
            "captured_prefills": EXPECTED_PREFILLS,
        },
        "eager_fp32": {"records": eager},
        "variants": {
            name: {
                "bucket_mode": "exact" if name.startswith("exact") else "bucket64",
                "graph_count": 3,
                "capture_setup_seconds": 0.05,
                "dtype_setup_seconds": 0.0,
                "peak_allocated_bytes": 100,
                "estimated_retained_graph_bytes": 20,
                "records": [_record(index) for index in range(EXPECTED_PREFILLS)],
            }
            for name in VARIANTS
        },
    }


def test_summarize_promotes_a_correct_candidate_that_clears_fixed_work_gate():
    report = summarize(_payload())

    assert report["promotion_pass"] is True
    assert report["decision"] == "PROMOTE_PREFILL_SCOUT"
    assert report["baseline_prefill_seconds"] == pytest.approx(2.61)
    assert report["variants"]["exact_fp32_graph"]["cold_request_saving_seconds"] == pytest.approx(
        2.61 - EXPECTED_PREFILLS * 10.1 / 1_000 - 0.05
    )


def test_fp32_reference_drift_blocks_fp32_candidate():
    payload = _payload()
    for name in VARIANTS:
        payload["variants"][name]["capture_setup_seconds"] = 3.0
    payload["variants"]["bucket64_fp32_graph"]["capture_setup_seconds"] = 0.0
    payload["variants"]["bucket64_fp32_graph"]["records"][2][
        "fp32_reference_exact"
    ] = False
    payload["variants"]["bucket64_fp32_graph"]["records"][2][
        "fp32_max_abs_drift"
    ] = 1e-6

    report = summarize(payload)

    candidate = report["variants"]["bucket64_fp32_graph"]
    assert candidate["performance_pass"] is True
    assert candidate["correctness_pass"] is False
    assert report["promotion_pass"] is False


def test_fp16_drift_is_reported_but_not_mislabeled_as_fp32_exactness():
    payload = _payload()
    payload["variants"]["exact_fp16_graph"]["records"][0][
        "fp32_reference_exact"
    ] = False
    payload["variants"]["exact_fp16_graph"]["records"][0][
        "fp32_max_abs_drift"
    ] = 0.25

    report = summarize(payload)

    assert report["variants"]["exact_fp16_graph"]["correctness_pass"] is True
    assert report["variants"]["exact_fp16_graph"]["fp32_max_abs_drift"] == 0.25


def test_missing_prefill_fails_loudly():
    payload = copy.deepcopy(_payload())
    payload["eager_fp32"]["records"].pop()

    with pytest.raises(ValueError, match="exactly 87"):
        summarize(payload)


def test_invalid_graph_manifest_fields_fail_loudly():
    payload = _payload()
    payload["variants"]["exact_fp32_graph"]["graph_count"] = 0
    with pytest.raises(ValueError, match="positive integer"):
        summarize(payload)

    payload = _payload()
    payload["variants"]["bucket64_fp16_graph"]["bucket_mode"] = "exact"
    with pytest.raises(ValueError, match="bucket_mode"):
        summarize(payload)
