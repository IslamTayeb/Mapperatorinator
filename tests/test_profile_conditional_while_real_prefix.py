from __future__ import annotations

import pytest

from utils.profile_conditional_while_real_prefix import (
    ACCEPTED_DISPATCH,
    VARIANTS,
    _compare_case,
    _validate_report,
    _variant_launches,
)


@pytest.mark.parametrize(
    ("variant", "steps", "expected"),
    (
        ("k1", 8, 8),
        ("k4", 8, 2),
        ("k8", 8, 1),
        ("while", 8, 1),
        ("k4", 3, 1),
        ("k8", 7, 1),
    ),
)
def test_variant_launch_count_is_explicit(variant, steps, expected):
    assert _variant_launches(variant, steps) == expected


def _row(variant: str, *, logical: int, physical: int, repeat: int) -> dict:
    del repeat
    return {
        "variant": variant,
        "logical_steps": logical,
        "physical_steps": physical,
        "wasted_steps": physical - logical,
        "visible_token_sha256": "tokens",
        "logical_cache_sha256": "logical-cache",
        "full_cache_sha256": (
            "exact-cache" if variant in {"k1", "while"} else f"{variant}-waste"
        ),
        "memory_stable": True,
        "cuda_event_ms": {
            "k1": 8.0,
            "k4": 5.0,
            "k8": 4.5,
            "while": 4.0,
        }[variant],
    }


def test_forced_stop_gate_requires_visible_and_logical_cache_parity():
    rows = []
    expected_physical = {"k1": 3, "k4": 4, "k8": 8, "while": 3}
    for repeat in range(2):
        for variant in VARIANTS:
            rows.append(
                _row(
                    variant,
                    logical=3,
                    physical=expected_physical[variant],
                    repeat=repeat,
                )
            )

    report = _compare_case(rows, expected_logical=3, fixed_steps=8)

    assert report["pass"] is True
    assert report["while_no_post_stop_waste"] is True
    assert report["expected_physical_steps"] == expected_physical
    assert report["while_speedup_vs_pct"]["k8"] == pytest.approx((4.5 - 4.0) / 4.5 * 100)
    assert report["reciprocal_cuda_ms"]["while"] == 4.0


def test_gate_fails_if_while_advances_after_stop():
    rows = []
    for repeat in range(2):
        for variant in VARIANTS:
            physical = {"k1": 3, "k4": 4, "k8": 8, "while": 4}[variant]
            rows.append(_row(variant, logical=3, physical=physical, repeat=repeat))

    report = _compare_case(rows, expected_logical=3, fixed_steps=8)

    assert report["pass"] is False
    assert report["while_no_post_stop_waste"] is False


def test_report_requires_accepted_dispatch_and_all_stops():
    case = {"pass": True}
    report = {
        "schema_version": 1,
        "metadata": {"accepted_dispatch": ACCEPTED_DISPATCH},
        "fixed_work": case,
        "forced_stop": {"1": case, "3": case, "7": case},
        "decision": {"pass": True},
    }
    _validate_report(report)

    report["metadata"]["accepted_dispatch"] = {**ACCEPTED_DISPATCH, "precision": "fp16"}
    with pytest.raises(RuntimeError, match="dispatch"):
        _validate_report(report)
