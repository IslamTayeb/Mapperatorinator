from __future__ import annotations

from pathlib import Path

import pytest

from utils.profile_shared_rope_scout import (
    REQUIRED_MAIN_SAVING_SECONDS,
    SCHEMA_VERSION,
    _validate_report,
    summarize_shared_rope,
)
from utils.profile_native_prefix_dtype_scout import ALL_BUCKETS


ROOT = Path(__file__).resolve().parents[1]


def _bucket(accepted_ms: float, candidate_ms: float, *, exact=True, calls=True):
    return {
        "timing": {
            "accepted_ms_per_call": accepted_ms,
            "shared_rope_ms_per_call": candidate_ms,
        },
        "short_loop_argmax_tokens": {
            "accepted": [1, 2, 3, 4],
            "shared_rope": [1, 2, 3, 4],
            "exact": exact,
        },
        "exact_pass": exact,
        "rope_call_accounting_pass": calls,
    }


def _report(*, saving_ms=1.0, count=1_000):
    buckets = {
        str(prefix): _bucket(2.0, 2.0 - saving_ms)
        for prefix in ALL_BUCKETS
    }
    summary = summarize_shared_rope(
        buckets,
        live_counts={prefix: count for prefix in ALL_BUCKETS},
        install_setup_seconds=0.0,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {"measured_buckets": list(ALL_BUCKETS)},
        "buckets": buckets,
        "summary": summary,
    }


def test_weighted_gate_charges_setup_and_unmeasured_buckets_get_zero_credit():
    buckets = {str(prefix): _bucket(2.0, 1.5) for prefix in ALL_BUCKETS}
    live_counts = {prefix: 100 for prefix in ALL_BUCKETS}
    live_counts[576] = 1_000
    live_counts[640] = 2_000
    summary = summarize_shared_rope(
        buckets,
        live_counts=live_counts,
        install_setup_seconds=0.047,
    )

    expected_replay_saving = sum(live_counts.values()) * 0.5 / 1_000.0
    assert summary["replay_saving_seconds"] == pytest.approx(expected_replay_saving)
    assert summary["projected_main_saving_seconds"] == pytest.approx(
        expected_replay_saving - 0.047
    )
    assert summary["unmeasured_buckets_assumed_saving_seconds"] == 0.0
    assert summary["required_main_saving_seconds"] == REQUIRED_MAIN_SAVING_SECONDS
    assert summary["promotion_pass"] is True
    assert "table row" in summary["follow_up_if_below_threshold"]


def test_exactness_or_call_accounting_blocks_promotion():
    report = _report(saving_ms=1.0)
    assert report["summary"]["promotion_pass"] is True

    report["buckets"]["128"]["exact_pass"] = False
    report["summary"] = summarize_shared_rope(
        report["buckets"],
        live_counts={prefix: 1_000 for prefix in ALL_BUCKETS},
        install_setup_seconds=0.0,
    )
    assert report["summary"]["promotion_pass"] is False

    report = _report(saving_ms=1.0)
    report["buckets"]["576"]["rope_call_accounting_pass"] = False
    report["summary"] = summarize_shared_rope(
        report["buckets"],
        live_counts={prefix: 1_000 for prefix in ALL_BUCKETS},
        install_setup_seconds=0.0,
    )
    assert report["summary"]["promotion_pass"] is False


def test_report_validation_fails_on_bad_coverage_timing_or_decision():
    report = _report()
    _validate_report(report)

    report["metadata"]["measured_buckets"] = [128]
    with pytest.raises(ValueError, match="every live"):
        _validate_report(report)

    report = _report()
    report["buckets"]["128"]["timing"]["shared_rope_ms_per_call"] = 0.0
    with pytest.raises(ValueError, match="invalid graph timing"):
        _validate_report(report)

    report = _report()
    report["summary"]["promotion_pass"] = not report["summary"]["promotion_pass"]
    with pytest.raises(ValueError, match="decision is inconsistent"):
        _validate_report(report)


def test_dcc_wrapper_is_exact_clean_opt_in_and_retains_negative_report():
    source = (ROOT / "scripts/dcc/profile_shared_rope_scout.sbatch").read_text()

    assert "#SBATCH --gres=gpu:2080:1" in source
    assert 'REPO=${MAPPERATORINATOR_REPO:?' in source
    assert 'COMMIT=${MAPPERATORINATOR_COMMIT:?' in source
    assert 'BRANCH=${MAPPERATORINATOR_BRANCH:?' in source
    assert 'REMOTE_REF=${MAPPERATORINATOR_REMOTE_REF:?' in source
    assert 'git -C "$REPO" status --porcelain' in source
    assert 'git -C "$REPO" rev-parse HEAD' in source
    assert 'git -C "$REPO" branch --show-current' in source
    assert 'git -C "$REPO" rev-parse "${REMOTE_REF}^{commit}"' in source
    assert "utils/profile_shared_rope_scout.py" in source
    assert "--warmup 100" in source
    assert "--iters 1000" in source
    assert 'PROFILE_EXIT" -ne 0 && "$PROFILE_EXIT" -ne 3' in source
    assert 'exit "$PROFILE_EXIT"' in source
    assert "valid_negative_component_gate" in source
    assert "invariant_or_setup_failure" in source
    assert "profile-exit-code.txt" in source
    assert "decision-class.txt" in source
    assert "shared-rope.json" in source
    assert "shared-rope.txt" in source
    assert "nvidia-smi.csv" in source
    assert "sha256sums.txt" in source
