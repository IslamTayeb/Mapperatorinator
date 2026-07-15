from __future__ import annotations

from pathlib import Path

import pytest

from utils.profile_shared_rope_scout import (
    REQUIRED_MAIN_SAVING_SECONDS,
    SCHEMA_VERSION,
    _strict_fp32_environment,
    _validate_live_accepted_graph_cache,
    _validate_report,
    decision_exit_code,
    summarize_shared_rope,
)
from utils.profile_native_prefix_dtype_scout import ALL_BUCKETS


ROOT = Path(__file__).resolve().parents[1]


def _live_graph_cache(*, changed_count: int | None = None):
    entries = {}
    for prefix in ALL_BUCKETS:
        entries[(prefix,)] = {
            "active_prefix_length": prefix,
            "graph": object(),
            "outputs": object(),
            "static_inputs": {},
            "decode_replays": changed_count if prefix == 128 and changed_count else 1,
        }
    return entries


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


def test_live_graph_manifest_accepts_fresh_positive_replay_counts():
    entries = _validate_live_accepted_graph_cache(
        _live_graph_cache(changed_count=123)
    )

    assert tuple(entries) == ALL_BUCKETS
    assert entries[128]["decode_replays"] == 123


def test_live_graph_manifest_rejects_nonpositive_counts_and_duplicate_prefixes():
    invalid = _live_graph_cache()
    invalid[(128,)]["decode_replays"] = 0
    with pytest.raises(RuntimeError, match="invalid decode_replays"):
        _validate_live_accepted_graph_cache(invalid)

    duplicate = _live_graph_cache()
    duplicate[("duplicate",)] = dict(duplicate[(128,)])
    with pytest.raises(RuntimeError, match="repeats prefix 128"):
        _validate_live_accepted_graph_cache(duplicate)


def test_short_loop_source_caps_replays_at_fixed_prefix_boundary():
    source = (ROOT / "utils/profile_shared_rope_scout.py").read_text()

    assert "safe_steps = min(steps, active_prefix_length - cache_position)" in source
    assert '"executed_steps": len(accepted_tokens)' in source


def test_component_profiler_requires_strict_fp32_environment(monkeypatch):
    import torch

    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    torch.set_float32_matmul_precision("highest")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)

    assert _strict_fp32_environment() == {
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "nvidia_tf32_override": "0",
    }

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    with pytest.raises(RuntimeError, match="strict FP32 environment"):
        _strict_fp32_environment()


def test_accepted_capture_requests_strict_fp32_setup():
    source = (ROOT / "utils/profile_native_prefix_dtype_scout.py").read_text()

    assert 'strict_fp32=(' in source
    assert 'args.inference_engine == "optimized" and args.precision == "fp32"' in source


def test_live_graph_manifest_requires_exact_current_bucket_topology():
    missing = _live_graph_cache()
    del missing[(ALL_BUCKETS[-1],)]
    with pytest.raises(RuntimeError, match="must contain current buckets"):
        _validate_live_accepted_graph_cache(missing)

    extra = _live_graph_cache()
    extra[(ALL_BUCKETS[-1] + 64,)] = {
        **extra[(ALL_BUCKETS[-1],)],
        "active_prefix_length": ALL_BUCKETS[-1] + 64,
    }
    with pytest.raises(RuntimeError, match="must contain current buckets"):
        _validate_live_accepted_graph_cache(extra)


def test_live_graph_manifest_requires_every_accepted_entry_field():
    missing = _live_graph_cache()
    del missing[(128,)]["outputs"]

    with pytest.raises(RuntimeError, match=r"prefix 128 is missing \['outputs'\]"):
        _validate_live_accepted_graph_cache(missing)


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
    assert decision_exit_code(report) == 0

    report["buckets"]["128"]["exact_pass"] = False
    report["summary"] = summarize_shared_rope(
        report["buckets"],
        live_counts={prefix: 1_000 for prefix in ALL_BUCKETS},
        install_setup_seconds=0.0,
    )
    assert report["summary"]["promotion_pass"] is False
    assert decision_exit_code(report) == 1

    report = _report(saving_ms=1.0)
    report["buckets"]["576"]["rope_call_accounting_pass"] = False
    report["summary"] = summarize_shared_rope(
        report["buckets"],
        live_counts={prefix: 1_000 for prefix in ALL_BUCKETS},
        install_setup_seconds=0.0,
    )
    assert report["summary"]["promotion_pass"] is False
    assert decision_exit_code(report) == 1


def test_performance_only_negative_uses_exit_three():
    report = _report(saving_ms=0.01)

    assert report["summary"]["exact_pass"] is True
    assert report["summary"]["rope_call_accounting_pass"] is True
    assert report["summary"]["performance_pass"] is False
    assert decision_exit_code(report) == 3


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
    assert "export NVIDIA_TF32_OVERRIDE=0" in source
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
