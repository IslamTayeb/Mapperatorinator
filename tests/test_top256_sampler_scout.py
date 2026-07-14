import json

import pytest

from utils.top256_sampler_scout import (
    fixed_physical_work,
    render_text,
    summarize_candidate,
)


def _component(
    *,
    baseline_ms: float = 0.071114,
    candidate_ms: float = 0.030,
    samples: int = 512,
    overflow_count: int = 0,
    exact: bool = True,
    baseline_capture: float = 0.1,
    candidate_capture: float = 0.1,
) -> dict:
    return {
        "correctness": {
            "samples": samples,
            "overflow_count": overflow_count,
            "selected_token_exact": exact,
            "counter_threshold_consumed_unchanged": exact,
        },
        "representatives": [
            {
                "baseline_ms_per_step_worst": baseline_ms,
                "candidate_ms_per_step_worst": candidate_ms,
                "baseline_capture_seconds": baseline_capture,
                "candidate_capture_seconds": candidate_capture,
                "selected_token_exact": exact,
            }
        ],
    }


def _physical(steps: float = 8294.0) -> dict:
    return {"fixed_physical_steps": steps}


def test_summary_passes_only_exact_zero_overflow_charged_gate() -> None:
    report = summarize_candidate(_component(), _physical())

    assert report["gate"]["promotion_pass"] is True
    assert report["gate"]["fallback_charge_ms_per_step"] == 0.0
    assert report["gate"]["fixed_main_saving_seconds"] == pytest.approx(
        (0.071114 - 0.030) * 8294 / 1000
    )
    assert report["production_wiring_changed"] is False
    assert report["decision"] == "retain_for_full_runtime_scout"


def test_summary_charges_overflow_fallback_and_capture_delta() -> None:
    report = summarize_candidate(
        _component(
            candidate_ms=0.02,
            overflow_count=128,
            candidate_capture=0.3,
        ),
        _physical(),
    )

    assert report["gate"]["observed_overflow_fraction"] == 0.25
    assert report["gate"]["fallback_charge_ms_per_step"] == pytest.approx(
        0.25 * 0.071114
    )
    assert report["gate"]["capture_setup_delta_seconds"] == pytest.approx(0.2)
    assert report["gate"]["overflow_pass"] is False
    assert report["gate"]["promotion_pass"] is False


def test_summary_blocks_a_mutated_counter_threshold() -> None:
    component = _component()
    component["correctness"]["counter_threshold_consumed_unchanged"] = False

    report = summarize_candidate(component, _physical())

    assert report["gate"]["correctness_pass"] is False
    assert report["gate"]["promotion_pass"] is False


def test_summary_stops_candidate_that_misses_speed_or_saving_gate() -> None:
    report = summarize_candidate(
        _component(candidate_ms=0.0349),
        _physical(8294),
    )

    assert report["gate"]["speed_pass"] is True
    assert report["gate"]["saving_pass"] is True
    too_slow = summarize_candidate(
        _component(candidate_ms=0.0351),
        _physical(8294),
    )
    assert too_slow["gate"]["speed_pass"] is False
    assert too_slow["gate"]["promotion_pass"] is False


def test_fixed_physical_work_uses_main_k4_profile(tmp_path) -> None:
    profile = {
        "generation": [
            {
                "profile_label": "timing_generation",
                "optimized_cuda_graphs": {
                    "k8_candidate": {
                        "logical_steps": 10,
                        "physical_steps": 12,
                        "wasted_steps": 2,
                    }
                },
            },
            {
                "profile_label": "main_generation",
                "optimized_cuda_graphs": {
                    "k8_candidate": {
                        "logical_steps": 100,
                        "physical_steps": 104,
                        "wasted_steps": 4,
                    }
                },
            },
        ]
    }
    path = tmp_path / "song.profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")

    result = fixed_physical_work(tmp_path, fixed_main_steps=8294)

    assert result["observed_logical_steps"] == 100
    assert result["observed_physical_steps"] == 104
    assert result["fixed_physical_steps"] == pytest.approx(8294 * 1.04)


def test_fixed_physical_work_rejects_missing_or_inconsistent_evidence(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="expected one inference profile"):
        fixed_physical_work(tmp_path)

    path = tmp_path / "song.profile.json"
    path.write_text(
        json.dumps(
            {
                "generation": [
                    {
                        "profile_label": "main_generation",
                        "optimized_cuda_graphs": {
                            "k8_candidate": {
                                "logical_steps": 10,
                                "physical_steps": 11,
                                "wasted_steps": 0,
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="inconsistent"):
        fixed_physical_work(tmp_path)


def test_text_report_is_plain_and_records_gate() -> None:
    text = render_text(summarize_candidate(_component(), _physical()))

    assert "decision=retain_for_full_runtime_scout" in text
    assert "selected_token_exact=True" in text
    assert "production_wiring_changed=False" in text
