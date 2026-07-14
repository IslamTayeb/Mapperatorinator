from __future__ import annotations

import copy

import pytest

from utils.summarize_complete_request_wall import summarize


STAGES = (
    ("compile_args", 0.0, 0.1),
    ("setup_inference_environment", 0.2, 0.3),
    ("load_main_model", 0.4, 1.4),
    ("build_generation_config", 1.5, 1.6),
    ("validate_inputs", 2.0, 2.1),
    ("setup_processors", 2.2, 2.4),
    ("audio_load", 2.5, 2.8),
    ("audio_segment", 2.9, 3.1),
    ("timing_context_generation", 3.2, 5.2),
    ("timing_context_postprocess", 5.3, 5.5),
    ("main_generation", 5.6, 12.6),
    ("merge_generated_events", 12.7, 12.9),
    ("resnap_events", 13.0, 13.1),
    ("postprocess_generate_osu", 13.2, 13.5),
    ("write_osu", 13.6, 13.8),
)


def _profile(role: str, *, scale: float = 1.0):
    pass_kind = "untraced_control" if role == "control" else "untraced_budget"
    stages = []
    for name, start, finish in STAGES:
        stages.append(
            {
                "name": name,
                "started_at_perf_counter_seconds": start * scale,
                "finished_at_perf_counter_seconds": finish * scale,
                "wall_seconds": (finish - start) * scale,
                "cuda_memory_allocated_mb": 1.0,
            }
        )
    return {
        "schema_version": 1,
        "metadata": {"profile_pass_kind": pass_kind},
        "stages": stages,
    }


def test_summarizes_complete_request_groups_and_paired_overhead():
    report = summarize(_profile("control"), _profile("budget", scale=1.1))

    control = report["control"]
    assert control["request_to_output"]["wall_seconds"] == pytest.approx(11.8)
    assert control["cold_in_process"]["wall_seconds"] == pytest.approx(13.8)
    assert control["groups"]["pre_request_setup_load"][
        "stage_wall_seconds"
    ] == pytest.approx(1.3)
    assert control["groups"]["audio_preparation"][
        "stage_wall_seconds"
    ] == pytest.approx(0.5)
    assert control["groups"]["timing_generation"][
        "stage_names"
    ] == ["timing_context_generation"]
    assert control["groups"]["main_generation"][
        "stage_wall_seconds"
    ] == pytest.approx(7.0)
    assert control["request_to_output"]["residual_gap_seconds"] == pytest.approx(
        1.0
    )
    paired = report["paired_overhead"]
    assert paired["request_to_output"]["delta_fraction_of_control"] == pytest.approx(
        0.1
    )
    assert paired["groups"]["main_generation"][
        "delta_seconds"
    ] == pytest.approx(0.7)
    assert control["stages"][0]["metadata"] == {"cuda_memory_allocated_mb": 1.0}


def test_accepts_super_timing_and_osz_paths():
    control = _profile("control")
    budget = _profile("budget")
    for payload in (control, budget):
        for stage in payload["stages"]:
            if stage["name"] == "timing_context_generation":
                stage["name"] = "super_timing_generation"
            elif stage["name"] == "timing_context_postprocess":
                stage["name"] = "super_timing_postprocess"
            elif stage["name"] == "write_osu":
                stage["name"] = "write_osz"

    report = summarize(control, budget)

    assert report["control"]["final_write_stage"] == "write_osz"
    assert report["control"]["groups"]["timing_generation"]["stage_names"] == [
        "super_timing_generation"
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda profile: profile["stages"].pop(0), "exactly one compile_args"),
        (
            lambda profile: profile["stages"].pop(
                next(
                    index
                    for index, stage in enumerate(profile["stages"])
                    if stage["name"] == "validate_inputs"
                )
            ),
            "exactly one validate_inputs",
        ),
        (
            lambda profile: profile["stages"].__setitem__(
                -1,
                {
                    **profile["stages"][-1],
                    "name": "postprocess_generate_osu",
                },
            ),
            "exactly one final write",
        ),
        (
            lambda profile: profile["stages"][1].update(
                started_at_perf_counter_seconds=0.05,
                wall_seconds=0.25,
            ),
            "stages overlap",
        ),
        (
            lambda profile: profile["stages"][3].update(name="unknown_stage"),
            "unsupported stage",
        ),
    ),
)
def test_fails_loudly_on_missing_overlap_or_unknown_stage(mutation, message):
    control = _profile("control")
    mutation(control)

    with pytest.raises(ValueError, match=message):
        summarize(control, _profile("budget"))


def test_rejects_different_stage_sequences():
    control = _profile("control")
    budget = _profile("budget")
    budget["stages"] = copy.deepcopy(budget["stages"])
    budget["stages"].pop(
        next(
            index
            for index, stage in enumerate(budget["stages"])
            if stage["name"] == "resnap_events"
        )
    )

    with pytest.raises(ValueError, match="stage sequences differ"):
        summarize(control, budget)


def test_empty_optional_group_uses_json_safe_null_fraction():
    control = _profile("control")
    budget = _profile("budget")
    for payload in (control, budget):
        payload["stages"] = [
            stage
            for stage in payload["stages"]
            if stage["name"] != "timing_context_postprocess"
        ]

    report = summarize(control, budget)

    assert report["paired_overhead"]["groups"]["timing_postprocess"][
        "delta_fraction_of_control"
    ] is None
