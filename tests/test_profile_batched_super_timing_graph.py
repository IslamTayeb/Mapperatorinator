from __future__ import annotations

from copy import deepcopy

from utils.profile_batched_super_timing_graph import compare_exact
from utils.summarize_batched_super_timing_graph import summarize


def _report(*, mode: str, batch: int, repetition: int, seconds: float):
    windows = [{
        "iteration": 0,
        "window_index": 0,
        "prompt_sha256": "prompt",
        "prompt_tokens": 2,
        "raw_generated_token_ids": [4, 2],
        "generated_tokens": 2,
    }]
    output = {
        "raw_histograms": {"beats": "raw"},
        "smoothed_histograms": {"beats": "smooth"},
        "tpbs": "tpbs",
        "measure_counts": "counts",
        "final_events": [["TIME_SHIFT", 10]],
        "final_event_times": [10],
    }
    return {
        "schema_version": 1,
        "metadata": {
            "mode": mode,
            "batch_size": batch,
            "precision": "fp32",
            "repetition": repetition,
            "phase": "full",
            "seed": 12345,
            "timer_iterations": 20,
            "timer_num_beams": 1,
            "timer_cfg_scale": 1.0,
            "audio_sha256": "audio",
            "model_path": "model",
            "git_commit": "commit",
            "torch_version": "torch",
            "cuda_version": "cuda",
            "cuda_device": "2080 Ti",
            "public_wiring": False,
            "order": (
                1
                if (repetition % 2 == 1) == (mode == "eager")
                else 2
            ),
        },
        "timing": {"complete_super_timing_seconds": seconds},
        "workload": {"audio_offsets_ms": [0], "windows": windows},
        "output": output,
        "gates": {"pass": True},
        "_path": f"{mode}-b{batch}-r{repetition}.json",
    }


def test_exact_comparison_covers_tokens_histograms_and_final_output():
    eager = _report(mode="eager", batch=4, repetition=1, seconds=10.0)
    graph = _report(mode="graph", batch=4, repetition=1, seconds=9.0)
    assert compare_exact(eager, graph)["pass"]

    changed = deepcopy(graph)
    changed["workload"]["windows"][0]["raw_generated_token_ids"] = [5, 2]
    comparison = compare_exact(eager, changed)
    assert not comparison["pass"]
    assert not comparison["checks"]["per_window_tokens"]


def test_ladder_requires_three_exact_deterministic_repetitions_and_five_pct():
    reports = []
    for repetition, eager_seconds, graph_seconds in (
        (1, 10.0, 9.0),
        (2, 10.2, 9.1),
        (3, 9.8, 8.9),
    ):
        reports.append(
            _report(
                mode="eager",
                batch=4,
                repetition=repetition,
                seconds=eager_seconds,
            )
        )
        reports.append(
            _report(
                mode="graph",
                batch=4,
                repetition=repetition,
                seconds=graph_seconds,
            )
        )

    result = summarize(reports, required_repetitions=3)

    assert result["variants"]["4"]["survives"]
    assert result["winning_batch_size"] == 4
    assert result["global_complete_wall_speedup_pct"] == 10.0
    assert result["promotion_pass"]
