from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import torch

from utils.profile_batched_super_timing_graph import _graph_summary, compare_exact
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


def _cache(batch_size: int):
    def layers():
        return [
            SimpleNamespace(
                keys=torch.zeros((batch_size, 2, 8, 4)),
                values=torch.zeros((batch_size, 2, 8, 4)),
            )
        ]

    return SimpleNamespace(
        self_attention_cache=SimpleNamespace(layers=layers()),
        cross_attention_cache=SimpleNamespace(layers=layers()),
    )


def test_graph_summary_proves_full_and_tail_cache_ownership():
    from osuT5.osuT5.inference.optimized.single.state import ProductionDecodeSession

    session = ProductionDecodeSession()
    session.caches = {
        (1, 4, 1, 1, "torch.float32", "cpu"): _cache(4),
        (1, 1, 1, 1, "torch.float32", "cpu"): _cache(1),
    }
    session.graph_cache[(4,)] = {
        "static_inputs": {"input_ids": torch.ones((4, 1), dtype=torch.long)},
        "active_prefix_length": 64,
        "capture_seconds": 0.1,
        "decode_replays": 3,
    }
    runtime = SimpleNamespace(new_context_state=lambda: session)

    summary = _graph_summary(runtime, expected_batch_sizes={1, 4})

    assert summary["cache_ownership_pass"]
    assert summary["observed_cache_batches_pass"]
    assert all(row["families_nonempty"] for row in summary["cache_states"])


def test_graph_summary_rejects_cross_state_cache_aliasing():
    from osuT5.osuT5.inference.optimized.single.state import ProductionDecodeSession

    first = _cache(1)
    second = _cache(1)
    second.cross_attention_cache.layers[0].values = (
        first.cross_attention_cache.layers[0].values
    )
    session = ProductionDecodeSession()
    session.caches = {
        (1, 1, 1, 1, "torch.float32", "cpu"): first,
        (2, 1, 1, 1, "torch.float32", "cpu"): second,
    }
    runtime = SimpleNamespace(new_context_state=lambda: session)

    summary = _graph_summary(runtime, expected_batch_sizes={1})

    assert not summary["cache_ownership_pass"]
    assert any(
        not tensor["unique_storage_across_states"]
        for row in summary["cache_states"]
        for tensor in row["tensors"]
    )
