from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "utils" / "inference_profile_metrics.py"
    spec = importlib.util.spec_from_file_location("inference_profile_metrics", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_first_record_breakdown_splits_cold_window_from_remaining_records():
    module = _load_module()
    profile = {
        "generation": [
            {
                "profile_label": "main_generation",
                "mode": "sequential",
                "context_type": "MAP",
                "sequence_index": 0,
                "generated_tokens": 100,
                "model_elapsed_seconds": 25.0,
                "wall_seconds": 26.0,
                "model_generate_cpu_elapsed_seconds": 25.0,
                "model_generate_cuda_event_seconds": 20.0,
                "model_generate_host_gap_seconds": 5.0,
            },
            {
                "profile_label": "main_generation",
                "mode": "sequential",
                "context_type": "MAP",
                "sequence_index": 1,
                "generated_tokens": 50,
                "model_elapsed_seconds": 0.5,
                "wall_seconds": 0.6,
                "model_generate_cpu_elapsed_seconds": 0.5,
                "model_generate_cuda_event_seconds": 0.4,
                "model_generate_host_gap_seconds": 0.1,
            },
            {
                "profile_label": "timing_context",
                "mode": "sequential",
                "context_type": "TIMING",
                "sequence_index": 0,
                "generated_tokens": 20,
                "model_elapsed_seconds": 1.0,
                "wall_seconds": 1.1,
            },
            {
                "profile_label": "main_generation",
                "mode": "sequential",
                "context_type": "MAP",
                "sequence_index": 2,
                "generated_tokens": 50,
                "model_elapsed_seconds": 0.5,
                "wall_seconds": 0.7,
                "model_generate_cpu_elapsed_seconds": 0.5,
                "model_generate_cuda_event_seconds": 0.3,
                "model_generate_host_gap_seconds": 0.2,
            },
        ]
    }

    breakdown = module.first_record_breakdown(profile, "main_generation")

    assert breakdown["records"] == 3
    assert breakdown["first_record"]["key"] == "MAP/sequential/seq0"
    assert breakdown["first_record"]["generated_tokens"] == 100
    assert breakdown["first_record"]["tokens_per_second"] == 4.0
    assert breakdown["first_record"]["model_generate_cuda_event_seconds"] == 20.0
    assert breakdown["first_record"]["model_generate_host_gap_seconds"] == 5.0
    assert breakdown["remaining_records"]["records"] == 2
    assert breakdown["remaining_records"]["generated_tokens"] == 100
    assert breakdown["remaining_records"]["model_elapsed_seconds"] == 1.0
    assert breakdown["remaining_records"]["model_generate_cpu_elapsed_seconds"] == 1.0
    assert breakdown["remaining_records"]["model_generate_cuda_event_seconds"] == 0.7
    assert round(breakdown["remaining_records"]["model_generate_host_gap_seconds"], 6) == 0.3
    assert breakdown["remaining_records"]["model_generate_records_with_cuda_event"] == 2
    assert breakdown["remaining_records"]["model_generate_cuda_event_fraction"] == 0.7
    assert round(breakdown["remaining_records"]["model_generate_host_gap_fraction"], 6) == 0.3
    assert breakdown["remaining_records"]["tokens_per_second"] == 100.0
