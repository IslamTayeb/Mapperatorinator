from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "osuT5" / "osuT5" / "inference" / "profiler.py"
    spec = importlib.util.spec_from_file_location("profiler", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generation_summary_aggregates_model_generate_cuda_ledger():
    module = _load_module()
    profiler = module.InferenceProfiler(enabled=True)
    profiler.record_generation(
        profile_label="main_generation",
        context_type="MAP",
        wall_seconds=1.2,
        model_elapsed_seconds=1.0,
        generated_tokens=100,
        model_generate_cpu_elapsed_seconds=1.0,
        model_generate_cuda_event_seconds=0.7,
        model_generate_host_gap_seconds=0.3,
    )
    profiler.record_generation(
        profile_label="main_generation",
        context_type="MAP",
        wall_seconds=2.2,
        model_elapsed_seconds=2.0,
        generated_tokens=200,
        model_generate_cpu_elapsed_seconds=2.0,
        model_generate_cuda_event_seconds=1.2,
        model_generate_host_gap_seconds=0.8,
    )

    summary = profiler._summary()["generation_by_label"]["main_generation"]

    assert summary["records"] == 2
    assert summary["generated_tokens"] == 300
    assert summary["model_elapsed_seconds"] == 3.0
    assert summary["model_generate_cpu_elapsed_seconds"] == 3.0
    assert summary["model_generate_cuda_event_seconds"] == 1.9
    assert summary["model_generate_host_gap_seconds"] == 1.1
    assert summary["model_generate_records_with_cuda_event"] == 2
    assert round(summary["model_generate_cuda_event_fraction"], 6) == round(1.9 / 3.0, 6)
    assert round(summary["model_generate_host_gap_fraction"], 6) == round(1.1 / 3.0, 6)
