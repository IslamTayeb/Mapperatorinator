from __future__ import annotations

import json

from osuT5.osuT5.inference.profiler import InferenceProfiler


def test_generation_summary_aggregates_model_time_and_tokens(tmp_path):
    profiler = InferenceProfiler(enabled=True)
    profiler.record_generation(
        profile_label="main_generation",
        context_type="MAP",
        wall_seconds=1.2,
        model_elapsed_seconds=1.0,
        generated_tokens=100,
        generated_token_ids=[1, 2],
    )
    profiler.record_generation(
        profile_label="main_generation",
        context_type="MAP",
        wall_seconds=2.2,
        model_elapsed_seconds=2.0,
        generated_tokens=200,
        generated_token_ids=[3, 4],
    )

    summary = profiler._summary()["generation_by_label"]["main_generation"]

    assert summary == {
        "records": 2,
        "wall_seconds": 3.4000000000000004,
        "model_elapsed_seconds": 3.0,
        "generated_tokens": 300,
        "tokens_per_second": 100.0,
    }

    output = profiler.write(tmp_path / "profile.json")
    assert output is not None
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["generation"][0]["generated_token_ids"] == [1, 2]
    assert "torch_profiles" not in payload


def test_disabled_profiler_does_not_write(tmp_path):
    profiler = InferenceProfiler()
    profiler.record_generation(profile_label="main_generation")

    assert profiler.write(tmp_path / "profile.json") is None
    assert not (tmp_path / "profile.json").exists()
