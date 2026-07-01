from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "utils" / "summarize_inference_profile.py"
    spec = importlib.util.spec_from_file_location("summarize_inference_profile", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _profile(*, tokens: list[int], tok_s: float, model_s: float, wall_s: float, seed: int = 12345):
    metadata = {
        key: "same"
        for key in [
            "model_path",
            "audio_path",
            "precision",
            "attn_implementation",
            "in_context",
            "output_type",
        ]
    }
    metadata.update({
        "seed": seed,
        "use_server": False,
        "parallel": False,
        "temperature": 1.0,
        "timing_temperature": 1.0,
        "mania_column_temperature": 1.0,
        "taiko_hit_temperature": 1.0,
        "timeshift_bias": 0,
        "top_p": 0.9,
        "top_k": 50,
        "do_sample": True,
        "num_beams": 1,
        "cfg_scale": 1.0,
        "lookback": 0,
        "lookahead": 0,
        "start_time": 71000,
        "end_time": 86000,
        "profile_record_token_ids": True,
    })
    return {
        "metadata": metadata,
        "stages": [
            {
                "name": "generation",
                "wall_seconds": wall_s + 1.0,
            }
        ],
        "generation": [
            {
                "profile_label": "main_generation",
                "mode": "sequential",
                "context_type": "MAP",
                "sequence_index": 0,
                "generated_tokens": len(tokens),
                "model_elapsed_seconds": model_s,
                "wall_seconds": wall_s,
                "tokens_per_second": tok_s,
                "generated_token_ids": tokens,
            }
        ],
        "summary": {
            "generation_by_label": {
                "main_generation": {
                    "generated_tokens": len(tokens),
                    "model_elapsed_seconds": model_s,
                    "wall_seconds": wall_s,
                    "tokens_per_second": tok_s,
                    "records": 1,
                }
            }
        },
    }


def test_compare_profiles_passes_equivalent_non_regression(tmp_path):
    module = _load_module()
    baseline = tmp_path / "baseline.profile.json"
    candidate = tmp_path / "candidate.profile.json"
    baseline.write_text(module.json.dumps(_profile(tokens=[1, 2, 3], tok_s=100, model_s=10, wall_s=11)))
    candidate.write_text(module.json.dumps(_profile(tokens=[1, 2, 3], tok_s=110, model_s=9, wall_s=10)))

    report = module.compare_profiles(baseline, candidate, label="main_generation")

    assert report["same_calculation"]["pass"]
    assert report["token_equivalence"]["pass"]
    assert report["performance"]["pass"]


def test_compare_profiles_reports_token_and_performance_failures(tmp_path):
    module = _load_module()
    baseline = tmp_path / "baseline.profile.json"
    candidate = tmp_path / "candidate.profile.json"
    baseline.write_text(module.json.dumps(_profile(tokens=[1, 2, 3], tok_s=100, model_s=10, wall_s=11)))
    candidate.write_text(module.json.dumps(_profile(tokens=[1, 4, 3], tok_s=90, model_s=12, wall_s=13)))

    report = module.compare_profiles(baseline, candidate, label="main_generation")

    assert report["same_calculation"]["pass"]
    assert not report["token_equivalence"]["pass"]
    assert report["token_equivalence"]["first_mismatch"] == 1
    assert not report["performance"]["pass"]
    assert not report["performance"]["metrics"]["tokens_per_second"]["pass"]
    assert not report["performance"]["metrics"]["model_elapsed_seconds"]["pass"]
    assert not report["performance"]["per_window"]["pass"]
