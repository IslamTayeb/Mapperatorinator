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


def _suite_manifest(*, hash_suffix: str = "same", warmed_tok_s: float = 120.0):
    runs = [
        {
            "run_index": 0,
            "repeat_index": 0,
            "song_index": 0,
            "main_generated_tokens": 100,
            "main_model_elapsed_seconds": 2.0,
            "main_wall_seconds": 2.5,
            "main_tokens_per_second": 50.0,
            "main_token_count": 100,
            "main_token_sha256": f"hash-cold-{hash_suffix}",
        },
        {
            "run_index": 1,
            "repeat_index": 1,
            "song_index": 0,
            "main_generated_tokens": 120,
            "main_model_elapsed_seconds": 1.0,
            "main_wall_seconds": 1.2,
            "main_tokens_per_second": warmed_tok_s,
            "main_token_count": 120,
            "main_token_sha256": f"hash-warm-{hash_suffix}",
        },
    ]
    return {
        "run_kind": "warm_repeat",
        "song_count": 1,
        "seed_step": 0,
        "runs": runs,
        "aggregate": {
            "all_runs": {
                "runs": 2,
                "generated_tokens": 220,
                "model_elapsed_seconds": 3.0,
                "wall_seconds": 3.7,
                "tokens_per_second": 73.333,
            },
            "warmed_runs": {
                "runs": 1,
                "generated_tokens": 120,
                "model_elapsed_seconds": 120 / warmed_tok_s,
                "wall_seconds": 1.2,
                "tokens_per_second": warmed_tok_s,
            },
        },
    }


def test_compare_suite_manifests_passes_warmed_non_regression(tmp_path):
    module = _load_module()
    baseline = tmp_path / "baseline-suite.json"
    candidate = tmp_path / "candidate-suite.json"
    baseline.write_text(module.json.dumps(_suite_manifest(warmed_tok_s=100.0)))
    candidate.write_text(module.json.dumps(_suite_manifest(warmed_tok_s=120.0)))

    report = module.compare_suite_manifests(baseline, candidate, scope="warmed_runs")

    assert report["shape"]["pass"]
    assert report["token_equivalence"]["pass"]
    assert report["performance"]["pass"]


def test_compare_suite_manifests_reports_hash_and_warmed_regressions(tmp_path):
    module = _load_module()
    baseline = tmp_path / "baseline-suite.json"
    candidate = tmp_path / "candidate-suite.json"
    baseline.write_text(module.json.dumps(_suite_manifest(hash_suffix="base", warmed_tok_s=100.0)))
    candidate.write_text(module.json.dumps(_suite_manifest(hash_suffix="cand", warmed_tok_s=80.0)))

    report = module.compare_suite_manifests(baseline, candidate, scope="warmed_runs")

    assert report["shape"]["pass"]
    assert not report["token_equivalence"]["pass"]
    assert report["token_equivalence"]["mismatches"]
    assert not report["performance"]["pass"]
    assert not report["performance"]["metrics"]["tokens_per_second"]["pass"]
