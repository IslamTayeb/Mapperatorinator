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


def _profile_with_timing(
    *,
    main_tokens: list[int],
    timing_tokens: list[int],
    main_tok_s: float,
    timing_tok_s: float,
    main_model_s: float,
    timing_model_s: float,
    main_wall_s: float,
    timing_wall_s: float,
):
    profile = _profile(tokens=main_tokens, tok_s=main_tok_s, model_s=main_model_s, wall_s=main_wall_s)
    profile["generation"].append({
        "profile_label": "timing_context",
        "mode": "sequential",
        "context_type": "TIMING",
        "sequence_index": 0,
        "generated_tokens": len(timing_tokens),
        "model_elapsed_seconds": timing_model_s,
        "wall_seconds": timing_wall_s,
        "tokens_per_second": timing_tok_s,
        "generated_token_ids": timing_tokens,
    })
    profile["summary"]["generation_by_label"]["timing_context"] = {
        "generated_tokens": len(timing_tokens),
        "model_elapsed_seconds": timing_model_s,
        "wall_seconds": timing_wall_s,
        "tokens_per_second": timing_tok_s,
        "records": 1,
    }
    profile["stages"][0]["wall_seconds"] = main_wall_s + timing_wall_s + 1.0
    return profile


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


def test_compare_profiles_for_labels_passes_when_main_and_timing_pass(tmp_path):
    module = _load_module()
    baseline = tmp_path / "baseline.profile.json"
    candidate = tmp_path / "candidate.profile.json"
    baseline.write_text(module.json.dumps(_profile_with_timing(
        main_tokens=[1, 2, 3],
        timing_tokens=[8, 9],
        main_tok_s=100,
        timing_tok_s=50,
        main_model_s=10,
        timing_model_s=4,
        main_wall_s=11,
        timing_wall_s=5,
    )))
    candidate.write_text(module.json.dumps(_profile_with_timing(
        main_tokens=[1, 2, 3],
        timing_tokens=[8, 9],
        main_tok_s=110,
        timing_tok_s=60,
        main_model_s=9,
        timing_model_s=3,
        main_wall_s=10,
        timing_wall_s=4,
    )))

    report = module.compare_profiles_for_labels(
        baseline,
        candidate,
        labels=["main_generation", "timing_context"],
    )

    assert report["same_calculation_pass"]
    assert report["token_equivalence_pass"]
    assert report["performance_pass"]
    assert report["reports"]["main_generation"]["performance"]["pass"]
    assert report["reports"]["timing_context"]["performance"]["pass"]


def test_compare_profiles_for_labels_fails_when_timing_regresses(tmp_path):
    module = _load_module()
    baseline = tmp_path / "baseline.profile.json"
    candidate = tmp_path / "candidate.profile.json"
    baseline.write_text(module.json.dumps(_profile_with_timing(
        main_tokens=[1, 2, 3],
        timing_tokens=[8, 9],
        main_tok_s=100,
        timing_tok_s=50,
        main_model_s=10,
        timing_model_s=4,
        main_wall_s=11,
        timing_wall_s=5,
    )))
    candidate.write_text(module.json.dumps(_profile_with_timing(
        main_tokens=[1, 2, 3],
        timing_tokens=[8, 9],
        main_tok_s=110,
        timing_tok_s=25,
        main_model_s=9,
        timing_model_s=8,
        main_wall_s=6,
        timing_wall_s=9,
    )))

    report = module.compare_profiles_for_labels(
        baseline,
        candidate,
        labels=["main_generation", "timing_context"],
    )

    assert report["same_calculation_pass"]
    assert report["token_equivalence_pass"]
    assert not report["performance_pass"]
    assert report["reports"]["main_generation"]["performance"]["pass"]
    assert not report["reports"]["timing_context"]["performance"]["pass"]


def _suite_manifest(
    *,
    hash_suffix: str = "same",
    warmed_tok_s: float = 120.0,
    warmed_first_tok_s: float = 100.0,
    warmed_remaining_tok_s: float = 140.0,
    timing_tok_s: float = 50.0,
):
    runs = [
        {
            "run_index": 0,
            "repeat_index": 0,
            "song_index": 0,
            "song_id": "salvalai",
            "audio_path": "/work/salvalai.mp3",
            "start_time": 71000,
            "end_time": 86000,
            "seed": 12345,
            "main_generated_tokens": 100,
            "main_model_elapsed_seconds": 2.0,
            "main_wall_seconds": 2.5,
            "main_tokens_per_second": 50.0,
            "main_token_count": 100,
            "main_token_sha256": f"hash-cold-{hash_suffix}",
            "main_first_record": {
                "records": 1,
                "generated_tokens": 50,
                "model_elapsed_seconds": 1.0,
                "wall_seconds": 1.1,
                "tokens_per_second": 50.0,
            },
            "main_remaining_records": {
                "records": 1,
                "generated_tokens": 50,
                "model_elapsed_seconds": 1.0,
                "wall_seconds": 1.1,
                "tokens_per_second": 50.0,
            },
            "timing_generated_tokens": 20,
            "timing_model_elapsed_seconds": 0.5,
            "timing_wall_seconds": 0.6,
            "timing_tokens_per_second": 40.0,
        },
        {
            "run_index": 1,
            "repeat_index": 1,
            "song_index": 0,
            "song_id": "salvalai",
            "audio_path": "/work/salvalai.mp3",
            "start_time": 71000,
            "end_time": 86000,
            "seed": 12345,
            "main_generated_tokens": 120,
            "main_model_elapsed_seconds": 1.0,
            "main_wall_seconds": 1.2,
            "main_tokens_per_second": warmed_tok_s,
            "main_token_count": 120,
            "main_token_sha256": f"hash-warm-{hash_suffix}",
            "main_first_record": {
                "records": 1,
                "generated_tokens": 40,
                "model_elapsed_seconds": 40 / warmed_first_tok_s,
                "wall_seconds": 0.5,
                "tokens_per_second": warmed_first_tok_s,
            },
            "main_remaining_records": {
                "records": 2,
                "generated_tokens": 80,
                "model_elapsed_seconds": 80 / warmed_remaining_tok_s,
                "wall_seconds": 0.7,
                "tokens_per_second": warmed_remaining_tok_s,
            },
            "timing_generated_tokens": 30,
            "timing_model_elapsed_seconds": 30 / timing_tok_s,
            "timing_wall_seconds": 0.7,
            "timing_tokens_per_second": timing_tok_s,
        },
    ]
    return {
        "schema_version": 3,
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
                "first_records": {
                    "runs": 2,
                    "records": 2,
                    "generated_tokens": 90,
                    "model_elapsed_seconds": 1.0 + 40 / warmed_first_tok_s,
                    "wall_seconds": 1.6,
                    "tokens_per_second": 90 / (1.0 + 40 / warmed_first_tok_s),
                },
                "remaining_records": {
                    "runs": 2,
                    "records": 3,
                    "generated_tokens": 130,
                    "model_elapsed_seconds": 1.0 + 80 / warmed_remaining_tok_s,
                    "wall_seconds": 1.8,
                    "tokens_per_second": 130 / (1.0 + 80 / warmed_remaining_tok_s),
                },
            },
            "warmed_runs": {
                "runs": 1,
                "generated_tokens": 120,
                "model_elapsed_seconds": 120 / warmed_tok_s,
                "wall_seconds": 1.2,
                "tokens_per_second": warmed_tok_s,
                "first_records": {
                    "runs": 1,
                    "records": 1,
                    "generated_tokens": 40,
                    "model_elapsed_seconds": 40 / warmed_first_tok_s,
                    "wall_seconds": 0.5,
                    "tokens_per_second": warmed_first_tok_s,
                },
                "remaining_records": {
                    "runs": 1,
                    "records": 2,
                    "generated_tokens": 80,
                    "model_elapsed_seconds": 80 / warmed_remaining_tok_s,
                    "wall_seconds": 0.7,
                    "tokens_per_second": warmed_remaining_tok_s,
                },
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
    assert report["scope_availability"]["pass"]
    assert report["token_equivalence"]["pass"]
    assert report["performance"]["pass"]
    assert report["segments"]["first_records"]["pass"]
    assert report["segments"]["remaining_records"]["pass"]
    assert report["timing_context"]["pass"]
    assert report["per_song"]["pass"]


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


def test_compare_suite_manifests_reports_segment_and_timing_regressions(tmp_path):
    module = _load_module()
    baseline = tmp_path / "baseline-suite.json"
    candidate = tmp_path / "candidate-suite.json"
    baseline.write_text(module.json.dumps(_suite_manifest(warmed_first_tok_s=100.0, timing_tok_s=50.0)))
    candidate.write_text(module.json.dumps(_suite_manifest(warmed_first_tok_s=80.0, timing_tok_s=25.0)))

    report = module.compare_suite_manifests(baseline, candidate, scope="warmed_runs")

    assert not report["segments"]["first_records"]["pass"]
    assert not report["segments"]["first_records"]["metrics"]["tokens_per_second"]["pass"]
    assert report["segments"]["remaining_records"]["pass"]
    assert not report["timing_context"]["pass"]
    assert not report["timing_context"]["metrics"]["tokens_per_second"]["pass"]


def _serial_suite_manifest(*, song1_tok_s: float = 100.0, song2_tok_s: float = 100.0):
    def run(index: int, song_index: int, tok_s: float):
        return {
            "run_index": index,
            "repeat_index": 0,
            "song_index": song_index,
            "song_id": f"song{song_index}",
            "audio_path": f"/work/song{song_index}.mp3",
            "start_time": 0,
            "end_time": 15000,
            "seed": 12345,
            "sequence_count": 10,
            "song_length_ms": 15000,
            "main_generated_tokens": 100,
            "main_model_elapsed_seconds": 100 / tok_s,
            "main_wall_seconds": 100 / tok_s,
            "main_tokens_per_second": tok_s,
            "main_token_count": 100,
            "main_token_sha256": f"hash-song{song_index}",
            "main_first_record": {
                "records": 1,
                "generated_tokens": 50,
                "model_elapsed_seconds": 50 / tok_s,
                "wall_seconds": 50 / tok_s,
                "tokens_per_second": tok_s,
            },
            "main_remaining_records": {
                "records": 1,
                "generated_tokens": 50,
                "model_elapsed_seconds": 50 / tok_s,
                "wall_seconds": 50 / tok_s,
                "tokens_per_second": tok_s,
            },
            "timing_generated_tokens": 10,
            "timing_model_elapsed_seconds": 0.1,
            "timing_wall_seconds": 0.1,
            "timing_tokens_per_second": 100.0,
        }

    runs = [run(0, 0, song1_tok_s), run(1, 1, song2_tok_s)]
    generated_tokens = sum(item["main_generated_tokens"] for item in runs)
    model_elapsed_seconds = sum(item["main_model_elapsed_seconds"] for item in runs)
    return {
        "schema_version": 3,
        "run_kind": "serial_multi_song",
        "song_count": 5,
        "seed_step": 0,
        "runs": runs,
        "aggregate": {
            "all_runs": {
                "runs": len(runs),
                "generated_tokens": generated_tokens,
                "model_elapsed_seconds": model_elapsed_seconds,
                "wall_seconds": model_elapsed_seconds,
                "tokens_per_second": generated_tokens / model_elapsed_seconds,
                "first_records": {
                    "runs": len(runs),
                    "records": len(runs),
                    "generated_tokens": 100,
                    "model_elapsed_seconds": sum(item["main_first_record"]["model_elapsed_seconds"] for item in runs),
                    "wall_seconds": sum(item["main_first_record"]["wall_seconds"] for item in runs),
                    "tokens_per_second": (
                        100 / sum(item["main_first_record"]["model_elapsed_seconds"] for item in runs)
                    ),
                },
                "remaining_records": {
                    "runs": len(runs),
                    "records": len(runs),
                    "generated_tokens": 100,
                    "model_elapsed_seconds": sum(item["main_remaining_records"]["model_elapsed_seconds"] for item in runs),
                    "wall_seconds": sum(item["main_remaining_records"]["wall_seconds"] for item in runs),
                    "tokens_per_second": (
                        100 / sum(item["main_remaining_records"]["model_elapsed_seconds"] for item in runs)
                    ),
                },
            },
            "warmed_runs": None,
        },
    }


def test_compare_suite_manifests_reports_per_song_regression_hidden_by_aggregate(tmp_path):
    module = _load_module()
    baseline = tmp_path / "baseline-suite.json"
    candidate = tmp_path / "candidate-suite.json"
    baseline.write_text(module.json.dumps(_serial_suite_manifest(song1_tok_s=100.0, song2_tok_s=100.0)))
    candidate.write_text(module.json.dumps(_serial_suite_manifest(song1_tok_s=300.0, song2_tok_s=90.0)))

    report = module.compare_suite_manifests(baseline, candidate, scope="all_runs")

    assert report["performance"]["pass"]
    assert not report["per_song"]["pass"]
    assert len(report["per_song"]["failed"]) == 1
    assert report["per_song"]["failed"][0]["song_index"] == 1


def test_compare_suite_manifests_reports_contract_mismatch(tmp_path):
    module = _load_module()
    baseline = tmp_path / "baseline-suite.json"
    candidate = tmp_path / "candidate-suite.json"
    base = _suite_manifest()
    cand = _suite_manifest()
    cand["runs"][0]["audio_path"] = "/work/different.mp3"
    baseline.write_text(module.json.dumps(base))
    candidate.write_text(module.json.dumps(cand))

    report = module.compare_suite_manifests(baseline, candidate, scope="warmed_runs")

    assert not report["shape"]["pass"]
    assert any(mismatch["key"] == "audio_path" for mismatch in report["shape"]["mismatches"])
