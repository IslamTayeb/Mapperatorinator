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


def _profile(
    *,
    tokens: list[int],
    tok_s: float,
    model_s: float,
    wall_s: float,
    seed: int = 12345,
    result_sha256: str = "same-output",
    result_size_bytes: int = 1234,
):
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
        "result_file_sha256": result_sha256,
        "result_file_size_bytes": result_size_bytes,
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
    assert report["output_artifact"]["pass"]
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


def test_compare_profiles_reports_output_artifact_mismatch(tmp_path):
    module = _load_module()
    baseline = tmp_path / "baseline.profile.json"
    candidate = tmp_path / "candidate.profile.json"
    baseline.write_text(module.json.dumps(_profile(
        tokens=[1, 2, 3],
        tok_s=100,
        model_s=10,
        wall_s=11,
        result_sha256="baseline-output",
        result_size_bytes=1234,
    )))
    candidate.write_text(module.json.dumps(_profile(
        tokens=[1, 2, 3],
        tok_s=110,
        model_s=9,
        wall_s=10,
        result_sha256="candidate-output",
        result_size_bytes=1235,
    )))

    report = module.compare_profiles(baseline, candidate, label="main_generation")

    assert not report["output_artifact"]["pass"]
    assert report["output_artifact"]["status"] == "FAIL"


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
    assert report["output_artifact_pass"]
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
            "result_file_sha256": f"result-cold-{hash_suffix}",
            "result_file_size_bytes": 1000,
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
            "result_file_sha256": f"result-warm-{hash_suffix}",
            "result_file_size_bytes": 1200,
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
    assert report["output_artifact_equivalence"]["pass"]
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
    assert not report["output_artifact_equivalence"]["pass"]
    assert report["output_artifact_equivalence"]["mismatches"]
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
            "result_file_sha256": f"result-song{song_index}",
            "result_file_size_bytes": 1000 + song_index,
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


def _static_server_manifest(
    *,
    tok_s: float = 120.0,
    scheduler_wall_seconds: float = 10.0,
    generated_tokens: int = 1200,
    result_class: str = "static_server_batch",
    server_batch_observed: bool = True,
    token_status: str = "not_checked_shared_server_rng",
    server_batch_timeout: float = 0.2,
    max_batch_size: int = 5,
):
    runs = [
        {
            "run_index": 0,
            "repeat_index": 0,
            "song_index": 0,
            "song_id": "song0",
            "audio_path": "/work/song0.mp3",
            "beatmap_path": "",
            "start_time": 71000,
            "end_time": 86000,
            "seed": 12345,
            "requested_seed": 12345,
            "sequence_count": 10,
            "song_length_ms": 15000,
            "token_equivalence_status": token_status,
            "main_generated_tokens": generated_tokens,
            "main_model_elapsed_seconds": generated_tokens / tok_s,
            "main_wall_seconds": generated_tokens / tok_s,
            "main_tokens_per_second": tok_s,
        }
    ]
    return {
        "schema_version": 1,
        "run_kind": "static_server_batch",
        "same_calculation": False,
        "throughput_claim_scope": "static_ipc_concurrent_full_song_requests",
        "token_equivalence_status": token_status,
        "song_count": 5,
        "repeats": 1,
        "max_workers": 5,
        "server_config_fingerprint": {
            "model_path": "same-model",
            "precision": "fp32",
            "attn_implementation": "sdpa",
            "max_batch_size": max_batch_size,
            "server_batch_timeout": server_batch_timeout,
            "inference_generation_compile": False,
        },
        "runs": runs,
        "aggregate": {
            "result_class": result_class,
            "server_batch_observed": server_batch_observed,
            "same_calculation": False,
            "throughput_claim_scope": "static_ipc_concurrent_full_song_requests",
            "token_equivalence_status": token_status,
            "main_generated_tokens": generated_tokens,
            "timing_generated_tokens": 0,
            "scheduler_wall_seconds": scheduler_wall_seconds,
            "request_wall_seconds_sum": scheduler_wall_seconds * 5,
            "request_wall_seconds_max": scheduler_wall_seconds,
            "request_wall_seconds_p95": scheduler_wall_seconds,
            "main_model_elapsed_seconds_sum": generated_tokens / tok_s,
            "timing_model_elapsed_seconds_sum": 0.0,
            "main_tokens_per_scheduler_second": generated_tokens / scheduler_wall_seconds,
            "timing_tokens_per_scheduler_second": 0.0,
            "main_tokens_per_request_model_second_attributed": tok_s,
            "timing_tokens_per_request_model_second_attributed": 0.0,
        },
    }


def test_compare_static_server_manifests_passes_operational_non_regression(tmp_path):
    module = _load_module()
    baseline = tmp_path / "baseline-static.json"
    candidate = tmp_path / "candidate-static.json"
    baseline.write_text(module.json.dumps(_static_server_manifest(tok_s=100.0, scheduler_wall_seconds=12.0)))
    candidate.write_text(module.json.dumps(_static_server_manifest(tok_s=110.0, scheduler_wall_seconds=10.0)))

    report = module.compare_static_server_manifests(baseline, candidate)

    assert report["contract"]["pass"]
    assert report["result_class"]["pass"]
    assert report["token_status"]["pass"]
    assert report["performance"]["pass"]


def test_compare_static_server_manifests_reports_regression_and_nonbatch(tmp_path):
    module = _load_module()
    baseline = tmp_path / "baseline-static.json"
    candidate = tmp_path / "candidate-static.json"
    baseline.write_text(module.json.dumps(_static_server_manifest(tok_s=100.0, scheduler_wall_seconds=10.0)))
    candidate.write_text(module.json.dumps(_static_server_manifest(
        tok_s=90.0,
        scheduler_wall_seconds=12.0,
        generated_tokens=900,
        result_class="static_server_no_batch_observed",
        server_batch_observed=False,
        token_status="PASS",
    )))

    report = module.compare_static_server_manifests(baseline, candidate)

    assert report["contract"]["pass"]
    assert not report["result_class"]["pass"]
    assert not report["token_status"]["pass"]
    assert not report["performance"]["pass"]
    assert not report["performance"]["generated_tokens_non_decreasing"]


def test_compare_static_server_manifests_can_allow_timeout_knob_change(tmp_path):
    module = _load_module()
    baseline = tmp_path / "baseline-static.json"
    candidate = tmp_path / "candidate-static.json"
    baseline.write_text(module.json.dumps(_static_server_manifest(server_batch_timeout=0.2)))
    candidate.write_text(module.json.dumps(_static_server_manifest(
        tok_s=130.0,
        scheduler_wall_seconds=9.0,
        server_batch_timeout=0.02,
    )))

    strict_report = module.compare_static_server_manifests(baseline, candidate)
    allowed_report = module.compare_static_server_manifests(
        baseline,
        candidate,
        allow_server_batch_timeout_change=True,
    )

    assert not strict_report["contract"]["pass"]
    assert allowed_report["contract"]["pass"]


def test_compare_static_server_manifests_can_allow_max_batch_size_knob_change(tmp_path):
    module = _load_module()
    baseline = tmp_path / "baseline-static.json"
    candidate = tmp_path / "candidate-static.json"
    baseline.write_text(module.json.dumps(_static_server_manifest(max_batch_size=5)))
    candidate.write_text(module.json.dumps(_static_server_manifest(
        tok_s=130.0,
        scheduler_wall_seconds=9.0,
        max_batch_size=10,
    )))

    strict_report = module.compare_static_server_manifests(baseline, candidate)
    allowed_report = module.compare_static_server_manifests(
        baseline,
        candidate,
        allow_server_max_batch_size_change=True,
    )

    assert not strict_report["contract"]["pass"]
    assert allowed_report["contract"]["pass"]


def _continuous_scheduler_manifest(
    *,
    token_hash: str = "same-token-hash",
    generated_tokens: int = 3,
    stop_reason: str = "eos",
    result_class: str = "continuous_scheduler_dry_run",
    model_generation_executed: bool = False,
    token_status: str = "scheduler_only_scripted_tokens",
    active_batch_size_histogram: dict | None = None,
    scheduler_cpu_wall_seconds: float = 0.01,
):
    request = {
        "request_id": "song0-window0",
        "prompt_tokens": 4,
        "max_new_tokens": 4,
        "eos_token_ids": [99],
        "planned_arrival_step": 0,
        "generated_tokens": [1, 2, 99][:generated_tokens],
        "generated_token_count": generated_tokens,
        "generated_token_sha256": token_hash,
        "stop_reason": stop_reason,
        "enqueue_step": 0,
        "activation_step": 0,
        "finish_step": 2,
        "queue_wait_steps": 0,
        "decode_steps": 3,
        "latency_steps": 3,
        "cache_slot_id": 0,
        "slot_generation": 1,
        "metadata": {"song_id": "song0", "window_index": 0},
        "token_equivalence_status": token_status,
        "initial_rng_state_hash": "rng-before",
        "final_rng_state_hash": "rng-after",
        "logits_processor_state_hash": "logits-state",
        "cache_state_hash": "cache-state",
    }
    histogram = active_batch_size_histogram or {"1": 3}
    return {
        "schema_version": 1,
        "run_kind": "continuous_scheduler_dry_run",
        "result_class": result_class,
        "model_generation_executed": model_generation_executed,
        "token_equivalence_status": token_status,
        "request_count": 1,
        "config": {
            "max_active_sequences": 1,
            "max_wait_ms": 0,
            "prefill_policy": "serial",
            "decode_order_policy": "arrival_order",
            "rng_policy": "serial_global",
        },
        "compatibility_key": [["do_sample", True]],
        "active_batch_size_histogram": histogram,
        "steps": [
            {
                "step_index": 0,
                "activated": [{"request_id": "song0-window0", "cache_slot_id": 0, "slot_generation": 1}],
                "decoded": [{"request_id": "song0-window0", "slot_id": 0, "slot_generation": 1, "token_id": 1, "stop_reason": None}],
                "finished": [],
                "active_batch_size": 1,
            },
            {
                "step_index": 1,
                "activated": [],
                "decoded": [{"request_id": "song0-window0", "slot_id": 0, "slot_generation": 1, "token_id": 2, "stop_reason": None}],
                "finished": [],
                "active_batch_size": 1,
            },
            {
                "step_index": 2,
                "activated": [],
                "decoded": [
                    {
                        "request_id": "song0-window0",
                        "slot_id": 0,
                        "slot_generation": 1,
                        "token_id": 99,
                        "stop_reason": stop_reason,
                    }
                ],
                "finished": [
                    {
                        "request_id": "song0-window0",
                        "cache_slot_id": 0,
                        "slot_generation": 1,
                        "finish_step": 2,
                        "decode_steps": 3,
                        "latency_steps": 3,
                        "stop_reason": stop_reason,
                        "generated_tokens": [1, 2, 99][:generated_tokens],
                    }
                ],
                "active_batch_size": 1,
            },
        ],
        "requests": [request],
        "cache_slot_events": [
            {"event": "acquire", "step_index": 0, "request_id": "song0-window0", "cache_slot_id": 0, "slot_generation": 1},
            {"event": "release", "step_index": 2, "request_id": "song0-window0", "cache_slot_id": 0, "slot_generation": 1, "stop_reason": stop_reason},
        ],
        "aggregate": {
            "result_class": result_class,
            "model_generation_executed": model_generation_executed,
            "request_count": 1,
            "completed_request_count": 1,
            "total_generated_tokens": generated_tokens,
            "scheduler_step_count": 3,
            "idle_step_count": 0,
            "scheduler_cpu_wall_seconds": scheduler_cpu_wall_seconds,
            "scheduler_tokens_per_cpu_second": generated_tokens / scheduler_cpu_wall_seconds,
            "active_batch_size_histogram": histogram,
            "planned_arrival_step_histogram": {"0": 1},
            "stop_reason_counts": {stop_reason: 1},
            "cache_slot_acquire_count": 1,
            "cache_slot_release_count": 1,
        },
    }


def test_compare_continuous_scheduler_manifests_passes_scheduler_only_equivalence(tmp_path):
    module = _load_module()
    baseline = tmp_path / "baseline-continuous.json"
    candidate = tmp_path / "candidate-continuous.json"
    baseline.write_text(module.json.dumps(_continuous_scheduler_manifest()))
    candidate.write_text(module.json.dumps(_continuous_scheduler_manifest(scheduler_cpu_wall_seconds=0.02)))

    report = module.compare_continuous_scheduler_manifests(baseline, candidate)

    assert report["contract"]["pass"]
    assert report["result_class"]["pass"]
    assert report["scripted_token_equivalence"]["pass"]
    assert report["state_ledger"]["pass"]
    assert report["scheduling_shape"]["pass"]
    assert not report["cpu_timing"]["pass"]


def test_compare_continuous_scheduler_manifests_reports_token_and_shape_mismatch(tmp_path):
    module = _load_module()
    baseline = tmp_path / "baseline-continuous.json"
    candidate = tmp_path / "candidate-continuous.json"
    baseline.write_text(module.json.dumps(_continuous_scheduler_manifest()))
    candidate.write_text(module.json.dumps(_continuous_scheduler_manifest(
        token_hash="different-token-hash",
        generated_tokens=2,
        stop_reason="max_new_tokens",
        active_batch_size_histogram={"1": 1, "2": 1},
    )))

    report = module.compare_continuous_scheduler_manifests(baseline, candidate)

    assert report["contract"]["pass"]
    assert report["result_class"]["pass"]
    assert not report["scripted_token_equivalence"]["pass"]
    assert report["state_ledger"]["pass"]
    assert not report["scheduling_shape"]["pass"]


def test_compare_continuous_scheduler_manifests_reports_state_ledger_mismatch(tmp_path):
    module = _load_module()
    baseline = tmp_path / "baseline-continuous.json"
    candidate = tmp_path / "candidate-continuous.json"
    base_manifest = _continuous_scheduler_manifest()
    candidate_manifest = _continuous_scheduler_manifest()
    candidate_manifest["requests"][0]["final_rng_state_hash"] = "different-rng-after"
    baseline.write_text(module.json.dumps(base_manifest))
    candidate.write_text(module.json.dumps(candidate_manifest))

    report = module.compare_continuous_scheduler_manifests(baseline, candidate)

    assert report["contract"]["pass"]
    assert report["result_class"]["pass"]
    assert report["scripted_token_equivalence"]["pass"]
    assert not report["state_ledger"]["pass"]
    assert report["scheduling_shape"]["pass"]


def test_compare_continuous_scheduler_manifests_rejects_model_backed_manifest(tmp_path):
    module = _load_module()
    baseline = tmp_path / "baseline-continuous.json"
    candidate = tmp_path / "candidate-continuous.json"
    baseline.write_text(module.json.dumps(_continuous_scheduler_manifest()))
    candidate.write_text(module.json.dumps(_continuous_scheduler_manifest(
        result_class="continuous_model_runtime",
        model_generation_executed=True,
        token_status="PASS",
    )))

    report = module.compare_continuous_scheduler_manifests(baseline, candidate)

    assert not report["result_class"]["pass"]
    assert not report["scripted_token_equivalence"]["pass"]
