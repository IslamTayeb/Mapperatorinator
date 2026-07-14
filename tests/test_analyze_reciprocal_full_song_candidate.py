import json
from pathlib import Path

import pytest

from utils.analyze_reciprocal_full_song_candidate import (
    CandidateAnalysisError,
    RUN_ORDER,
    analyze,
    text_report,
)


STAGE_NAMES = (
    "compile_args",
    "setup_inference_environment",
    "load_main_model",
    "load_timing_model",
    "build_generation_config",
    "validate_inputs",
    "setup_processors",
    "audio_load",
    "audio_segment",
    "timing_context_generation",
    "timing_context_postprocess",
    "main_generation",
    "merge_generated_events",
    "resnap_events",
    "postprocess_generate_osu",
    "write_osu",
)


def _osu(*, timing_points: int = 1, hit_objects: int = 1) -> str:
    timing = "\n".join(
        f"{index * 500},500,4,2,1,50,1,0" for index in range(timing_points)
    )
    objects = "\n".join(
        f"64,192,{1000 + index * 100},1,0,0:0:0:0:" for index in range(hit_objects)
    )
    return (
        "osu file format v14\n\n[TimingPoints]\n"
        + timing
        + "\n\n[HitObjects]\n"
        + objects
        + "\n"
    )


def _stages(*, timing_wall: float, main_wall: float) -> list[dict]:
    walls = {
        "compile_args": 0.2,
        "setup_inference_environment": 0.1,
        "load_main_model": 1.0,
        "load_timing_model": 1.1,
        "build_generation_config": 0.1,
        "validate_inputs": 0.1,
        "setup_processors": 0.1,
        "audio_load": 0.2,
        "audio_segment": 0.2,
        "timing_context_generation": timing_wall,
        "timing_context_postprocess": 0.3,
        "main_generation": main_wall,
        "merge_generated_events": 0.1,
        "resnap_events": 0.1,
        "postprocess_generate_osu": 0.2,
        "write_osu": 0.1,
    }
    records = []
    cursor = 10.0
    for index, name in enumerate(STAGE_NAMES):
        wall = walls[name]
        records.append(
            {
                "name": name,
                "wall_seconds": wall,
                "started_at_perf_counter_seconds": cursor,
                "finished_at_perf_counter_seconds": cursor + wall,
                "cuda_memory_allocated_mb": 1000.0 + index,
                "cuda_max_memory_allocated_mb": 1500.0 + index,
            }
        )
        cursor += wall + 0.01
    return records


def _generation_record(
    label: str,
    *,
    tokens: list[int],
    model_seconds: float,
    precision: str,
    split_kv: bool,
) -> dict:
    dispatch = {
        "native_q1_rope_cache_self_attention": 12,
        "native_q1_self_attention": 0,
        "q1_bmm_cross_attention": 24,
        "native_cross_mlp_tail": 12,
    }
    if split_kv:
        dispatch["native_q1_rope_cache_self_attention_split_kv_8"] = 12
        dispatch[
            "native_q1_rope_cache_self_attention_split_kv_8_prefix_640"
        ] = 12
    return {
        "profile_label": label,
        "context_type": "timing" if label == "timing_context" else "map",
        "mode": "generation",
        "sequence_index": 0,
        "precision": precision,
        "generated_tokens": len(tokens),
        "generated_tokens_per_sample": [len(tokens)],
        "generated_token_ids": tokens,
        "model_elapsed_seconds": model_seconds,
        "wall_seconds": model_seconds + 0.1,
        "decode_graph_capture_seconds_delta": 0.05,
        "decoder_loop_backend": "active_prefix_cuda_graph",
        "torch_compile_enabled": False,
        "generation_compile_enabled": False,
        "optimized_effective_config_version": "accepted",
        "optimized_dispatch_capture_hits": dispatch,
        "optimized_cuda_graphs": {
            "graph_count": 1,
            "decode_replays": len(tokens) - 1,
            "capture_seconds": 0.05,
            "buckets": {"128": {"graph_count": 1, "decode_replays": len(tokens) - 1, "capture_seconds": 0.05}},
        },
        "native_cross_mlp_tail_requested": True,
        "native_cross_mlp_tail_enabled": True,
        "cuda_memory_allocated_mb": 1400.0,
        "cuda_max_memory_allocated_mb": 1600.0,
    }


def _profile(
    result_path: Path,
    *,
    main_model_seconds: float,
    timing_model_seconds: float = 2.0,
    main_tokens: list[int] | None = None,
    timing_tokens: list[int] | None = None,
    precision: str = "fp32",
    split_kv: bool = False,
) -> dict:
    main_tokens = main_tokens or [10, 11, 12, 13]
    timing_tokens = timing_tokens or [1, 2]
    effective = {
        "version": "accepted",
        "precision": precision,
        "batch_size": 1,
        "decoder_loop_backend": "active_prefix_cuda_graph",
        "q1_bmm_cross_attention": True,
        "native_decode_kernels": True,
        "native_q1_self_attention": True,
        "native_q1_rope_cache_self_attention": True,
    }
    return {
        "schema_version": 1,
        "metadata": {
            "profile_pass_kind": "untraced_control",
            "authoritative_performance": True,
            "model_path": "model",
            "audio_path": "/audio/song.mp3",
            "beatmap_path": "",
            "seed": 12345,
            "precision": precision,
            "attn_implementation": "sdpa",
            "inference_engine": "optimized",
            "use_server": False,
            "parallel": False,
            "max_batch_size": 32,
            "temperature": 0.9,
            "timing_temperature": 0.1,
            "mania_column_temperature": 0.8,
            "taiko_hit_temperature": 0.8,
            "timeshift_bias": 0.0,
            "top_p": 0.9,
            "top_k": 0,
            "do_sample": True,
            "num_beams": 1,
            "cfg_scale": 1.0,
            "lookback": 0.5,
            "lookahead": 0.4,
            "start_time": None,
            "end_time": None,
            "in_context": [],
            "output_type": ["timing", "map", "sv"],
            "sequence_count": 1,
            "song_length_ms": 1000.0,
            "optimized_effective_config_version": "accepted",
            "optimized_effective_config": effective,
            "optimized_runtime_owner": "osuT5.osuT5.inference.optimized.single.engine",
            "optimized_result_class": "documented-drift",
            "result_file_path": str(result_path),
        },
        "stages": _stages(
            timing_wall=timing_model_seconds + 0.4,
            main_wall=main_model_seconds + 0.5,
        ),
        "generation": [
            _generation_record(
                "timing_context",
                tokens=timing_tokens,
                model_seconds=timing_model_seconds,
                precision=precision,
                split_kv=split_kv,
            ),
            _generation_record(
                "main_generation",
                tokens=main_tokens,
                model_seconds=main_model_seconds,
                precision=precision,
                split_kv=split_kv,
            ),
        ],
    }


def _write_run(
    root: Path,
    role: str,
    *,
    main_seconds: float,
    osu: str | None = None,
    **profile_kwargs,
) -> Path:
    osu_path = root / f"{role}.osu"
    osu_path.write_text(osu or _osu(), encoding="utf-8")
    profile_path = root / f"{role}.profile.json"
    profile_path.write_text(
        json.dumps(
            _profile(osu_path, main_model_seconds=main_seconds, **profile_kwargs)
        ),
        encoding="utf-8",
    )
    return profile_path


def _four_runs(tmp_path: Path, **candidate_kwargs) -> dict[str, Path]:
    return {
        "baseline_first": _write_run(
            tmp_path, "baseline_first", main_seconds=10.0
        ),
        "candidate_first": _write_run(
            tmp_path,
            "candidate_first",
            main_seconds=8.0,
            **candidate_kwargs,
        ),
        "candidate_second": _write_run(
            tmp_path,
            "candidate_second",
            main_seconds=8.4,
            **candidate_kwargs,
        ),
        "baseline_second": _write_run(
            tmp_path, "baseline_second", main_seconds=10.4
        ),
    }


def test_exact_fp32_reports_reciprocal_order_aware_metrics(tmp_path: Path) -> None:
    report = analyze(_four_runs(tmp_path))

    assert report["mode"] == "exact-fp32"
    assert report["run_order"] == list(RUN_ORDER)
    assert report["parity"]["cross_candidate_exact"] is True
    main = report["metrics"]["main_model_seconds"]
    assert main["baseline_median"] == pytest.approx(10.2)
    assert main["candidate_median"] == pytest.approx(8.2)
    assert main["improvement"] == pytest.approx(2.0)
    assert main["reciprocal_improvement_median"] == pytest.approx(2.0)
    assert main["reciprocal_improvement_range"] == pytest.approx(0.0)
    tps = report["metrics"]["main_tps"]
    assert tps["direction"] == "higher_is_better"
    assert tps["improvement"] > 0
    assert "metric.complete_request_wall_seconds=" in text_report(report)


def test_exact_dispatch_delta_requires_a_used_explicit_pattern(tmp_path: Path) -> None:
    profiles = _four_runs(tmp_path, split_kv=True)
    with pytest.raises(CandidateAnalysisError, match="undeclared"):
        analyze(profiles)

    patterns = [
        "records.*[[]0].optimized_dispatch_capture_hits."
        "native_q1_rope_cache_self_attention*"
    ]
    report = analyze(profiles, allowed_dispatch_deltas=patterns)
    topology = report["parity"]["dispatch_cache_topology"]
    assert topology["pass"] is True
    assert topology["differences"]

    with pytest.raises(CandidateAnalysisError, match="unused"):
        analyze(
            profiles,
            allowed_dispatch_deltas=[*patterns, "records.*.does_not_exist"],
        )


def test_exact_mode_rejects_token_or_final_map_divergence(tmp_path: Path) -> None:
    profiles = _four_runs(tmp_path, main_tokens=[10, 99, 12, 13])
    with pytest.raises(CandidateAnalysisError, match="tokens, stopping, or final OSU"):
        analyze(profiles)

    profiles = _four_runs(tmp_path, osu=_osu(hit_objects=2))
    with pytest.raises(CandidateAnalysisError, match="tokens, stopping, or final OSU"):
        analyze(profiles)


def test_relaxed_mode_reports_token_stopping_structure_and_map_divergence(
    tmp_path: Path,
) -> None:
    profiles = _four_runs(
        tmp_path,
        precision="fp16",
        main_tokens=[10, 99, 12],
        timing_tokens=[1, 9],
        osu=_osu(timing_points=2, hit_objects=2),
    )
    report = analyze(profiles, mode="relaxed")

    parity = report["parity"]
    assert parity["claim"] == "relaxed-nonexact"
    assert parity["cross_candidate_exact"] is False
    assert parity["token_and_stopping_divergence"]["main_generation"][
        "aligned_mismatches"
    ] == 1
    assert not parity["token_and_stopping_divergence"]["main_generation"][
        "stopping_equal"
    ]
    assert parity["output_divergence"]["scalar_deltas"]["timing_points"] == 1
    assert parity["output_divergence"]["scalar_deltas"]["hit_objects"] == 1
    assert parity["output_divergence"]["final_map_equal"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda profile: profile.update(schema_version=2), "schema_version"),
        (
            lambda profile: profile["stages"][0].update(wall_seconds=float("nan")),
            "finite",
        ),
        (
            lambda profile: profile["metadata"].update(
                authoritative_performance=False
            ),
            "authoritative_performance",
        ),
    ],
)
def test_malformed_profiles_fail_loudly(tmp_path: Path, mutation, message: str) -> None:
    profiles = _four_runs(tmp_path)
    path = profiles["candidate_first"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CandidateAnalysisError, match=message):
        analyze(profiles)


def test_stage_schema_must_match_across_all_four_runs(tmp_path: Path) -> None:
    profiles = _four_runs(tmp_path)
    path = profiles["candidate_second"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["stages"][10]["name"] = "super_timing_postprocess"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CandidateAnalysisError, match="stage sequence"):
        analyze(profiles)
