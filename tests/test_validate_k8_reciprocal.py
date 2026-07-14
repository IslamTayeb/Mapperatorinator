from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from utils import summarize_inference_profile as exact
from utils import validate_k8_reciprocal as gate


def _osu(path: Path, *, x: int) -> Path:
    path.write_text(
        "osu file format v14\n\n"
        "[TimingPoints]\n0,500,4,2,1,70,1,0\n\n"
        f"[HitObjects]\n{x},192,1000,1,0,0:0:0:0:\n",
        encoding="utf-8",
    )
    return path


def _k8(*, logical: int = 3, physical: int = 9, block_replays: int = 1) -> dict:
    wasted = physical - logical
    return {
        "block_size": 8,
        "prefill_steps": 1,
        "eligible_steps": 8 * block_replays,
        "block_replays": block_replays,
        "remainder_steps": physical - 1 - 8 * block_replays,
        "status_reads": 2,
        "copy_calls": 4,
        "copy_bytes": 64,
        "capture_seconds": 0.1,
        "peak_vram_bytes": 1024,
        "physical_steps": physical,
        "logical_steps": logical,
        "wasted_steps": wasted,
        "rng_policy": gate.K8_POLICY,
        "rng_exact": False,
        "rng_drift": "documented_counter_based_per_window",
        "rng_request_seed": 12345,
        "rng_window_identity": 1,
        "rng_early_eos_isolation": True,
        "sampling_mode": "sample",
        "prompt_seed_d2h_copy_calls": 1,
        "prompt_seed_d2h_copy_bytes": 16,
        "prompt_seed_setup_seconds": 0.01,
        "processor_signature_d2h_copy_calls": 2,
        "processor_signature_d2h_copy_bytes": 32,
        "processor_signature_setup_seconds": 0.02,
        "parent_backend": "cuda_python_child_graphs",
        "capture_state_restore_synchronized": True,
    }


def _profile(
    result: Path,
    *,
    candidate: bool,
    main_tokens: list[int],
    main_seconds: float,
    k8: dict | None = None,
) -> dict:
    content = result.read_bytes()
    metadata = {key: "same" for key in exact.CONTRACT_METADATA_KEYS}
    metadata.update(
        inference_engine="optimized",
        precision="fp32",
        seed=12345,
        use_server=False,
        parallel=False,
        profile_pass_kind="untraced_control",
        authoritative_performance=True,
        profile_detail_ranges=False,
        profile_cuda_capture=False,
        result_file_path=str(result),
        result_file_sha256=hashlib.sha256(content).hexdigest(),
        result_file_size_bytes=len(content),
    )
    main = {
        "profile_label": "main_generation",
        "context_type": "MAP",
        "mode": "sequential",
        "sequence_index": 0,
        "generated_tokens": len(main_tokens),
        "generated_token_ids": main_tokens,
        "model_elapsed_seconds": main_seconds,
        "wall_seconds": main_seconds + 0.1,
        "precision": "fp32",
        "optimized_cuda_graphs": {"graph_count": 1, "decode_replays": 3},
    }
    if candidate:
        main["optimized_cuda_graphs"]["k8_candidate"] = k8 or _k8(
            logical=len(main_tokens)
        )
    timing = {
        "profile_label": "timing_context",
        "context_type": "TIMING",
        "mode": "sequential",
        "sequence_index": 0,
        "generated_tokens": 2,
        "generated_token_ids": [8, 9],
        "model_elapsed_seconds": 0.5,
        "wall_seconds": 0.6,
        "precision": "fp32",
        "optimized_cuda_graphs": {"graph_count": 1, "decode_replays": 2},
    }
    if candidate:
        timing["optimized_cuda_graphs"]["k8_candidate"] = _k8(
            logical=2,
            physical=9,
        )
    return {
        "schema_version": 1,
        "metadata": metadata,
        "summary": {"stage_wall_seconds": {"inference": 2.0}},
        "stages": [{"name": "inference", "wall_seconds": 2.0}],
        "generation": [timing, main],
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _profiles(tmp_path: Path, *, diverge: bool = True) -> dict[str, Path]:
    baseline_osu = _osu(tmp_path / "baseline.osu", x=64)
    candidate_osu = _osu(tmp_path / "candidate.osu", x=128 if diverge else 64)
    baseline_tokens = [1, 2, 3]
    candidate_tokens = [1, 7, 3] if diverge else baseline_tokens
    return {
        "baseline_first": _write(
            tmp_path / "baseline-first.json",
            _profile(
                baseline_osu,
                candidate=False,
                main_tokens=baseline_tokens,
                main_seconds=1.0,
            ),
        ),
        "candidate_second": _write(
            tmp_path / "candidate-second.json",
            _profile(
                candidate_osu,
                candidate=True,
                main_tokens=candidate_tokens,
                main_seconds=0.8,
            ),
        ),
        "candidate_first": _write(
            tmp_path / "candidate-first.json",
            _profile(
                candidate_osu,
                candidate=True,
                main_tokens=candidate_tokens,
                main_seconds=0.8,
            ),
        ),
        "baseline_second": _write(
            tmp_path / "baseline-second.json",
            _profile(
                baseline_osu,
                candidate=False,
                main_tokens=baseline_tokens,
                main_seconds=1.0,
            ),
        ),
    }


def _summarize(paths: dict[str, Path], *, scope: str = "smoke") -> dict:
    return gate.summarize(
        paths,
        scope=scope,
        minimum_smoke_speedup_pct=0.0,
        full_fixed_work_saving_seconds=0.1,
    )


def test_changed_tokens_are_reported_but_do_not_fake_fixed_work(tmp_path):
    report = _summarize(_profiles(tmp_path, diverge=True))

    assert report["feasibility_pass"]
    assert report["candidate_repeatability"]["pass"]
    assert not report["fixed_work"]["available"]
    assert report["fixed_work"]["mean_main_model_seconds_saving"] is None
    assert report["orders"]["baseline_first"]["main_generation"][
        "work_comparability"
    ] == "changed_work"
    assert not report["orders"]["baseline_first"]["output"]["artifact_equal"]
    assert report["orders"]["baseline_first"]["output"]["structure"]["equal"]
    assert report["orders"]["baseline_first"]["output"]["structure"][
        "numeric_deltas"
    ]["hit_objects"] == 0
    assert "equal" in report["orders"]["baseline_first"]["graph_cache"]
    candidate = report["orders"]["baseline_first"]["main_generation"]["candidate"]
    assert candidate["logical_steps"] == 3
    assert candidate["physical_steps"] == 9
    assert candidate["wasted_steps"] == 6


def test_exact_fixed_work_can_pass_full_saving_gate(tmp_path):
    report = _summarize(_profiles(tmp_path, diverge=False), scope="full")

    assert report["feasibility_pass"]
    assert report["fixed_work"]["available"]
    assert math.isclose(
        report["fixed_work"]["mean_main_model_seconds_saving"],
        0.2,
    )
    assert report["fixed_work"]["promotion_pass"]


def test_missing_main_k8_replay_fails(tmp_path):
    paths = _profiles(tmp_path)
    payload = json.loads(paths["candidate_first"].read_text())
    payload["generation"][1]["optimized_cuda_graphs"].pop("k8_candidate")
    _write(paths["candidate_first"], payload)

    report = _summarize(paths)

    assert not report["feasibility_pass"]
    assert any("no main K8" in failure for failure in report["failures"])


def test_timing_k8_is_profiled_independently(tmp_path):
    report = _summarize(_profiles(tmp_path))

    timing = report["candidate_k8_totals"]["candidate_first"]["timing_context"]
    assert timing["records"] == 1
    assert timing["block_replays"] == 1
    candidate = report["orders"]["baseline_first"]["timing_context"]["candidate"]
    assert candidate["logical_steps"] == 2
    assert candidate["physical_steps"] == 9


def test_timing_must_not_silently_omit_k8(tmp_path):
    paths = _profiles(tmp_path)
    payload = json.loads(paths["candidate_first"].read_text())
    payload["generation"][0]["optimized_cuda_graphs"].pop("k8_candidate")
    _write(paths["candidate_first"], payload)

    report = _summarize(paths)

    assert not report["feasibility_pass"]
    assert any("silently omitted K8" in failure for failure in report["failures"])


def test_timing_explicit_fallback_reason_is_reported(tmp_path):
    paths = _profiles(tmp_path)
    for name in ("candidate_first", "candidate_second"):
        payload = json.loads(paths[name].read_text())
        graphs = payload["generation"][0]["optimized_cuda_graphs"]
        graphs.pop("k8_candidate")
        graphs["k8_fallback_reason"] = "timing prompt never reached one full K8 block"
        _write(paths[name], payload)

    report = _summarize(paths)

    assert report["feasibility_pass"]
    timing = report["candidate_k8_totals"]["candidate_first"]["timing_context"]
    assert timing["records"] == 0
    assert timing["fallback_records"] == 1


def test_accounting_mismatch_fails_loudly(tmp_path):
    paths = _profiles(tmp_path)
    payload = json.loads(paths["candidate_first"].read_text())
    payload["generation"][1]["optimized_cuda_graphs"]["k8_candidate"][
        "physical_steps"
    ] = 10
    _write(paths["candidate_first"], payload)

    report = _summarize(paths)

    assert not report["feasibility_pass"]
    assert any("physical work diverged" in failure for failure in report["failures"])


def test_candidate_repeatability_is_a_hard_gate(tmp_path):
    paths = _profiles(tmp_path)
    payload = json.loads(paths["candidate_first"].read_text())
    payload["generation"][1]["generated_token_ids"] = [1, 6, 3]
    _write(paths["candidate_first"], payload)

    report = _summarize(paths)

    assert not report["feasibility_pass"]
    assert "candidate reciprocal repeats are not deterministic" in report["failures"]


def test_logical_steps_must_match_generated_tokens(tmp_path):
    paths = _profiles(tmp_path)
    payload = json.loads(paths["candidate_first"].read_text())
    stats = payload["generation"][1]["optimized_cuda_graphs"]["k8_candidate"]
    stats["logical_steps"] = 4
    stats["wasted_steps"] = 5
    _write(paths["candidate_first"], payload)

    report = _summarize(paths)

    assert not report["feasibility_pass"]
    assert any("does not equal generated_tokens" in failure for failure in report["failures"])


def test_candidate_requires_synchronized_capture_restore_evidence(tmp_path):
    paths = _profiles(tmp_path)
    payload = json.loads(paths["candidate_first"].read_text())
    stats = payload["generation"][1]["optimized_cuda_graphs"]["k8_candidate"]
    stats.pop("capture_state_restore_synchronized")
    _write(paths["candidate_first"], payload)

    report = _summarize(paths)

    assert not report["feasibility_pass"]
    assert any("lacks synchronized capture restore" in failure for failure in report["failures"])


def test_candidate_graph_cache_signature_must_repeat(tmp_path):
    paths = _profiles(tmp_path)
    payload = json.loads(paths["candidate_first"].read_text())
    payload["generation"][1]["optimized_cuda_graphs"]["decode_replays"] += 1
    _write(paths["candidate_first"], payload)

    report = _summarize(paths)

    assert not report["feasibility_pass"]
    assert not report["graph_cache_repeatability"]["candidate_pass"]
    assert "candidate reciprocal graph-cache signatures differ" in report["failures"]
