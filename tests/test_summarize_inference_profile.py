from __future__ import annotations

import json
from pathlib import Path

from utils import summarize_inference_profile as profiles


def _profile(
    *,
    main_tokens: list[int] | None = None,
    timing_tokens: list[int] | None = None,
    model_scale: float = 1.0,
    wall_scale: float = 1.0,
    output_hash: str = "same-output",
    seed: int = 12345,
) -> dict:
    main_tokens = [1, 2, 3] if main_tokens is None else main_tokens
    timing_tokens = [8, 9] if timing_tokens is None else timing_tokens
    metadata = {
        key: "same"
        for key in profiles.CONTRACT_METADATA_KEYS
    }
    metadata.update(
        seed=seed,
        use_server=False,
        parallel=False,
        result_file_sha256=output_hash,
        result_file_size_bytes=1234,
    )
    generation = []
    for label, context, tokens, model_seconds, wall_seconds in (
        ("main_generation", "MAP", main_tokens, 1.0 * model_scale, 1.1 * wall_scale),
        ("timing_context", "TIMING", timing_tokens, 0.5 * model_scale, 0.6 * wall_scale),
    ):
        generation.append(
            {
                "profile_label": label,
                "mode": "sequential",
                "context_type": context,
                "sequence_index": 0,
                "generated_tokens": len(tokens),
                "generated_token_ids": tokens,
                "model_elapsed_seconds": model_seconds,
                "wall_seconds": wall_seconds,
            }
        )
    return {
        "metadata": metadata,
        "stages": [{"name": "inference", "wall_seconds": 2.0 * wall_scale}],
        "generation": generation,
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_direct_comparison_passes_exact_faster_candidate(tmp_path):
    baseline = _write(tmp_path / "baseline.json", _profile())
    candidate = _write(
        tmp_path / "candidate.json",
        _profile(model_scale=0.9, wall_scale=0.9),
    )

    report = profiles.compare_profiles_for_labels(
        baseline,
        candidate,
        labels=list(profiles.DEFAULT_LABELS),
        regression_tolerance_pct=0.0,
    )

    assert report["same_calculation_pass"]
    assert report["token_equivalence_pass"]
    assert report["output_artifact_pass"]
    assert report["performance_pass"]


def test_comparison_reports_exactness_and_contract_failures(tmp_path):
    baseline = _write(tmp_path / "baseline.json", _profile())
    candidate = _write(
        tmp_path / "candidate.json",
        _profile(main_tokens=[1, 7, 3], output_hash="different", seed=9),
    )

    report = profiles.compare_profiles(
        baseline,
        candidate,
        label="main_generation",
    )

    assert not report["same_calculation"]["pass"]
    assert report["same_calculation"]["mismatches"] == [
        {"key": "seed", "baseline": 12345, "candidate": 9}
    ]
    assert not report["token_equivalence"]["pass"]
    assert report["token_equivalence"]["first_mismatch"] == 1
    assert report["output_artifact"]["status"] == "FAIL"


def test_missing_tokens_or_artifact_never_count_as_exact(tmp_path):
    baseline_profile = _profile()
    candidate_profile = _profile()
    del candidate_profile["generation"][0]["generated_token_ids"]
    del candidate_profile["metadata"]["result_file_sha256"]
    baseline = _write(tmp_path / "baseline.json", baseline_profile)
    candidate = _write(tmp_path / "candidate.json", candidate_profile)

    report = profiles.compare_profiles(baseline, candidate, label="main_generation")

    assert report["token_equivalence"]["status"] == "not_checked"
    assert not report["token_equivalence"]["pass"]
    assert report["output_artifact"]["status"] == "not_checked"
    assert not report["output_artifact"]["pass"]


def test_five_percent_boundary_is_inclusive(tmp_path):
    baseline = _write(tmp_path / "baseline.json", _profile())
    candidate = _write(
        tmp_path / "candidate.json",
        _profile(model_scale=1.05, wall_scale=1.05),
    )

    report = profiles.compare_profiles(
        baseline,
        candidate,
        label="main_generation",
        regression_tolerance_pct=5.0,
    )

    assert report["performance"]["metrics"]["model_elapsed_seconds"]["pass"]
    assert report["performance"]["metrics"]["outer_wall_seconds"]["pass"]
    assert report["performance"]["pass"]


def test_record_shape_and_counts_must_match(tmp_path):
    baseline_profile = _profile()
    candidate_profile = _profile()
    candidate_profile["generation"][0]["sequence_index"] = 1
    baseline = _write(tmp_path / "baseline.json", baseline_profile)
    candidate = _write(tmp_path / "candidate.json", candidate_profile)

    report = profiles.compare_profiles(baseline, candidate, label="main_generation")

    assert not report["performance"]["records_match"]
    assert not report["performance"]["pass"]


def test_reciprocal_comparison_requires_both_orders(tmp_path):
    baseline_first = _write(tmp_path / "baseline-first.json", _profile())
    candidate_second = _write(tmp_path / "candidate-second.json", _profile(model_scale=0.9, wall_scale=0.9))
    candidate_first = _write(tmp_path / "candidate-first.json", _profile(model_scale=1.1, wall_scale=1.1))
    baseline_second = _write(tmp_path / "baseline-second.json", _profile())

    report = profiles.compare_reciprocal_profiles(
        baseline_first,
        candidate_second,
        candidate_first,
        baseline_second,
        labels=list(profiles.DEFAULT_LABELS),
        regression_tolerance_pct=5.0,
    )

    assert report["same_calculation_pass"]
    assert report["token_equivalence_pass"]
    assert report["output_artifact_pass"]
    assert not report["performance_pass"]
    assert report["orders"]["baseline_first"]["performance_pass"]
    assert not report["orders"]["candidate_first"]["performance_pass"]
