import json
from pathlib import Path

import pytest

from utils.fixed_seed_inference import (
    DETERMINISM_POLICY_VERSION,
    REQUIRED_CUBLAS_WORKSPACE_CONFIG,
    SEED_POLICY_SOURCE_COMMIT,
    SEED_POLICY_VERSION,
    reset_rng,
    rng_state_fingerprints,
    seed_fingerprint,
    stage_seed,
)
from utils.validate_reciprocal_stage_seed_profiles import (
    ROLES,
    StageSeedProfileError,
    validate_profiles,
)


def _profile(path: Path, *, mutate=None) -> Path:
    labels = {}
    metadata = {
        "stage_seed_policy": SEED_POLICY_VERSION,
        "stage_seed_policy_source_commit": SEED_POLICY_SOURCE_COMMIT,
        "reciprocal_seed_policy": SEED_POLICY_VERSION,
        "deterministic_algorithm_policy": DETERMINISM_POLICY_VERSION,
        "torch_deterministic_algorithms": True,
        "cublas_workspace_config": REQUIRED_CUBLAS_WORKSPACE_CONFIG,
    }
    for label in ("timing_context", "main_generation"):
        seed = stage_seed(12345, label)
        reset_rng(seed)
        states = rng_state_fingerprints()
        states.setdefault("cuda", "1" * 64)
        labels[label] = {
            "seed": seed,
            "seed_sha256": seed_fingerprint(12345, label, seed),
            "rng_state_sha256": states,
        }
        metadata[f"reciprocal_seed_{label}"] = seed
        for device, digest in states.items():
            metadata[f"reciprocal_rng_{device}_sha256_{label}"] = digest
    metadata["profile_label_seed_fingerprints"] = labels
    payload = {"schema_version": 1, "metadata": metadata}
    if mutate is not None:
        mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {role: _profile(tmp_path / f"{role}.json") for role in ROLES}


def test_validates_same_stage_seed_and_rng_fingerprints_for_all_roles(
    tmp_path: Path,
) -> None:
    report = validate_profiles(_paths(tmp_path), base_seed=12345)

    assert report["pass"] is True
    assert report["stage_seed_policy"] == SEED_POLICY_VERSION
    assert set(report["roles"]) == set(ROLES)
    assert set(report["profile_label_seed_fingerprints"]) == {
        "timing_context",
        "main_generation",
    }


def test_rejects_missing_standard_policy_before_analysis(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    payload = json.loads(paths["candidate_first"].read_text(encoding="utf-8"))
    payload["metadata"].pop("stage_seed_policy")
    paths["candidate_first"].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StageSeedProfileError, match="stage_seed_policy"):
        validate_profiles(paths, base_seed=12345)


def test_rejects_cross_process_rng_fingerprint_drift(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    payload = json.loads(paths["baseline_second"].read_text(encoding="utf-8"))
    record = payload["metadata"]["profile_label_seed_fingerprints"][
        "main_generation"
    ]
    record["rng_state_sha256"]["cuda"] = "2" * 64
    payload["metadata"]["reciprocal_rng_cuda_sha256_main_generation"] = "2" * 64
    paths["baseline_second"].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StageSeedProfileError, match="differ"):
        validate_profiles(paths, base_seed=12345)


def test_rejects_missing_deterministic_math_policy(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    payload = json.loads(paths["candidate_second"].read_text(encoding="utf-8"))
    payload["metadata"]["torch_deterministic_algorithms"] = False
    paths["candidate_second"].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StageSeedProfileError, match="torch_deterministic_algorithms"):
        validate_profiles(paths, base_seed=12345)
