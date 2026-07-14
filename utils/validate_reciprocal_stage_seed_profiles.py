"""Validate fixed-stage RNG and deterministic-math evidence before comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from utils.fixed_seed_inference import (
        DETERMINISM_POLICY_VERSION,
        REQUIRED_CUBLAS_WORKSPACE_CONFIG,
        SEED_POLICY_SOURCE_COMMIT,
        SEED_POLICY_VERSION,
        seed_fingerprint,
        stage_seed,
    )
except ModuleNotFoundError:  # Direct ``python utils/...py`` execution.
    from fixed_seed_inference import (  # type: ignore[no-redef]
        DETERMINISM_POLICY_VERSION,
        REQUIRED_CUBLAS_WORKSPACE_CONFIG,
        SEED_POLICY_SOURCE_COMMIT,
        SEED_POLICY_VERSION,
        seed_fingerprint,
        stage_seed,
    )


SCHEMA_VERSION = "mapperatorinator.reciprocal-stage-seed-profiles.v1"
ROLES = (
    "baseline_first",
    "candidate_first",
    "candidate_second",
    "baseline_second",
)
LABELS = ("timing_context", "main_generation")
DEVICES = ("cpu", "cuda")


class StageSeedProfileError(ValueError):
    pass


def _object(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageSeedProfileError(f"{name} must be an object")
    return value


def _sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StageSeedProfileError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _load_metadata(path: Path, *, role: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageSeedProfileError(f"{role} profile is unreadable: {path}") from exc
    root = _object(payload, name=f"{role}.profile")
    if root.get("schema_version") != 1:
        raise StageSeedProfileError(f"{role} profile schema_version must be 1")
    return _object(root.get("metadata"), name=f"{role}.metadata")


def _parse_role(path: Path, *, role: str, base_seed: int) -> dict[str, Any]:
    metadata = _load_metadata(path, role=role)
    expected_scalars = {
        "stage_seed_policy": SEED_POLICY_VERSION,
        "stage_seed_policy_source_commit": SEED_POLICY_SOURCE_COMMIT,
        "reciprocal_seed_policy": SEED_POLICY_VERSION,
        "deterministic_algorithm_policy": DETERMINISM_POLICY_VERSION,
        "torch_deterministic_algorithms": True,
        "cublas_workspace_config": REQUIRED_CUBLAS_WORKSPACE_CONFIG,
    }
    for key, expected in expected_scalars.items():
        if metadata.get(key) != expected:
            raise StageSeedProfileError(
                f"{role}.metadata.{key} must be {expected!r}, "
                f"got {metadata.get(key)!r}"
            )

    raw_labels = _object(
        metadata.get("profile_label_seed_fingerprints"),
        name=f"{role}.metadata.profile_label_seed_fingerprints",
    )
    if set(raw_labels) != set(LABELS):
        raise StageSeedProfileError(
            f"{role} profile-label seed fingerprints must cover {list(LABELS)}"
        )
    labels: dict[str, Any] = {}
    for label in LABELS:
        record = _object(raw_labels[label], name=f"{role}.{label}")
        expected_seed = stage_seed(base_seed, label)
        if record.get("seed") != expected_seed:
            raise StageSeedProfileError(f"{role}.{label} derived seed is wrong")
        expected_seed_sha = seed_fingerprint(base_seed, label, expected_seed)
        seed_sha = _sha256(
            record.get("seed_sha256"),
            name=f"{role}.{label}.seed_sha256",
        )
        if seed_sha != expected_seed_sha:
            raise StageSeedProfileError(f"{role}.{label} seed fingerprint is wrong")
        states = _object(
            record.get("rng_state_sha256"),
            name=f"{role}.{label}.rng_state_sha256",
        )
        if set(states) != set(DEVICES):
            raise StageSeedProfileError(
                f"{role}.{label} RNG state must cover CPU and CUDA"
            )
        parsed_states = {
            device: _sha256(
                states[device],
                name=f"{role}.{label}.rng_state_sha256.{device}",
            )
            for device in DEVICES
        }
        if metadata.get(f"reciprocal_seed_{label}") != expected_seed:
            raise StageSeedProfileError(f"{role}.{label} flat derived seed is wrong")
        for device, digest in parsed_states.items():
            if metadata.get(f"reciprocal_rng_{device}_sha256_{label}") != digest:
                raise StageSeedProfileError(
                    f"{role}.{label} flat {device} RNG fingerprint disagrees"
                )
        labels[label] = {
            "seed": expected_seed,
            "seed_sha256": seed_sha,
            "rng_state_sha256": parsed_states,
        }
    return {
        "profile": str(path.resolve()),
        "stage_seed_policy": SEED_POLICY_VERSION,
        "profile_label_seed_fingerprints": labels,
        "deterministic_algorithm_policy": DETERMINISM_POLICY_VERSION,
        "cublas_workspace_config": REQUIRED_CUBLAS_WORKSPACE_CONFIG,
    }


def validate_profiles(
    profile_paths: Mapping[str, Path],
    *,
    base_seed: int,
) -> dict[str, Any]:
    if set(profile_paths) != set(ROLES):
        raise StageSeedProfileError(f"profile roles must be exactly {list(ROLES)}")
    roles = {
        role: _parse_role(profile_paths[role], role=role, base_seed=base_seed)
        for role in ROLES
    }
    reference = roles[ROLES[0]]["profile_label_seed_fingerprints"]
    for role in ROLES[1:]:
        if roles[role]["profile_label_seed_fingerprints"] != reference:
            raise StageSeedProfileError(
                f"profile-label seed/RNG fingerprints differ for {role}"
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "pass": True,
        "base_seed": base_seed,
        "stage_seed_policy": SEED_POLICY_VERSION,
        "stage_seed_policy_source_commit": SEED_POLICY_SOURCE_COMMIT,
        "profile_label_seed_fingerprints": reference,
        "deterministic_algorithm_policy": DETERMINISM_POLICY_VERSION,
        "cublas_workspace_config": REQUIRED_CUBLAS_WORKSPACE_CONFIG,
        "roles": roles,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for role in ROLES:
        parser.add_argument(
            f"--{role.replace('_', '-')}-profile",
            type=Path,
            required=True,
        )
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate_profiles(
        {role: getattr(args, f"{role}_profile") for role in ROLES},
        base_seed=args.base_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
