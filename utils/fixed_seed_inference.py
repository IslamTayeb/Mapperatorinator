"""Benchmark-only stage-local RNG control for reciprocal inference runs."""

from __future__ import annotations

import hashlib
import os
import random
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch


SEED_POLICY_VERSION = "reciprocal-stage-seed-v1"
SEED_POLICY_SOURCE_COMMIT = "5d95977be1e7649a4a1c769ff87d754f92653dd3"
DETERMINISM_POLICY_VERSION = "reciprocal-cuda-determinism-v1"
REQUIRED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def stage_seed(base_seed: int, profile_label: str) -> int:
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer")
    if not isinstance(profile_label, str) or not profile_label:
        raise ValueError("profile_label must be a non-empty string")
    digest = hashlib.sha256(
        f"{SEED_POLICY_VERSION}:{base_seed}:{profile_label}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def reset_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state_fingerprints() -> dict[str, str]:
    fingerprints = {
        "cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
    }
    if torch.cuda.is_available():
        fingerprints["cuda"] = hashlib.sha256(
            torch.cuda.get_rng_state().cpu().numpy().tobytes()
        ).hexdigest()
    return fingerprints


def seed_fingerprint(base_seed: int, profile_label: str, seed: int) -> str:
    material = (
        f"{SEED_POLICY_VERSION}:{base_seed}:{profile_label}:{seed}".encode("utf-8")
    )
    return hashlib.sha256(material).hexdigest()


@contextmanager
def deterministic_inference_algorithms() -> Iterator[None]:
    """Enable fail-loud deterministic CUDA math for reciprocal processes."""

    if torch.cuda.is_available() and (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") != REQUIRED_CUBLAS_WORKSPACE_CONFIG
    ):
        raise RuntimeError(
            "deterministic reciprocal inference requires "
            f"CUBLAS_WORKSPACE_CONFIG={REQUIRED_CUBLAS_WORKSPACE_CONFIG}"
        )
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.use_deterministic_algorithms(
            previous_enabled,
            warn_only=previous_warn_only,
        )


@contextmanager
def fixed_seed_processor_generation(inference_module, *, base_seed: int) -> Iterator[None]:
    """Reset RNG at each named generation stage and restore the class method."""

    processor = inference_module.Processor
    original_generate = processor.generate
    profile_label_seed_fingerprints: dict[str, dict[str, object]] = {}

    def generate_with_fixed_stage_seed(self, *args, **kwargs):
        profile_label = kwargs.get("profile_label")
        if not isinstance(profile_label, str) or not profile_label:
            raise RuntimeError(
                "reciprocal fixed-seed inference requires an explicit profile_label"
            )
        seed = stage_seed(base_seed, profile_label)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        reset_rng(seed)
        fingerprints = rng_state_fingerprints()
        profile_label_seed_fingerprints[profile_label] = {
            "seed": seed,
            "seed_sha256": seed_fingerprint(base_seed, profile_label, seed),
            "rng_state_sha256": dict(sorted(fingerprints.items())),
        }
        profiler = getattr(self, "profiler", None)
        set_metadata = getattr(profiler, "set_metadata", None)
        if callable(set_metadata):
            set_metadata(
                stage_seed_policy=SEED_POLICY_VERSION,
                stage_seed_policy_source_commit=SEED_POLICY_SOURCE_COMMIT,
                profile_label_seed_fingerprints=dict(
                    sorted(profile_label_seed_fingerprints.items())
                ),
                deterministic_algorithm_policy=DETERMINISM_POLICY_VERSION,
                torch_deterministic_algorithms=(
                    torch.are_deterministic_algorithms_enabled()
                ),
                cublas_workspace_config=os.environ.get(
                    "CUBLAS_WORKSPACE_CONFIG"
                ),
                reciprocal_seed_policy=SEED_POLICY_VERSION,
                **{f"reciprocal_seed_{profile_label}": seed},
                **{
                    f"reciprocal_rng_{device}_sha256_{profile_label}": digest
                    for device, digest in fingerprints.items()
                },
            )
        return original_generate(self, *args, **kwargs)

    processor.generate = generate_with_fixed_stage_seed
    try:
        yield
    finally:
        processor.generate = original_generate
