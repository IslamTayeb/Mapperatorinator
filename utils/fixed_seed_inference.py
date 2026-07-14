"""Benchmark-only stage-local RNG control for reciprocal inference runs."""

from __future__ import annotations

import hashlib
import random
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch


SEED_POLICY_VERSION = "reciprocal-stage-seed-v1"


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


@contextmanager
def fixed_seed_processor_generation(inference_module, *, base_seed: int) -> Iterator[None]:
    """Reset RNG at each named generation stage and restore the class method."""

    processor = inference_module.Processor
    original_generate = processor.generate

    def generate_with_fixed_stage_seed(self, *args, **kwargs):
        profile_label = kwargs.get("profile_label")
        if not isinstance(profile_label, str) or not profile_label:
            raise RuntimeError(
                "reciprocal fixed-seed inference requires an explicit profile_label"
            )
        seed = stage_seed(base_seed, profile_label)
        reset_rng(seed)
        fingerprints = rng_state_fingerprints()
        profiler = getattr(self, "profiler", None)
        set_metadata = getattr(profiler, "set_metadata", None)
        if callable(set_metadata):
            set_metadata(
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
