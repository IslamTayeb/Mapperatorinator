#!/usr/bin/env python3
"""Run one W3 reciprocal role with opt-in B16 encoder prefill.

Baseline leaves the accepted Processor path cold. Candidate installs
``install_batched_encoder_prefill_candidate`` for the request only.
Nothing here is imported by selectors, V32, or the accepted runtime.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import pickle
import random
import runpy
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_NAME = "tiger_b16_batched_encoder_prefill"
SCHEMA_VERSION = 1


def _ensure_repo_import_path() -> Path:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return REPO_ROOT


def _parse_runner_args(argv: list[str]) -> tuple[str, Path, int, str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--storage", choices=("cpu", "cuda"), default="cpu")
    parsed, remaining = parser.parse_known_args(argv)
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    if not remaining:
        raise SystemExit("inference Hydra arguments are required")
    if isinstance(parsed.batch_size, bool) or parsed.batch_size <= 0:
        raise SystemExit("batch-size must be a positive integer")
    return (
        parsed.mode,
        parsed.evidence_manifest,
        parsed.batch_size,
        parsed.storage,
        remaining,
    )


def _sha256_pickle(value: object) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def _sha256_rng_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def _rng_fingerprint() -> dict[str, object]:
    return {
        "python_sha256": _sha256_pickle(random.getstate()),
        "numpy_sha256": _sha256_pickle(np.random.get_state()),
        "torch_cpu_sha256": _sha256_rng_tensor(torch.random.get_rng_state()),
        "torch_cuda_sha256": [
            _sha256_rng_tensor(state) for state in torch.cuda.get_rng_state_all()
        ],
    }


def main() -> None:
    mode, manifest_path, batch_size, storage, inference_args = _parse_runner_args(
        sys.argv[1:]
    )
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_path}")

    _ensure_repo_import_path()

    from accelerate import utils as accelerate_utils
    from osuT5.osuT5.inference.processor import Processor
    from osuT5.osuT5.inference.optimized.scout.batched_encoder_prefill import (
        install_batched_encoder_prefill_candidate,
    )

    after_seed: dict[str, object] | None = None
    original_set_seed = accelerate_utils.set_seed

    def recorded_set_seed(*args, **kwargs) -> None:
        nonlocal after_seed
        original_set_seed(*args, **kwargs)
        after_seed = _rng_fingerprint()

    inference_script = REPO_ROOT / "inference.py"
    accelerate_utils.set_seed = recorded_set_seed
    evidence: dict[str, object] | None = None
    try:
        if mode == "candidate":
            with install_batched_encoder_prefill_candidate(
                Processor,
                batch_size=batch_size,
                storage=storage,
            ) as manager:
                sys.argv = [str(inference_script), *inference_args]
                runpy.run_path(str(inference_script), run_name="__main__")
                evidence = manager.finalize()
        else:
            with nullcontext():
                sys.argv = [str(inference_script), *inference_args]
                runpy.run_path(str(inference_script), run_name="__main__")
        after_inference = _rng_fingerprint()
    finally:
        accelerate_utils.set_seed = original_set_seed

    if after_seed is None:
        raise RuntimeError("inference never initialized the RNG environment")
    if mode == "candidate":
        if not isinstance(evidence, dict):
            raise RuntimeError("candidate never finalized encoder-prefill evidence")
        if int(evidence.get("store_count", 0)) < 1:
            raise RuntimeError("candidate encoder prefill produced no stores")
        labels = evidence.get("labels_completed")
        if labels != ["timing_context", "main_generation"]:
            raise RuntimeError(
                f"candidate labels_completed={labels!r}; expected timing then main"
            )
    elif evidence is not None:
        raise RuntimeError("baseline unexpectedly produced candidate evidence")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "role": mode,
                "candidate": CANDIDATE_NAME,
                "enabled": mode == "candidate",
                "batch_size": batch_size,
                "storage": storage,
                "result_class": "documented-drift",
                "production_default": False,
                "tiger_pr": 120,
                "prefill_evidence": evidence,
                "rng_after_seed": after_seed,
                "rng_after_inference": after_inference,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
