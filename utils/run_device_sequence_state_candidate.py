#!/usr/bin/env python3
"""Run one reciprocal role and record activation plus RNG evidence."""

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


def _ensure_repo_import_path() -> Path:
    """Make direct file execution resolve the repository's Python packages."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return REPO_ROOT


def _parse_runner_args(argv: list[str]) -> tuple[str, Path, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parsed, remaining = parser.parse_known_args(argv)
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    if not remaining:
        raise SystemExit("inference Hydra arguments are required")
    return parsed.mode, parsed.evidence_manifest, remaining


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
    mode, manifest_path, inference_args = _parse_runner_args(sys.argv[1:])
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_path}")

    _ensure_repo_import_path()

    from accelerate import utils as accelerate_utils
    from osuT5.osuT5.inference.optimized.scout.device_sequence_state import (
        DeviceSequenceStateActivation,
        device_sequence_state_candidate_context,
    )

    after_seed: dict[str, object] | None = None
    original_set_seed = accelerate_utils.set_seed

    def recorded_set_seed(*args, **kwargs) -> None:
        nonlocal after_seed
        original_set_seed(*args, **kwargs)
        after_seed = _rng_fingerprint()

    activation = DeviceSequenceStateActivation()
    context = (
        device_sequence_state_candidate_context()
        if mode == "candidate"
        else nullcontext(activation)
    )
    inference_script = REPO_ROOT / "inference.py"
    accelerate_utils.set_seed = recorded_set_seed
    try:
        with context as activation:
            sys.argv = [str(inference_script), *inference_args]
            runpy.run_path(str(inference_script), run_name="__main__")
        after_inference = _rng_fingerprint()
    finally:
        accelerate_utils.set_seed = original_set_seed

    if after_seed is None:
        raise RuntimeError("inference never initialized the RNG environment")
    if mode == "candidate" and activation.calls <= 0:
        raise RuntimeError("candidate inference never invoked the optimized decode loop")
    if mode == "baseline" and activation.calls != 0:
        raise RuntimeError("baseline unexpectedly activated the candidate decode loop")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "role": mode,
                "candidate": "preallocated_batch1_device_sequence_state",
                "enabled": mode == "candidate",
                "decode_loop_calls": activation.calls,
                "decoder_layer_replaced": False,
                "accepted_dispatch_replaced": False,
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
