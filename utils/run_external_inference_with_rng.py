#!/usr/bin/env python3
"""Run one clean worktree's inference entrypoint and fingerprint RNG state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import random
import runpy
import subprocess
import sys

import numpy as np
import torch


def _parse_args(argv: list[str]) -> tuple[Path, str, str, Path, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parsed, remaining = parser.parse_known_args(argv)
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    if not remaining:
        raise SystemExit("inference Hydra arguments are required")
    return (
        parsed.repo.resolve(),
        parsed.expected_commit,
        parsed.role,
        parsed.evidence_manifest,
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


def _validate_repo(repo: Path, expected_commit: str) -> Path:
    inference_script = repo / "inference.py"
    if not inference_script.is_file():
        raise FileNotFoundError(f"target worktree has no inference.py: {repo}")
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = completed.stdout.strip()
    if actual != expected_commit:
        raise RuntimeError(
            f"target worktree commit changed: expected {expected_commit}, got {actual}"
        )
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError(f"target worktree is dirty: {repo}")
    return inference_script


def main() -> None:
    repo, expected_commit, role, evidence_path, inference_args = _parse_args(
        sys.argv[1:]
    )
    if evidence_path.exists():
        raise FileExistsError(f"refusing to overwrite {evidence_path}")
    inference_script = _validate_repo(repo, expected_commit)
    sys.path.insert(0, str(repo))

    from accelerate import utils as accelerate_utils

    rng_after_seed: dict[str, object] | None = None
    original_set_seed = accelerate_utils.set_seed

    def recorded_set_seed(*args, **kwargs) -> None:
        nonlocal rng_after_seed
        original_set_seed(*args, **kwargs)
        rng_after_seed = _rng_fingerprint()

    accelerate_utils.set_seed = recorded_set_seed
    try:
        sys.argv = [str(inference_script), *inference_args]
        runpy.run_path(str(inference_script), run_name="__main__")
        rng_after_inference = _rng_fingerprint()
    finally:
        accelerate_utils.set_seed = original_set_seed

    if rng_after_seed is None:
        raise RuntimeError("target inference never initialized RNG state")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "role": role,
                "repo": str(repo),
                "commit": expected_commit,
                "rng_after_seed": rng_after_seed,
                "rng_after_inference": rng_after_inference,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
