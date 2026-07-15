#!/usr/bin/env python3
"""Run normal inference with the opt-in preallocated batch-one state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_repo_import_path() -> Path:
    """Make direct file execution resolve the repository's Python packages."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return REPO_ROOT


def _parse_runner_args(argv: list[str]) -> tuple[Path, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--activation-manifest", type=Path, required=True)
    parsed, remaining = parser.parse_known_args(argv)
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    if not remaining:
        raise SystemExit("inference Hydra arguments are required")
    return parsed.activation_manifest, remaining


def main() -> None:
    manifest_path, inference_args = _parse_runner_args(sys.argv[1:])
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_path}")

    _ensure_repo_import_path()

    from osuT5.osuT5.inference.optimized.scout.device_sequence_state import (
        device_sequence_state_candidate_context,
    )

    inference_script = REPO_ROOT / "inference.py"
    with device_sequence_state_candidate_context() as activation:
        sys.argv = [str(inference_script), *inference_args]
        runpy.run_path(str(inference_script), run_name="__main__")

    if activation.calls <= 0:
        raise RuntimeError("candidate inference never invoked the optimized decode loop")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate": "preallocated_batch1_device_sequence_state",
                "enabled": True,
                "decode_loop_calls": activation.calls,
                "decoder_layer_replaced": False,
                "accepted_dispatch_replaced": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
