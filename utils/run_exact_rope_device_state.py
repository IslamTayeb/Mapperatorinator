#!/usr/bin/env python3
"""Run shared decoder RoPE plus exact preallocated batch-one decode state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--rope-evidence-path", type=Path, required=True)
    parser.add_argument("--activation-manifest", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args(argv)


def _activation_payload(calls: int) -> dict[str, object]:
    if isinstance(calls, bool) or not isinstance(calls, int) or calls <= 0:
        raise RuntimeError("combined candidate never invoked the optimized decode loop")
    return {
        "schema_version": 1,
        "candidate": "shared_rope_plus_preallocated_batch1_device_sequence_state",
        "enabled": True,
        "decode_loop_calls": calls,
        "decoder_layer_replaced": False,
        "accepted_dispatch_replaced": False,
        "shared_rope_enabled": True,
        "device_sequence_state_enabled": True,
    }


def run(
    *,
    config_name: str,
    overrides: list[str],
    rope_evidence_path: Path,
    activation_manifest: Path,
) -> None:
    if rope_evidence_path == activation_manifest:
        raise ValueError("RoPE evidence and activation manifest paths must differ")
    if rope_evidence_path.exists():
        raise FileExistsError(f"refusing to overwrite {rope_evidence_path}")
    if activation_manifest.exists():
        raise FileExistsError(f"refusing to overwrite {activation_manifest}")

    from osuT5.osuT5.inference.optimized.scout.device_sequence_state import (
        device_sequence_state_candidate_context,
    )
    from utils.run_strict_shared_rope import run as run_shared_rope

    with device_sequence_state_candidate_context() as activation:
        run_shared_rope(config_name, overrides, rope_evidence_path)

    payload = _activation_payload(activation.calls)
    activation_manifest.parent.mkdir(parents=True, exist_ok=True)
    activation_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parsed = _parse_args(sys.argv[1:])
    run(
        config_name=parsed.config_name,
        overrides=parsed.overrides,
        rope_evidence_path=parsed.rope_evidence_path,
        activation_manifest=parsed.activation_manifest,
    )


if __name__ == "__main__":
    main()
