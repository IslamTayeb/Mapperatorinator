#!/usr/bin/env python3
"""Reciprocal tip + PYTORCH_CUDA_ALLOC_CONF=expandable_segments candidate."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CANDIDATE_ALLOC_CONF = "expandable_segments:True"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--rope-evidence-path", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args(argv)


def _alloc_conf() -> str | None:
    raw = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _require_mode_alloc_conf(mode: str) -> str | None:
    conf = _alloc_conf()
    if mode == "candidate":
        if conf != CANDIDATE_ALLOC_CONF:
            raise RuntimeError(
                "cuda-alloc-expandable candidate requires "
                f"PYTORCH_CUDA_ALLOC_CONF={CANDIDATE_ALLOC_CONF!r}, got {conf!r}"
            )
        return conf
    if conf is not None:
        raise RuntimeError(
            "cuda-alloc-expandable baseline requires unset "
            f"PYTORCH_CUDA_ALLOC_CONF, got {conf!r}"
        )
    return None


def _activation_payload(
    *,
    mode: str,
    calls: int,
    alloc_conf: str | None,
    rng_after_seed: dict[str, Any],
    rng_after_inference: dict[str, Any],
) -> dict[str, object]:
    if mode not in {"baseline", "candidate"}:
        raise ValueError("cuda-alloc-expandable mode must be baseline or candidate")
    if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
        raise RuntimeError("cuda-alloc-expandable decode-loop count is invalid")
    if calls <= 0:
        raise RuntimeError(
            "cuda-alloc-expandable requires device-sequence-state decode calls"
        )
    enabled = mode == "candidate"
    if enabled and alloc_conf != CANDIDATE_ALLOC_CONF:
        raise RuntimeError("candidate alloc conf mismatch in evidence payload")
    if (not enabled) and alloc_conf is not None:
        raise RuntimeError("baseline alloc conf must be null in evidence payload")
    return {
        "schema_version": 1,
        "role": mode,
        "candidate": "exact_device_state_plus_cuda_alloc_expandable",
        "enabled": enabled,
        "decode_loop_calls": calls,
        "decoder_layer_replaced": False,
        "accepted_dispatch_replaced": False,
        "shared_rope_enabled": True,
        "device_sequence_state_enabled": True,
        "cuda_alloc_expandable_enabled": enabled,
        "cuda_alloc_conf": alloc_conf,
        "rng_after_seed": rng_after_seed,
        "rng_after_inference": rng_after_inference,
    }


def _run_setup_and_fingerprint(
    setup: Any,
    seed: int,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    from utils.run_device_sequence_state_candidate import _rng_fingerprint

    setup(seed, *args, **kwargs)
    return _rng_fingerprint()


def run(
    *,
    mode: str,
    config_name: str,
    overrides: list[str],
    rope_evidence_path: Path,
    evidence_manifest: Path,
) -> None:
    if rope_evidence_path == evidence_manifest:
        raise ValueError("RoPE evidence and activation manifest paths must differ")
    for path in (rope_evidence_path, evidence_manifest):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    alloc_conf = _require_mode_alloc_conf(mode)

    import inference as inference_module
    from osuT5.osuT5.inference.optimized.scout.device_sequence_state import (
        device_sequence_state_candidate_context,
    )
    from utils.run_device_sequence_state_candidate import _rng_fingerprint
    from utils.run_strict_shared_rope import run as run_shared_rope

    rng_after_seed: dict[str, Any] | None = None
    original_setup = inference_module.setup_inference_environment

    def recorded_setup(seed: int, *args: Any, **kwargs: Any) -> None:
        nonlocal rng_after_seed
        rng_after_seed = _run_setup_and_fingerprint(
            original_setup,
            seed,
            *args,
            **kwargs,
        )

    inference_module.setup_inference_environment = recorded_setup
    try:
        with ExitStack() as stack:
            activation = stack.enter_context(device_sequence_state_candidate_context())
            run_shared_rope(config_name, overrides, rope_evidence_path)
        rng_after_inference = _rng_fingerprint()
    finally:
        inference_module.setup_inference_environment = original_setup

    if rng_after_seed is None:
        raise RuntimeError(
            "cuda-alloc-expandable inference never initialized the RNG environment"
        )
    payload = _activation_payload(
        mode=mode,
        calls=activation.calls,
        alloc_conf=alloc_conf,
        rng_after_seed=rng_after_seed,
        rng_after_inference=rng_after_inference,
    )
    evidence_manifest.parent.mkdir(parents=True, exist_ok=True)
    evidence_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parsed = _parse_args(sys.argv[1:])
    run(
        mode=parsed.mode,
        config_name=parsed.config_name,
        overrides=parsed.overrides,
        rope_evidence_path=parsed.rope_evidence_path,
        evidence_manifest=parsed.evidence_manifest,
    )


if __name__ == "__main__":
    main()
