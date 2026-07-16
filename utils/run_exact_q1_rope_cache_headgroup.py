#!/usr/bin/env python3
"""Reciprocal runner: accepted tip ± q1 RoPE/cache head-group CTAs."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager, nullcontext
import json
from pathlib import Path
import sys
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--rope-evidence-path", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args(argv)


def _activation_payload(
    *,
    mode: str,
    decode_loop_calls: int,
    headgroup_calls: int,
    rng_after_seed: dict[str, Any],
    rng_after_inference: dict[str, Any],
) -> dict[str, object]:
    if mode not in {"baseline", "candidate"}:
        raise ValueError("mode must be baseline or candidate")
    if (headgroup_calls > 0) != (mode == "candidate"):
        raise RuntimeError("q1 RoPE/cache headgroup activation does not match its role")
    if decode_loop_calls <= 0:
        raise RuntimeError("tip device-state decode loop never ran")
    return {
        "schema_version": 1,
        "role": mode,
        "candidate": "exact_q1_rope_cache_headgroup_on_tip",
        "enabled": mode == "candidate",
        "decode_loop_calls": decode_loop_calls,
        "headgroup_profile_calls": headgroup_calls,
        "shared_rope_enabled": True,
        "device_sequence_state_enabled": True,
        "native_q1_rope_cache_headgroup_enabled": mode == "candidate",
        "rng_after_seed": rng_after_seed,
        "rng_after_inference": rng_after_inference,
    }


@contextmanager
def _stacked(*contexts) -> Iterator[tuple[Any, ...]]:
    with ExitStack() as stack:
        yield tuple(stack.enter_context(ctx) for ctx in contexts)


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

    import inference as inference_module
    from osuT5.osuT5.inference.optimized.scout.device_sequence_state import (
        device_sequence_state_candidate_context,
    )
    from osuT5.osuT5.inference.optimized.scout.q1_rope_cache_headgroup import (
        Q1RopeCacheHeadgroupActivation,
        q1_rope_cache_headgroup_candidate_context,
    )
    from utils.run_device_sequence_state_candidate import _rng_fingerprint
    from utils.run_exact_rope_device_state import _run_setup_and_fingerprint
    from utils.run_strict_shared_rope import run as run_shared_rope

    rng_after_seed: dict[str, Any] | None = None
    original_setup = inference_module.setup_inference_environment

    def recorded_setup(seed: int, *args: Any, **kwargs: Any) -> None:
        nonlocal rng_after_seed
        rng_after_seed = _run_setup_and_fingerprint(
            original_setup, seed, *args, **kwargs
        )

    headgroup_context = (
        q1_rope_cache_headgroup_candidate_context()
        if mode == "candidate"
        else nullcontext(Q1RopeCacheHeadgroupActivation())
    )
    inference_module.setup_inference_environment = recorded_setup
    try:
        with _stacked(
            device_sequence_state_candidate_context(),
            headgroup_context,
        ) as (device_activation, headgroup_activation):
            run_shared_rope(config_name, overrides, rope_evidence_path)
        rng_after_inference = _rng_fingerprint()
    finally:
        inference_module.setup_inference_environment = original_setup

    if rng_after_seed is None:
        raise RuntimeError("q1 RoPE/cache headgroup inference never initialized RNG")
    payload = _activation_payload(
        mode=mode,
        decode_loop_calls=int(device_activation.calls),
        headgroup_calls=int(headgroup_activation.calls),
        rng_after_seed=rng_after_seed,
        rng_after_inference=rng_after_inference,
    )
    evidence_manifest.parent.mkdir(parents=True, exist_ok=True)
    evidence_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    run(
        mode=args.mode,
        config_name=args.config_name,
        overrides=args.overrides,
        rope_evidence_path=args.rope_evidence_path,
        evidence_manifest=args.evidence_manifest,
    )


if __name__ == "__main__":
    main()
