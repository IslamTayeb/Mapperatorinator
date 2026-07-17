#!/usr/bin/env python3
"""Reciprocal exact baseline-glue v2 (W-BASE / §46) on tip composition."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from typing import Any


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
    calls: int,
    rng_after_seed: dict[str, Any],
    rng_after_inference: dict[str, Any],
    sampling_tail_hits: int,
) -> dict[str, object]:
    if mode not in {"baseline", "candidate"}:
        raise ValueError("baseline-glue-v2 mode must be baseline or candidate")
    if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
        raise RuntimeError("baseline-glue-v2 decode-loop count is invalid")
    if calls <= 0:
        raise RuntimeError(
            "baseline-glue-v2 requires device-sequence-state decode calls"
        )
    if mode == "candidate" and sampling_tail_hits <= 0:
        raise RuntimeError(
            "baseline-glue-v2 candidate requires sampling-tail hits > 0"
        )
    return {
        "schema_version": 1,
        "role": mode,
        "candidate": "exact_baseline_glue_v2_sampling_tail",
        "enabled": mode == "candidate",
        "decode_loop_calls": calls,
        "decoder_layer_replaced": False,
        "accepted_dispatch_replaced": False,
        "shared_rope_enabled": True,
        "device_sequence_state_enabled": True,
        "pinned_eos_flag_enabled": mode == "candidate",
        "sampling_tail_cuda_graph_enabled": mode == "candidate",
        "sampling_tail_cuda_graph_hits": (
            sampling_tail_hits if mode == "candidate" else 0
        ),
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

    import inference as inference_module
    from osuT5.osuT5.inference.optimized.kernels.sampling_tail_cuda_graph import (
        sampling_tail_cuda_graph_hits,
    )
    from osuT5.osuT5.inference.optimized.scout.device_sequence_state import (
        baseline_glue_v2_candidate_context,
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
            if mode == "candidate":
                activation = stack.enter_context(baseline_glue_v2_candidate_context())
            else:
                # Tip composition: shared RoPE + device sequence state.
                activation = stack.enter_context(
                    device_sequence_state_candidate_context()
                )
            run_shared_rope(config_name, overrides, rope_evidence_path)
            hits = int(sampling_tail_cuda_graph_hits())
        rng_after_inference = _rng_fingerprint()
    finally:
        inference_module.setup_inference_environment = original_setup

    if rng_after_seed is None:
        raise RuntimeError(
            "baseline-glue-v2 inference never initialized the RNG environment"
        )
    payload = _activation_payload(
        mode=mode,
        calls=activation.calls,
        rng_after_seed=rng_after_seed,
        rng_after_inference=rng_after_inference,
        sampling_tail_hits=hits if mode == "candidate" else 0,
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
