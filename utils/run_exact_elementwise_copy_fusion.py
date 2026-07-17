#!/usr/bin/env python3
"""Reciprocal exact elementwise/copy fusion on shared-RoPE + device-state tip."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
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
    calls: int,
    rng_after_seed: dict[str, Any],
    rng_after_inference: dict[str, Any],
    fusion_stats: dict[str, Any],
) -> dict[str, object]:
    if mode not in {"baseline", "candidate"}:
        raise ValueError("elementwise-fusion mode must be baseline or candidate")
    if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
        raise RuntimeError("elementwise-fusion decode-loop count is invalid")
    if calls <= 0:
        raise RuntimeError(
            "elementwise-fusion requires device-sequence-state decode calls"
        )
    return {
        "schema_version": 1,
        "role": mode,
        "candidate": "exact_device_state_plus_elementwise_copy_fusion",
        "enabled": mode == "candidate",
        "decode_loop_calls": calls,
        "decoder_layer_replaced": False,
        "accepted_dispatch_replaced": False,
        "shared_rope_enabled": True,
        "device_sequence_state_enabled": True,
        "elementwise_copy_fusion_enabled": mode == "candidate",
        "elementwise_fusion_stats": fusion_stats,
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
    import utils.run_strict_shared_rope as shared_rope_runner
    from osuT5.osuT5.inference.optimized.kernels.elementwise_fusion import (
        elementwise_fusion_candidate_context,
        elementwise_fusion_stats,
        reset_elementwise_fusion_stats,
    )
    from osuT5.osuT5.inference.optimized.scout.device_sequence_state import (
        device_sequence_state_candidate_context,
    )
    from utils.run_device_sequence_state_candidate import _rng_fingerprint
    from utils.run_strict_shared_rope import run as run_shared_rope

    rng_after_seed: dict[str, Any] | None = None
    original_setup = inference_module.setup_inference_environment
    original_shared_context = shared_rope_runner.shared_decoder_rope_context

    def recorded_setup(seed: int, *args: Any, **kwargs: Any) -> None:
        nonlocal rng_after_seed
        rng_after_seed = _run_setup_and_fingerprint(
            original_setup,
            seed,
            *args,
            **kwargs,
        )

    @contextmanager
    def shared_with_elementwise_fusion(
        model,
        *,
        stats=None,
    ) -> Iterator[Any]:
        with ExitStack() as stack:
            if mode == "candidate":
                stack.enter_context(elementwise_fusion_candidate_context(model))
            else:
                reset_elementwise_fusion_stats()
            value = stack.enter_context(
                original_shared_context(model, stats=stats)
            )
            yield value

    inference_module.setup_inference_environment = recorded_setup
    shared_rope_runner.shared_decoder_rope_context = shared_with_elementwise_fusion
    try:
        with device_sequence_state_candidate_context() as activation:
            run_shared_rope(config_name, overrides, rope_evidence_path)
        rng_after_inference = _rng_fingerprint()
        fusion_stats = elementwise_fusion_stats().as_dict()
    finally:
        inference_module.setup_inference_environment = original_setup
        shared_rope_runner.shared_decoder_rope_context = original_shared_context

    if rng_after_seed is None:
        raise RuntimeError(
            "elementwise-fusion inference never initialized the RNG environment"
        )
    if mode == "candidate":
        if fusion_stats.get("rope_epilogue_hits", 0) <= 0:
            raise RuntimeError("elementwise fusion RoPE epilogue never engaged")
        if fusion_stats.get("attn_pack_hits", 0) <= 0:
            raise RuntimeError("elementwise fusion attn pack never engaged")

    payload = _activation_payload(
        mode=mode,
        calls=activation.calls,
        rng_after_seed=rng_after_seed,
        rng_after_inference=rng_after_inference,
        fusion_stats=fusion_stats,
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
