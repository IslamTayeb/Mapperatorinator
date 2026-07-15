"""Run a fresh strict-FP32 control or opt-in batched encoder store."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _load_args(config_name: str, overrides: list[str]):
    import config  # noqa: F401
    import hydra
    from omegaconf import DictConfig, OmegaConf

    config_dir = REPO_ROOT / "configs" / "inference"
    resolved_name = config_name.removeprefix("inference/")
    with hydra.initialize_config_dir(
        config_dir=str(config_dir),
        version_base="1.1",
    ):
        cfg = hydra.compose(config_name=resolved_name, overrides=overrides)
    return OmegaConf.to_object(cfg) if isinstance(cfg, DictConfig) else cfg


def _validate_args(args, *, mode: str, batch_size: int) -> None:
    expected = {
        "inference_engine": "optimized",
        "attn_implementation": "sdpa",
        "device": "cuda",
        "use_server": False,
        "parallel": False,
        "cfg_scale": 1.0,
        "num_beams": 1,
        "super_timing": False,
        "generate_positions": False,
        "profile_inference": True,
    }
    failures = {
        key: {"expected": value, "actual": getattr(args, key)}
        for key, value in expected.items()
        if getattr(args, key) != value
    }
    if failures:
        raise ValueError(f"encoder-store reciprocal args changed: {failures}")
    if args.precision not in {"fp32", "fp16"}:
        raise ValueError("encoder-store gate requires precision=fp32 or fp16")
    if mode not in {"baseline", "candidate"}:
        raise ValueError("mode must be baseline or candidate")
    if args.profile_pass_kind not in {"untraced_control", "exactness_audit"}:
        raise ValueError(
            "encoder-store strict gate requires untraced_control or "
            "exactness_audit"
        )
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0 or batch_size > int(args.max_batch_size):
        raise ValueError(
            f"batch_size must be within 1..max_batch_size={args.max_batch_size}"
        )


def run(
    config_name: str,
    overrides: list[str],
    *,
    mode: str,
    batch_size: int,
    evidence_path: Path,
) -> dict:
    import inference

    args = _load_args(config_name, overrides)
    _validate_args(args, mode=mode, batch_size=batch_size)
    started = time.perf_counter()
    manager = None
    if mode == "candidate":
        from osuT5.osuT5.inference.optimized.scout.encoder_store import (
            install_batched_encoder_store_candidate,
        )

        store_context = install_batched_encoder_store_candidate(
            inference.Processor,
            batch_size=batch_size,
            precision=args.precision,
        )
    else:
        from contextlib import nullcontext

        store_context = nullcontext(None)

    with store_context as manager:
        inference.main(args)
    run_wall_seconds = time.perf_counter() - started
    if not math.isfinite(run_wall_seconds) or run_wall_seconds <= 0:
        raise RuntimeError("encoder-store runner produced an invalid wall time")
    payload = {
        "schema_version": 1,
        "mode": mode,
        "config_name": config_name,
        "seed": int(args.seed),
        "batch_size": batch_size if mode == "candidate" else None,
        "profile_pass_kind": args.profile_pass_kind,
        "precision": args.precision,
        "strict_fp32": args.precision == "fp32",
        "stage_seed_override": False,
        "run_wall_seconds": run_wall_seconds,
        "encoder_store": manager.finalize() if manager is not None else None,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--evidence-path", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    parsed = parser.parse_args()
    payload = run(
        parsed.config_name,
        parsed.overrides,
        mode=parsed.mode,
        batch_size=parsed.batch_size,
        evidence_path=parsed.evidence_path,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
