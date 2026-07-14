"""Run a fixed-seed baseline or opt-in exact main-encoder overlap."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import ExitStack, nullcontext
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.run_k4_shared_rope_approximate_weight_only import (  # noqa: E402
    COMPOSITION_VERSION,
    run as run_current_topology,
)


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


def _validate_args(args, *, mode: str) -> None:
    expected = {
        "inference_engine": "optimized",
        "precision": "fp32",
        "attn_implementation": "sdpa",
        "device": "cuda",
        "use_server": False,
        "parallel": False,
        "cfg_scale": 1.0,
        "num_beams": 1,
        "super_timing": False,
        "generate_positions": False,
        "profile_inference": True,
        "auto_select_gamemode_model": True,
    }
    failures = {
        key: {"expected": value, "actual": getattr(args, key)}
        for key, value in expected.items()
        if getattr(args, key) != value
    }
    if failures:
        raise ValueError(f"exact encoder-overlap args changed: {failures}")
    if mode not in {"baseline", "candidate"}:
        raise ValueError("mode must be baseline or candidate")


def run(
    config_name: str,
    overrides: list[str],
    *,
    mode: str,
    evidence_path: Path,
    initialization_path: Path,
) -> dict:
    import inference

    args = _load_args(config_name, overrides)
    _validate_args(args, mode=mode)
    from osuT5.osuT5.inference.optimized.scout.encoder_overlap import (
        install_encoder_material_audit,
        install_exact_main_encoder_overlap,
    )

    started = time.perf_counter()
    with ExitStack() as stack:
        audit = stack.enter_context(
            install_encoder_material_audit(inference.Processor)
        )
        manager = stack.enter_context(
            install_exact_main_encoder_overlap(inference.Processor)
            if mode == "candidate"
            else nullcontext(None)
        )
        run_current_topology(
            config_name,
            overrides,
            initialization_path,
            graph_remainders=True,
        )
    run_wall_seconds = time.perf_counter() - started
    if not math.isfinite(run_wall_seconds) or run_wall_seconds <= 0:
        raise RuntimeError("encoder-overlap runner produced invalid wall time")
    payload = {
        "schema_version": 1,
        "mode": mode,
        "config_name": config_name,
        "seed": int(args.seed),
        "run_wall_seconds": run_wall_seconds,
        "combined_runtime": COMPOSITION_VERSION,
        "initialization_path": str(initialization_path),
        "material_audit": audit.finalize(),
        "encoder_overlap": manager.finalize() if manager is not None else None,
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
    parser.add_argument("--evidence-path", type=Path, required=True)
    parser.add_argument("--initialization-path", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    parsed = parser.parse_args()
    payload = run(
        parsed.config_name,
        parsed.overrides,
        mode=parsed.mode,
        evidence_path=parsed.evidence_path,
        initialization_path=parsed.initialization_path,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
