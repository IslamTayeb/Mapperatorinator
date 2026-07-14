"""Run ordinary inference with benchmark-only stage-local RNG control."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    from .fixed_seed_inference import fixed_seed_processor_generation
except ImportError:  # Direct ``python utils/run_fixed_seed_inference.py`` execution.
    from fixed_seed_inference import fixed_seed_processor_generation


def _target_repo(value: Path | None) -> Path:
    path = REPO_ROOT if value is None else value.expanduser().resolve()
    if not (path / "inference.py").is_file() or not (
        path / "configs" / "inference"
    ).is_dir():
        raise ValueError(f"target repo lacks inference.py/configs/inference: {path}")
    return path


def _load_args(config_name: str, overrides: list[str], *, repo_root: Path):
    import config  # noqa: F401
    import hydra
    from omegaconf import DictConfig, OmegaConf

    config_dir = repo_root / "configs" / "inference"
    resolved_name = config_name.removeprefix("inference/")
    with hydra.initialize_config_dir(
        config_dir=str(config_dir),
        version_base="1.1",
    ):
        cfg = hydra.compose(config_name=resolved_name, overrides=overrides)
    return OmegaConf.to_object(cfg) if isinstance(cfg, DictConfig) else cfg


def run(config_name: str, overrides: list[str], *, target_repo: Path | None = None) -> None:
    target = _target_repo(target_repo)
    target_text = str(target)
    if target_text in sys.path:
        sys.path.remove(target_text)
    sys.path.insert(0, target_text)
    importlib.invalidate_caches()
    inference = importlib.import_module("inference")
    loaded_path = Path(inference.__file__).resolve()
    expected_path = (target / "inference.py").resolve()
    if loaded_path != expected_path:
        raise RuntimeError(
            f"loaded inference module from {loaded_path}, expected {expected_path}"
        )

    args = _load_args(config_name, overrides, repo_root=target)
    if not args.profile_inference:
        raise ValueError("fixed-seed reciprocal inference requires profile_inference=true")
    if args.super_timing or args.generate_positions:
        raise ValueError(
            "fixed-seed reciprocal runner supports Processor timing/main only; "
            "super_timing and generate_positions must be false"
        )
    with fixed_seed_processor_generation(inference, base_seed=args.seed):
        inference.main(args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--target-repo", type=Path)
    parser.add_argument("overrides", nargs="*")
    parsed = parser.parse_args()
    run(parsed.config_name, parsed.overrides, target_repo=parsed.target_repo)


if __name__ == "__main__":
    main()
