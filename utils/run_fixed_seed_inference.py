"""Run ordinary inference with benchmark-only stage-local RNG control."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.fixed_seed_inference import fixed_seed_processor_generation
from utils.run_approximate_weight_only import _load_args


def _target_repo(value: Path | None) -> Path:
    path = REPO_ROOT if value is None else value.expanduser().resolve()
    if not (path / "inference.py").is_file() or not (path / "configs" / "inference").is_dir():
        raise ValueError(f"target repo lacks inference.py/configs/inference: {path}")
    return path


def run(config_name: str, overrides: list[str], *, target_repo: Path | None = None) -> None:
    target = _target_repo(target_repo)
    target_text = str(target)
    if target_text in sys.path:
        sys.path.remove(target_text)
    sys.path.insert(0, target_text)
    importlib.invalidate_caches()
    inference = importlib.import_module("inference")
    loaded_path = Path(inference.__file__).resolve()
    if loaded_path != (target / "inference.py").resolve():
        raise RuntimeError(
            f"loaded inference module from {loaded_path}, expected {target / 'inference.py'}"
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
