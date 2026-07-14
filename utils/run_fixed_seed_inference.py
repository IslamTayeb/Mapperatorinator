"""Run ordinary inference with benchmark-only stage-local RNG control."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.fixed_seed_inference import fixed_seed_processor_generation
from utils.run_approximate_weight_only import _load_args


def run(config_name: str, overrides: list[str]) -> None:
    import inference

    args = _load_args(config_name, overrides)
    if not args.profile_inference:
        raise ValueError("fixed-seed reciprocal inference requires profile_inference=true")
    with fixed_seed_processor_generation(inference, base_seed=args.seed):
        inference.main(args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("overrides", nargs="*")
    parsed = parser.parse_args()
    run(parsed.config_name, parsed.overrides)


if __name__ == "__main__":
    main()
