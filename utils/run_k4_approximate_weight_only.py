"""Run the opt-in K=4 decode loop with the mixed-weight CUDA runtime."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.inference.optimized.single.k8_runtime import (  # noqa: E402
    install_k8_candidate,
)
from utils.run_approximate_weight_only import run as run_weight_only  # noqa: E402


def run(config_name: str, overrides: list[str], output_init_json: Path) -> None:
    """Compose the two independently measured candidates without global leakage."""

    with install_k8_candidate(block_size=4):
        run_weight_only(config_name, overrides, output_init_json)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--output-init-json", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    parsed = parser.parse_args()
    run(parsed.config_name, parsed.overrides, parsed.output_init_json)


if __name__ == "__main__":
    main()
