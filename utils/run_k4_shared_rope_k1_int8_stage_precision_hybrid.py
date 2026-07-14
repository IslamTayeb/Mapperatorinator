"""Run the K1-remainder + INT8 main topology with specialized FP16 K1 timing."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.run_k4_shared_rope_k1_int8_stage_aware_control import (  # noqa: E402
    COMPOSITION_VERSION,
    INITIALIZER,
)
from utils.run_k4_stage_precision_hybrid import run as run_hybrid  # noqa: E402
from utils.run_main_shared_rope_delegate import run_with_main_shared_rope  # noqa: E402


def _run_delegate(config_name: str, overrides: list[str], output_init_json: Path) -> None:
    run_hybrid(
        config_name,
        overrides,
        output_init_json,
        initializer_name=INITIALIZER,
        graph_remainders=True,
    )


def run(config_name: str, overrides: list[str], output_init_json: Path) -> None:
    run_with_main_shared_rope(
        _run_delegate,
        config_name,
        overrides,
        output_init_json,
        arm="candidate",
        composition_version=COMPOSITION_VERSION,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--output-init-json", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    run(args.config_name, args.overrides, args.output_init_json)


if __name__ == "__main__":
    main()
