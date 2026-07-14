"""Run the K1-remainder + INT8 main topology with specialized FP16 K1 timing."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

COMPOSITION_VERSION = (
    "k4-split-kv-mixed-weight-shared-rope-k1-remainder-int8-mlp-"
    "stage-precision-v1"
)
INITIALIZER = "initialize_approximate_int8_mlp_weight_only"


def _hybrid_runner():
    from utils.run_k4_stage_precision_hybrid import run

    return run


def _shared_runner():
    from utils.run_main_shared_rope_delegate import run_with_main_shared_rope

    return run_with_main_shared_rope


def _run_delegate(config_name: str, overrides: list[str], output_init_json: Path) -> None:
    _hybrid_runner()(
        config_name,
        overrides,
        output_init_json,
        initializer_name=INITIALIZER,
        graph_remainders=True,
    )


def run(config_name: str, overrides: list[str], output_init_json: Path) -> None:
    _shared_runner()(
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
