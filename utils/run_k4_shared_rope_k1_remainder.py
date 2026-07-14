"""Run shared-RoPE K4 with captured one-step prefix-boundary remainders."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.run_k4_shared_rope_approximate_weight_only import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--output-init-json", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    parsed = parser.parse_args()
    run(
        parsed.config_name,
        parsed.overrides,
        parsed.output_init_json,
        graph_remainders=True,
    )


if __name__ == "__main__":
    main()
