"""Run the opt-in exactness-gated split-8 cross-attention candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

from osuT5.osuT5.inference.optimized.kernels.weight_only_runtime import CROSS_SPLIT8
from utils.run_k4_shared_rope_cross_candidate import run


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
        mode=CROSS_SPLIT8,
    )


if __name__ == "__main__":
    main()
