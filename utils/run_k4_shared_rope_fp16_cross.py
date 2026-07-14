"""Run the opt-in FP16-packed cross-projection full-song candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

from osuT5.osuT5.inference.optimized.kernels.weight_only_runtime import (
    CROSS_FP16_PACKED,
)
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
        mode=CROSS_FP16_PACKED,
    )


if __name__ == "__main__":
    main()
