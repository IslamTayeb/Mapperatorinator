from __future__ import annotations

import argparse
from pathlib import Path

from osuT5.osuT5.inference.optimized.single.timing_precision_matrix import FULL_FP16
from utils.run_selected_timing_precision import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--output-init-json", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    run(args.config_name, args.overrides, args.output_init_json, mode=FULL_FP16)


if __name__ == "__main__":
    main()
