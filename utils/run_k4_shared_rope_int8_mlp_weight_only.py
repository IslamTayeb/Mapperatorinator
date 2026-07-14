"""Compose K=4, split-KV, mixed weights, shared RoPE, and INT8 MLP."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.run_approximate_int8_mlp_weight_only import run as run_int8_weight_only
from utils.run_k4_shared_rope_approximate_weight_only import run as run_combined


COMPOSITION_VERSION = "k4-split-kv-mixed-weight-shared-rope-int8-mlp-v1"


def run(config_name: str, overrides: list[str], output_init_json: Path) -> None:
    run_combined(
        config_name,
        overrides,
        output_init_json,
        weight_runner=run_int8_weight_only,
        composition_version=COMPOSITION_VERSION,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--output-init-json", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    parsed = parser.parse_args()
    run(parsed.config_name, parsed.overrides, parsed.output_init_json)


if __name__ == "__main__":
    main()
