"""Run K=4 mixed weights with request-local decoder-mask reuse enabled."""

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
    with install_k8_candidate(
        block_size=4,
        reuse_decoder_attention_mask=True,
    ):
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
