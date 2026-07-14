"""Add request-local decoder-mask reuse to the shared-RoPE K4 runtime."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.run_k4_shared_rope_approximate_weight_only import (  # noqa: E402
    run as run_shared_rope,
)


def run(config_name: str, overrides: list[str], output_init_json: Path) -> None:
    run_shared_rope(
        config_name,
        overrides,
        output_init_json,
        reuse_decoder_attention_mask=True,
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
