"""Selected shared-arena stack with opt-in split-KV num_splits override."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.inference.optimized.kernels.q1_attention import (  # noqa: E402
    _ALLOWED_SPLIT_KV_SPLITS,
    _SPLIT_KV_SPLITS_ENV,
    configured_split_kv_q1_splits,
)
from osuT5.osuT5.inference.optimized.kernels.weight_only_runtime import (  # noqa: E402
    CROSS_FP16_PACKED,
)
from utils.run_k4_shared_rope_cross_candidate import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--output-init-json", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    parsed = parser.parse_args()

    raw = os.environ.get(_SPLIT_KV_SPLITS_ENV)
    if raw is None or raw == "":
        raise SystemExit(
            f"candidate requires {_SPLIT_KV_SPLITS_ENV} in "
            f"{sorted(_ALLOWED_SPLIT_KV_SPLITS)}"
        )
    splits = configured_split_kv_q1_splits()
    if splits == 8:
        raise SystemExit(
            f"candidate {_SPLIT_KV_SPLITS_ENV} must differ from control default 8"
        )

    run(
        parsed.config_name,
        parsed.overrides,
        parsed.output_init_json,
        mode=CROSS_FP16_PACKED,
        shared_static_input_arena=True,
    )


if __name__ == "__main__":
    main()
