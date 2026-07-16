"""Selected shared-arena control + opt-in cross out layout fusion.

Compares against ``run_k4_shared_rope_fp16_cross_shared_arena.py``.
Does **not** enable compiled-cross BMM.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.inference.optimized.kernels.cross_out_epilogue import (  # noqa: E402
    LAYOUT_FUSED_ENV,
)
from osuT5.osuT5.inference.optimized.kernels.weight_only_runtime import (  # noqa: E402
    CROSS_FP16_PACKED,
)
from utils.run_k4_shared_rope_cross_candidate import run  # noqa: E402


def _enrich_layout_fused_evidence(output_init_json: Path) -> None:
    try:
        payload = json.loads(output_init_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("layout-fused initialization evidence is invalid") from exc
    if not isinstance(payload, dict):
        raise TypeError("layout-fused initialization evidence must be an object")
    if payload.get("compiled_cross_bmm") is not None:
        raise RuntimeError("layout-fused candidate unexpectedly enabled compiled-cross")
    if "cross_out_layout_fused" in payload:
        raise RuntimeError("layout-fused evidence already present")
    payload["cross_out_layout_fused"] = {
        "enabled": True,
        "env": LAYOUT_FUSED_ENV,
        "variant": "heads_view_fused",
        "compiled_cross_composed": False,
        "incremental_control": (
            "k4-split-kv-mixed-weight-shared-rope-k1-remainder-int8-mlp-"
            "fp16_packed_projections-shared-arena-cross-out-layout-fused-v1"
        ),
    }
    output_init_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--output-init-json", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    parsed = parser.parse_args()
    previous = os.environ.get(LAYOUT_FUSED_ENV)
    os.environ[LAYOUT_FUSED_ENV] = "1"
    try:
        run(
            parsed.config_name,
            parsed.overrides,
            parsed.output_init_json,
            mode=CROSS_FP16_PACKED,
            shared_static_input_arena=True,
            compiled_cross_bmm=False,
        )
    finally:
        if previous is None:
            os.environ.pop(LAYOUT_FUSED_ENV, None)
        else:
            os.environ[LAYOUT_FUSED_ENV] = previous
    _enrich_layout_fused_evidence(parsed.output_init_json)


if __name__ == "__main__":
    main()
