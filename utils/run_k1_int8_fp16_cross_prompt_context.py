"""Run the selected runtime with exact main prompt/context preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.inference.optimized.kernels.weight_only_runtime import (  # noqa: E402
    CROSS_FP16_PACKED,
)
from osuT5.osuT5.inference.optimized.scout.prompt_context import (  # noqa: E402
    install_exact_prompt_context_preparation,
)
from utils.run_k4_shared_rope_cross_candidate import run as run_cross  # noqa: E402


COMPOSITION_VERSION = (
    "k4-split-kv-mixed-weight-shared-rope-k1-remainder-int8-mlp-"
    "fp16_packed_projections-exact-prompt-context-v1"
)


def _enrich_evidence(output_init_json: Path, prompt_summary: dict) -> None:
    try:
        payload = json.loads(output_init_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("prompt-context initialization evidence is invalid") from exc
    if not isinstance(payload, dict):
        raise TypeError("prompt-context initialization evidence must be an object")
    expected_base = (
        "k4-split-kv-mixed-weight-shared-rope-k1-remainder-int8-mlp-"
        "fp16_packed_projections-v1"
    )
    if payload.get("combined_runtime") != expected_base:
        raise RuntimeError("prompt-context base composition identity is wrong")
    if "prompt_context_preparation" in payload:
        raise RuntimeError("prompt-context evidence already exists")
    if prompt_summary.get("scope") != "main_generation_only":
        raise RuntimeError("prompt-context evidence scope is wrong")
    if prompt_summary.get("exactness_required") is not True:
        raise RuntimeError("prompt-context evidence must require exactness")
    payload["prompt_context_preparation"] = prompt_summary
    payload["prompt_context_incremental_control"] = expected_base
    payload["combined_runtime"] = COMPOSITION_VERSION
    output_init_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(config_name: str, overrides: list[str], output_init_json: Path) -> None:
    with install_exact_prompt_context_preparation() as prompt_state:
        run_cross(
            config_name,
            overrides,
            output_init_json,
            mode=CROSS_FP16_PACKED,
        )
    _enrich_evidence(output_init_json, prompt_state.summary())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--output-init-json", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    parsed = parser.parse_args()
    run(parsed.config_name, parsed.overrides, parsed.output_init_json)


if __name__ == "__main__":
    main()
