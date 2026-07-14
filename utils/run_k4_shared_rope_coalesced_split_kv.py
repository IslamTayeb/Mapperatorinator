"""Add the opt-in coalesced split-KV self kernel to the accepted shared stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.inference.optimized.kernels import q1_attention  # noqa: E402
from osuT5.osuT5.inference.optimized.kernels import weight_only_runtime  # noqa: E402
from utils.run_k4_shared_rope_approximate_weight_only import (  # noqa: E402
    run as run_shared_stack,
)


COMPOSITION_VERSION = "k4-split-kv-coalesced-mixed-weight-shared-rope-v1"


def _enrich_initialization_evidence(output_init_json: Path) -> None:
    try:
        payload = json.loads(output_init_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("coalesced split-KV initialization evidence is invalid") from exc
    if not isinstance(payload, dict):
        raise TypeError("coalesced split-KV initialization evidence must be an object")
    if payload.get("combined_runtime") != (
        "k4-split-kv-mixed-weight-shared-rope-v1"
    ):
        raise RuntimeError("coalesced split-KV runner lost its exact shared-stack base")
    if "coalesced_split_kv" in payload:
        raise RuntimeError("coalesced split-KV evidence already exists")
    payload["combined_runtime"] = COMPOSITION_VERSION
    payload["coalesced_split_kv"] = {
        "version": "sm75-q1-split-kv-8-coalesced-v1",
        "scope": "main-model-mixed-weight-self-attention-only",
        "accepted_short_prefix_fallback": True,
        "exactness_claim": False,
        "required_dispatch_label": (
            "native_q1_rope_cache_self_attention_split_kv_8_coalesced"
        ),
    }
    output_init_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(config_name: str, overrides: list[str], output_init_json: Path) -> None:
    """Patch only the model-owned mixed-weight self-attention hook."""

    original_attention = weight_only_runtime.native_q1_rope_cache_attention
    original_variant = weight_only_runtime.native_q1_rope_cache_attention_variant
    weight_only_runtime.native_q1_rope_cache_attention = (
        q1_attention.native_q1_rope_cache_attention_coalesced
    )
    weight_only_runtime.native_q1_rope_cache_attention_variant = (
        q1_attention.native_q1_rope_cache_attention_coalesced_variant
    )
    try:
        run_shared_stack(config_name, overrides, output_init_json)
    finally:
        weight_only_runtime.native_q1_rope_cache_attention = original_attention
        weight_only_runtime.native_q1_rope_cache_attention_variant = original_variant
    _enrich_initialization_evidence(output_init_json)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--output-init-json", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    parsed = parser.parse_args()
    run(parsed.config_name, parsed.overrides, parsed.output_init_json)


if __name__ == "__main__":
    main()
