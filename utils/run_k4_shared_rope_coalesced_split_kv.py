"""Add opt-in coalesced split-KV above K1, INT8, and one owned cross mode."""

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
from osuT5.osuT5.inference.optimized.kernels.weight_only_runtime import (  # noqa: E402
    CROSS_ACCEPTED,
    CROSS_MODES,
)
from utils.run_k4_shared_rope_cross_candidate import (  # noqa: E402
    run as run_cross_candidate,
)
from utils.run_k4_shared_rope_k1_remainder_int8_mlp_weight_only import (  # noqa: E402
    COMPOSITION_VERSION as K1_INT8_COMPOSITION_VERSION,
    run as run_k1_int8,
)


COMPOSITION_VERSION = (
    "k4-split-kv-mixed-weight-shared-rope-k1-remainder-int8-mlp-"
    "coalesced-split-kv-v1"
)


def _base_composition_version(cross_mode: str) -> str:
    if cross_mode == CROSS_ACCEPTED:
        return K1_INT8_COMPOSITION_VERSION
    return (
        "k4-split-kv-mixed-weight-shared-rope-k1-remainder-int8-mlp-"
        f"{cross_mode}-v1"
    )


def _composition_version(cross_mode: str) -> str:
    if cross_mode == CROSS_ACCEPTED:
        return COMPOSITION_VERSION
    return f"{_base_composition_version(cross_mode).removesuffix('-v1')}-coalesced-split-kv-v1"


def _enrich_initialization_evidence(
    output_init_json: Path,
    *,
    cross_mode: str,
) -> None:
    try:
        payload = json.loads(output_init_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("coalesced split-KV initialization evidence is invalid") from exc
    if not isinstance(payload, dict):
        raise TypeError("coalesced split-KV initialization evidence must be an object")
    expected_base = _base_composition_version(cross_mode)
    if payload.get("combined_runtime") != expected_base:
        raise RuntimeError("coalesced split-KV runner lost its requested base topology")
    if "coalesced_split_kv" in payload:
        raise RuntimeError("coalesced split-KV evidence already exists")
    payload["combined_runtime"] = _composition_version(cross_mode)
    payload["coalesced_split_kv"] = {
        "version": "sm75-q1-split-kv-8-coalesced-v1",
        "scope": "main-model-mixed-weight-self-attention-only",
        "accepted_short_prefix_fallback": True,
        "cross_mode": cross_mode,
        "result_class": "documented-drift",
        "exactness_claim": False,
        "incremental_exactness_claim": False,
        "original_decoder_forward_required": True,
        "required_dispatch_label": (
            "native_q1_rope_cache_self_attention_split_kv_8_coalesced"
        ),
    }
    output_init_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(
    config_name: str,
    overrides: list[str],
    output_init_json: Path,
    *,
    cross_mode: str = CROSS_ACCEPTED,
) -> None:
    """Patch only self-attention while retaining one requested cross mode."""

    if cross_mode not in CROSS_MODES:
        raise ValueError(f"cross_mode must be one of {sorted(CROSS_MODES)}")

    original_attention = weight_only_runtime.native_q1_rope_cache_attention
    original_variant = weight_only_runtime.native_q1_rope_cache_attention_variant
    weight_only_runtime.native_q1_rope_cache_attention = (
        q1_attention.native_q1_rope_cache_attention_coalesced
    )
    weight_only_runtime.native_q1_rope_cache_attention_variant = (
        q1_attention.native_q1_rope_cache_attention_coalesced_variant
    )
    try:
        if cross_mode == CROSS_ACCEPTED:
            run_k1_int8(config_name, overrides, output_init_json)
        else:
            run_cross_candidate(
                config_name,
                overrides,
                output_init_json,
                mode=cross_mode,
            )
    finally:
        weight_only_runtime.native_q1_rope_cache_attention = original_attention
        weight_only_runtime.native_q1_rope_cache_attention_variant = original_variant
    _enrich_initialization_evidence(output_init_json, cross_mode=cross_mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--output-init-json", type=Path, required=True)
    parser.add_argument("--cross-mode", choices=sorted(CROSS_MODES), default=CROSS_ACCEPTED)
    parser.add_argument("overrides", nargs="*")
    parsed = parser.parse_args()
    run(
        parsed.config_name,
        parsed.overrides,
        parsed.output_init_json,
        cross_mode=parsed.cross_mode,
    )


if __name__ == "__main__":
    main()
