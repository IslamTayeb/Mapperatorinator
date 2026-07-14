"""Compose one cross delta with K4/K1, shared RoPE, mixed weights, and INT8 MLP."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.inference.optimized.kernels.weight_only_runtime import (  # noqa: E402
    CROSS_FP16_PACKED,
)
from utils.run_approximate_weight_only import run_with_initializer  # noqa: E402
from utils.run_k4_shared_rope_approximate_weight_only import (  # noqa: E402
    run as run_combined,
)


CROSS_CANDIDATE_MODES = frozenset({CROSS_FP16_PACKED})


def _enrich_evidence(
    output_init_json: Path,
    *,
    mode: str,
    composition_version: str,
) -> None:
    try:
        payload = json.loads(output_init_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("cross candidate initialization evidence is invalid") from exc
    if not isinstance(payload, dict):
        raise TypeError("cross candidate initialization evidence must be an object")
    cross = payload.get("cross_candidate")
    if not isinstance(cross, dict) or cross.get("mode") != mode:
        raise RuntimeError("cross candidate initialization mode was not recorded")
    if payload.get("combined_runtime") != composition_version:
        raise RuntimeError("cross candidate composition identity was not recorded")
    if not isinstance(payload.get("shared_rope"), dict):
        raise RuntimeError("cross candidate shared-RoPE evidence is missing")
    if "cross_runtime" in payload:
        raise RuntimeError("cross candidate evidence already contains cross wiring")
    payload["cross_runtime"] = {
        **cross,
        "incremental_control": (
            "k4-split-kv-mixed-weight-shared-rope-k1-remainder-int8-mlp-v1"
        ),
        "incremental_exactness_required": True,
        "packed_projection_delta_only": True,
        "accepted_q1_bmm_required": True,
        "original_decoder_forward_required": True,
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
    mode: str,
    shared_static_input_arena: bool = False,
    transition_timing: bool = False,
) -> None:
    if mode not in CROSS_CANDIDATE_MODES:
        raise ValueError(
            f"cross candidate mode must be one of {sorted(CROSS_CANDIDATE_MODES)}"
        )
    composition_version = (
        "k4-split-kv-mixed-weight-shared-rope-k1-remainder-int8-mlp-"
        f"{mode}-v1"
    )

    def weight_runner(
        nested_config_name: str,
        nested_overrides: list[str],
        nested_output_init_json: Path,
    ) -> None:
        run_with_initializer(
            nested_config_name,
            nested_overrides,
            nested_output_init_json,
            initializer_name="initialize_approximate_int8_mlp_weight_only_cross",
            initializer_kwargs={"mode": mode},
        )

    run_combined(
        config_name,
        overrides,
        output_init_json,
        graph_remainders=True,
        weight_runner=weight_runner,
        composition_version=composition_version,
        shared_static_input_arena=shared_static_input_arena,
        transition_timing=transition_timing,
    )
    _enrich_evidence(
        output_init_json,
        mode=mode,
        composition_version=composition_version,
    )


__all__ = ["CROSS_CANDIDATE_MODES", "run"]
