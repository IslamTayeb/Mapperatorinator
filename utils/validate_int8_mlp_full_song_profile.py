"""Validate incremental INT8 MLP dispatch on the combined full-song runtime."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROLES = ("baseline", "candidate")
LABELS = ("timing_context", "main_generation")
COUNTER = "int8_weight_mlp_tail"
OVERLAY_VERSION = "per-row-symmetric-int8-mlp-v1"


class Int8MlpProfileError(ValueError):
    pass


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Int8MlpProfileError(f"{name} must be an object")
    return value


def validate_profile(payload: Any, *, role: str) -> dict[str, Any]:
    if role not in ROLES:
        raise Int8MlpProfileError(f"role must be one of {ROLES}")
    root = _object(payload, name="profile")
    if root.get("schema_version") != 1:
        raise Int8MlpProfileError("profile.schema_version must be 1")
    metadata = _object(root.get("metadata"), name="profile.metadata")
    weight_metadata = _object(
        metadata.get("optimized_approximate_weight_only"),
        name="profile.metadata.optimized_approximate_weight_only",
    )
    overlay = weight_metadata.get("int8_mlp_overlay")
    candidate = role == "candidate"
    if candidate:
        overlay = _object(overlay, name="int8_mlp_overlay")
        required = {
            "version": OVERLAY_VERSION,
            "result_class": "documented-drift",
            "exactness_claim": False,
            "scope": "main-model-decoder-mlp-only",
            "composition_control": "k4-split-kv-mixed-weight-shared-rope-v1",
            "replaces_effective_regions": ["mlp_fc1", "mlp_fc2"],
            "retained_unused_fp16_regions": ["mlp_fc1", "mlp_fc2"],
            "fp32_activations_norm_bias_reductions_residual_outputs": True,
            "quantization": "symmetric-per-output-row",
            "dispatch_counter": COUNTER,
        }
        failures = {
            key: {"expected": expected, "actual": overlay.get(key)}
            for key, expected in required.items()
            if overlay.get(key) != expected
        }
        for key in (
            "extension_init_seconds",
            "weight_pack_seconds",
            "packed_weight_bytes",
        ):
            value = overlay.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                failures[key] = {"expected": "finite non-negative", "actual": value}
        if failures:
            raise Int8MlpProfileError(f"invalid INT8 MLP metadata: {failures}")
    elif overlay is not None:
        raise Int8MlpProfileError("incremental control must not initialize INT8 MLP")

    generation = root.get("generation")
    if not isinstance(generation, list) or not generation:
        raise Int8MlpProfileError("profile.generation must be a non-empty list")
    totals = {label: 0 for label in LABELS}
    base_mlp_totals = {label: 0 for label in LABELS}
    seen = {label: 0 for label in LABELS}
    for index, raw in enumerate(generation):
        record = _object(raw, name=f"profile.generation[{index}]")
        label = record.get("profile_label")
        if label not in totals:
            continue
        seen[label] += 1
        hits = _object(
            record.get("optimized_dispatch_capture_hits"),
            name=f"profile.generation[{index}].hits",
        )
        int8_count = hits.get(COUNTER, 0)
        base_count = hits.get("weight_only_mlp_tail", 0)
        for key, value in ((COUNTER, int8_count), ("weight_only_mlp_tail", base_count)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise Int8MlpProfileError(f"{label}.{key} must be non-negative integer")
        totals[label] += int8_count
        base_mlp_totals[label] += base_count

    missing = [label for label, count in seen.items() if count <= 0]
    if missing:
        raise Int8MlpProfileError(f"profile is missing labels {missing}")
    if totals["timing_context"] != 0:
        raise Int8MlpProfileError("timing context must not execute INT8 MLP")
    if candidate:
        if totals["main_generation"] <= 0:
            raise Int8MlpProfileError("candidate main generation did not execute INT8 MLP")
        if totals["main_generation"] != base_mlp_totals["main_generation"]:
            raise Int8MlpProfileError(
                "INT8 MLP count must equal the mixed-runtime MLP hook count"
            )
    elif any(totals.values()):
        raise Int8MlpProfileError("incremental control executed INT8 MLP")
    return {
        "schema_version": 1,
        "role": role,
        "overlay_present": overlay is not None,
        "dispatch_totals": totals,
        "mixed_mlp_hook_totals": base_mlp_totals,
        "pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    parsed = parser.parse_args()
    payload = json.loads(parsed.profile.read_text(encoding="utf-8"))
    print(json.dumps(validate_profile(payload, role=parsed.role), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
