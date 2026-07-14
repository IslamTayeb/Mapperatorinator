"""Validate composed shared-RoPE/stage-precision initialization evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.run_main_shared_rope_delegate import (  # noqa: E402
    SHARED_STAGE_COMPOSITION_VERSION,
)


HYBRID_VERSION = "timing-fp16-main-mixed-fp32-k4-v1"


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def validate(payload: Any, *, role: str) -> dict[str, Any]:
    if role not in {"control", "candidate"}:
        raise ValueError("role must be control or candidate")
    root = _object(payload, name="initialization")
    if root.get("combined_runtime") != SHARED_STAGE_COMPOSITION_VERSION:
        raise ValueError("combined runtime version is missing or stale")
    if root.get("composition_arm") != role:
        raise ValueError("composition arm disagrees with validation role")
    if root.get("result_class") != "documented-drift":
        raise ValueError("mixed-weight result class must remain documented-drift")
    if root.get("exactness_claim") is not False:
        raise ValueError("combined runtime must not claim overall exactness")

    shared = _object(root.get("shared_rope"), name="initialization.shared_rope")
    expected_shared = {
        "version": "shared-decoder-rope-v1",
        "scope": "main-model-only",
        "incremental_exactness_claim": True,
        "original_decoder_forward_required": True,
    }
    failures = {
        key: {"expected": expected, "actual": shared.get(key)}
        for key, expected in expected_shared.items()
        if shared.get(key) != expected
    }
    if failures:
        raise ValueError(f"shared RoPE declarations are invalid: {failures}")
    stats = _object(shared.get("stats"), name="initialization.shared_rope.stats")
    integer_fields = (
        "module_count",
        "group_count",
        "forwards",
        "computes",
        "reuses",
        "expected_computes",
        "expected_reuses",
        "eliminated_per_forward",
    )
    for field in integer_fields:
        value = stats.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"shared RoPE {field} must be a positive integer")
    if stats["module_count"] != 12 or stats["eliminated_per_forward"] != 11:
        raise ValueError("shared RoPE must cover all twelve decoder layers")
    if stats["computes"] != stats["expected_computes"]:
        raise ValueError("shared RoPE compute accounting changed")
    if stats["reuses"] != stats["expected_reuses"]:
        raise ValueError("shared RoPE reuse accounting changed")

    hybrid = root.get("stage_precision_hybrid")
    if role == "control":
        if hybrid is not None:
            raise ValueError("control must not initialize a stage-precision hybrid")
        blocks = _object(
            root.get("decode_block_sizes"),
            name="initialization.decode_block_sizes",
        )
        if blocks != {"timing_context": 1, "main_generation": 4}:
            raise ValueError("control K1/K4 block declarations changed")
    else:
        hybrid = _object(hybrid, name="initialization.stage_precision_hybrid")
        if hybrid.get("version") != HYBRID_VERSION:
            raise ValueError("candidate stage-precision version is missing or stale")
        if hybrid.get("result_class") != "documented-drift":
            raise ValueError("candidate stage precision result class changed")
        if hybrid.get("exactness_claim") is not False:
            raise ValueError("candidate stage precision must not claim exactness")
        if hybrid.get("decode_block_sizes") != {
            "timing_context": 1,
            "main_generation": 4,
        }:
            raise ValueError("candidate K1/K4 block declarations changed")

    setup = {}
    for field in (
        "initialization_wall_seconds",
        "initialization_unattributed_seconds",
        "extension_init_seconds",
        "weight_pack_seconds",
    ):
        value = root.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be numeric")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0.0:
            raise ValueError(f"{field} must be finite and non-negative")
        setup[field] = parsed

    memory = _object(
        root.get("initialization_cuda_memory"),
        name="initialization.initialization_cuda_memory",
    )
    memory_fields = (
        "allocated_bytes_before",
        "allocated_bytes_after",
        "allocated_bytes_delta",
        "reserved_bytes_before",
        "reserved_bytes_after",
        "reserved_bytes_delta",
        "max_allocated_bytes_before",
        "max_allocated_bytes_after",
        "max_allocated_bytes_delta",
    )
    for field in memory_fields:
        value = memory.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"initialization CUDA memory {field} must be an integer")

    return {
        "schema_version": 1,
        "role": role,
        "combined_runtime": SHARED_STAGE_COMPOSITION_VERSION,
        "shared_rope": {
            "forwards": stats["forwards"],
            "computes": stats["computes"],
            "reuses": stats["reuses"],
            "eliminated_per_forward": stats["eliminated_per_forward"],
        },
        "setup_seconds": setup,
        "initialization_cuda_memory": {
            field: memory[field] for field in memory_fields
        },
        "pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--role", choices=("control", "candidate"), required=True)
    args = parser.parse_args()
    report = validate(
        json.loads(args.initialization.read_text(encoding="utf-8")),
        role=args.role,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
