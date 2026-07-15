"""Opt-in strict-FP32 inference with exact incremental prompt preparation."""

from __future__ import annotations

import os
import sys
from typing import Any


def _require_strict_fp32_cli(argv: list[str]) -> None:
    overrides = {}
    for value in argv:
        if "=" not in value:
            continue
        key, raw = value.split("=", 1)
        overrides[key.lstrip("+")] = raw
    required = {
        "precision": "fp32",
        "inference_engine": "optimized",
    }
    failures = {
        key: {"expected": expected, "actual": overrides.get(key)}
        for key, expected in required.items()
        if overrides.get(key) != expected
    }
    if failures:
        raise RuntimeError(
            "strict FP32 prompt-context inference requires explicit overrides: "
            f"{failures}"
        )


def run() -> Any:
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError(
            "strict FP32 prompt-context inference requires NVIDIA_TF32_OVERRIDE=0 "
            "before Python starts"
        )
    _require_strict_fp32_cli(sys.argv[1:])

    from inference import main as inference_main
    from osuT5.osuT5.inference.optimized.scout.prompt_context import (
        install_exact_prompt_context_preparation,
    )

    with install_exact_prompt_context_preparation() as state:
        result = inference_main()
    state.summary()
    return result


if __name__ == "__main__":
    run()
