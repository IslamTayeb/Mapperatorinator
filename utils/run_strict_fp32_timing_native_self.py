"""Opt only the standalone FP32 timing model into accepted native self kernels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RUNNER_VERSION = "strict-fp32-timing-native-self-runner-v1"


def _require_strict_fp32_environment() -> None:
    failures: dict[str, Any] = {}
    actual = {
        "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    expected = {
        "NVIDIA_TF32_OVERRIDE": "0",
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }
    for key, value in expected.items():
        if actual[key] != value:
            failures[key] = {"expected": value, "actual": actual[key]}
    if failures:
        raise RuntimeError(
            f"strict FP32 timing-native-self environment mismatch: {failures}"
        )


def _validate_overrides(overrides: list[str]) -> None:
    fixed = {
        "device": "cuda",
        "precision": "fp32",
        "attn_implementation": "sdpa",
        "inference_engine": "optimized",
        "use_server": "false",
        "parallel": "false",
        "cfg_scale": "1.0",
        "num_beams": "1",
        "super_timing": "false",
    }
    supplied = {
        key: value
        for item in overrides
        if "=" in item
        for key, value in [item.split("=", 1)]
        if key in fixed
    }
    failures = {
        key: {"expected": expected, "actual": supplied[key]}
        for key, expected in fixed.items()
        if key in supplied and supplied[key].lower() != expected
    }
    if failures:
        raise ValueError(
            f"strict FP32 timing-native-self runner rejected overrides: {failures}"
        )


def _write_initialization(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite timing-native-self initialization: {path}"
        )
    payload = {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "timing_native_self": metadata,
        "model_loads": {
            "count": 2,
            "main_auto_select_gamemode_model": True,
            "timing_auto_select_gamemode_model": False,
            "owners_distinct": True,
        },
        "strict_fp32": {
            "NVIDIA_TF32_OVERRIDE": "0",
            "float32_matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(config_name: str, overrides: list[str], output_init_json: Path) -> None:
    """Run inference after installing one request-scoped loader interception."""

    _validate_overrides(overrides)
    import inference
    from osuT5.osuT5.inference.engine_binding import InferenceEngineBinding
    from osuT5.osuT5.inference.optimized.single.engine import OptimizedSingleRuntime

    original_loader = inference.load_model_with_engine
    original_argv = sys.argv[:]
    load_count = 0
    main_model: torch.nn.Module | None = None
    timing_metadata: dict[str, Any] | None = None

    def candidate_loader(*loader_args, **loader_kwargs):
        nonlocal load_count, main_model, timing_metadata
        if load_count >= 2:
            raise RuntimeError(
                "strict FP32 timing-native-self runner observed an extra model load"
            )
        _require_strict_fp32_environment()
        binding, tokenizer = original_loader(*loader_args, **loader_kwargs)
        if not isinstance(binding, InferenceEngineBinding):
            raise TypeError(
                "strict FP32 timing-native-self runner requires optimized bindings"
            )
        if type(binding.runtime) is not OptimizedSingleRuntime:
            raise TypeError(
                "strict FP32 timing-native-self runner requires the single runtime"
            )
        if binding.runtime.preset.precision != "fp32":
            raise TypeError(
                "strict FP32 timing-native-self runner requires the FP32 preset"
            )
        if binding.raw_model.dtype != torch.float32:
            raise TypeError(
                "strict FP32 timing-native-self runner requires FP32 model storage"
            )
        auto_select = loader_kwargs.get("auto_select_gamemode_model")
        if load_count == 0:
            if auto_select is not True:
                raise RuntimeError("first model load must be the selected main model")
            main_model = binding.raw_model
            if binding.runtime._strict_fp32_timing_native_self_owner is not None:
                raise RuntimeError(
                    "main model unexpectedly owns timing-native-self policy"
                )
        else:
            if auto_select is not False:
                raise RuntimeError("second model load must be the base timing model")
            if binding.raw_model is main_model:
                raise RuntimeError(
                    "timing-native-self runner requires a standalone timing model"
                )
            timing_metadata = (
                binding.runtime.initialize_strict_fp32_timing_native_self(
                    binding.raw_model
                )
            )
        load_count += 1
        return binding, tokenizer

    inference.load_model_with_engine = candidate_loader
    sys.argv = [
        str(REPO_ROOT / "inference.py"),
        "--config-name",
        config_name,
        *overrides,
    ]
    try:
        inference.main()
    finally:
        inference.load_model_with_engine = original_loader
        sys.argv = original_argv
    if load_count != 2 or timing_metadata is None:
        raise RuntimeError(
            "strict FP32 timing-native-self runner requires exactly one distinct "
            "main model and one timing model"
        )
    _write_initialization(output_init_json, timing_metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--output-init-json", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    run(args.config_name, args.overrides, args.output_init_json)


if __name__ == "__main__":
    main()
