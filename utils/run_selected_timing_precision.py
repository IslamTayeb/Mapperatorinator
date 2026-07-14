"""Run one timing-precision arm on the unchanged selected main stack."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.inference.engine_binding import InferenceEngineBinding  # noqa: E402
from osuT5.osuT5.inference.optimized.kernels.weight_only_runtime import (  # noqa: E402
    CROSS_FP16_PACKED,
)
from osuT5.osuT5.inference.optimized.single.timing_precision_matrix import (  # noqa: E402
    FP16_WEIGHTS_FP32_STATE,
    FULL_FP16,
    TIMING_PRECISION_MODES,
    TimingPrecisionMatrixRuntime,
)
from utils.run_k4_shared_rope_cross_candidate import run as run_selected  # noqa: E402
from utils.run_selected_timing_native_self import (  # noqa: E402
    SELECTED_MAIN_COMPOSITION,
)


COMPOSITION_VERSION = "selected-main-timing-precision-matrix-v1"


def _timed_binding(
    binding: InferenceEngineBinding,
    *,
    mode: str,
) -> tuple[InferenceEngineBinding, dict]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("timing precision initialization requires CUDA")
    torch.cuda.synchronize()
    allocated_before = int(torch.cuda.memory_allocated())
    reserved_before = int(torch.cuda.memory_reserved())
    peak_before = int(torch.cuda.max_memory_allocated())
    started = time.perf_counter()
    wrapped, metadata = TimingPrecisionMatrixRuntime.from_binding(
        binding,
        mode=mode,
    )
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    if not math.isfinite(wall) or wall < 0.0:
        raise RuntimeError("timing precision initialization wall is invalid")
    allocated_after = int(torch.cuda.memory_allocated())
    reserved_after = int(torch.cuda.memory_reserved())
    peak_after = int(torch.cuda.max_memory_allocated())
    return wrapped, {
        **metadata,
        "initialization_wall_seconds": wall,
        "initialization_cuda_memory": {
            "allocated_bytes_before": allocated_before,
            "allocated_bytes_after": allocated_after,
            "allocated_bytes_delta": allocated_after - allocated_before,
            "reserved_bytes_before": reserved_before,
            "reserved_bytes_after": reserved_after,
            "reserved_bytes_delta": reserved_after - reserved_before,
            "max_allocated_bytes_before": peak_before,
            "max_allocated_bytes_after": peak_after,
            "max_allocated_bytes_delta": peak_after - peak_before,
        },
    }


def _enrich_initialization(path: Path, *, mode: str, timing: dict) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("selected initialization evidence is invalid") from exc
    if not isinstance(payload, dict):
        raise TypeError("selected initialization evidence must be an object")
    if payload.get("combined_runtime") != SELECTED_MAIN_COMPOSITION:
        raise RuntimeError("selected main composition changed")
    if payload.get("cross_candidate", {}).get("mode") != CROSS_FP16_PACKED:
        raise RuntimeError("selected FP16-packed main cross candidate is missing")
    if "timing_precision_matrix" in payload:
        raise RuntimeError("timing precision evidence already exists")
    payload["timing_precision_matrix"] = {
        **timing,
        "composition_version": COMPOSITION_VERSION,
        "incremental_control": SELECTED_MAIN_COMPOSITION,
        "timing_model_distinct_from_main": True,
        "fixed_main_tokens": 8_294,
        "fixed_timing_tokens": 821,
        "complete_request_wall_required": True,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(
    config_name: str,
    overrides: list[str],
    output_init_json: Path,
    *,
    mode: str,
) -> None:
    if mode not in TIMING_PRECISION_MODES:
        raise ValueError(
            f"timing precision mode must be one of {sorted(TIMING_PRECISION_MODES)}"
        )
    import inference

    original_loader = inference.load_model_with_engine
    load_count = 0
    main_model = None
    timing_metadata: dict | None = None

    def timing_loader(*loader_args, **loader_kwargs):
        nonlocal load_count, main_model, timing_metadata
        if load_count > 1:
            raise RuntimeError("timing precision runner observed an extra model load")
        auto_select = loader_kwargs.get("auto_select_gamemode_model")
        effective_kwargs = dict(loader_kwargs)
        if load_count == 1 and mode == FULL_FP16:
            effective_kwargs["precision"] = "fp16"
        binding, tokenizer = original_loader(*loader_args, **effective_kwargs)
        if not isinstance(binding, InferenceEngineBinding):
            raise TypeError("timing precision runner requires optimized bindings")
        if load_count == 0:
            if auto_select is not True:
                raise RuntimeError("first precision-matrix load must be main")
            main_model = binding.raw_model
        else:
            if auto_select is not False:
                raise RuntimeError("second precision-matrix load must be timing")
            if binding.raw_model is main_model:
                raise RuntimeError("timing precision runner requires distinct models")
            binding, timing_metadata = _timed_binding(binding, mode=mode)
        load_count += 1
        return binding, tokenizer

    inference.load_model_with_engine = timing_loader
    try:
        run_selected(
            config_name,
            overrides,
            output_init_json,
            mode=CROSS_FP16_PACKED,
        )
    finally:
        inference.load_model_with_engine = original_loader
    if load_count != 2 or timing_metadata is None:
        raise RuntimeError("timing precision runner requires two distinct model loads")
    _enrich_initialization(
        output_init_json,
        mode=mode,
        timing=timing_metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--output-init-json", type=Path, required=True)
    parser.add_argument("--timing-mode", choices=sorted(TIMING_PRECISION_MODES), required=True)
    parser.add_argument("overrides", nargs="*")
    parsed = parser.parse_args()
    run(
        parsed.config_name,
        parsed.overrides,
        parsed.output_init_json,
        mode=parsed.timing_mode,
    )


if __name__ == "__main__":
    main()
