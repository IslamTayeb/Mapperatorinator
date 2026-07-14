"""Compose FP32 timing-native-self over the selected K1+INT8+cross runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.inference.engine_binding import InferenceEngineBinding  # noqa: E402
from osuT5.osuT5.inference.optimized.kernels.weight_only_runtime import (  # noqa: E402
    CROSS_FP16_PACKED,
)
from osuT5.osuT5.inference.optimized.single.engine import (  # noqa: E402
    OptimizedSingleRuntime,
)
from utils.run_k4_shared_rope_cross_candidate import run as run_composed  # noqa: E402


COMPOSITION_VERSION = "k1-int8-fp16-cross-fp32-timing-native-self-v1"


def _enrich_initialization(path: Path, timing_metadata: dict) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("composed initialization evidence is invalid") from exc
    if not isinstance(payload, dict):
        raise TypeError("composed initialization evidence must be an object")
    if payload.get("cross_candidate", {}).get("mode") != CROSS_FP16_PACKED:
        raise RuntimeError("selected FP16-packed cross candidate is missing")
    if not isinstance(payload.get("cross_runtime"), dict):
        raise RuntimeError("cross runtime evidence is missing")
    if "timing_native_self_scout" in payload:
        raise RuntimeError("timing-native-self evidence already exists")
    payload["timing_native_self_scout"] = {
        **timing_metadata,
        "composition_version": COMPOSITION_VERSION,
        "incremental_control": payload.get("combined_runtime"),
        "timing_model_distinct_from_main": True,
        "fixed_timing_work_required": True,
        "complete_request_wall_required": True,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(config_name: str, overrides: list[str], output_init_json: Path) -> None:
    import inference

    original_loader = inference.load_model_with_engine
    load_count = 0
    main_model = None
    timing_metadata: dict | None = None

    def scout_loader(*loader_args, **loader_kwargs):
        nonlocal load_count, main_model, timing_metadata
        if load_count > 1:
            raise RuntimeError("timing-native-self scout observed an extra model load")
        binding, tokenizer = original_loader(*loader_args, **loader_kwargs)
        if not isinstance(binding, InferenceEngineBinding):
            raise TypeError("timing-native-self scout requires an optimized binding")
        if type(binding.runtime) is not OptimizedSingleRuntime:
            raise TypeError("timing-native-self scout requires the single runtime")
        auto_select = loader_kwargs.get("auto_select_gamemode_model")
        if load_count == 0:
            if auto_select is not True:
                raise RuntimeError("first scout load must be the main gamemode model")
            main_model = binding.raw_model
            if binding.runtime._timing_native_self_owner is not None:
                raise RuntimeError("main runtime unexpectedly enabled timing-native-self")
        else:
            if auto_select is not False:
                raise RuntimeError("second scout load must be the base timing model")
            if binding.raw_model is main_model:
                raise RuntimeError("timing-native-self scout requires distinct models")
            if binding.runtime._approximate_weight_only_state is not None:
                raise RuntimeError("timing runtime unexpectedly owns mixed weights")
            initializer = getattr(
                binding.runtime,
                "initialize_fp32_timing_native_self",
                None,
            )
            if not callable(initializer):
                raise RuntimeError("timing runtime lacks the native-self initializer")
            timing_metadata = initializer(binding.raw_model)
            if not isinstance(timing_metadata, dict):
                raise TypeError("timing-native-self initializer returned invalid metadata")
        load_count += 1
        return binding, tokenizer

    inference.load_model_with_engine = scout_loader
    try:
        run_composed(
            config_name,
            overrides,
            output_init_json,
            mode=CROSS_FP16_PACKED,
        )
    finally:
        inference.load_model_with_engine = original_loader
    if load_count != 2 or timing_metadata is None:
        raise RuntimeError(
            "timing-native-self scout requires exactly one main and one timing model"
        )
    _enrich_initialization(output_init_json, timing_metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--output-init-json", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    parsed = parser.parse_args()
    run(parsed.config_name, parsed.overrides, parsed.output_init_json)


if __name__ == "__main__":
    main()
