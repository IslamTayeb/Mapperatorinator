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
from utils.run_k4_shared_rope_cross_candidate import run as run_selected  # noqa: E402


COMPOSITION_VERSION = "k1-int8-fp16-cross-fp32-timing-native-self-v1"
SELECTED_MAIN_COMPOSITION = (
    "k4-split-kv-mixed-weight-shared-rope-k1-remainder-int8-mlp-"
    "fp16_packed_projections-v1"
)


def _enrich_initialization(path: Path, timing_metadata: dict) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("selected initialization evidence is invalid") from exc
    if not isinstance(payload, dict):
        raise TypeError("selected initialization evidence must be an object")
    if payload.get("combined_runtime") != SELECTED_MAIN_COMPOSITION:
        raise RuntimeError("selected K4/K1/shared-RoPE/INT8 composition is missing")
    if payload.get("cross_candidate", {}).get("mode") != CROSS_FP16_PACKED:
        raise RuntimeError("selected FP16-packed cross candidate is missing")
    cross_runtime = payload.get("cross_runtime")
    if not isinstance(cross_runtime, dict):
        raise RuntimeError("selected cross runtime evidence is missing")
    if cross_runtime.get("accepted_q1_bmm_required") is not True:
        raise RuntimeError("selected q1 BMM cross topology is missing")
    if cross_runtime.get("original_decoder_forward_required") is not True:
        raise RuntimeError("selected original decoder forward is missing")
    if timing_metadata.get("result_class") != "exact":
        raise RuntimeError("timing-native-self delta must be exact")
    if timing_metadata.get("exactness_claim") is not True:
        raise RuntimeError("timing-native-self exactness claim is missing")
    if "timing_native_self" in payload:
        raise RuntimeError("timing-native-self evidence already exists")
    payload["timing_native_self"] = {
        **timing_metadata,
        "composition_version": COMPOSITION_VERSION,
        "incremental_control": SELECTED_MAIN_COMPOSITION,
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

    def timing_loader(*loader_args, **loader_kwargs):
        nonlocal load_count, main_model, timing_metadata
        if load_count > 1:
            raise RuntimeError("selected runner observed an extra model load")
        binding, tokenizer = original_loader(*loader_args, **loader_kwargs)
        if not isinstance(binding, InferenceEngineBinding):
            raise TypeError("selected runner requires an optimized binding")
        if type(binding.runtime) is not OptimizedSingleRuntime:
            raise TypeError("selected runner requires the single runtime")
        auto_select = loader_kwargs.get("auto_select_gamemode_model")
        if load_count == 0:
            if auto_select is not True:
                raise RuntimeError("first selected load must be the main model")
            main_model = binding.raw_model
            if binding.runtime._timing_native_self_owner is not None:
                raise RuntimeError("main runtime unexpectedly enabled timing-native-self")
        else:
            if auto_select is not False:
                raise RuntimeError("second selected load must be the timing model")
            if binding.raw_model is main_model:
                raise RuntimeError("selected runner requires distinct model owners")
            if binding.runtime._approximate_weight_only_state is not None:
                raise RuntimeError("timing runtime unexpectedly owns mixed weights")
            timing_metadata = binding.runtime.initialize_fp32_timing_native_self(
                binding.raw_model
            )
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
        raise RuntimeError(
            "selected runner requires exactly one main and one timing model"
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
