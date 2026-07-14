"""Run the mixed-FP32 control with K1 timing and K4 main decoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.event import ContextType  # noqa: E402
from osuT5.osuT5.inference.engine_binding import InferenceEngineBinding  # noqa: E402
from osuT5.osuT5.inference.optimized.single.engine import (  # noqa: E402
    OPTIMIZED_PRESETS,
    OptimizedSingleRuntime,
)
from osuT5.osuT5.inference.optimized.single.k8_runtime import (  # noqa: E402
    install_k8_candidate,
    k8_block_size_context,
)
from utils.fixed_seed_inference import fixed_seed_processor_generation  # noqa: E402
from utils.run_approximate_weight_only import (  # noqa: E402
    _initialize_with_evidence,
    _load_args,
)


class _TimingK1Runtime:
    def __init__(self, runtime: OptimizedSingleRuntime, model):
        if type(runtime) is not OptimizedSingleRuntime:
            raise TypeError("stage-aware control requires optimized single runtime")
        if runtime.preset is not OPTIMIZED_PRESETS["fp32"]:
            raise RuntimeError("stage-aware control timing runtime must be FP32")
        if runtime._approximate_weight_only_state is not None:
            raise RuntimeError("stage-aware control timing model must not own mixed weights")
        self._runtime = runtime
        self._model = model

    def new_context_state(self):
        return self._runtime.new_context_state()

    def generate_window(self, **kwargs):
        if kwargs.get("model") is not self._model:
            raise RuntimeError("stage-aware timing control received wrong model identity")
        generate_kwargs = kwargs.get("generate_kwargs")
        if not isinstance(generate_kwargs, dict):
            raise TypeError("stage-aware timing control requires generate_kwargs")
        context = generate_kwargs.get("context_type")
        if context is None or ContextType(context) != ContextType.TIMING:
            raise RuntimeError("stage-aware timing control may execute only timing context")
        with k8_block_size_context(1):
            return self._runtime.generate_window(**kwargs)

    def profile_metadata(self) -> dict[str, Any]:
        return self._runtime.profile_metadata()


def run(
    config_name: str,
    overrides: list[str],
    output_init_json: Path,
    *,
    initializer_name: str = "initialize_approximate_weight_only",
    graph_remainders: bool = False,
) -> None:
    import inference

    if not isinstance(initializer_name, str) or not initializer_name:
        raise ValueError("stage-aware initializer name must be non-empty")
    if not isinstance(graph_remainders, bool):
        raise TypeError("graph_remainders must be boolean")
    args = _load_args(config_name, overrides)
    if args.inference_engine != "optimized" or args.precision != "fp32":
        raise ValueError("stage-aware control requires optimized FP32 inference")
    if args.use_server or not args.profile_inference:
        raise ValueError("stage-aware control requires local profiled inference")
    if args.super_timing or args.generate_positions:
        raise ValueError("stage-aware control supports Processor timing/main only")
    if not inference.should_load_separate_timing_model(args):
        raise RuntimeError("stage-aware control requires a distinct timing model")

    original_loader = inference.load_model_with_engine
    load_count = 0
    main_model = None
    timing_model = None
    initialization = None

    def control_loader(*loader_args, **loader_kwargs):
        nonlocal load_count, main_model, timing_model, initialization
        if load_count > 1:
            raise RuntimeError("stage-aware control observed unexpected model load")
        if loader_kwargs.get("precision") != "fp32":
            raise RuntimeError("stage-aware control loader request must remain FP32")
        auto_select = loader_kwargs.get("auto_select_gamemode_model")
        expected = load_count == 0
        if auto_select is not expected:
            raise RuntimeError("stage-aware control model load order changed")
        binding, tokenizer = original_loader(*loader_args, **loader_kwargs)
        if not isinstance(binding, InferenceEngineBinding):
            raise TypeError("stage-aware control requires optimized bindings")
        if load_count == 0:
            initializer = getattr(binding.runtime, initializer_name, None)
            if not callable(initializer):
                raise RuntimeError(
                    "main control runtime lacks requested mixed-weight initialization "
                    f"{initializer_name!r}"
                )
            initialization = _initialize_with_evidence(initializer, binding.raw_model)
            main_model = binding.raw_model
            result = binding, tokenizer
        else:
            if binding.raw_model is main_model:
                raise RuntimeError("control timing and main models must be distinct")
            timing_model = binding.raw_model
            result = (
                InferenceEngineBinding(
                    raw_model=binding.raw_model,
                    runtime=_TimingK1Runtime(binding.runtime, binding.raw_model),
                ),
                tokenizer,
            )
        load_count += 1
        return result

    inference.load_model_with_engine = control_loader
    try:
        install_options = {"block_size": 4}
        if graph_remainders:
            install_options["graph_remainders"] = True
        with install_k8_candidate(**install_options):
            with fixed_seed_processor_generation(inference, base_seed=args.seed):
                inference.main(args)
        if load_count != 2 or main_model is None or timing_model is None:
            raise RuntimeError("stage-aware control did not load exactly two models")
        if initialization is None:
            raise RuntimeError("stage-aware control did not initialize main weights")
        payload = dict(initialization)
        payload["decode_block_sizes"] = {
            "timing_context": 1,
            "main_generation": 4,
        }
        payload["graph_remainders"] = graph_remainders
        output_init_json.parent.mkdir(parents=True, exist_ok=True)
        output_init_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        inference.load_model_with_engine = original_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--output-init-json", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    run(args.config_name, args.overrides, args.output_init_json)


if __name__ == "__main__":
    main()
