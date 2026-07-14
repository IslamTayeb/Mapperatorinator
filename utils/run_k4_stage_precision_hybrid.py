"""Run K4 with FP16 timing and mixed-weight FP32 main generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.inference.engine_binding import InferenceEngineBinding  # noqa: E402
from osuT5.osuT5.inference.optimized.single.k8_runtime import (  # noqa: E402
    install_k8_candidate,
)
from osuT5.osuT5.inference.optimized.single.stage_precision_hybrid import (  # noqa: E402
    StagePrecisionHybridCoordinator,
)
from utils.fixed_seed_inference import fixed_seed_processor_generation  # noqa: E402
from utils.run_approximate_weight_only import (  # noqa: E402
    _initialize_with_evidence,
    _load_args,
)


def _write_initialization(
    path: Path,
    *,
    initialization: dict,
    coordinator: StagePrecisionHybridCoordinator,
) -> None:
    payload = dict(initialization)
    payload["stage_precision_hybrid"] = coordinator.metadata()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(config_name: str, overrides: list[str], output_init_json: Path) -> None:
    import inference

    args = _load_args(config_name, overrides)
    if args.inference_engine != "optimized":
        raise ValueError("stage-precision hybrid requires inference_engine=optimized")
    if args.precision != "fp32":
        raise ValueError("stage-precision hybrid public request must remain fp32")
    if args.use_server:
        raise ValueError("stage-precision hybrid requires use_server=false")
    if not args.profile_inference:
        raise ValueError("stage-precision hybrid requires profile_inference=true")
    if args.super_timing or args.generate_positions:
        raise ValueError(
            "stage-precision hybrid supports Processor timing/main only"
        )
    if not inference.should_load_separate_timing_model(args):
        raise RuntimeError(
            "stage-precision hybrid requires a distinct base timing model"
        )

    original_loader = inference.load_model_with_engine
    coordinator = StagePrecisionHybridCoordinator()
    load_count = 0
    initialization: dict | None = None

    def hybrid_loader(*loader_args, **loader_kwargs):
        nonlocal load_count, initialization
        if load_count > 1:
            raise RuntimeError("stage-precision hybrid observed an unexpected model load")
        requested_precision = loader_kwargs.get("precision")
        if requested_precision != "fp32":
            raise RuntimeError(
                "stage-precision hybrid expected every public loader request as fp32"
            )
        auto_select = loader_kwargs.get("auto_select_gamemode_model")
        if load_count == 0:
            if auto_select is not True:
                raise RuntimeError(
                    "stage-precision hybrid first load must be the gamemode main model"
                )
            binding, tokenizer = original_loader(*loader_args, **loader_kwargs)
            if not isinstance(binding, InferenceEngineBinding):
                raise TypeError("main loader did not return an optimized binding")
            initializer = getattr(
                binding.runtime,
                "initialize_approximate_weight_only",
                None,
            )
            if not callable(initializer):
                raise RuntimeError("main runtime lacks mixed-weight initialization")
            initialization = _initialize_with_evidence(
                initializer,
                binding.raw_model,
            )
            result = coordinator.bind_main(binding), tokenizer
        else:
            if auto_select is not False:
                raise RuntimeError(
                    "stage-precision hybrid second load must be the base timing model"
                )
            timing_kwargs = dict(loader_kwargs)
            timing_kwargs["precision"] = "fp16"
            binding, tokenizer = original_loader(*loader_args, **timing_kwargs)
            if not isinstance(binding, InferenceEngineBinding):
                raise TypeError("timing loader did not return an optimized binding")
            result = coordinator.bind_timing(binding), tokenizer
        load_count += 1
        return result

    inference.load_model_with_engine = hybrid_loader
    try:
        with install_k8_candidate(block_size=4):
            with fixed_seed_processor_generation(inference, base_seed=args.seed):
                inference.main(args)
        coordinator.assert_complete()
        if load_count != 2:
            raise RuntimeError(
                f"stage-precision hybrid expected two model loads, got {load_count}"
            )
        if initialization is None:
            raise RuntimeError("main mixed-weight initialization did not run")
        _write_initialization(
            output_init_json,
            initialization=initialization,
            coordinator=coordinator,
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
