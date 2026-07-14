"""Run two selected-stack songs in one process with persistent graph workspaces."""

from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.inference.engine_binding import InferenceEngineBinding  # noqa: E402
from osuT5.osuT5.inference.optimized.kernels.weight_only_runtime import (  # noqa: E402
    CROSS_FP16_PACKED,
)
from osuT5.osuT5.inference.optimized.scout.shared_rope import (  # noqa: E402
    SharedRopeStats,
    shared_decoder_rope_context,
)
from osuT5.osuT5.inference.optimized.single.engine import (  # noqa: E402
    OptimizedSingleRuntime,
)
from osuT5.osuT5.inference.optimized.single.k8_runtime import (  # noqa: E402
    install_k8_candidate,
)
from utils.fixed_seed_inference import fixed_seed_processor_generation  # noqa: E402
from utils.run_approximate_weight_only import (  # noqa: E402
    _initialize_with_evidence,
    _load_args,
)


TOPOLOGY_VERSION = "selected-k4-k1-int8-fp16-cross-persistent-graphs-v1"


def _profile_path(result_path: Path) -> Path:
    path = Path(f"{result_path}.profile.json")
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"persistent graph profile is missing: {path}")
    return path


def _validate_args(args) -> None:
    if args.inference_engine != "optimized" or args.precision != "fp32":
        raise ValueError("persistent graph scout requires optimized FP32 inference")
    if args.use_server or not args.profile_inference:
        raise ValueError("persistent graph scout requires local profiled inference")
    if args.super_timing or args.generate_positions:
        raise ValueError("persistent graph scout supports Processor timing/main only")


def run(
    config_name: str,
    overrides: list[str],
    *,
    output_init_json: Path,
    output_manifest: Path,
) -> None:
    import inference

    args = _load_args(config_name, overrides)
    _validate_args(args)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    original_loader = inference.load_model_with_engine
    stack = ExitStack()
    shared_stats = SharedRopeStats()
    cached: list[tuple[InferenceEngineBinding, Any]] = []
    load_index = 0
    main_init: dict[str, Any] | None = None
    timing_init: dict[str, Any] | None = None
    pool_initial: dict[str, Any] = {}

    def persistent_loader(*loader_args, **loader_kwargs):
        nonlocal load_index, main_init, timing_init
        role = load_index % 2
        pass_index = load_index // 2
        auto_select = loader_kwargs.get("auto_select_gamemode_model")
        expected_auto_select = role == 0
        if auto_select is not expected_auto_select:
            raise RuntimeError("persistent graph model load order changed")
        if pass_index > 1:
            raise RuntimeError("persistent graph scout observed an extra inference pass")
        if pass_index == 1:
            binding, tokenizer = cached[role]
            load_index += 1
            return binding, tokenizer

        binding, tokenizer = original_loader(*loader_args, **loader_kwargs)
        if not isinstance(binding, InferenceEngineBinding):
            raise TypeError("persistent graph scout requires optimized bindings")
        if type(binding.runtime) is not OptimizedSingleRuntime:
            raise TypeError("persistent graph scout requires single runtimes")
        if role == 0:
            stack.enter_context(
                shared_decoder_rope_context(binding.raw_model, stats=shared_stats)
            )
            main_init = _initialize_with_evidence(
                lambda model: (
                    binding.runtime.initialize_approximate_int8_mlp_weight_only_cross(
                        model,
                        mode=CROSS_FP16_PACKED,
                    )
                ),
                binding.raw_model,
            )
            pool_initial["main"] = (
                binding.runtime.initialize_persistent_graph_workspace_scout(
                    binding.raw_model,
                    topology_signature=(
                        TOPOLOGY_VERSION,
                        "main",
                        "block_size=4",
                        "graph_remainders=true",
                        "shared_rope=true",
                    ),
                )
            )
        else:
            if binding.raw_model is cached[0][0].raw_model:
                raise RuntimeError("persistent graph scout requires a distinct timing model")
            timing_init = binding.runtime.initialize_fp32_timing_native_self(
                binding.raw_model
            )
            pool_initial["timing"] = (
                binding.runtime.initialize_persistent_graph_workspace_scout(
                    binding.raw_model,
                    topology_signature=(
                        TOPOLOGY_VERSION,
                        "timing",
                        "block_size=4",
                        "graph_remainders=true",
                        "timing_native_self=true",
                    ),
                )
            )
        cached.append((binding, tokenizer))
        load_index += 1
        return binding, tokenizer

    inference.load_model_with_engine = persistent_loader
    results: dict[str, Any] = {}
    final_pool_summary: dict[str, Any] = {}
    close_completed = False
    try:
        with install_k8_candidate(block_size=4, graph_remainders=True):
            with fixed_seed_processor_generation(inference, base_seed=args.seed):
                for label in ("cold", "warm"):
                    pass_args = copy.deepcopy(args)
                    pass_args.output_path = str(output_manifest.parent / label)
                    Path(pass_args.output_path).mkdir(parents=True, exist_ok=True)
                    started = time.perf_counter()
                    _, result_path = inference.main(pass_args)
                    process_call_wall_seconds = time.perf_counter() - started
                    result_path = Path(result_path)
                    results[label] = {
                        "result_path": str(result_path),
                        "profile_path": str(_profile_path(result_path)),
                        "process_call_wall_seconds": process_call_wall_seconds,
                    }
            for role, (binding, _) in zip(("main", "timing"), cached, strict=True):
                final_pool_summary[role] = (
                    binding.runtime._persistent_graph_workspace_pool.summary()
                )
            for binding, _ in cached:
                binding.runtime.close_persistent_graph_workspace_scout()
            close_completed = True
    finally:
        inference.load_model_with_engine = original_loader
        for binding, _ in cached:
            binding.runtime.close_persistent_graph_workspace_scout()
        stack.close()

    if load_index != 4 or main_init is None or timing_init is None:
        raise RuntimeError("persistent graph scout did not complete two model passes")
    initialization = {
        "schema_version": 1,
        "topology_version": TOPOLOGY_VERSION,
        "main": main_init,
        "timing": timing_init,
        "pool_initial": pool_initial,
        "shared_rope": shared_stats.as_dict(),
    }
    output_init_json.parent.mkdir(parents=True, exist_ok=True)
    output_init_json.write_text(
        json.dumps(initialization, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "topology_version": TOPOLOGY_VERSION,
                "initialization_path": str(output_init_json),
                "results": results,
                "final_pool_summary": final_pool_summary,
                "close_completed": close_completed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--output-init-json", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    parsed = parser.parse_args()
    run(
        parsed.config_name,
        parsed.overrides,
        output_init_json=parsed.output_init_json,
        output_manifest=parsed.output_manifest,
    )


if __name__ == "__main__":
    main()
