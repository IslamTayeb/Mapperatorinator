"""Run inference with explicit initialization of the mixed-weight candidate."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.fixed_seed_inference import fixed_seed_processor_generation


def _load_args(config_name: str, overrides: list[str]):
    import config  # noqa: F401
    import hydra
    from omegaconf import DictConfig, OmegaConf

    config_dir = REPO_ROOT / "configs" / "inference"
    resolved_name = config_name.removeprefix("inference/")
    with hydra.initialize_config_dir(
        config_dir=str(config_dir),
        version_base="1.1",
    ):
        cfg = hydra.compose(config_name=resolved_name, overrides=overrides)
    return OmegaConf.to_object(cfg) if isinstance(cfg, DictConfig) else cfg


def _initialize_with_evidence(initializer, model) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("weight-only initialization evidence requires CUDA")
    torch.cuda.synchronize()
    allocated_before = int(torch.cuda.memory_allocated())
    reserved_before = int(torch.cuda.memory_reserved())
    peak_before = int(torch.cuda.max_memory_allocated())
    started = time.perf_counter()
    metadata = initializer(model)
    torch.cuda.synchronize()
    finished = time.perf_counter()
    if not isinstance(metadata, dict):
        raise TypeError("weight-only initializer must return metadata")
    initialization_wall = finished - started
    extension_wall = metadata.get("extension_init_seconds")
    pack_wall = metadata.get("weight_pack_seconds")
    for name, value in (
        ("initialization_wall_seconds", initialization_wall),
        ("extension_init_seconds", extension_wall),
        ("weight_pack_seconds", pack_wall),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise RuntimeError(f"{name} must be finite and non-negative")
    unattributed = initialization_wall - float(extension_wall) - float(pack_wall)
    if unattributed < -1e-3:
        raise RuntimeError("weight-only component walls exceed initialization wall")
    allocated_after = int(torch.cuda.memory_allocated())
    reserved_after = int(torch.cuda.memory_reserved())
    peak_after = int(torch.cuda.max_memory_allocated())
    return {
        **metadata,
        "initialization_wall_seconds": initialization_wall,
        "initialization_unattributed_seconds": max(0.0, unattributed),
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


def run(config_name: str, overrides: list[str], output_init_json: Path) -> None:
    import inference

    args = _load_args(config_name, overrides)
    if args.inference_engine != "optimized":
        raise ValueError("weight-only candidate requires inference_engine=optimized")
    if args.precision != "fp32":
        raise ValueError("weight-only candidate requires precision=fp32")
    if args.use_server:
        raise ValueError("weight-only candidate requires use_server=false")
    if not args.profile_inference:
        raise ValueError("weight-only validation requires profile_inference=true")

    original_loader = inference.load_model_with_engine
    initialized = False
    init_metadata = None

    def candidate_loader(*loader_args, **loader_kwargs):
        nonlocal initialized, init_metadata
        binding, tokenizer = original_loader(*loader_args, **loader_kwargs)
        if not initialized:
            initializer = getattr(
                binding.runtime,
                "initialize_approximate_weight_only",
                None,
            )
            if initializer is None:
                raise RuntimeError(
                    "loaded runtime does not expose weight-only initialization"
                )
            init_metadata = _initialize_with_evidence(
                initializer,
                binding.raw_model,
            )
            output_init_json.parent.mkdir(parents=True, exist_ok=True)
            output_init_json.write_text(
                json.dumps(init_metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            initialized = True
        return binding, tokenizer

    inference.load_model_with_engine = candidate_loader
    try:
        with fixed_seed_processor_generation(inference, base_seed=args.seed):
            inference.main(args)
    finally:
        inference.load_model_with_engine = original_loader
    if not initialized or init_metadata is None:
        raise RuntimeError("weight-only candidate model was never initialized")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--output-init-json", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    parsed = parser.parse_args()
    run(parsed.config_name, parsed.overrides, parsed.output_init_json)


if __name__ == "__main__":
    main()
