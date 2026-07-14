"""Run inference with explicit initialization of the mixed-weight candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def run(config_name: str, overrides: list[str], output_init_json: Path) -> None:
    import inference

    args = _load_args(config_name, overrides)
    if args.inference_engine != "optimized":
        raise ValueError("weight-only candidate requires inference_engine=optimized")
    if args.precision != "fp32":
        raise ValueError("weight-only candidate requires precision=fp32")
    if args.use_server:
        raise ValueError("weight-only candidate requires use_server=false")

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
            init_metadata = initializer(binding.raw_model)
            initialized = True
        return binding, tokenizer

    inference.load_model_with_engine = candidate_loader
    try:
        inference.main(args)
    finally:
        inference.load_model_with_engine = original_loader
    if not initialized or init_metadata is None:
        raise RuntimeError("weight-only candidate model was never initialized")
    output_init_json.parent.mkdir(parents=True, exist_ok=True)
    output_init_json.write_text(
        json.dumps(init_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--output-init-json", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    parsed = parser.parse_args()
    run(parsed.config_name, parsed.overrides, parsed.output_init_json)


if __name__ == "__main__":
    main()
