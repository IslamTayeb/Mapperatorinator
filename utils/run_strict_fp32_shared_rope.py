"""Run the exact FP32 shared-RoPE candidate without changing production wiring."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.inference.optimized.scout.shared_rope import (  # noqa: E402
    SharedRopeStats,
    shared_decoder_rope_context,
)


CANDIDATE_VERSION = "strict-fp32-main-shared-rope-v1"


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


def _validate_args(args: Any) -> None:
    requirements = {
        "inference_engine": "optimized",
        "precision": "fp32",
        "attn_implementation": "sdpa",
        "device": "cuda",
        "use_server": False,
        "parallel": False,
        "cfg_scale": 1.0,
        "num_beams": 1,
        "profile_inference": True,
        "super_timing": False,
        "generate_positions": False,
    }
    failures = {
        name: (getattr(args, name, None), expected)
        for name, expected in requirements.items()
        if getattr(args, name, None) != expected
    }
    if failures:
        raise ValueError(f"strict FP32 shared-RoPE requirements changed: {failures}")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError(
            "strict FP32 shared-RoPE requires NVIDIA_TF32_OVERRIDE=0 before startup"
        )


def _strict_environment_evidence() -> dict[str, Any]:
    import torch

    evidence = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
    }
    if evidence != {
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "nvidia_tf32_override": "0",
    }:
        raise RuntimeError(f"strict FP32 environment is not active: {evidence}")
    return evidence


def _validated_shared_rope_evidence(stats: SharedRopeStats) -> dict[str, Any]:
    evidence = stats.as_dict()
    if evidence["module_count"] <= 1 or evidence["group_count"] <= 0:
        raise RuntimeError("shared RoPE did not discover a reusable decoder group")
    if evidence["forwards"] <= 0:
        raise RuntimeError("shared RoPE did not observe a main-model forward")
    if evidence["computes"] != evidence["expected_computes"]:
        raise RuntimeError("shared RoPE compute accounting is inconsistent")
    if evidence["reuses"] != evidence["expected_reuses"]:
        raise RuntimeError("shared RoPE reuse accounting is inconsistent")
    if evidence["reuses"] <= 0:
        raise RuntimeError("shared RoPE did not eliminate a redundant computation")
    return {
        "version": CANDIDATE_VERSION,
        "scope": "main-model-only",
        "precision": "fp32",
        "original_decoder_forward_required": True,
        "production_wiring": False,
        "component_exactness_precondition": True,
        "full_song_exactness_claim": False,
        "stats": evidence,
    }


def run(
    config_name: str,
    overrides: list[str],
    evidence_path: Path,
) -> None:
    """Install shared RoPE only around the first (main) optimized FP32 model."""

    import inference

    args = _load_args(config_name, overrides)
    _validate_args(args)
    original_loader = inference.load_model_with_engine
    stack = ExitStack()
    stats = SharedRopeStats()
    loaded_bindings = 0
    strict_environment: dict[str, Any] | None = None

    def candidate_loader(*loader_args, **loader_kwargs):
        nonlocal loaded_bindings, strict_environment
        binding, tokenizer = original_loader(*loader_args, **loader_kwargs)
        loaded_bindings += 1
        if loaded_bindings == 1:
            strict_environment = _strict_environment_evidence()
            stack.enter_context(
                shared_decoder_rope_context(binding.raw_model, stats=stats)
            )
        return binding, tokenizer

    inference.load_model_with_engine = candidate_loader
    try:
        inference.main(args)
    finally:
        inference.load_model_with_engine = original_loader
        stack.close()
    if loaded_bindings < 1 or strict_environment is None:
        raise RuntimeError("strict FP32 shared-RoPE main model was never loaded")

    payload = {
        "schema_version": 1,
        "candidate": _validated_shared_rope_evidence(stats),
        "strict_fp32_environment": strict_environment,
        "loaded_bindings": loaded_bindings,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--evidence-path", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    parsed = parser.parse_args()
    run(parsed.config_name, parsed.overrides, parsed.evidence_path)


if __name__ == "__main__":
    main()
