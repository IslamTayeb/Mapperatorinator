"""Run the dtype-generic shared-RoPE candidate without production wiring."""

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


CANDIDATE_VERSION = "strict-shared-rope-dtype-generic-v2"
SUPPORTED_PRECISIONS = frozenset({"fp32", "fp16"})


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
    precision = getattr(args, "precision", None)
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(
            "strict shared-RoPE precision must be one of: fp16, fp32"
        )
    requirements = {
        "inference_engine": "optimized",
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
        raise ValueError(f"strict shared-RoPE requirements changed: {failures}")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError(
            "strict shared-RoPE requires NVIDIA_TF32_OVERRIDE=0 before startup"
        )


def _strict_environment_evidence(precision: str) -> dict[str, Any]:
    import torch

    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError("environment evidence precision must be fp16 or fp32")
    evidence = {
        "precision": precision,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
    }
    required = {
        "precision": precision,
        "nvidia_tf32_override": "0",
    }
    if precision == "fp32":
        required.update(
            {
                "float32_matmul_precision": "highest",
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
            }
        )
    failures = {
        key: {"expected": value, "actual": evidence.get(key)}
        for key, value in required.items()
        if evidence.get(key) != value
    }
    if failures:
        raise RuntimeError(
            f"strict {precision} environment is not active: {failures}"
        )
    return evidence


def _validated_shared_rope_evidence(
    stats: SharedRopeStats,
    *,
    precision: str,
    preset: dict[str, Any],
) -> dict[str, Any]:
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError("shared-RoPE evidence precision must be fp16 or fp32")
    expected_version = {
        "fp32": "accepted-fp32-native-cross-mlp-289-v3",
        "fp16": "accepted-fp16-all-fused-v2",
    }[precision]
    required_dispatch = {
        "q1_bmm_cross_attention": True,
        "native_q1_self_attention": True,
        "native_q1_rope_cache_self_attention": True,
    }
    preset_failures = {
        "version": preset.get("version") != expected_version,
        "precision": preset.get("precision") != precision,
        **{
            name: preset.get(name) is not expected
            for name, expected in required_dispatch.items()
        },
    }
    failed_preset_fields = sorted(
        name for name, failed in preset_failures.items() if failed
    )
    if failed_preset_fields:
        raise RuntimeError(
            "shared RoPE did not retain the accepted specialized topology: "
            f"{failed_preset_fields}"
        )
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
        "precision": precision,
        "dtype_generic": True,
        "original_decoder_forward_required": True,
        "production_wiring": False,
        "component_exactness_precondition": True,
        "full_song_exactness_claim": False,
        "accepted_preset": preset,
        "retained_specialized_dispatch": required_dispatch,
        "stats": evidence,
    }


def run(
    config_name: str,
    overrides: list[str],
    evidence_path: Path,
) -> None:
    """Install shared RoPE only around the first optimized main model."""

    import inference

    args = _load_args(config_name, overrides)
    _validate_args(args)
    original_loader = inference.load_model_with_engine
    stack = ExitStack()
    stats = SharedRopeStats()
    loaded_bindings = 0
    strict_environment: dict[str, Any] | None = None
    accepted_preset: dict[str, Any] | None = None

    def candidate_loader(*loader_args, **loader_kwargs):
        nonlocal loaded_bindings, strict_environment, accepted_preset
        binding, tokenizer = original_loader(*loader_args, **loader_kwargs)
        loaded_bindings += 1
        if loaded_bindings == 1:
            runtime = getattr(binding, "runtime", None)
            preset = getattr(runtime, "preset", None)
            profile_metadata = getattr(runtime, "profile_metadata", None)
            if preset is None or not callable(profile_metadata):
                raise RuntimeError(
                    "shared RoPE requires the accepted optimized single runtime"
                )
            if getattr(preset, "precision", None) != args.precision:
                raise RuntimeError("shared RoPE loader precision changed")
            import torch

            expected_dtype = {
                "fp32": torch.float32,
                "fp16": torch.float16,
            }[args.precision]
            if getattr(binding.raw_model, "dtype", None) != expected_dtype:
                raise TypeError(
                    "shared RoPE model storage dtype does not match precision"
                )
            accepted_preset = profile_metadata()["optimized_effective_config"]
            strict_environment = _strict_environment_evidence(args.precision)
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
    if (
        loaded_bindings < 1
        or strict_environment is None
        or accepted_preset is None
    ):
        raise RuntimeError("strict shared-RoPE main model was never loaded")

    payload = {
        "schema_version": 1,
        "candidate": _validated_shared_rope_evidence(
            stats,
            precision=args.precision,
            preset=accepted_preset,
        ),
        "strict_environment": strict_environment,
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
