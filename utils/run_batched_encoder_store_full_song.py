"""Run fixed-seed baseline or opt-in batched encoder-store inference."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.fixed_seed_inference import fixed_seed_processor_generation  # noqa: E402
from utils.run_approximate_weight_only import _initialize_with_evidence  # noqa: E402


SELECTED_STACK_VERSION = "selected-k4-k1-int8-fp16-cross-7569d50-v1"


def _load_profile(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("selected encoder-store profile must be an object")
    return payload


def _validate_selected_profile(profile: dict, shared_rope: dict) -> dict:
    from utils.validate_int8_mlp_full_song_profile import (
        validate_profile as validate_int8,
    )
    from utils.validate_k1_remainder_profile import validate as validate_k1
    from utils.validate_k4_profile_contract import validate as validate_k4
    from utils.validate_weight_only_full_song_profile import validate_profile

    if shared_rope.get("computes") != shared_rope.get("expected_computes"):
        raise RuntimeError("selected encoder-store shared-RoPE compute count changed")
    if shared_rope.get("reuses") != shared_rope.get("expected_reuses"):
        raise RuntimeError("selected encoder-store shared-RoPE reuse count changed")
    if int(shared_rope.get("reuses", 0)) <= 0:
        raise RuntimeError("selected encoder-store did not reuse decoder RoPE")
    return {
        "version": SELECTED_STACK_VERSION,
        "weight_only": validate_profile(
            profile,
            role="candidate",
            cross_mode="fp16_packed_projections",
        ),
        "int8_mlp": validate_int8(profile, role="candidate"),
        "k1_remainder": validate_k1(profile, role="candidate"),
        "k4": validate_k4(profile, role="candidate", block_size=4),
        "shared_rope": shared_rope,
    }


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


def _validate_args(args, *, mode: str, batch_size: int) -> None:
    expected = {
        "inference_engine": "optimized",
        "precision": "fp32",
        "attn_implementation": "sdpa",
        "device": "cuda",
        "use_server": False,
        "parallel": False,
        "cfg_scale": 1.0,
        "num_beams": 1,
        "super_timing": False,
        "generate_positions": False,
        "profile_inference": True,
    }
    failures = {
        key: {"expected": value, "actual": getattr(args, key)}
        for key, value in expected.items()
        if getattr(args, key) != value
    }
    if failures:
        raise ValueError(f"encoder-store reciprocal args changed: {failures}")
    if mode not in {"baseline", "candidate"}:
        raise ValueError("mode must be baseline or candidate")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0 or batch_size > int(args.max_batch_size):
        raise ValueError(
            f"batch_size must be within 1..max_batch_size={args.max_batch_size}"
        )


def run(
    config_name: str,
    overrides: list[str],
    *,
    mode: str,
    batch_size: int,
    evidence_path: Path,
) -> dict:
    import inference

    args = _load_args(config_name, overrides)
    _validate_args(args, mode=mode, batch_size=batch_size)
    started = time.perf_counter()
    manager = None
    from osuT5.osuT5.inference.optimized.kernels.weight_only_runtime import (
        CROSS_FP16_PACKED,
    )
    from osuT5.osuT5.inference.optimized.scout.shared_rope import (
        SharedRopeStats,
        shared_decoder_rope_context,
    )
    from osuT5.osuT5.inference.optimized.single.k8_runtime import (
        install_k8_candidate,
    )

    original_loader = inference.load_model_with_engine
    stack = ExitStack()
    shared_stats = SharedRopeStats()
    selected_initialization = None
    loaded_bindings = 0

    def selected_loader(*loader_args, **loader_kwargs):
        nonlocal selected_initialization, loaded_bindings
        binding, tokenizer = original_loader(*loader_args, **loader_kwargs)
        loaded_bindings += 1
        if loaded_bindings == 1:
            stack.enter_context(
                shared_decoder_rope_context(binding.raw_model, stats=shared_stats)
            )
            selected_initialization = _initialize_with_evidence(
                lambda model: (
                    binding.runtime.initialize_approximate_int8_mlp_weight_only_cross(
                        model,
                        mode=CROSS_FP16_PACKED,
                    )
                ),
                binding.raw_model,
            )
        return binding, tokenizer

    inference.load_model_with_engine = selected_loader
    if mode == "candidate":
        from osuT5.osuT5.inference.optimized.scout.encoder_store import (
            install_batched_encoder_store_candidate,
        )

        store_context = install_batched_encoder_store_candidate(
            inference.Processor,
            batch_size=batch_size,
        )
    else:
        from contextlib import nullcontext

        store_context = nullcontext(None)

    result_path = None
    finalized_store = None
    try:
        stack.enter_context(
            install_k8_candidate(block_size=4, graph_remainders=True)
        )
        with store_context as manager:
            with fixed_seed_processor_generation(inference, base_seed=args.seed):
                _, result_path = inference.main(args)
        finalized_store = manager.finalize() if manager is not None else None
    finally:
        inference.load_model_with_engine = original_loader
        stack.close()
    run_wall_seconds = time.perf_counter() - started
    if not math.isfinite(run_wall_seconds) or run_wall_seconds <= 0:
        raise RuntimeError("encoder-store runner produced an invalid wall time")
    if loaded_bindings < 2 or selected_initialization is None or result_path is None:
        raise RuntimeError("encoder-store runner did not install the selected stack")
    profile_path = Path(f"{result_path}.profile.json")
    if not profile_path.is_file() or profile_path.stat().st_size <= 0:
        raise RuntimeError("encoder-store runner profile is missing")
    selected_topology = _validate_selected_profile(
        _load_profile(profile_path),
        shared_stats.as_dict(),
    )
    payload = {
        "schema_version": 2,
        "mode": mode,
        "config_name": config_name,
        "seed": int(args.seed),
        "batch_size": batch_size if mode == "candidate" else None,
        "run_wall_seconds": run_wall_seconds,
        "encoder_store": finalized_store,
        "selected_initialization": selected_initialization,
        "selected_topology": selected_topology,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--evidence-path", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    parsed = parser.parse_args()
    payload = run(
        parsed.config_name,
        parsed.overrides,
        mode=parsed.mode,
        batch_size=parsed.batch_size,
        evidence_path=parsed.evidence_path,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
