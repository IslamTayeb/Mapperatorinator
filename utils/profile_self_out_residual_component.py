"""Component projection scout for self-attn out_proj+residual fusion.

Loads real V32 decoder self-attn Wo weights (optimized engine once), times the
accepted tip path against ``native_one_token_linear_residual`` on contiguous
``[1, H, 1, D]`` attn out, and projects full-song main-model saving.

Evidence mode is ``component_projection`` — not e2e TPS. No INT8 paths.
No production wiring.

Gate (documented): ``sizing_pass`` if projected main-model save ≥ 5% of the
graduated tip main_model_s (FP16 ~1.07 s / FP32 ~1.32 s), not the historical
cross-MLP ``SAVING_TARGET_SECONDS=1.412``. Correctness: max abs drift ≤ 1e-3.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Graduated tip 55949274 authoritative reciprocals (SALVALAI).
TIP_COMMIT = "55949274"
TIP_MAIN = {
    "fp16": {"main_model_s": 21.330, "tokens": 7809},
    "fp32": {"main_model_s": 26.494, "tokens": 8294},
}
SIZING_FRACTION_OF_TIP = 0.05
MAX_ABS_DRIFT = 1e-3
CANDIDATES = ("native_fused",)
EXPECTED_HEADS = 12
EXPECTED_HEAD_DIM = 64
EXPECTED_HIDDEN = EXPECTED_HEADS * EXPECTED_HEAD_DIM


def _git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def saving_target_seconds(precision: str) -> float:
    tip = TIP_MAIN[precision]
    return float(tip["main_model_s"]) * SIZING_FRACTION_OF_TIP


def _max_abs(left: Any, right: Any) -> float:
    return float((left - right).abs().max().item())


def _time_callable(
    callable_: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
) -> tuple[float, Any, bool]:
    import torch

    callable_()
    torch.cuda.synchronize()
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    allocated_before = int(torch.cuda.memory_allocated())
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    last = None
    for _ in range(iters):
        last = callable_()
    end.record()
    torch.cuda.synchronize()
    if last is None:
        raise RuntimeError("timing loop produced no output")
    last = last.detach()
    allocated_after = int(torch.cuda.memory_allocated())
    ms_per_call = float(start.elapsed_time(end)) / float(iters)
    memory_stable = allocated_after <= allocated_before + last.numel() * last.element_size()
    return ms_per_call, last, memory_stable


def summarize_variant(
    *,
    name: str,
    accepted_ms: float,
    candidate_ms: float,
    drift: float,
    memory_stable: bool,
    finite: bool,
    precision: str,
    decoder_layers: int,
) -> dict[str, Any]:
    if precision not in TIP_MAIN:
        raise ValueError(f"unsupported precision {precision!r}")
    if min(accepted_ms, candidate_ms) <= 0 or not all(
        math.isfinite(value) for value in (accepted_ms, candidate_ms, drift)
    ):
        raise ValueError(f"{name} has invalid timing or drift")
    tip = TIP_MAIN[precision]
    tokens = int(tip["tokens"])
    target = saving_target_seconds(precision)
    accepted_seconds = decoder_layers * tokens * accepted_ms / 1000.0
    candidate_seconds = decoder_layers * tokens * candidate_ms / 1000.0
    saving = accepted_seconds - candidate_seconds
    region_fraction = (accepted_ms - candidate_ms) / accepted_ms
    correctness_pass = finite and memory_stable and drift <= MAX_ABS_DRIFT
    sizing_pass = saving >= target
    return {
        "candidate": name,
        "exactness_class": "exact",
        "accepted_ms_per_call": accepted_ms,
        "candidate_ms_per_call": candidate_ms,
        "local_speedup": accepted_ms / candidate_ms,
        "region_relative_saving": region_fraction,
        "weighted_accepted_seconds": accepted_seconds,
        "weighted_candidate_seconds": candidate_seconds,
        "projected_main_saving_seconds": saving,
        "saving_target_seconds": target,
        "saving_target_basis": (
            f"{SIZING_FRACTION_OF_TIP:.0%} of graduated tip {TIP_COMMIT} "
            f"{precision} main_model_s={tip['main_model_s']}"
        ),
        "sizing_pass": sizing_pass,
        "max_abs_drift_vs_accepted": drift,
        "max_abs_drift_gate": MAX_ABS_DRIFT,
        "finite_outputs": finite,
        "memory_stable": memory_stable,
        "correctness_pass": correctness_pass,
        "component_pass": correctness_pass and sizing_pass,
        "promotion_pass": False,
        "evidence_mode": "component_projection",
    }


def summarize_component(
    variants: dict[str, dict[str, Any]],
    *,
    precision: str,
) -> dict[str, Any]:
    if not variants:
        raise ValueError("component summary requires variants")
    ranked = sorted(
        variants.values(),
        key=lambda row: float(row["projected_main_saving_seconds"]),
        reverse=True,
    )
    best = ranked[0]
    any_component_pass = any(bool(row["component_pass"]) for row in variants.values())
    return {
        "best_candidate": best["candidate"],
        "best_projected_main_saving_seconds": best["projected_main_saving_seconds"],
        "best_region_relative_saving": best["region_relative_saving"],
        "saving_target_seconds": saving_target_seconds(precision),
        "saving_target_basis": best["saving_target_basis"],
        "any_component_pass": any_component_pass,
        "promotion_pass": False,
        "evidence_mode": "component_projection",
        "variants": variants,
        "stop_condition": (
            None
            if any_component_pass
            else "STOP_SELF_OUT_RESIDUAL_FUSION_COMPONENT"
        ),
    }


def _load_args(config_name: str, overrides: list[str]):
    import hydra
    from omegaconf import DictConfig, OmegaConf

    __import__("config")
    config_dir = REPO_ROOT / "configs" / "inference"
    resolved_name = config_name.removeprefix("inference/")
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = hydra.compose(config_name=resolved_name, overrides=overrides)
    return OmegaConf.to_object(cfg) if isinstance(cfg, DictConfig) else cfg


def _resolve_component_audio_path() -> Path:
    """Satisfy compile_paths without requiring the Mac-local profile_salvalai path.

    Component profiling only loads Wo weights; audio is never decoded.
    Prefer MAPPERATORINATOR_AUDIO / DCC SALVALAI, else a disposable dummy .mp3.
    """
    candidates: list[Path] = []
    env = os.environ.get("MAPPERATORINATOR_AUDIO")
    if env:
        candidates.append(Path(env))
    candidates.append(Path("/work/imt11/Mapperatorinator/data/salvalai.mp3"))
    for path in candidates:
        if path.is_file():
            return path
    dummy_dir = Path(tempfile.mkdtemp(prefix="self_out_residual_component_"))
    dummy = dummy_dir / "dummy.mp3"
    dummy.write_bytes(b"")
    return dummy


def _extract_self_wo_layers(model) -> list[dict[str, Any]]:
    from osuT5.osuT5.model.custom_transformers.modeling_varwhisper import (
        VarWhisperDecoderLayer,
    )

    layers: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, VarWhisperDecoderLayer):
            continue
        self_attn = module.self_attn
        wo = self_attn.Wo
        layers.append(
            {
                "name": name,
                "layer_idx": int(self_attn.layer_idx),
                "num_heads": int(self_attn.num_heads),
                "head_dim": int(self_attn.head_dim),
                "hidden": int(self_attn.all_head_size),
                "weight": wo.weight.detach(),
                "bias": None if wo.bias is None else wo.bias.detach(),
            }
        )
    if not layers:
        raise RuntimeError("no VarWhisperDecoderLayer self-attn Wo weights found")
    layers.sort(key=lambda row: int(row["layer_idx"]))
    return layers


def _load_real_wo_layers(*, precision: str, model_path: str | None) -> list[dict[str, Any]]:
    import torch

    from inference import (
        compile_args,
        load_model_with_engine,
        setup_inference_environment,
    )

    # profile_salvalai.yaml pins a Mac-local audio_path; component loading only
    # needs model weights, but compile_args still validates audio existence.
    audio_path = _resolve_component_audio_path()
    overrides = [
        "device=cuda",
        f"precision={precision}",
        "attn_implementation=sdpa",
        "inference_engine=optimized",
        "use_server=false",
        "parallel=false",
        "cfg_scale=1.0",
        "num_beams=1",
        "seed=12345",
        "profile_inference=false",
        f"audio_path={audio_path}",
        f"output_path={audio_path.parent}",
    ]
    if model_path:
        overrides.append(f"model_path={model_path}")
    args = _load_args("profile_salvalai", overrides)
    compile_args(args, verbose=False)
    setup_inference_environment(
        args.seed,
        strict_fp32=(
            args.inference_engine == "optimized" and args.precision == "fp32"
        ),
    )
    binding, _tokenizer = load_model_with_engine(
        args.model_path,
        args.train,
        args.device,
        max_batch_size=args.max_batch_size,
        use_server=False,
        precision=args.precision,
        attn_implementation=args.attn_implementation,
        lora_path=args.lora_path,
        gamemode=args.gamemode,
        auto_select_gamemode_model=args.auto_select_gamemode_model,
        inference_engine="optimized",
    )
    model = getattr(binding, "model", binding)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("optimized engine did not expose a torch.nn.Module")
    model.eval()
    return _extract_self_wo_layers(model)


def profile_component(
    *,
    precision: str,
    warmup: int,
    iters: int,
    seed: int,
    model_path: str | None,
) -> dict[str, Any]:
    import torch

    from osuT5.osuT5.inference.optimized.kernels.decoder_layer import (
        preload_native_decoder_layer,
    )
    from osuT5.osuT5.inference.optimized.scout.self_out_residual_fusion import (
        accepted_self_out_residual,
        native_fused_self_out_residual,
        scout_metadata,
    )

    if precision not in TIP_MAIN:
        raise ValueError("precision must be fp16 or fp32")
    if not torch.cuda.is_available():
        raise RuntimeError("self out residual component requires CUDA")
    if warmup < 1 or iters < 1:
        raise ValueError("warmup and iters must be positive")

    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    if capability != (7, 5):
        raise RuntimeError(
            f"component requires SM75, got sm_{capability[0]}{capability[1]}"
        )
    dtype = torch.float16 if precision == "fp16" else torch.float32

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    load_started = time.perf_counter()
    wo_layers = _load_real_wo_layers(precision=precision, model_path=model_path)
    load_seconds = time.perf_counter() - load_started

    shapes = {
        "num_layers": len(wo_layers),
        "num_heads": [int(row["num_heads"]) for row in wo_layers],
        "head_dim": [int(row["head_dim"]) for row in wo_layers],
        "hidden": [int(row["hidden"]) for row in wo_layers],
        "weight_shapes": [list(row["weight"].shape) for row in wo_layers],
        "has_bias": [row["bias"] is not None for row in wo_layers],
    }
    if any(h != EXPECTED_HEADS for h in shapes["num_heads"]):
        raise RuntimeError(f"unexpected head counts: {shapes['num_heads']}")
    if any(d != EXPECTED_HEAD_DIM for d in shapes["head_dim"]):
        raise RuntimeError(f"unexpected head dims: {shapes['head_dim']}")
    if any(h != EXPECTED_HIDDEN for h in shapes["hidden"]):
        raise RuntimeError(f"unexpected hidden sizes: {shapes['hidden']}")

    with torch.no_grad():
        attn = torch.randn(
            (1, EXPECTED_HEADS, 1, EXPECTED_HEAD_DIM),
            device=device,
            dtype=dtype,
        ).contiguous()
        residual = torch.randn(
            (1, 1, EXPECTED_HIDDEN),
            device=device,
            dtype=dtype,
        ).contiguous()

        # Mean timing across real per-layer Wo weights (same shapes).
        weights = [
            row["weight"].to(device=device, dtype=dtype).contiguous()
            for row in wo_layers
        ]
        biases = [
            None
            if row["bias"] is None
            else row["bias"].to(device=device, dtype=dtype).contiguous()
            for row in wo_layers
        ]

        preload_started = time.perf_counter()
        preload_native_decoder_layer()
        torch.cuda.synchronize()
        preload_seconds = time.perf_counter() - preload_started

        def accepted_all_layers() -> torch.Tensor:
            out = residual
            for weight, bias in zip(weights, biases, strict=True):
                out = accepted_self_out_residual(attn, residual, weight, bias)
            return out

        def candidate_all_layers() -> torch.Tensor:
            out = residual
            for weight, bias in zip(weights, biases, strict=True):
                out = native_fused_self_out_residual(attn, residual, weight, bias)
            return out

        # Time per-layer mean by dividing the all-layers callable by layer count.
        accepted_ms_all, accepted_out, accepted_stable = _time_callable(
            accepted_all_layers,
            warmup=warmup,
            iters=iters,
        )
        if not accepted_stable:
            raise RuntimeError("accepted self_out timing allocated memory")
        accepted_ms = accepted_ms_all / float(len(wo_layers))

        candidate_ms_all, candidate_out, memory_stable = _time_callable(
            candidate_all_layers,
            warmup=warmup,
            iters=iters,
        )
        candidate_ms = candidate_ms_all / float(len(wo_layers))
        drift = _max_abs(accepted_out, candidate_out)
        finite = bool(torch.isfinite(candidate_out).all().item())

        variants = {
            "native_fused": summarize_variant(
                name="native_fused",
                accepted_ms=accepted_ms,
                candidate_ms=candidate_ms,
                drift=drift,
                memory_stable=memory_stable,
                finite=finite,
                precision=precision,
                decoder_layers=len(wo_layers),
            )
        }

    summary = summarize_component(variants, precision=precision)
    tip = TIP_MAIN[precision]
    return {
        "schema_version": 1,
        "metadata": {
            "commit": _git_head(),
            "base_tip": f"codex/exact-shared-rope-device-state@{TIP_COMMIT}",
            "candidate_scope": "component_only_no_production_wiring",
            "evidence_mode": "component_projection",
            "precision": precision,
            "hypothesis": (
                "Self-attn out_proj+residual: skip accepted "
                "transpose+contiguous+view materialization by feeding "
                "contiguous [1,H,1,D] into native_one_token_linear_residual; "
                f"≥{SIZING_FRACTION_OF_TIP:.0%} of graduated tip {precision} "
                f"main_model (~{saving_target_seconds(precision):.2f}s)"
            ),
            "exactness_claim": True,
            "note": (
                "Real Wo weights + synthetic activations; labeled "
                "component_projection — not production / e2e TPS."
            ),
            "scout": scout_metadata(),
            "warmup": warmup,
            "iters": iters,
            "seed": seed,
            "model_load_seconds": load_seconds,
            "extension_preload_seconds": preload_seconds,
            "fixed_main_generated_tokens": tip["tokens"],
            "tip_main_model_s": tip["main_model_s"],
            "decoder_layers": len(wo_layers),
            "verified_shapes": shapes,
            "expected_shapes": {
                "num_heads": EXPECTED_HEADS,
                "head_dim": EXPECTED_HEAD_DIM,
                "hidden": EXPECTED_HIDDEN,
            },
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_capability": f"sm_{capability[0]}{capability[1]}",
            "int8": False,
        },
        "summary": summary,
    }


def _write_text(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    meta = result["metadata"]
    lines = [
        "Self-attn out_proj+residual fusion component scout",
        f"commit={meta['commit']}",
        f"base_tip={meta['base_tip']}",
        f"precision={meta['precision']}",
        f"evidence_mode={meta['evidence_mode']}",
        f"best_candidate={summary['best_candidate']}",
        f"best_projected_main_saving_seconds="
        f"{summary['best_projected_main_saving_seconds']:.9f}",
        f"saving_target_seconds={summary['saving_target_seconds']:.9f}",
        f"saving_target_basis={summary['saving_target_basis']}",
        f"any_component_pass={str(summary['any_component_pass']).lower()}",
        f"promotion_pass={str(summary['promotion_pass']).lower()}",
        f"stop_condition={summary['stop_condition']}",
    ]
    for name, row in summary["variants"].items():
        lines.append(
            f"variant={name} accepted_ms={row['accepted_ms_per_call']:.6f} "
            f"candidate_ms={row['candidate_ms_per_call']:.6f} "
            f"saving={row['projected_main_saving_seconds']:.6f} "
            f"region={row['region_relative_saving']:.4f} "
            f"drift={row['max_abs_drift_vs_accepted']:.6g} "
            f"component_pass={str(row['component_pass']).lower()}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args(argv)

    result = profile_component(
        precision=args.precision,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
        model_path=args.model_path,
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_text(args.report_path.with_suffix(".txt"), result)

    summary = result["summary"]
    print(json.dumps({"summary": summary}, indent=2))
    if not summary["any_component_pass"]:
        print("STOP_SELF_OUT_RESIDUAL_FUSION_COMPONENT", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
