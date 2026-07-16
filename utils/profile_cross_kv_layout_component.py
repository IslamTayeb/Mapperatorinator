"""Cheap component gate for TRT-LLM ContiguousKv / XQA-shape cross-KV scout.

Synthetic fixed-shape tensors matching selected+compiled-cross warmup
(``[1,12,1,64]`` Q, ``[1,12,1024,64]`` K/V).  No full-song capture.

Hypothesis: fused/static cross-KV layout + XQA-style single-pass online decode
saves ≥0.2s main-model **or** ≥5% of the measured accepted cross region on top
of selected+compiled-cross.  Not production TPS.

Exactness:
- ``pretransposed_k_bmm``: exact vs ``_q1_bmm_cross_attention``
- fused online variants: documented drift ≤ ``MAX_ABS_DRIFT``
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DECODER_LAYERS = 12
HEADS = 12
KV_LENGTH = 1024
HEAD_DIM = 64
# Campaign residual after compiled-cross graduation (~0.2s / ~6 TPS @ 494.27).
SAVING_TARGET_SECONDS = 0.2
REGION_RELATIVE_GATE = 0.05  # ≥5% of measured accepted cross region
MAX_ABS_DRIFT = 1e-3
# Contiguous pre-transposed K can differ from a non-contiguous transpose view by
# a few ULPs under FP32 BMM; still treated as exact-vs-accepted math intent.
EXACT_MAX_ABS_DRIFT = 1e-6
# Caching allocator / cuBLAS workspace can jitter a few pages across iters.
MEMORY_STABLE_SLACK_BYTES = 1 << 20  # 1 MiB
FIXED_MAIN_GENERATED_TOKENS = 8_532  # selected-stack SALVALAI main tokens
BASE_TIP = "codex/500tps-arena-compiled-cross-last-mile@0dbab9e5"
CANDIDATES = (
    "pretransposed_k_bmm",
    "fused_online_fp32",
    "fused_online_fp16_kv",
)


def _git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _max_abs(left: Any, right: Any) -> float:
    return float((left - right).abs().max().item())


def _time_callable(
    callable_: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
) -> tuple[float, Any, bool]:
    import torch

    # Warm first so cuBLAS/allocator workspace settles before the stability check.
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
    allocated_after = int(torch.cuda.memory_allocated())
    ms_per_call = float(start.elapsed_time(end)) / float(iters)
    memory_stable = allocated_after <= allocated_before + MEMORY_STABLE_SLACK_BYTES
    return ms_per_call, last.detach(), memory_stable


def summarize_variant(
    *,
    name: str,
    accepted_ms: float,
    candidate_ms: float,
    drift: float,
    exactness_class: str,
    memory_stable: bool,
    finite: bool,
) -> dict[str, Any]:
    if min(accepted_ms, candidate_ms) <= 0 or not all(
        math.isfinite(value) for value in (accepted_ms, candidate_ms, drift)
    ):
        raise ValueError(f"{name} has invalid timing or drift")
    accepted_seconds = (
        DECODER_LAYERS * FIXED_MAIN_GENERATED_TOKENS * accepted_ms / 1000.0
    )
    candidate_seconds = (
        DECODER_LAYERS * FIXED_MAIN_GENERATED_TOKENS * candidate_ms / 1000.0
    )
    saving = accepted_seconds - candidate_seconds
    region_fraction = (accepted_ms - candidate_ms) / accepted_ms
    drift_gate = EXACT_MAX_ABS_DRIFT if exactness_class == "exact" else MAX_ABS_DRIFT
    correctness_pass = finite and memory_stable and drift <= drift_gate
    region_pass = region_fraction >= REGION_RELATIVE_GATE
    e2e_pass = saving >= SAVING_TARGET_SECONDS
    sizing_pass = region_pass or e2e_pass
    return {
        "candidate": name,
        "exactness_class": exactness_class,
        "accepted_ms_per_call": accepted_ms,
        "candidate_ms_per_call": candidate_ms,
        "local_speedup": accepted_ms / candidate_ms,
        "region_relative_saving": region_fraction,
        "region_relative_gate": REGION_RELATIVE_GATE,
        "region_pass": region_pass,
        "weighted_accepted_seconds_12_layers": accepted_seconds,
        "weighted_candidate_seconds_12_layers": candidate_seconds,
        "projected_main_saving_seconds": saving,
        "saving_target_seconds": SAVING_TARGET_SECONDS,
        "e2e_pass": e2e_pass,
        "sizing_pass": sizing_pass,
        "max_abs_drift_vs_accepted_bmm": drift,
        "max_abs_drift_gate": drift_gate,
        "finite_outputs": finite,
        "memory_stable": memory_stable,
        "correctness_pass": correctness_pass,
        "component_pass": correctness_pass and sizing_pass,
        "promotion_pass": False,
    }


def summarize_component(variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
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
        "saving_target_seconds": SAVING_TARGET_SECONDS,
        "region_relative_gate": REGION_RELATIVE_GATE,
        "any_component_pass": any_component_pass,
        "promotion_pass": False,
        "variants": variants,
        "stop_condition": (
            None
            if any_component_pass
            else "STOP_CROSS_KV_LAYOUT_COMPONENT"
        ),
    }


def profile_component(*, warmup: int, iters: int, seed: int) -> dict[str, Any]:
    import torch

    from osuT5.osuT5.inference.optimized.kernels.dispatch import (
        _q1_bmm_cross_attention,
    )
    from osuT5.osuT5.inference.optimized.scout.cross_kv_layout import (
        fused_online_cross_attention,
        pack_static_cross_kv,
        preload_cross_kv_layout_scout,
        pretransposed_k_bmm_cross_attention,
        scout_metadata,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("cross-KV layout component requires CUDA")
    if warmup < 1 or iters < 1:
        raise ValueError("warmup and iters must be positive")
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    if capability != (7, 5):
        raise RuntimeError(
            f"component requires SM75, got sm_{capability[0]}{capability[1]}"
        )

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        query = torch.randn(
            (1, HEADS, 1, HEAD_DIM), device=device, dtype=torch.float32
        ).contiguous()
        keys = torch.randn(
            (1, HEADS, KV_LENGTH, HEAD_DIM), device=device, dtype=torch.float32
        ).contiguous()
        values = torch.randn(
            (1, HEADS, KV_LENGTH, HEAD_DIM), device=device, dtype=torch.float32
        ).contiguous()

        pack_started = time.perf_counter()
        pack = pack_static_cross_kv(keys, values)
        torch.cuda.synchronize()
        pack_seconds = time.perf_counter() - pack_started

        preload_started = time.perf_counter()
        preload_cross_kv_layout_scout()
        torch.cuda.synchronize()
        preload_seconds = time.perf_counter() - preload_started

        accepted_ms, accepted_out, accepted_stable = _time_callable(
            lambda: _q1_bmm_cross_attention(
                query, keys, values, expected_dtype=torch.float32
            ),
            warmup=warmup,
            iters=iters,
        )
        # Accepted BMM is the reference timer; do not abort the scout on allocator
        # jitter (still recorded). Candidate memory_stable remains a correctness bit.
        if not accepted_stable:
            print(
                "warning: accepted BMM timing exceeded memory slack "
                f"({MEMORY_STABLE_SLACK_BYTES} bytes); continuing",
                file=sys.stderr,
            )

        callables: dict[str, tuple[str, Callable[[], Any]]] = {
            "pretransposed_k_bmm": (
                "exact",
                lambda: pretransposed_k_bmm_cross_attention(query, pack),
            ),
            "fused_online_fp32": (
                "relaxed-documented-drift",
                lambda: fused_online_cross_attention(query, pack.fused_kv),
            ),
            "fused_online_fp16_kv": (
                "relaxed-documented-drift",
                lambda: fused_online_cross_attention(query, pack.fused_kv_fp16),
            ),
        }
        variants: dict[str, dict[str, Any]] = {}
        for name in CANDIDATES:
            exactness_class, callable_ = callables[name]
            candidate_ms, candidate_out, memory_stable = _time_callable(
                callable_,
                warmup=warmup,
                iters=iters,
            )
            drift = _max_abs(accepted_out, candidate_out)
            finite = bool(torch.isfinite(candidate_out).all().item())
            variants[name] = summarize_variant(
                name=name,
                accepted_ms=accepted_ms,
                candidate_ms=candidate_ms,
                drift=drift,
                exactness_class=exactness_class,
                memory_stable=memory_stable,
                finite=finite,
            )

    summary = summarize_component(variants)
    return {
        "schema_version": 1,
        "metadata": {
            "commit": _git_head(),
            "base_tip": BASE_TIP,
            "candidate_scope": "component_only_no_production_wiring",
            "hypothesis": (
                "TRT-LLM ContiguousKv / XQA-shape single-request cross decode "
                "on static fused encoder KV saves ≥0.2s main or ≥5% of the "
                "accepted cross-BMM region vs selected+compiled-cross math"
            ),
            "exactness_claim": False,
            "note": (
                "Synthetic fixed-shape component; not production throughput. "
                "pretransposed_k_bmm is exact-vs-BMM; fused online is relaxed."
            ),
            "scout": scout_metadata(),
            "warmup": warmup,
            "iters": iters,
            "seed": seed,
            "pack_setup_seconds": pack_seconds,
            "extension_preload_seconds": preload_seconds,
            "fixed_main_generated_tokens": FIXED_MAIN_GENERATED_TOKENS,
            "decoder_layers": DECODER_LAYERS,
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_capability": f"sm_{capability[0]}{capability[1]}",
        },
        "memory": {
            "source_kv_bytes": keys.numel() * keys.element_size()
            + values.numel() * values.element_size(),
            "key_pretransposed_bytes": pack.key_pretransposed.numel()
            * pack.key_pretransposed.element_size(),
            "fused_kv_fp32_bytes": pack.fused_kv.numel() * pack.fused_kv.element_size(),
            "fused_kv_fp16_bytes": pack.fused_kv_fp16.numel()
            * pack.fused_kv_fp16.element_size(),
        },
        "summary": summary,
    }


def _write_text(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    lines = [
        "TRT-LLM ContiguousKv / XQA-shape cross-KV layout component scout",
        f"commit={result['metadata']['commit']}",
        f"base_tip={result['metadata']['base_tip']}",
        f"best_candidate={summary['best_candidate']}",
        f"best_projected_main_saving_seconds="
        f"{summary['best_projected_main_saving_seconds']:.9f}",
        f"best_region_relative_saving={summary['best_region_relative_saving']:.6f}",
        f"any_component_pass={str(summary['any_component_pass']).lower()}",
        f"promotion_pass={str(summary['promotion_pass']).lower()}",
        f"stop_condition={summary['stop_condition']}",
    ]
    for name, row in summary["variants"].items():
        lines.append(
            f"variant={name} candidate_ms={row['candidate_ms_per_call']:.6f} "
            f"saving={row['projected_main_saving_seconds']:.6f} "
            f"region={row['region_relative_saving']:.4f} "
            f"drift={row['max_abs_drift_vs_accepted_bmm']:.6g} "
            f"component_pass={str(row['component_pass']).lower()}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args(argv)

    result = profile_component(
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_text(args.report_path.with_suffix(".txt"), result)

    summary = result["summary"]
    print(json.dumps({"summary": summary}, indent=2))
    if not summary["any_component_pass"]:
        print("STOP_CROSS_KV_LAYOUT_COMPONENT", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
