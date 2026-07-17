#!/usr/bin/env python3
"""Component bitwise + timing probe for §30 elementwise/copy fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--scale", type=float, default=1.0)
    return parser.parse_args(argv)


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_ms(fn, iters: int, warmup: int) -> float:
    import torch

    for _ in range(warmup):
        fn()
    _sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - start) * 1e3 / iters


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv or sys.argv[1:])
    import torch

    from osuT5.osuT5.inference.optimized.kernels.elementwise_fusion import (
        fused_pack_q1_attn_out,
        fused_residual_add,
        fused_rope_epilogue,
        reference_rope_epilogue,
        reset_elementwise_fusion_stats,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("elementwise fusion component requires CUDA")

    device = torch.device("cuda")
    half = args.head_dim // 2
    results: dict[str, object] = {
        "schema_version": 1,
        "component": "elementwise_copy_fusion",
        "iters": args.iters,
        "warmup": args.warmup,
        "geometry": {
            "head_dim": args.head_dim,
            "num_heads": args.num_heads,
            "hidden": args.hidden,
            "half": half,
            "scale": args.scale,
        },
        "checks": {},
        "timing_ms": {},
    }

    # --- RoPE epilogue bitwise ---
    reset_elementwise_fusion_stats()
    freqs = torch.randn(1, 1, half, device=device, dtype=torch.float32)
    for out_dtype, name in (
        (torch.float32, "fp32"),
        (torch.float16, "fp16"),
    ):
        ref_cos, ref_sin = reference_rope_epilogue(
            freqs,
            attention_scaling=args.scale,
            out_dtype=out_dtype,
        )
        cand_cos, cand_sin = fused_rope_epilogue(
            freqs,
            attention_scaling=args.scale,
            out_dtype=out_dtype,
        )
        exact = bool(
            torch.equal(ref_cos, cand_cos) and torch.equal(ref_sin, cand_sin)
        )
        max_abs = float(
            torch.maximum(
                (ref_cos.float() - cand_cos.float()).abs().max(),
                (ref_sin.float() - cand_sin.float()).abs().max(),
            ).item()
        )
        results["checks"][f"rope_epilogue_{name}"] = {
            "exact": exact,
            "max_abs": max_abs,
        }

    # --- attn pack bitwise ---
    bh1d = torch.randn(
        1,
        args.num_heads,
        1,
        args.head_dim,
        device=device,
        dtype=torch.float16,
    )
    ref_pack = bh1d.transpose(1, 2).contiguous().view(1, 1, args.hidden)
    cand_pack = fused_pack_q1_attn_out(
        bh1d,
        batch_size=1,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        all_head_size=args.hidden,
    )
    results["checks"]["attn_pack_fp16"] = {
        "exact": bool(torch.equal(ref_pack, cand_pack)),
    }

    # --- residual add bitwise ---
    residual = torch.randn(1, 1, args.hidden, device=device, dtype=torch.float16)
    hidden = torch.randn(1, 1, args.hidden, device=device, dtype=torch.float16)
    ref_add = residual + hidden
    cand_add = fused_residual_add(residual, hidden)
    results["checks"]["residual_add_fp16"] = {
        "exact": bool(torch.equal(ref_add, cand_add)),
    }

    # --- timing (fp16 rope epilogue + pack; decode-shaped) ---
    def eager_rope():
        reference_rope_epilogue(
            freqs,
            attention_scaling=args.scale,
            out_dtype=torch.float16,
        )

    def fused_rope():
        fused_rope_epilogue(
            freqs,
            attention_scaling=args.scale,
            out_dtype=torch.float16,
        )

    def eager_pack():
        bh1d.transpose(1, 2).contiguous().view(1, 1, args.hidden)

    def fused_pack():
        fused_pack_q1_attn_out(
            bh1d,
            batch_size=1,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            all_head_size=args.hidden,
        )

    rope_eager_ms = _time_ms(eager_rope, args.iters, args.warmup)
    rope_fused_ms = _time_ms(fused_rope, args.iters, args.warmup)
    pack_eager_ms = _time_ms(eager_pack, args.iters, args.warmup)
    pack_fused_ms = _time_ms(fused_pack, args.iters, args.warmup)
    # Tip path: shared RoPE → 1 rope compute / token; pack once / layer (12).
    layers = 12
    saving_ms_per_token = (
        (rope_eager_ms - rope_fused_ms)
        + layers * (pack_eager_ms - pack_fused_ms)
    )
    results["timing_ms"] = {
        "rope_eager": rope_eager_ms,
        "rope_fused": rope_fused_ms,
        "pack_eager": pack_eager_ms,
        "pack_fused": pack_fused_ms,
        "projected_saving_ms_per_token": saving_ms_per_token,
        "budget_ms_per_token": 0.15,
        "clears_budget": saving_ms_per_token >= 0.15,
    }

    all_exact = all(
        bool(check.get("exact")) for check in results["checks"].values()
    )
    results["pass"] = all_exact
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not all_exact:
        raise SystemExit("elementwise_fusion_component=FAIL exactness")
    print(
        "elementwise_fusion_component=PASS "
        f"saving_ms_per_token={saving_ms_per_token:.4f} "
        f"clears_budget={saving_ms_per_token >= 0.15}"
    )


if __name__ == "__main__":
    main()
