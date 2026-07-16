"""Attention-only microbench for split-KV num_splits in {4,6,8,12,16}."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from osuT5.osuT5.inference.optimized.kernels.q1_attention import (  # noqa: E402
    _ALLOWED_SPLIT_KV_SPLITS,
    _SPLIT_KV_SPLITS_ENV,
    preload_native_q1_attention,
)


SWEEP = (4, 6, 8, 12, 16)
PREFIXES = (192, 448, 640, 832)
HEADS = 12
HEAD_DIM = 64
WARMUPS = 8
ITERS = 32
# Continue to sealed reciprocal only if best non-8 projects >= this.
MIN_PROJECTED_SAVE_SECONDS = 0.05
FIXED_MAIN_DECODE_REPLAYS = 8_207
DECODER_LAYERS = 12


def _git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _median_ms(samples: list[float]) -> float:
    return float(statistics.median(samples)) * 1_000.0


def _time_split_count(
    *,
    extension: Any,
    split_count: int,
    prefix: int,
    qkv: torch.Tensor,
    cache_keys: torch.Tensor,
    cache_values: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache_position: torch.Tensor,
) -> float:
    stream = torch.cuda.current_stream()
    for _ in range(WARMUPS):
        extension.split_kv_q1_rope_cache_attention(
            qkv,
            cache_keys,
            cache_values,
            cos,
            sin,
            cache_position,
            None,
            prefix,
            split_count,
        )
    stream.synchronize()
    samples: list[float] = []
    for _ in range(ITERS):
        stream.synchronize()
        started = time.perf_counter()
        extension.split_kv_q1_rope_cache_attention(
            qkv,
            cache_keys,
            cache_values,
            cos,
            sin,
            cache_position,
            None,
            prefix,
            split_count,
        )
        stream.synchronize()
        samples.append(time.perf_counter() - started)
    return _median_ms(samples)


def profile_component() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("split-KV num_splits profiler requires CUDA")
    if torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError("split-KV num_splits profiler requires SM75")

    # Force a clean extension load; split count is a kernel argument.
    os.environ.pop(_SPLIT_KV_SPLITS_ENV, None)
    extension = preload_native_q1_attention()

    device = torch.device("cuda")
    dtype = torch.float32
    max_prefix = max(PREFIXES)
    results: dict[str, Any] = {"prefixes": {}, "by_split": {}}

    for prefix in PREFIXES:
        qkv = torch.randn(1, 1, 3, HEADS, HEAD_DIM, device=device, dtype=dtype)
        cache_keys = torch.randn(
            1, HEADS, max_prefix, HEAD_DIM, device=device, dtype=dtype
        )
        cache_values = torch.randn(
            1, HEADS, max_prefix, HEAD_DIM, device=device, dtype=dtype
        )
        cos = torch.randn(1, 1, HEAD_DIM, device=device, dtype=dtype)
        sin = torch.randn(1, 1, HEAD_DIM, device=device, dtype=dtype)
        cache_position = torch.tensor([prefix - 1], device=device, dtype=torch.int64)
        prefix_row: dict[str, float] = {}
        for split_count in SWEEP:
            if prefix < split_count:
                continue
            prefix_row[str(split_count)] = _time_split_count(
                extension=extension,
                split_count=split_count,
                prefix=prefix,
                qkv=qkv,
                cache_keys=cache_keys,
                cache_values=cache_values,
                cos=cos,
                sin=sin,
                cache_position=cache_position,
            )
        results["prefixes"][str(prefix)] = prefix_row

    # Equal-weight mean across measured prefixes for ranking.
    for split_count in SWEEP:
        samples = [
            row[str(split_count)]
            for row in results["prefixes"].values()
            if str(split_count) in row
        ]
        mean_ms = float(statistics.mean(samples)) if samples else float("inf")
        results["by_split"][str(split_count)] = {
            "mean_ms": mean_ms,
            "prefix_count": len(samples),
        }

    control_ms = results["by_split"]["8"]["mean_ms"]
    ranked = sorted(
        (
            (int(split), payload["mean_ms"])
            for split, payload in results["by_split"].items()
        ),
        key=lambda item: item[1],
    )
    best_split, best_ms = ranked[0]
    # Prefer the fastest non-default when control wins the microbench by noise.
    non_default = [(s, ms) for s, ms in ranked if s != 8]
    chosen_split, chosen_ms = (best_split, best_ms) if best_split != 8 else non_default[0]
    delta_ms = control_ms - chosen_ms
    # Diagnostic only — reciprocal wall is the sealed gate.
    projected_save_s = (delta_ms / 1_000.0) * DECODER_LAYERS * FIXED_MAIN_DECODE_REPLAYS
    continue_reciprocal = True

    return {
        "schema_version": 1,
        "candidate": "split_kv_num_splits_sweep",
        "commit": _git_head(),
        "allowed_splits": list(SWEEP),
        "control_split": 8,
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "warmups": WARMUPS,
        "iters": ITERS,
        "measurements": results,
        "ranked_splits": [
            {"split": split, "mean_ms": mean_ms} for split, mean_ms in ranked
        ],
        "best_split": best_split,
        "chosen_split": chosen_split,
        "control_mean_ms": control_ms,
        "best_mean_ms": best_ms,
        "chosen_mean_ms": chosen_ms,
        "delta_vs_control_ms": delta_ms,
        "projected_full_song_save_seconds": projected_save_s,
        "min_projected_save_seconds": MIN_PROJECTED_SAVE_SECONDS,
        "continue_reciprocal": continue_reciprocal,
        "stop_rule": (
            "Always pick fastest non-8 for one sealed reciprocal; promote only if "
            f"sealed wall <= baseline-{MIN_PROJECTED_SAVE_SECONDS}s with OK tokens"
        ),
        "result_class": "component_scout_documented_drift",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = profile_component()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "split-KV num_splits component: "
        f"best={report['best_split']} "
        f"chosen={report['chosen_split']} "
        f"delta_ms={report['delta_vs_control_ms']:.3f} "
        f"projected_s={report['projected_full_song_save_seconds']:.4f} "
        f"continue={report['continue_reciprocal']}"
    )
    # Emit machine-readable choice for the wrapper sbatch.
    print(f"CHOSEN_SPLIT={report['chosen_split']}")


if __name__ == "__main__":
    main()
