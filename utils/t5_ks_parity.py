#!/usr/bin/env python3
"""T5 / TIER1(c) KS distribution parity on .osu sets (CPU).

Reuses the §44 metric set and α=0.01 Bonferroni family control from
docs/inference_evidence_packs.md. Does not invent relaxed thresholds (§34).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "utils") not in sys.path:
    sys.path.insert(0, str(REPO / "utils"))

from t5_beatmap_metrics import (  # noqa: E402
    BeatmapMetrics,
    extract_metrics,
    find_osu_files,
    type_samples,
)


def _ks_2samp(a: list[float] | list[int], b: list[float] | list[int]) -> dict[str, Any]:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.size == 0 or bb.size == 0:
        return {
            "statistic": None,
            "pvalue": None,
            "n_a": int(aa.size),
            "n_b": int(bb.size),
            "error": "empty_sample",
        }
    try:
        from scipy.stats import ks_2samp
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "scipy is required for KS parity (pip/conda env with scipy)"
        ) from exc
    res = ks_2samp(aa, bb, alternative="two-sided", mode="auto")
    return {
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "n_a": int(aa.size),
        "n_b": int(bb.size),
        "error": None,
    }


def _load_dir_metrics(root: Path, *, density_bin_ms: int) -> list[BeatmapMetrics]:
    files = find_osu_files(root)
    return [extract_metrics(p, density_bin_ms=density_bin_ms) for p in files]


def _metric_samples(metrics: list[BeatmapMetrics], name: str) -> list[float]:
    if name == "ho_count":
        return [float(m.ho_count) for m in metrics]
    if name == "density":
        out: list[float] = []
        for m in metrics:
            out.extend(float(x) for x in m.density_bins)
        return out
    if name == "ho_type":
        return [float(x) for x in type_samples(metrics)]
    if name == "timeshift":
        out = []
        for m in metrics:
            out.extend(float(x) for x in m.timeshifts_ms)
        return out
    if name == "slider_length":
        out = []
        for m in metrics:
            out.extend(float(x) for x in m.slider_lengths)
        return out
    raise ValueError(f"unknown metric {name}")


DEFAULT_METRICS = ("ho_count", "density", "ho_type", "timeshift", "slider_length")


def compare_sets(
    baseline_root: Path,
    candidate_root: Path,
    *,
    alpha: float = 0.01,
    density_bin_ms: int = 1000,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    song_id: str | None = None,
) -> dict[str, Any]:
    base = _load_dir_metrics(baseline_root, density_bin_ms=density_bin_ms)
    cand = _load_dir_metrics(candidate_root, density_bin_ms=density_bin_ms)
    tests: dict[str, Any] = {}
    for name in metrics:
        tests[name] = _ks_2samp(_metric_samples(base, name), _metric_samples(cand, name))

    n_tests = len(metrics)
    bonferroni_alpha = alpha / n_tests if n_tests else alpha
    fails = []
    for name, row in tests.items():
        p = row.get("pvalue")
        if p is None or (isinstance(p, float) and (math.isnan(p) or p < bonferroni_alpha)):
            fails.append(name)

    return {
        "song_id": song_id,
        "baseline_root": str(baseline_root),
        "candidate_root": str(candidate_root),
        "n_baseline_osu": len(base),
        "n_candidate_osu": len(cand),
        "alpha": alpha,
        "bonferroni_alpha": bonferroni_alpha,
        "n_tests": n_tests,
        "metrics": tests,
        "pass": len(fails) == 0 and len(base) > 0 and len(cand) > 0,
        "fail_metrics": fails,
    }


def compare_pack_layout(
    pack_root: Path,
    *,
    baseline_engine: str = "optimized",
    candidate_engine: str = "candidate",
    alpha: float = 0.01,
    density_bin_ms: int = 1000,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> dict[str, Any]:
    """Expect pack_root/osu/{engine}/{song_id}/seed_*/**/*.osu."""
    osu_root = pack_root / "osu"
    songs = sorted(
        {
            p.name
            for eng in (baseline_engine, candidate_engine)
            for p in (osu_root / eng).glob("*")
            if p.is_dir()
        }
    )
    per_song = []
    for song_id in songs:
        per_song.append(
            compare_sets(
                osu_root / baseline_engine / song_id,
                osu_root / candidate_engine / song_id,
                alpha=alpha,
                density_bin_ms=density_bin_ms,
                metrics=metrics,
                song_id=song_id,
            )
        )

    n_family = max(1, len(songs) * len(metrics))
    family_alpha = alpha / n_family
    family_fails = []
    for song_row in per_song:
        for name, row in song_row["metrics"].items():
            p = row.get("pvalue")
            if p is None or (isinstance(p, float) and (math.isnan(p) or p < family_alpha)):
                family_fails.append(f"{song_row['song_id']}:{name}")

    return {
        "pack_root": str(pack_root),
        "baseline_engine": baseline_engine,
        "candidate_engine": candidate_engine,
        "songs": songs,
        "alpha": alpha,
        "family_bonferroni_alpha": family_alpha,
        "n_family_tests": n_family,
        "per_song": per_song,
        "pass": len(songs) > 0 and len(family_fails) == 0 and all(s["pass"] for s in per_song),
        "fail_tests": family_fails,
        "status": "PASS" if (len(songs) > 0 and len(family_fails) == 0 and all(s["pass"] for s in per_song)) else (
            "AWAITING" if len(songs) == 0 else "FAIL"
        ),
        "note": (
            "T5 / TIER1(c) KS gate only. Not a 500 claim. "
            "α=0.01 with Bonferroni; no relaxed thresholds (§34)."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack-root", type=Path, help="Pack root with osu/{engine}/{song}/...")
    ap.add_argument("--baseline-root", type=Path, help="Single-set baseline .osu root")
    ap.add_argument("--candidate-root", type=Path, help="Single-set candidate .osu root")
    ap.add_argument("--baseline-engine", default="optimized")
    ap.add_argument("--candidate-engine", default="candidate")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--density-bin-ms", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--song-id", default=None)
    ap.add_argument("--require-pass", action="store_true")
    args = ap.parse_args()

    if args.pack_root is not None:
        payload = compare_pack_layout(
            args.pack_root,
            baseline_engine=args.baseline_engine,
            candidate_engine=args.candidate_engine,
            alpha=args.alpha,
            density_bin_ms=args.density_bin_ms,
        )
    else:
        if args.baseline_root is None or args.candidate_root is None:
            raise SystemExit("need --pack-root or both --baseline-root and --candidate-root")
        payload = compare_sets(
            args.baseline_root,
            args.candidate_root,
            alpha=args.alpha,
            density_bin_ms=args.density_bin_ms,
            song_id=args.song_id,
        )
        payload["status"] = "PASS" if payload.get("pass") else "FAIL"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"pass": payload.get("pass"), "status": payload.get("status"), "out": str(args.out)}))
    if args.require_pass and not payload.get("pass"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
