#!/usr/bin/env python3
"""Pure-Python .osu metric extractors for T5 / TIER1(c) KS parity (CPU-only)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


# osu! hit-object type bits
_CIRCLE = 1
_SLIDER = 2
_SPINNER = 8
_HOLD = 128


@dataclass(frozen=True)
class HitObjectLite:
    time_ms: int
    kind: str  # circle | slider | spinner | hold | other
    slider_length: float | None


@dataclass
class BeatmapMetrics:
    path: str
    ho_count: int
    type_counts: dict[str, int]
    timeshifts_ms: list[float]
    slider_lengths: list[float]
    density_bins: list[float]  # HO count per density_bin_ms window
    density_bin_ms: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_hit_objects(text: str) -> list[HitObjectLite]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    in_ho = False
    out: list[HitObjectLite] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_ho = line.lower() == "[hitobjects]"
            continue
        if not in_ho:
            continue
        parts = line.split(",")
        if len(parts) < 4:
            continue
        try:
            time_ms = int(float(parts[2]))
            type_bits = int(float(parts[3]))
        except ValueError:
            continue
        if type_bits & _SLIDER:
            kind = "slider"
            slider_length = None
            # x,y,time,type,hitSound,curves,slides,length,...
            if len(parts) >= 8:
                try:
                    slider_length = float(parts[7])
                except ValueError:
                    slider_length = None
            out.append(HitObjectLite(time_ms, kind, slider_length))
        elif type_bits & _SPINNER:
            out.append(HitObjectLite(time_ms, "spinner", None))
        elif type_bits & _HOLD:
            out.append(HitObjectLite(time_ms, "hold", None))
        elif type_bits & _CIRCLE:
            out.append(HitObjectLite(time_ms, "circle", None))
        else:
            out.append(HitObjectLite(time_ms, "other", None))
    out.sort(key=lambda h: h.time_ms)
    return out


def extract_metrics(osu_path: Path | str, *, density_bin_ms: int = 1000) -> BeatmapMetrics:
    path = Path(osu_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    hos = _parse_hit_objects(text)
    type_counts = {"circle": 0, "slider": 0, "spinner": 0, "hold": 0, "other": 0}
    timeshifts: list[float] = []
    slider_lengths: list[float] = []
    for i, ho in enumerate(hos):
        type_counts[ho.kind] = type_counts.get(ho.kind, 0) + 1
        if i > 0:
            timeshifts.append(float(ho.time_ms - hos[i - 1].time_ms))
        if ho.kind == "slider" and ho.slider_length is not None:
            slider_lengths.append(float(ho.slider_length))

    density_bins: list[float] = []
    if hos and density_bin_ms > 0:
        t0 = hos[0].time_ms
        t1 = hos[-1].time_ms
        n_bins = max(1, (t1 - t0) // density_bin_ms + 1)
        density_bins = [0.0] * int(n_bins)
        for ho in hos:
            idx = min(int((ho.time_ms - t0) // density_bin_ms), len(density_bins) - 1)
            density_bins[idx] += 1.0

    return BeatmapMetrics(
        path=str(path),
        ho_count=len(hos),
        type_counts=type_counts,
        timeshifts_ms=timeshifts,
        slider_lengths=slider_lengths,
        density_bins=density_bins,
        density_bin_ms=int(density_bin_ms),
    )


def find_osu_files(root: Path | str) -> list[Path]:
    root_p = Path(root)
    if not root_p.exists():
        return []
    cands = sorted(p for p in root_p.rglob("*.osu") if p.is_file())
    # Prefer assembled beatmaps over intermediates when present.
    preferred = [p for p in cands if "candidate" not in p.name.lower()]
    return preferred or cands


def metrics_from_dir(
    root: Path | str, *, density_bin_ms: int = 1000
) -> list[BeatmapMetrics]:
    return [extract_metrics(p, density_bin_ms=density_bin_ms) for p in find_osu_files(root)]


def type_samples(metrics: Iterable[BeatmapMetrics]) -> list[int]:
    """Encode HO types as integers for KS (pooled across maps)."""
    mapping = {"circle": 0, "slider": 1, "spinner": 2, "hold": 3, "other": 4}
    samples: list[int] = []
    for m in metrics:
        for kind, count in m.type_counts.items():
            samples.extend([mapping.get(kind, 4)] * int(count))
    return samples
