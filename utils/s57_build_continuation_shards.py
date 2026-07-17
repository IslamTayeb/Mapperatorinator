#!/usr/bin/env python3
"""§57 — CPU shard prep for continuation(+timing) distill.

Primary domain mismatch (§56): tip-dump / long map₀ E ≠ short lookback
map_rest (turbo med gen≈7) + timing (E≡1). Build hard-label shards that
tag map0 / map_rest / timing and a held-out split.

CPU-only. No TPS claim. No turbo runtime wire.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


S55_DEFAULT = Path(
    "/work/imt11/Mapperatorinator/runs/s55-turbo-fp16-scout-50152542"
)
TIP_SALVALAI_DEFAULT = Path(
    "/work/imt11/Mapperatorinator/runs/exact-rope-device-state-fp16-auth-49964133"
)
FIVE_SONG = {
    "pegasus": Path(
        "/work/imt11/Mapperatorinator/runs/five-song-profile-20260702-232516-8a2de72/"
        "after_separate/official/pegasus/pegasus.profile.json"
    ),
    "lambada": Path(
        "/work/imt11/Mapperatorinator/runs/five-song-profile-20260702-232516-8a2de72/"
        "after_separate/official/lambada/lambada.profile.json"
    ),
}


def find_profile(run_root: Path) -> Path:
    if run_root.is_file() and run_root.name.endswith(".profile.json"):
        return run_root
    cands = sorted(run_root.glob("**/beatmap*.osu.profile.json"))
    if not cands:
        cands = sorted(run_root.glob("**/*.profile.json"))
    if not cands:
        raise SystemExit(f"no profile json under {run_root}")
    for p in cands:
        if "candidate_first" in str(p):
            return p
    return cands[0]


def extract_windows(profile: Path, *, song_id: str, source: str) -> list[dict]:
    payload = json.loads(profile.read_text())
    rows: list[dict] = []
    for rec in payload.get("generation", []):
        ctx = str(rec.get("context_type"))
        if ctx not in ("map", "timing"):
            continue
        ids = rec.get("generated_token_ids")
        if not (isinstance(ids, list) and ids and isinstance(ids[0], int)):
            continue
        seq = int(rec.get("sequence_index"))
        if ctx == "map":
            role = "map0" if seq == 0 else "map_rest"
        else:
            role = "timing"
        rows.append(
            {
                "song_id": song_id,
                "source": source,
                "role": role,
                "context_type": ctx,
                "sequence_index": seq,
                "generated_token_ids": [int(x) for x in ids],
                "generated_tokens": int(rec.get("generated_tokens") or len(ids)),
                "prompt_tokens": rec.get("prompt_tokens"),
                "frame_time_ms": rec.get("frame_time_ms"),
                "trim_lookback": rec.get("trim_lookback"),
                "trim_lookahead": rec.get("trim_lookahead"),
            }
        )
    rows.sort(key=lambda r: (r["context_type"], r["sequence_index"]))
    return rows


def assign_split(
    rows: list[dict],
    *,
    holdout_frac: float,
    holdout_min: int,
) -> None:
    """Hold out last N map_rest + last N timing by sequence_index (per song/source)."""
    by_key: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        key = (r["song_id"], r["source"], r["role"])
        by_key.setdefault(key, []).append(r)
    for key, group in by_key.items():
        role = key[2]
        group.sort(key=lambda r: r["sequence_index"])
        if role == "map0":
            for r in group:
                r["split"] = "train"  # keep map0 for anti-regression; eval separately
            continue
        n = len(group)
        n_hold = max(holdout_min, int(round(n * holdout_frac)))
        n_hold = min(n_hold, max(0, n - 1))  # keep ≥1 train if possible
        for i, r in enumerate(group):
            r["split"] = "holdout" if i >= n - n_hold else "train"


def summarize(rows: list[dict]) -> dict:
    def _slice(pred):
        xs = [r for r in rows if pred(r)]
        return {
            "n_windows": len(xs),
            "n_tokens": sum(r["generated_tokens"] for r in xs),
            "median_gen": (
                sorted(r["generated_tokens"] for r in xs)[len(xs) // 2] if xs else None
            ),
        }

    return {
        "all": _slice(lambda _r: True),
        "by_role": {
            role: _slice(lambda r, role=role: r["role"] == role)
            for role in ("map0", "map_rest", "timing")
        },
        "by_split": {
            split: _slice(lambda r, split=split: r.get("split") == split)
            for split in ("train", "holdout")
        },
        "train_map_rest": _slice(
            lambda r: r.get("split") == "train" and r["role"] == "map_rest"
        ),
        "holdout_map_rest": _slice(
            lambda r: r.get("split") == "holdout" and r["role"] == "map_rest"
        ),
        "train_timing": _slice(
            lambda r: r.get("split") == "train" and r["role"] == "timing"
        ),
        "holdout_timing": _slice(
            lambda r: r.get("split") == "holdout" and r["role"] == "timing"
        ),
        "role_counts": dict(Counter(r["role"] for r in rows)),
        "source_counts": dict(Counter(r["source"] for r in rows)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--s55-run", type=Path, default=S55_DEFAULT)
    ap.add_argument("--tip-salvalai-run", type=Path, default=TIP_SALVALAI_DEFAULT)
    ap.add_argument("--holdout-frac", type=float, default=0.25)
    ap.add_argument("--holdout-min", type=int, default=16)
    ap.add_argument(
        "--include-five-song-timing",
        action="store_true",
        help="Add pegasus/lambada timing windows (long map_rest excluded)",
    )
    ap.add_argument(
        "--include-tip-long-rest",
        action="store_true",
        help="Also index tip long map_rest (low train weight later; not primary)",
    )
    args = ap.parse_args()

    rows: list[dict] = []
    sources: list[dict] = []

    s55_profile = find_profile(args.s55_run)
    s55_rows = extract_windows(s55_profile, song_id="salvalai", source="s55_turbo")
    rows.extend(s55_rows)
    sources.append(
        {
            "song_id": "salvalai",
            "source": "s55_turbo",
            "profile": str(s55_profile),
            "n_windows": len(s55_rows),
            "primary": True,
            "note": "short map_rest (med≈7) + timing — binding §56 domain",
        }
    )

    # Tip map0 only by default (anti-regression); tip long-rest optional
    tip_profile = find_profile(args.tip_salvalai_run)
    tip_rows = extract_windows(tip_profile, song_id="salvalai", source="tip_auth")
    if args.include_tip_long_rest:
        keep = tip_rows
    else:
        keep = [r for r in tip_rows if r["role"] == "map0"]
    # Avoid duplicating map0 if s55 already has it — keep tip map0 as separate source
    # for teacher-aligned long first window; s55 map0 also kept (turbo-committed).
    rows.extend(keep)
    sources.append(
        {
            "song_id": "salvalai",
            "source": "tip_auth",
            "profile": str(tip_profile),
            "n_windows": len(keep),
            "primary": False,
            "note": "tip map0 keep" + (" + long rest" if args.include_tip_long_rest else ""),
        }
    )

    if args.include_five_song_timing:
        for song_id, path in FIVE_SONG.items():
            if not path.exists():
                print(f"WARN: missing {path}", flush=True)
                continue
            song_rows = [
                r
                for r in extract_windows(path, song_id=song_id, source="five_song")
                if r["role"] == "timing"
            ]
            rows.extend(song_rows)
            sources.append(
                {
                    "song_id": song_id,
                    "source": "five_song",
                    "profile": str(path),
                    "n_windows": len(song_rows),
                    "primary": False,
                    "note": "timing-only multi-song (map_rest is long tip-style; excluded)",
                }
            )

    assign_split(rows, holdout_frac=args.holdout_frac, holdout_min=args.holdout_min)
    summary = summarize(rows)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    shards_path = out_dir / "continuation_shards.json"
    index_path = out_dir / "continuation_shards_index.json"

    payload = {
        "schema": "s57-continuation-shards-v1",
        "tip_commit": "55949274",
        "n_windows": len(rows),
        "n_tokens": sum(r["generated_tokens"] for r in rows),
        "holdout_frac": args.holdout_frac,
        "holdout_min": args.holdout_min,
        "sources": sources,
        "summary": summary,
        "windows": rows,
        "note": (
            "Hard-label shards for §57 continuation(+timing) CE/KL. "
            "Primary short map_rest/timing from s55 turbo scout. "
            "Offline held-out E claimed separately; no runtime wire."
        ),
    }
    shards_path.write_text(json.dumps(payload, indent=2) + "\n")

    index = {
        "schema": "s57-continuation-shards-index-v1",
        "shards": str(shards_path),
        "tip_commit": "55949274",
        "sources": sources,
        "summary": summary,
        "gates": {
            "map_rest_E_aim": 1.7,
            "map_rest_E_floor_for_map_wE_1.7": 1.63,
            "map_rest_delta_E_min_vs_s55": 0.15,
            "timing_must_rise_or_exit_E_gate": True,
            "wire_runtime": False,
        },
        "baseline_s55": {
            "map0_wE": 1.919,
            "map_rest_wE": 1.121,
            "timing_wE": 1.0,
            "map_wE": 1.311,
            "full_wE": 1.181,
            "job": "50152542",
        },
    }
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(index, indent=2), flush=True)
    print(f"WROTE {shards_path}", flush=True)
    print(f"WROTE {index_path}", flush=True)


if __name__ == "__main__":
    main()
