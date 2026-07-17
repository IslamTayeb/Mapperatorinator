#!/usr/bin/env python3
"""§37 — build tiny-draft distill shards from tip profile dumps.

Extracts map-window generated_token_ids (+ metadata) for CE/KL training.
Does not claim TPS. Not INT8.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_profile(run_root: Path) -> Path:
    cands = sorted(run_root.glob("**/beatmap*.osu.profile.json"))
    if not cands:
        raise SystemExit(f"no profile json under {run_root}")
    for p in cands:
        if "candidate_first" in str(p):
            return p
    return cands[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dump-run",
        type=Path,
        default=Path("/work/imt11/Mapperatorinator/runs/exact-rope-device-state-fp16-auth-49964133"),
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--context", default="map", choices=["map", "timing", "all"])
    args = ap.parse_args()

    profile = find_profile(args.dump_run)
    payload = json.loads(profile.read_text())
    meta = payload.get("metadata") or {}
    rows = []
    for rec in payload.get("generation", []):
        ctx = str(rec.get("context_type"))
        if args.context != "all" and ctx != args.context:
            continue
        ids = rec.get("generated_token_ids")
        if not (isinstance(ids, list) and ids and isinstance(ids[0], int)):
            continue
        rows.append(
            {
                "context_type": ctx,
                "sequence_index": int(rec.get("sequence_index")),
                "generated_token_ids": [int(x) for x in ids],
                "generated_tokens": int(rec.get("generated_tokens") or len(ids)),
                "prompt_tokens": rec.get("prompt_tokens"),
                "frame_time_ms": rec.get("frame_time_ms"),
            }
        )
    rows.sort(key=lambda r: (r["context_type"], r["sequence_index"]))
    out = {
        "schema": "s37-distill-shards-v1",
        "source_profile": str(profile),
        "dump_run": str(args.dump_run),
        "tip_commit_meta": meta.get("git_commit"),
        "model_path": meta.get("model_path"),
        "precision": meta.get("precision"),
        "temperature": meta.get("temperature"),
        "top_p": meta.get("top_p"),
        "n_windows": len(rows),
        "n_tokens": sum(r["generated_tokens"] for r in rows),
        "windows": rows,
        "note": "Hard-label shard for §37 tiny-draft CE. Acceptance/TPS claimed separately.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out} windows={out['n_windows']} tokens={out['n_tokens']}")


if __name__ == "__main__":
    main()
