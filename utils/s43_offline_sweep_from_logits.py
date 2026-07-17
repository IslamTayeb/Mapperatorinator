#!/usr/bin/env python3
"""§43 — CPU-only re-sweep from saved logit dumps (no GPU, no turbo wiring)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(os.environ.get("PROBE_REPO", ".")).resolve()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "s43_draft_quality_suite", REPO / "utils" / "s43_draft_quality_suite.py"
)
assert spec and spec.loader
s43 = importlib.util.module_from_spec(spec)
# Avoid executing GPU main; load only helpers by reading suite after partial import is hard.
# Instead duplicate thin wrappers via exec of functions — import module but guard main.
sys.modules["s43_draft_quality_suite"] = s43
spec.loader.exec_module(s43)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logit-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--held-out", default="ela-ke-leitada,nube-negra")
    ap.add_argument("--prefix", default="tip2layer", help="logit file prefix")
    ap.add_argument("--draft-rel-cost", type=float, default=2.0 / 12.0)
    ap.add_argument("--draft-name", default="tip_2layer_salvalai")
    ap.add_argument("--top-p", type=float, default=0.9)
    args = ap.parse_args()

    held_out = [s.strip() for s in args.held_out.split(",") if s.strip()]
    temperatures = [0.7, 0.9, 1.1]
    gammas = list(range(3, 9))
    ks = [1, 2, 4]
    all_rows = []
    for sid in held_out:
        path = args.logit_dir / f"{args.prefix}_{sid}.pt"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        pack = torch.load(path, map_location="cpu")
        alphas = s43.alphas_from_logit_pack(pack, 0.9, args.top_p)
        rows = s43.acceptance_vs_cost_table(
            alphas,
            draft_name=args.draft_name,
            draft_rel_cost=args.draft_rel_cost,
            temperatures=temperatures,
            gammas=gammas,
            ks=ks,
            top_p=args.top_p,
            logit_pack=pack,
        )
        for r in rows:
            r["song_id"] = sid
        all_rows.extend(rows)

    agg: dict[tuple, list[float]] = defaultdict(list)
    agg_e: dict[tuple, list[float]] = defaultdict(list)
    meta = {}
    for r in all_rows:
        key = (r["draft"], r["temperature"], r["gamma"], r["K"])
        agg[key].append(r["E_over_cost"])
        agg_e[key].append(r["E_accepted_per_step"])
        meta[key] = r
    agg_rows = []
    for key, vals in agg.items():
        base = meta[key]
        agg_rows.append(
            {
                "draft": key[0],
                "temperature": key[1],
                "gamma": key[2],
                "K": key[3],
                "held_out_mean_E": sum(agg_e[key]) / len(agg_e[key]),
                "held_out_mean_E_over_cost": sum(vals) / len(vals),
                "cost_verify_units": base["cost_verify_units"],
            }
        )
    rec = s43.pick_recommendation(
        [
            {
                **r,
                "E_accepted_per_step": r["held_out_mean_E"],
                "E_over_cost": r["held_out_mean_E_over_cost"],
            }
            for r in agg_rows
        ]
    )
    out = {
        "schema": "s43-offline-sweep-v1",
        "held_out": held_out,
        "rows": all_rows,
        "aggregate": sorted(agg_rows, key=lambda r: r["held_out_mean_E_over_cost"], reverse=True),
        "recommendation": rec,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"recommendation": rec, "top5": out["aggregate"][:5]}, indent=2))
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
