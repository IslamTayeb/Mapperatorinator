# §46 Baseline glue-elimination v2 (W-BASE) — handoff

**Status:** **STOP_NO_PROMOTE** — exact PASS; budget MISS  
**Branch / WT:** `codex/exact-baseline-glue-v2` / DCC `.../Mapperatorinator-worktrees/exact-baseline-glue-v2`  
**Code tip:** `3987bd5c` (gate-parser fix may follow)  
**Base / campaign tip:** `55949274` / FP16 **366.11** — **unchanged**  
**Not:** bare §29 retry — persistent cross-window sample cache + separate small tail graph

## Decision

| Gate | Result |
| --- | --- |
| Smoke exact (tok + RNG + hits) | **PASS** — tok **1322**=1322; RNG equal; hits **1322** |
| Glue cut ≥0.15 ms/tok | **MISS** — **−0.003** ms/tok (base 297.63 → cand 297.33 TPS on smoke15) |
| e2e scout ≥450 | **not submitted** (kill criterion) |
| Tip graduate / merge / 500 claim | **No** |

**Kill:** cannot show ≥0.15 ms/tok glue cut with persistent caches → **STOP_NO_PROMOTE**. Do not bare-retry sample-graph shape tweaks.

## Evidence

| Job | Node | Result |
| --- | --- | --- |
| **`50150069`** | z25-21 | smoke COMPLETED analysis FAILED only on stage-schema parser; artifacts valid |

Run root: `/work/imt11/Mapperatorinator/runs/baseline-glue-v2-smoke-fp16-50150069/`  
Corrected gate: `smoke_gate.json` (model_elapsed sum).

| Metric | Baseline (tip composition) | Candidate |
| --- | --- | --- |
| tokens | 1322 | 1322 |
| model_s | 4.442 | 4.446 |
| TPS | 297.63 | 297.33 |
| sampling-tail hits | 0 | 1322 |
| RNG / tokens | — | equal |

## What landed

1. Persistent sampling-tail CUDA graph (temp + top-p + softmax + multinomial; Philox register + RNG-restore capture)
2. Tip device token buffer (`preallocated_batch1_state`)
3. Pinned-flag EOS (async D2H + event)
4. Session-static sample workspaces in `shared_graph_cache` (cross-window)

## Why no headroom

Separate small tail graph is bit-exact and persistent, but host gap + second graph launch ≈ cost of eager TopP/sample on tip composition. Matches §29b lesson (Philox-in-graph possible; no bit-exact ms/tok headroom).

## Revisit

Only with **new** measured ≥0.15 ms/tok evidence on tip composition — e.g. collapsing forward+sample into one replay without per-window capture tax, or removing the host round-trip another way. Not another isolated sample-graph tweak. Not turbo.

## Scripts

- `scripts/dcc/profile_baseline_glue_v2_smoke.sbatch`
- `scripts/dcc/profile_baseline_glue_v2_scout.sbatch` (unused after kill)
- `utils/run_exact_baseline_glue_v2.py`
