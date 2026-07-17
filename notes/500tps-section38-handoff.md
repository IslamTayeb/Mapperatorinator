# §38 TIER2 fused decoder step — handoff

**Status:** **OPEN** (rung 1 harvested; quality partial) — 2026-07-17  
**Branch / WT:** `codex/turbo-tier2-fused-step` @ `/work/projects/Mapperatorinator-worktrees/turbo-tier2-fused-step`  
**Base tip:** `55949274`  
**Campaign tip unchanged:** `55949274` / FP16 **366.11** — **no 500 claim**; **no tip graduate**; **no merge**.


## Live jobs (rung 1)

| Job | Prec | Status |
| --- | --- | --- |
| `50149339` | FP16 | COMPLETED gate-fail (top1 PASS, rel FAIL) |
| `50149340` | FP32 | COMPLETED gate-fail (top1 PASS, rel FAIL) |
| Code tip | `c77aab55` | |
| Artifact root | `/work/imt11/Mapperatorinator/runs/s38-tier2-logit-agreement-{fp16,fp32}-$JOB/` | |

## Not this lever

§39 hybrid / `turbo_mixed`; speculative re-open without c_verify≤1.2× and c_draft≤0.15×; INT8-as-FP16; merge to main; bare-retry Track A structure STOP scouts.

## Wake

1. Harvest rung-1 `logit_agreement.json` (max_rel, top1, positions, `quality_gate_looks_reachable`).  
2. If reachable → implement CUDA 7-kernel collapse; else diagnose numerics / STOP.  
3. Tip stays `55949274` / **366.11** until song-wall + TIER2(+TIER1c) green.
