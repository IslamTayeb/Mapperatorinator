# §38 TIER2 fused decoder step — handoff

**Status:** **OPEN** (rung-1b quality PASS on ~8k; expand 100k + microbench SUBMITTED) — 2026-07-17  
**Branch / WT:** `codex/turbo-tier2-fused-step` @ `/work/projects/Mapperatorinator-worktrees/turbo-tier2-fused-step`  
**Code tip:** see `git rev-parse HEAD` (fusion `24ee13bd`+)  
**Base tip:** `55949274`  
**Campaign tip unchanged:** `55949274` / FP16 **366.11** — **no 500 claim**; **no tip graduate**; **no merge**.


## Rung 1b harvest (true 7-stage) — QUALITY PASS @ ~8k

| Job | Prec | positions | max_rel | top1 | gates |
| --- | --- | --- | --- | --- | --- |
| `50149619` | FP16 | 7809 | **0.0** | **1.0** | rel+top1 **PASS**; 100k pending |
| `50149620` | FP32 | 7809 | **0.0** | **1.0** | rel+top1 **PASS**; 100k pending |
| Code | `24ee13bd` | | | | `quality_gate_looks_reachable=true` |
| Artifact | `/work/imt11/Mapperatorinator/runs/s38-tier2-logit-agreement-{fp16,fp32}-$JOB/` | | | | |

## Prior rung 1a (blanket wrap — superseded FAIL)

| Job | Prec | Result |
| --- | --- | --- |
| `50149339` | FP16 | top1 0.9983; max_rel **30384** FAIL |
| `50149340` | FP32 | top1 1.0; max_rel **179** FAIL |

## Live follow-ups

| Job / script | Purpose |
| --- | --- |
| `jobs/s38-tier2-logit-100k.sbatch` | Expand ≥100k via lambada hybrid seeds (forced tokens) |
| `jobs/s38-tier2-microbench.sbatch` | ≥0.15 ms/token budget smoke vs tip |

## Implementation (`24ee13bd`)

7-stage patched `VarWhisperDecoderLayer.forward` behind turbo only:
norm+Wqkv → q1 → Wo+res → cross → fc1 → fc2+res → glue.  
One-token uses tip native CUDA (fp32 reductions). Multi-token storage-dtype / module norms (teacher-forced matched tip → max_rel 0).  
Preset `turbo-tier2-fused-step-s38-v2`. Optimized default untouched.

## Wake / next

1. Harvest 100k + microbench jobs.
2. If 100k quality PASS **and** microbench ≥0.15 ms/tok → reciprocal / TIER1c path.
3. If microbench MISS → STOP budget; keep quality evidence; revisit with deeper CUDA fusion.
4. Tip stays `55949274` / **366.11**.

## Not this lever

§39 hybrid / `turbo_mixed`; speculative re-open; INT8-as-FP16; merge to main.
