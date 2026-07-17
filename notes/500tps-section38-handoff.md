# §38 TIER2 fused decoder step — handoff

**Status:** **PARKED** (strategy shift) — was **STOP_NO_PROMOTE** (quality PASS; perf budget MISS) — 2026-07-17  
**Branch / WT:** `codex/turbo-tier2-fused-step` @ `/work/projects/Mapperatorinator-worktrees/turbo-tier2-fused-step`  
**Code tip:** `8b9c396e` (fusion `24ee13bd`)  
**Base tip:** `55949274`  
**Campaign tip unchanged:** `55949274` / FP16 **366.11** — **no 500 claim**; **no tip graduate**; **no merge**; **no reciprocal**.

**Park reason:** turbo deep-research package supersedes §38 as primary. No more §38 jobs. Siblings own §46/§47/§48/§49.


## Decision

| Gate | Result |
| --- | --- |
| TIER2 quality (max_rel≤1e-2 ∧ top1≥99.5% ∧ n≥100k) | **PASS** |
| Perf budget ≥0.15 ms/token vs tip (before reciprocal) | **MISS** — saved **0.0069** ms/tok |
| Promote / merge / 500 claim | **No** |
| Current posture | **PARKED** — do not grind fusion; do not submit §38 jobs |

**Revisit:** only if deep-research track explicitly reopens Tier-2 fusion with a deeper one-token CUDA launch-collapse (≥0.15 ms/tok measured). Do not bare-retry the Python 7-stage rearrange.


## Quality evidence

| Job | Prec | n | max_rel | top1 | gate |
| --- | --- | --- | --- | --- | --- |
| `50149619` | FP16 | 7809 | 0.0 | 1.0 | rel+top1 PASS |
| `50149620` | FP32 | 7809 | 0.0 | 1.0 | rel+top1 PASS |
| `50149733` | FP16 | **102055** | **0.0** | **1.0** | **tier2_quality_gate_pass=true** (9× lambada hybrid seeds) |

Artifacts: `/work/imt11/Mapperatorinator/runs/s38-tier2-logit-agreement-*` / `s38-tier2-logit-100k-fp16-50149733/`.


## Perf evidence

| Job | Prec | tip ms/tok | turbo ms/tok | saved | budget |
| --- | --- | --- | --- | --- | --- |
| `50149734` | FP16 | 0.1005 | 0.0936 | **0.0069** | need ≥0.15 → **MISS** |

Note: teacher-forced full-seq proxy (`use_cache=False`). One-token decode microbench preferred on revisit.


## Superseded

Rung 1a blanket fp32 Linear wrap (`50149339`/`50149340`) — top1 PASS, max_rel FAIL — not true fusion.


## Not this lever

§39 hybrid / `turbo_mixed`; speculative re-open; INT8-as-FP16; merge; reciprocal without ≥0.15 ms/tok.
