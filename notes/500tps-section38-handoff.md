# §38 TIER2 fused decoder step — handoff

**Status:** **OPEN** (rung-1b 7-stage fusion probe SUBMITTED) — 2026-07-17  
**Branch / WT:** `codex/turbo-tier2-fused-step` @ `/work/projects/Mapperatorinator-worktrees/turbo-tier2-fused-step`  
**Code tip:** `24ee13bd`  
**Base tip:** `55949274`  
**Campaign tip unchanged:** `55949274` / FP16 **366.11** — **no 500 claim**; **no tip graduate**; **no merge**.


## Live jobs (rung 1b — true 7-stage)

| Job | Prec | Status |
| --- | --- | --- |
| `50149619` | FP16 | PENDING/RUNNING |
| `50149620` | FP32 | PENDING/RUNNING |
| Code tip | `24ee13bd` | |
| Artifact root | `/work/imt11/Mapperatorinator/runs/s38-tier2-logit-agreement-{fp16,fp32}-$JOB/` | |

## Prior rung 1a (blanket wrap — superseded)

| Job | Prec | Result |
| --- | --- | --- |
| `50149339` | FP16 | top1 **0.9983** PASS; max_rel **30384** FAIL; n=7809 |
| `50149340` | FP32 | top1 **1.0** PASS; max_rel **179** FAIL; n=7809 |
| Code | `c77aab55` | blanket fp32 Linear/RMSNorm wrap — **not** true 7-kernel |

## What changed @ `24ee13bd`

- Replaced blanket Linear/RMSNorm wraps with patched **7-stage** `VarWhisperDecoderLayer.forward`:
  1. norm+Wqkv  2. q1  3. Wo+res  4. cross block  5. fc1  6. fc2+res  7. glue
- One-token: tip native CUDA kernels (fp32 reductions inside).
- Multi-token (teacher-forced probe): storage-dtype GEMM + module RMSNorm (closer to tip HF).
- Preset `turbo-tier2-fused-step-s38-v2`. Opt-in turbo only; optimized default untouched.

## Wake / next

1. Harvest `50149619`/`50149620` `logit_agreement.json` (max_rel, top1, positions, gates).
2. If quality PASS on ≥~8k → expand ≥100k multi-song; then microbench ≥0.15 ms/tok before reciprocal.
3. If quality still FAIL → **STOP** revisit (per-op fp32 accumulate only on reductions; no further blanket wraps).
4. Tip stays `55949274` / **366.11** until song-wall + TIER2(+TIER1c) green.

## Not this lever

§39 hybrid / `turbo_mixed`; speculative re-open without c_verify≤1.2× and c_draft≤0.15×; INT8-as-FP16; merge to main; bare-retry Track A structure STOP scouts.
