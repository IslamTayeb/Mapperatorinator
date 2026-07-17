# §38 TIER2 fused decoder step — handoff

**Status:** **OPEN** (rung 1 submitted / in flight) — 2026-07-17  
**Branch / WT:** `codex/turbo-tier2-fused-step` @ `/work/projects/Mapperatorinator-worktrees/turbo-tier2-fused-step`  
**Base tip:** `55949274`  
**Campaign tip unchanged:** `55949274` / FP16 **366.11** — **no 500 claim**; **no tip graduate**; **no merge**.


## Live jobs (rung 1)

| Job | Prec | Status |
| --- | --- | --- |
| `50149260` | FP16 | PENDING |
| `50149261` | FP32 | PENDING |
| Code tip | `9d27d4bc` | |
| Artifact root | `/work/imt11/Mapperatorinator/runs/s38-tier2-logit-agreement-{fp16,fp32}-$JOB/` | |

## Pivot

Track C speculative (§45) **STOP_DEAD_END** (~44.5 TPS; optimistic ceiling ~310 ≪ tip).  
§38 is the primary Track C path: **TIER2 relaxed-numerics fused decoder step** behind `inference_engine=turbo` (immutable preset). Bit-exact `optimized` default is untouched.

## Budget claim (before full reciprocal)

| Fact | Value | Source |
| --- | --- | --- |
| Launches / token (node path) | **~400** | combo handoff / tip node path |
| Elementwise + memory family | **~30%** (~0.64 ms/tok) | nsight `49966210` (elem ~344 µs + mem ~295 µs) |
| gemm_gemv_projection | **~26%** | nsight `49966210` |
| native_q1 | **~19%** | nsight `49966210` |
| fused MLP (fc1/fc2) | **~15%** | already fused; neighbor-fuse target |
| Gap tip→500 (FP16) | **≥0.6 ms/token** (~0.73) | 21.330 s → ≤20.264 s |
| **§38 claim before recip** | **≥0.15 ms/token** | collapse ~400 launches/glue into 7 kernels/layer × 12 layers + glue |

7-kernel layer target: **norm+Wqkv, q1, Wo+res, cross block, fc1, fc2+res, glue**; fp32 accumulate.

## Gates (no 500 without these)

1. Teacher-forced max rel logit delta ≤ **1e-2** AND top-1 ≥ **99.5%** over ≥ **100k** positions.  
2. Then TIER1(c) 30-seed KS + fixed-seed gallery.  
3. Perf: aim ≥5% vs tip then →500; **song wall wins**.

## Scaffold

| Piece | Path |
| --- | --- |
| Turbo preset (TIER2 fused) | `osuT5/osuT5/inference/turbo/` |
| fp32-accumulate reference | `osuT5/osuT5/inference/turbo/fused_step.py` |
| Selector | `inference.py` (`v32\|optimized\|turbo`) |
| Rung-1 probe | `utils/s38_tier2_logit_agreement_probe.py` |
| DCC job | `jobs/s38-tier2-logit-agreement.sbatch` |
| Env kill-switch | `MAPPERATORINATOR_TURBO_TIER2_FUSED=0` |

**Opt-in only** via `inference_engine=turbo`. Never changes `optimized` bit-exact default.

## Rung ladder

| Rung | What | Status |
| ---: | --- | --- |
| 1 | Component / teacher-forced logit agreement on real tensors | **OPEN** (this turn) |
| 2 | CUDA 7-kernel fusion (after quality looks reachable) | queued |
| 3 | Short loop / smoke | queued |
| 4 | Full-song wall vs tip (≥5%) | queued |
| 5 | TIER1(c) + gallery → 500 discussion | queued |

## Not this lever

§39 hybrid / `turbo_mixed`; speculative re-open without c_verify≤1.2× and c_draft≤0.15×; INT8-as-FP16; merge to main; bare-retry Track A structure STOP scouts.

## Wake

1. Harvest rung-1 `logit_agreement.json` (max_rel, top1, positions, `quality_gate_looks_reachable`).  
2. If reachable → implement CUDA 7-kernel collapse; else diagnose numerics / STOP.  
3. Tip stays `55949274` / **366.11** until song-wall + TIER2(+TIER1c) green.
