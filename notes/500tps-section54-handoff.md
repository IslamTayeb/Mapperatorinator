# §54 Verify kernels Stage A/B — handoff (Track 2)

**Status:** **Stage A PASS** → Stage B authorized  
**Track:** Endgame **Track 2** (500-enabler A)  
**Numbering:** **§54** — package-original “§51 verify” renamed to avoid collision with vacated §51 / §53 EAGLE  
**Branch / WT:** `codex/turbo-verify-fused-mrow` @ **`1faca2db`**  
**Campaign tip:** `55949274` / FP16 **366.11** — **no merge**; **no 500 claim**  
**Package:** `notes/500tps-turbo-endgame-package.md`

## Measured constants (binding)

| Constant | Value |
| --- | --- |
| Q1 step | 1.842–1.853 ms |
| Unfused graph-native K=3 verify | **3.075 ms** (1.669×) — **§48 sealed; do not re-grind** |
| Draft chain γ=3 | 1.257 ms (§49) |

## Stage plan

| Stage | Work | Gate | Kill |
| --- | --- | --- | --- |
| **A** | m≤8-row extensions of owned warp-group kernels (`fc1_gelu`, `fc2_residual`, one-token linears, rope+KV-cache-write fusion) around SDPA | in-loop c_verify ≤**1.35×** (~2.5 ms) | — |
| **B** | m-row split-KV rope-cache attention (per-row KV lengths; row-tiled; avoid per-(head,row) m× KV re-read) | ≤**1.25×** (~2.3 ms) | **>1.45×** after Stage B |

**Mandatory per stage:** (a) re-measure runtime E (ΔE ≥ **−0.05**); (b) greedy TIER1a canary unaffected. Grid: Stage B + current draft → **505–554** TPS.

## Stage A harvest (tip `1faca2db`)

| Metric | Value | Evidence |
| --- | --- | --- |
| Decision | **STAGE_A_PASS** | `50152588` |
| Path | **`graph_native_k`** | prepare_inputs=0; graph_native_replays=258 |
| Q1 step | **1.852 ms** | optimized cuda_graph |
| in-loop c_verify | **2.467 ms** | CUDA-event avg |
| ratio | **1.332×** | vs Q1; interim ≤1.35 / abs ≤2.5 ms |
| E (accepted/verify) | **1.842** | ΔE **−0.013** ≥ −0.05 vs same-job unfused control E=1.855 |
| Unfused control c_verify (same tip, MROW=0) | 2.618 ms | not a re-grind of sealed §48 3.075 |
| TIER1a canary | **PASS** | `50154498` argmax 2213 / s40 ref; `canary_path_status=PASS` |
| Stage B | **Y** (authorized) | interim gate hit + ΔE + canary |

Artifacts (DCC):  
`/work/imt11/Mapperatorinator/runs/s54-mrow-stage-a-50152588/`  
`/work/imt11/Mapperatorinator/runs/s54-mrow-canary-50154498/`

### Jobs

| Job | Role | Commit | Result |
| --- | --- | --- | --- |
| `50151205` | Stage A (pre-import-fix) | `cd71b686` | FAILED `osuT5.runtime_profiling` import |
| `50151206` | Canary | `cd71b686` | CANCELLED |
| `50152588` | Stage A auth | `1faca2db` | **STAGE_A_PASS** 2.467 ms / **1.332×** |
| `50154498` | TIER1a canary | `1faca2db` | **PASS** |
| `50180928`/`50180929` | Relaunch confirm | `1faca2db` | CANCELLED (redundant; auth harvest already tip-matched) |

## What landed (Stage A)

| Piece | Change |
| --- | --- |
| Kernels | `osuT5/.../optimized/kernels/m_row.py` — m≤8 fc1_gelu / fc2_residual / rmsnorm_linear / linear_residual / rope+KV-cache-write |
| Turbo wire | `turbo/mrow_verify.py` + `teacher_aligned_runtime_context` arms m-row for K>1; m=1 falls through aligned q1 (canary) |
| Probe | `utils/s54_mrow_verify_stage_a.py` + `jobs/s54-mrow-*.sbatch` |
| Import fix | `1faca2db` — `runtime_profiling` path |

## Stage B next

Replace Stage A SDPA attention core with **row-tiled m-row split-KV** (per-row KV lengths; reuse KV across rows in-block). Gate ≤1.25× (~2.3 ms); kill >1.45×. Re-measure E + canary. No §48 unfused grind. Tip unchanged.

## Standing

No merge to main. No §48 unfused grind. No trees. Tip stays `55949274` / **366.11**. No 500 claim.
