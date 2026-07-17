# §54 Verify kernels Stage A/B — handoff (Track 2)

**Status:** **Stage A PASS** · **Stage B MISS** (no kill; STOP Stage B grind)  
**Track:** Endgame **Track 2** (500-enabler A)  
**Numbering:** **§54** — package-original “§51 verify” renamed to avoid collision with vacated §51 / §53 EAGLE  
**Branch / WT:** `codex/turbo-verify-fused-mrow` @ **`9d91c5d2`** (+ default-off follow-up)  
**Campaign tip:** `55949274` / FP16 **366.11** — **no merge**; **no 500 claim**  
**Package:** `notes/500tps-turbo-endgame-package.md`

## Measured constants (binding)

| Constant | Value |
| --- | --- |
| Q1 step | 1.842–1.853 ms |
| Unfused graph-native K=3 verify | **3.075 ms** (1.669×) — **§48 sealed; do not re-grind** |
| Draft chain γ=3 | 1.257 ms (§49) |
| **Stage A auth c_verify** | **2.467 ms** (**1.332×**) — keep |

## Stage plan

| Stage | Work | Gate | Kill | Result |
| --- | --- | --- | --- | --- |
| **A** | m≤8-row warp-group islands + rope/KV-cache-write around SDPA | ≤**1.35×** (~2.5 ms) | — | **PASS** |
| **B** | row-tiled m-row split-KV (per-row KV lengths) | ≤**1.25×** (~2.3 ms) | **>1.45×** | **MISS** 1.412× (2.602 ms) |

**Mandatory per stage:** (a) re-measure runtime E (ΔE ≥ **−0.05**); (b) greedy TIER1a canary unaffected.

## Stage A harvest (`1faca2db`)

| Metric | Value | Evidence |
| --- | --- | --- |
| Decision | **STAGE_A_PASS** | `50152588` |
| Path | **`graph_native_k`** | prepare_inputs=0; graph_native_replays=258 |
| Q1 step | **1.852 ms** | optimized cuda_graph |
| in-loop c_verify | **2.467 ms** | CUDA-event avg |
| ratio | **1.332×** | vs Q1; interim ≤1.35 / abs ≤2.5 ms |
| E (accepted/verify) | **1.842** | ΔE **−0.013** ≥ −0.05 vs unfused control E=1.855 |
| TIER1a canary | **PASS** | `50154498` argmax 2213 |

## Stage B harvest (`9d91c5d2`)

| Metric | Value | Evidence |
| --- | --- | --- |
| Decision | **STAGE_B_MISS** | `50181231` (scavenger-gpu / 2080 Ti) |
| Path | **`graph_native_k`** | prepare_inputs=0; replays=262 |
| Q1 step | **1.843 ms** | same job |
| in-loop c_verify | **2.602 ms** | **1.412×** — gate ≤1.25 / ≤2.3 **MISS**; kill >1.45 **not hit** |
| vs Stage A control (same job) | 2.577 ms | Stage B **regress** vs SDPA core |
| E | **1.816** | ΔE **−0.039** ≥ −0.05 vs Stage A control E=1.855 |
| TIER1a canary | **PASS** | `50181232` |

**STOP Stage B grind.** Naive row-tiled split-KV does not beat Stage A SDPA on SM75; keep Stage A as the verify path (`MAPPERATORINATOR_TURBO_VERIFY_MROW_STAGE_B` default **off**). Revisit only with a different attention mechanism (shared-arena / flash-decode-style, not this partial/merge tax).

## Jobs

| Job | Role | Commit | Result |
| --- | --- | --- | --- |
| `50151205` | Stage A | `cd71b686` | FAILED import |
| `50152588` | Stage A auth | `1faca2db` | **PASS** 2.467 ms / **1.332×** |
| `50154498` | Canary | `1faca2db` | **PASS** |
| `50180928`/`29` | Relaunch confirm | `1faca2db` | CANCELLED (redundant) |
| `50181079`/`80` | Stage B (gpu-common) | `9d91c5d2` | CANCELLED (queue) |
| `50181231` | Stage B auth | `9d91c5d2` | **MISS** 2.602 ms / **1.412×** |
| `50181232` | Canary | `9d91c5d2` | **PASS** |

## Standing

No merge to main. No §48 unfused grind. Tip stays `55949274` / **366.11**. No 500 claim. Stage A is the retained Track-2 verify win.
