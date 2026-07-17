# §45 Combined turbo perf integrator — handoff

**Status:** **STOP_DEAD_END** (Track C speculative) — 2026-07-17  
**Branch / WT:** `codex/turbo-integrator` @ `/work/projects/Mapperatorinator-worktrees/turbo-integrator`  
**Integration tip:** instrumentation + diagnose (see git log); scout tip `70599188`  
**Campaign tip unchanged:** `55949274` / FP16 **366.11** — **no 500 claim**; **no tip graduate**; **no merge**.

## Absorbs

| Source | What |
| --- | --- |
| §41 | Eager-native aligned teacher verify; c_verify best **1.686×** MISS (≤1.2×) |
| §43 | Perf draft **1-layer · K=1 · γ=3 · temp=0.9** · E[acc]≈**1.97** held-out |
| §42 | GATE MISS c_draft **1.08 ms/tok** (0.36× main) — merge-OK; **not** in scout tip |

## Gates

| Gate | Job | Result |
| --- | --- | --- |
| TIER1a ≥500×3 | `50148575` @ `d7ed9f73` | **PASS** |
| FP16 SALVALAI scout | `50148770` @ `70599188` | **44.52** main_tps — STOP |
| Full TIER1 / §44 | — | **do not run** |
| §46 re-scout | — | **not submitted** (no path to ≥384) |

## Scout harvest (`50148770`)

| Metric | Value |
| --- | --- |
| main_tps | **44.52** (≪ tip **366.11**) |
| main_model_s / tokens | **155.36 s** / **6917** |
| Sustained main (excl. odd windows) | median **~23.0 ms/tok** (~43–45 TPS) |
| First timing window | 49 tok / **154.4 s** (warmup/compile; not in main_tps) |
| accepted/verify in profile | **absent** (profiler whitelist; fixed in diagnose tip) |
| Artifact | `/work/imt11/Mapperatorinator/runs/s45-turbo-fp16-scout-50148770/` |

## Root cause (measured + code)

1. **Crop-to-L + full rebuild every step** (`turbo_kv_commit_mode=crop_to_L_full_rebuild`): after draft+verify advance KV, caches are cropped to `L` and commit is replayed on teacher **and** draft. Keep-accepted-KV (`600f4f14`) was reverted — TIER1a mismatch@110 on smoke `50146929`. Doubles teacher work vs Leviathan keep-prefix.
2. **§42 draft fastpath not in tip** — draft is eager DynamicCache HF path (c_draft gate already 0.36× miss even with graphs).
3. **§41 c_verify ≥1.686× Q=1** — K-forward not ≤1.2×; sequential Q=1 used for canary (expensive).
4. **Profiler dropped turbo stats** — `accepted_per_verify` null in scout summary (now whitelisted + step timers).

## Structural bound (why no §46)

Tip `c_main` ≈ **2.73 ms/tok** (366.11 TPS). With K=1, γ=3, E[acc]≈**1.97**:

| Assumptions | Cost / step | ms/tok | TPS ceiling |
| --- | ---: | ---: | ---: |
| Ideal: §42 c_draft 1.08×3 + §41 c_verify 3.125; **no rebuild**; zero host | 6.37 ms | **3.23** | **~310** |
| Same + crop/rebuild ≈2× teacher | ~9.5 ms | **~4.8** | **~208** |
| Need ≥384 TPS (≤2.60 ms/tok) | need `(3·c_d + c_v)/1.97 < 2.60` | — | needs **both** c_verify≤1.2× **and** c_draft≪1.08 ms |

Even the optimistic no-rebuild ceiling (**~310 TPS**) is **below tip 366** and **below ≥384**. Speculation cannot beat tip at measured gate misses. Raising K only helps if c_verify≤1.2× (currently false).

## Decision

**STOP Track C speculative (draft/verify Leviathan) as 500 path.**  
TIER1a remains PASS evidence for alignment work; do not tip-graduate turbo; do not claim 500.

## Next (recommended)

1. **§38 TIER2 fused step** (relaxed fused decoder) — primary Track C pivot for 500.  
2. Reopen speculative **only if** c_verify≤1.2× **and** c_draft≤0.15× c_main (or E[acc] rises enough to clear the bound) with keep-prefix KV that still passes TIER1a.  
3. No §46 FP16 scout; no §44 harness for this tip.

## Instrumentation (diagnose tip)

- `speculate.py`: draft / verify / rebuild / host_gap ms totals + per-verify.  
- `processor.py`: whitelist turbo_* into profile JSON.  
- Env: `MAPPERATORINATOR_TURBO_STEP_PROFILE=1` (also on when `sync_model_timing`).

## Not this lever

§39 hybrid / turbo_mixed; INT8-as-FP16; merge to main.
