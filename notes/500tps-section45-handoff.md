# §45 Combined turbo perf integrator — handoff

**Status:** **EXPLAINED / REOPENED** (not dead-end) — TURBO DEEP-RESEARCH PACKAGE 2026-07-17  
**Branch / WT:** `codex/turbo-integrator` @ `/work/projects/Mapperatorinator-worktrees/turbo-integrator`  
**Integration tip:** instrumentation + diagnose; scout tip `70599188`  
**Campaign tip unchanged:** `55949274` / FP16 **366.11** — **no 500 claim**; **no tip graduate**; **no merge**.

## Absorbs

| Source | What |
| --- | --- |
| §41 | Eager-native aligned teacher verify; c_verify best **1.686×** MISS (≤1.2×) |
| §43 | Perf draft **1-layer · K=1 · γ=3 · temp=0.9** · E[acc]≈**1.97** held-out |
| §42 | GATE MISS c_draft **1.08 ms/tok** (0.36× main) — merge-OK; **not** in scout tip |
| Deep-research | cycle dissection + machinery map + production survey → reopen with W-KV/W-VG/W-DG + W-BASE |

## Gates (historical scout)

| Gate | Job | Result |
| --- | --- | --- |
| TIER1a ≥500×3 | `50148575` @ `d7ed9f73` | **PASS** |
| FP16 SALVALAI scout | `50148770` @ `70599188` | **44.52** main_tps — explained below |
| Full TIER1 / §44 | — | **do not run until ≥384 scout** |

## Scout harvest (`50148770`)

| Metric | Value |
| --- | --- |
| main_tps | **44.52** (≪ tip **366.11**) |
| main_model_s / tokens | **155.36 s** / **6917** |
| Sustained main (excl. odd windows) | median **~23.0 ms/tok** (~43–45 TPS; ~**45 ms/cycle**) |
| First timing window | 49 tok / **154.4 s** (warmup/compile; not in main_tps) |
| Artifact | `/work/imt11/Mapperatorinator/runs/s45-turbo-fp16-scout-50148770/` |

## Why 44.5 is explained (not a dead-end)

Scout measured an **unoptimized sampled path** where every shortcut was silently off:

1. **Verify graphs hard-off for K>1** — `verify_fastpath.py` gate; `_VERIFY_CUDA_GRAPH=1` did not mean the hot path was graphed.
2. **TEACHER_ALIGNED inert under `do_sample`** — aligned teacher only arms under greedy.
3. **Crop-to-L KV rebuild** — recomputes accepted KV every cycle (~12–16 ms/cycle; biggest single tax).
4. **Eager draft ×4** (`speculate.py`) — third forward discarded.
5. **Host syncs/sorts** — ~10 `.item()` syncs + 7 vocab sorts per cycle.

Measured constants that still stand: tip step **1.853 ms**; graphed K=5 verify **3.125 ms**; §42 draft **1.08 ms/tok**; E held-out ≈**1.97–2.44**.

## Ceilings (projections — never production TPS)

| Path | Condition | Ceiling |
| --- | --- | ---: |
| Turbo (all three) | W-KV + W-VG + W-DG land | **~570 TPS** (~1.7 ms/tok) |
| Turbo (any miss) | any of W-KV/W-VG/W-DG fails | **~310 TPS** (below tip) |
| Baseline glue | no speculation; kill 0.88 ms/tok loop glue | **~540 TPS** (1/1.853 ms) |

Strategy: run turbo fixes **and** baseline-glue in parallel; **first ≥450 scout carries**. If post-rung turbo ceiling &lt;420 → shift budget to W-BASE + EAGLE-head.

## RULING — keep-accepted-KV fp16 ULP

| Claim | Binding |
| --- | --- |
| Sampled turbo keep-accepted-KV fp16 ULP (verify-written vs fused-kernel rows; `600f4f14` revert) | **DOCUMENTED DRIFT** under §34; TIER1 KS covers it |
| Greedy TIER1a canary | keeps **crop-rebuild / aligned-Q1** mode — certification unaffected |
| Process | **Do not block sampled-path keep-KV on bit-parity again** |

## Parallel workers (next)

| Worker | Ledger § | Branch (plan) | Gate sketch |
| --- | --- | --- | --- |
| **W-BASE** | §46 | baseline glue v2 | e2e scout ≥450; ceiling ~540 |
| **W-KV** | §47 | `codex/turbo-keep-accepted-kv` | teacher fwd/cycle==1; −10 ms/cycle vs §45 |
| **W-VG** | §48 | graph-native K=γ verify | in-loop c_verify ≤1.2× |
| **W-DG** | §49 | graphed draft chain | 3 drafts ≤1.2 ms total |
| **W-ACC** | §50 | config/margin | **queued** until cycle &lt;10 ms |
| **W-CERT** | §44 | TIER1 pack | only on ≥384 scout |

Integrator serializes merges. Persistent caches across windows **everywhere**.

## Process changes (binding)

1. Scouts must **LOG path hit counters** (draft/verify/rebuild graphs) + accepted/verify + per-phase NVTX ms.
2. **Recompute structural ceiling** after every rung.
3. **Kill criteria:** verify &gt;1.35× after graph-native attempt; draft chain &gt;2 ms; E(real runtime) &lt;1.7 → STOP that worker with number; no grinding.
4. No §44 / no 500 claim until scout ≥**384** sustained; song wall wins; tip stays `55949274`/366.11.

## Parked

- **§38 TIER2 fused step** — PARKED behind §46–§50 (quality PASS / microbench MISS; revisit only with deeper one-token CUDA collapse ≥0.15 ms/tok).

## Instrumentation (diagnose tip)

- `speculate.py`: draft / verify / rebuild / host_gap ms totals + per-verify.
- `processor.py`: whitelist turbo_* into profile JSON.
- Env: `MAPPERATORINATOR_TURBO_STEP_PROFILE=1` (also on when `sync_model_timing`).

## Not this lever

§39 hybrid / turbo_mixed; INT8-as-FP16; merge to main; claiming 500 from projections.
