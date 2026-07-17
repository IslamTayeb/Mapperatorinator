# §42 Draft fast path (W3) — optimized kernels + CUDA graph

**Status:** OPEN — bench job submitted.
**Branch / WT:** `codex/turbo-draft-fastpath` / `turbo-draft-fastpath`
**Tip commit:** `ffdf6a23` (from turbo scaffold `3bfd7bdb`)
**Draft ckpt:** `/work/imt11/Mapperatorinator/runs/s37-tiny-draft-train-50146289/draft_train.pt` (2-layer)
**Campaign tip (unchanged):** `55949274` / FP16 **366.11** (~2.731 ms/token). No 500 claim without TIER1.

## Scope

Run the §37 2-layer draft on the optimized runtime’s q1 / device-state machinery with **its own** CUDA graph cache. Turbo-only; do not change bit-exact `optimized` default behavior.

**Not this work:** §40 canary, §43 train/quality, §39 hybrid TIER3, live FP16 scout `50147054`.

## Target

`c_draft` wall per draft token ≤ **0.15×** of a main step ≈ **≤0.4 ms** on 2080 Ti.

## Implementation

| Piece | Path |
| --- | --- |
| Runner | `osuT5/osuT5/inference/turbo/draft_fastpath.py` |
| Speculative wire | `osuT5/osuT5/inference/turbo/speculate.py` (`MAPPERATORINATOR_TURBO_DRAFT_FASTPATH`, default on) |
| Bench | `utils/s42_draft_fastpath_bench.py` |
| Sbatch | `jobs/s42-draft-fastpath-bench.sbatch` |

Reuses (import only): `ProductionDecodeSession`, `_capture_decode_cuda_graph`, active-prefix context, `generation_profile_context` q1 / cross-MLP hooks.

## Jobs

| Job | Role | Node | Result |
| --- | --- | --- | --- |
| `50147479` | FP16 c_draft microbench | TBD | PENDING |

## Measured (fill on harvest)

| Metric | Value |
| --- | --- |
| c_draft ms/token (median) | TBD |
| c_main ms/token (same-node) | TBD |
| ratio vs measured main | TBD |
| ratio vs tip 2.731 ms | TBD |
| gate ≤0.15× / ≤0.4 ms | TBD |

## Decision

Pending harvest of `50147479`.
