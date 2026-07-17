# §42 Draft fast path (W3) — optimized kernels + CUDA graph

**Status:** MEASURED — GATE MISS vs ≤0.15× / ≤0.4 ms target.
**Branch / WT:** `codex/turbo-draft-fastpath` / `turbo-draft-fastpath`
**Tip commit:** `c11c62c7`
**Draft ckpt:** `/work/imt11/Mapperatorinator/runs/s37-tiny-draft-train-50146289/draft_train.pt` (2-layer)
**Campaign tip (unchanged):** `55949274` / FP16 **366.11** (~2.731 ms/token). No 500 claim.

## Measured (job `50148590`, z25-21, 2080 Ti, FP16)

| Metric | Value |
| --- | ---: |
| **c_draft ms/token (median)** | **1.0815** |
| c_main ms/token (same-node, 12-layer teacher) | 2.995 |
| ratio vs measured main | **0.361×** |
| ratio vs tip 2.731 ms | 0.396× |
| gate ≤0.15× / ≤0.4 ms | **MISS** |

Per-prefix draft (CUDA-graph `decode_token`): prefix 128 → **1.0815** ms (eager 1.219); prefix 256 → **0.855** ms (eager 13.25 warmup outlier on teacher path; draft graph used).

`used_cuda_graph=true`; dispatch: q1 self + q1 cross BMM + cross-MLP (fused q1-RoPE disabled — draft lacks tip shared-rope cos/sin `[1,1,D]`).

## Decision

**STOP_NO_PROMOTE** on the ≤0.15× / ≤0.4 ms bar. Fastpath is wired and graphs replay, but c_draft ≈ **1.08 ms** is ~2.7× above the absolute target. Revisit when fused RoPE/device-state is available for the draft session or a thinner draft lands from §43.

## Jobs

| Job | Role | Result |
| --- | --- | --- |
| `50147957` | CUDA flag diagnose | all native flags OK (single step) |
| `50148590` | c_draft microbench | **DONE** — 1.0815 ms/tok, gate miss |
