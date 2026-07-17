# §49 Graphed draft chain (W-DG)

**Status:** MEASURED — **MISS_UNDER_KILL** (no grind)  
**Branch / WT:** `codex/turbo-graphed-draft-chain` / `turbo-graphed-draft-chain`  
**Code tip:** `9fa531ab`  
**Base:** §42 tip `e6bf3bb2` (c_draft 1.08 ms/tok GATE MISS)  
**Campaign tip (unchanged):** `55949274` / FP16 **366.11**. No 500 claim.

## Goal

Merge/port §42 draft fastpath onto StaticCache, then chain **γ=3** draft steps in
**ONE** CUDA graph including:

- in-graph sampling (Philox-safe multinomial per §29c)
- embedding feedback (`decoder_input_ids` → model embed → next forward)
- keep-KV: γ-th forward logits stay in `logits_ws` for next cycle's first seed

Persistent chain-graph cache on the decode session (cross-window).

## Gate

| Criterion | Threshold | Result |
| --- | ---: | --- |
| PASS | 3 drafts ≤ **1.2 ms** total | **MISS** |
| KILL | draft chain > **2.0 ms** | not triggered |

## Measured (job `50150142`, z25-20, 2080 Ti, FP16)

Pure CUDA-graph replay only (`timing_mode=pure_graph_replay`).  
Draft: §43 **1-layer** `draft_1layer.pt`, γ=3, temp=0.9.

| Prefix | ms/chain (median) | ms/tok | gate ≤1.2 | kill >2 |
| ---: | ---: | ---: | --- | --- |
| 128 | **1.219** | 0.406 | miss | no |
| 256 | **1.257** | 0.419 | miss | no |
| **median** | **1.257** | **0.419** | **MISS** | no |

Hit counters: `draft_chain_graph_replay` live; persistent cache size 1 per runner.  
Native dispatch: q1 self + q1 cross BMM + cross-MLP (fused q1-RoPE off, same as §42).

### Contaminated prior (do not use for gate)

Job `50149974` @ `27af9612` timed rewind+replay → 2.36 ms (false KILL). Superseded by pure-replay fix.

## vs §42

| | §42 per-token graph | §49 γ=3 chain |
| --- | ---: | ---: |
| ms/tok | 1.08 | **0.42** |
| ms for 3 drafts | ~3.24 (3× serial) | **1.26** (one graph) |

Chaining amortizes launch tax (~2.6× better than 3× §42), but still **+57 µs** over the 1.2 ms PASS bar.

## Decision

**STOP_NO_PROMOTE / MISS_UNDER_KILL.** Record **1.257 ms** γ=3 chain. Do not grind.  
Revisit when fused RoPE/device-state lands for draft, or a thinner draft / tighter in-graph path clears ≤1.2 without harness games. Tip stays `55949274` / **366.11**.

## Artifacts

| Item | Path |
| --- | --- |
| Runtime | `osuT5/osuT5/inference/turbo/draft_chain_graph.py` |
| §42 port | `…/draft_fastpath.py` |
| N-layer draft | `…/draft.py` |
| Bench | `utils/s49_draft_chain_bench.py` |
| Sbatch | `jobs/s49-draft-chain-bench.sbatch` |
| Run | `/work/imt11/Mapperatorinator/runs/s49-draft-chain-50150142/` |

## Jobs

| Job | Role | Result |
| --- | --- | --- |
| `50149974` | harness-inclusive (superseded) | false KILL 2.36 ms |
| `50150142` | pure graph replay gate | **1.257 ms** MISS_UNDER_KILL |
