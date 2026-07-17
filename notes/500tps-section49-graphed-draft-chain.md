# §49 Graphed draft chain (W-DG)

**Status:** OPEN — measuring  
**Branch / WT:** `codex/turbo-graphed-draft-chain` / `turbo-graphed-draft-chain`  
**Base:** §42 tip `e6bf3bb2` (c_draft 1.08 ms/tok GATE MISS)  
**Campaign tip (unchanged):** `55949274` / FP16 **366.11**. No 500 claim.

## Goal

Merge/port §42 draft fastpath onto StaticCache, then chain **γ=3** draft steps in
**ONE** CUDA graph including:

- in-graph sampling (Philox-safe multinomial per §29c)
- embedding feedback (`decoder_input_ids` → model embed → next forward)
- keep-KV: γ-th forward logits become next cycle's first seed (formerly discarded)

Persistent chain-graph cache on the decode session (cross-window; never rebuild
per window).

## Gate

| Criterion | Threshold |
| --- | ---: |
| PASS | 3 drafts ≤ **1.2 ms** total |
| KILL | draft chain > **2.0 ms** |

## Default draft

§43 perf: **1-layer · γ=3 · temp=0.9**  
`/work/imt11/Mapperatorinator/runs/s43-draft-quality-50147299/draft_1layer.pt`

## Artifacts

| Item | Path |
| --- | --- |
| Runtime | `osuT5/osuT5/inference/turbo/draft_chain_graph.py` |
| §42 port | `…/draft_fastpath.py` (from `e6bf3bb2`) |
| N-layer draft | `…/draft.py` (1-layer capable) |
| Bench | `utils/s49_draft_chain_bench.py` |
| Sbatch | `jobs/s49-draft-chain-bench.sbatch` |

## Jobs

| Job | Role | Result |
| --- | --- | --- |
| _(pending)_ | γ=3 chain microbench | — |

## Decision

_(fill after bench)_
