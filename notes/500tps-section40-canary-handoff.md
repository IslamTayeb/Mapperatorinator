# §40 TIER1a canary fix — handoff mirror

**Status:** **STOP_ESCALATE** (2026-07-17)  
**Branch / WT:** `codex/turbo-canary-fix` @ `/work/projects/Mapperatorinator-worktrees/turbo-canary-fix`  
**Base turbo tip:** `3bfd7bdb` (KV commit replay). **§40 tip:** `d3b8b2ca` (logit dump + STOP handoff only; no runtime fix). Campaign tip unchanged: `55949274` / **366.11**.  
**Do not** tip-graduate turbo or claim 500. **Do not** harvest/duplicate §39 or scout `50147054`.

## Classification

**Not a rejection-rule / indexing bug.**  
**Not K-batch vs one-token divergence inside turbo.**

At smoke mismatch abs=**110** (prompt_len=108, gen_index=2; turbo=**2236** vs optimized=**2213**):

| Path | argmax @110 | notes |
| --- | --- | --- |
| Turbo one-token eager (HF DynamicCache) | **2236** | top2 gap ≈ **0.008** logit vs 2213 |
| Turbo batched K-verify (forced opt prefix) | **2236** | argmax matches eager; small FP16 noise only |
| Optimized (`active_prefix_cuda_graph`) | **2213** | canary reference |

Artifact: `/work/imt11/Mapperatorinator/runs/s40-tier1a-logit-dump-50147276/logit_dump.json`  
Probe: `utils/s40_tier1a_logit_dump.py` / `jobs/s40-tier1a-logit-dump.sbatch`

Prior canaries still FAIL@110 after KV fix: `50146929` (`600f4f14`), `50147161` (`3bfd7bdb`).

## Decision

**STOP** — unfixable as a turbo Leviathan/verify-row bug. Matching optimized greedy requires a **shared exact teacher path** (graph-aligned / batch-invariant teacher verify that reproduces optimized logits), owned as its own ledger § (recommend **§41**). Classic batch-invariant kernels alone are **insufficient** here: eager already matches batched inside turbo.

## Gate

Greedy canary ≥500 tok × 3 seeds vs optimized: **FAIL / not re-run** after STOP.  
No 500 claim. Full TIER1 still required after teacher-parity lands.

## Jobs

| Job | Role | Result |
| --- | --- | --- |
| `50146929` | smoke @ `600f4f14` | TIER1a FAIL@110 |
| `50147161` | smoke @ `3bfd7bdb` KV fix | TIER1a FAIL@110 (same window) |
| `50147276` | §40 logit dump | classification → cross-engine numerics |

## Next owner

§41 (or coordinator-assigned): exact-shared / graph-aligned teacher verify for TIER1a; do not burn more KV-commit or rejection-index patches for this mismatch.
