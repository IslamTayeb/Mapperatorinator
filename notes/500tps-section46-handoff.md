# §46 Baseline glue-elimination v2 (W-BASE) — handoff

**Status:** **OPEN** — implementation landed; smoke/scout pending  
**Branch / WT:** `codex/exact-baseline-glue-v2`  
**Base tip:** `55949274` / FP16 **366.11**  
**Not:** bare §29 retry — persistent cross-window sample cache + separate small tail graph

## What

Independent no-speculation path targeting ~0.88 ms/tok non-model glue:

1. Persistent sampling-tail CUDA graph (temp + top-p sort + softmax + multinomial; Philox register + RNG-restore capture warmup)
2. Device token buffer (tip composition `preallocated_batch1_state`)
3. Pinned-flag EOS (async D2H + event) instead of `unfinished.max() == 0`

## Opt-in

- `baseline_glue_v2_candidate_context()` → `preallocated_batch1_state` + `pinned_eos_flag` + `sampling_tail_cuda_graph`
- V32 cold default unchanged; production engine kwargs unchanged until graduate

## Gates

| Gate | Bar |
| --- | --- |
| Smoke | token IDs + RNG equal; sampling-tail hits > 0 |
| Kill | cannot show ≥**0.15 ms/tok** glue cut with persistent caches |
| Scout | e2e ≥**450** TPS; if bitwise holds → EXACT tip graduate candidate |

## Scripts

- Smoke: `scripts/dcc/profile_baseline_glue_v2_smoke.sbatch` (`profile_salvalai_smoke15`)
- Scout: `scripts/dcc/profile_baseline_glue_v2_scout.sbatch` (`profile_salvalai`)
- Runner: `utils/run_exact_baseline_glue_v2.py` (tip composition baseline = shared-RoPE + device-state)

Unique `TMPDIR` / `TORCH_EXTENSIONS_DIR` per job id. Prefer serial GPU vs turbo workers (≤2 total).

## Campaign tip

Unchanged until scout passes: `55949274` / **366.11**. No 500 claim. No merge.
