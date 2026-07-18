# T2 Certain Full-Map Wins — handoff

**Status:** **WIRED — MEASURE PENDING** (2026-07-18)  
**Package:** Pivot execution **T2** (−~23% full-map before compile)  
**Branch / WT:** `codex/t2-fullmap-wins` @ *(fill after commit)*  
**Local WT:** `/work/projects/Mapperatorinator-worktrees/t2-fullmap-wins`  
**DCC WT:** `/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/t2-fullmap-wins`  
**Base:** tiger PR #120 mirror `d01cdd27`  
**Optimized tip FROZEN:** `55949274` / FP16 **366.11** — **do not modify**; **no merge**; **no push to PR #120**  
**T4 turbo:** **PARKED** (§57b E≪2.2) — not wired  
**Sibling T1:** rails may share `codex/turbo-on-tiger-pr120`; this track is a **clean tiger branch** (no turbo)

## Intent

Cut full-map **ms/map-token** on tiger-base before any torch.compile (T3):

1. **Hoist warmup/captures** (`session_warmup_captures`) — encoder cudnn + common CUDA-graph buckets run after model load (cold_start), not inside measured map.
2. **Encoder-precompute dedupe** (`encoder_precompute_dedupe`) — process-wide cache keyed by `(model id, frames fingerprint, cond)`; same-model timing→map with matching windows reuses hidden states.
3. **Timing-stride tune** (`timing_lookback` / `timing_lookahead`) — coarser timing windows (candidate: 0.3/0.3 → stride 0.4 vs default 0.1).

**Gate:** real full-map wall improvement; **no silent fast-path latch** (capture failure raises unless `MAPPERATORINATOR_ALLOW_CAPTURE_FALLBACK=1`; jobs force SDPA).

## Baseline (W1 A5000 tiger fp16 `50181770`)

| Metric | Value |
| --- | ---: |
| exclude wall | **30.85 s** |
| map tokens | **7526** |
| **ms/map-token** | **4.10** |
| main (map+sv) | 21.51 s / ~**349.9 TPS** |
| timing | 4.31 s (~14%) |
| cold_start proxy | 142.2 s |

## Candidate flags

```
session_warmup_captures=true
encoder_precompute_dedupe=true
timing_lookback=0.3
timing_lookahead=0.3
fast_decoder_loop=true
attn_implementation=sdpa
```

## Jobs

| Variant | Job | Artifact |
| --- | --- | --- |
| baseline | *(fill)* | `/work/imt11/Mapperatorinator/runs/t2-fullmap-baseline-fp16-<job>/` |
| t2 | *(fill)* | `/work/imt11/Mapperatorinator/runs/t2-fullmap-t2-fp16-<job>/` |

Harness: `scripts/dcc/t2_fullmap_cell.py` + `scripts/dcc/t2_fullmap_wins.sbatch`

## Results

| Variant | ms/map-token | main_tps | cold_start_s | Decision |
| --- | ---: | ---: | ---: | --- |
| baseline | | | | |
| t2 | | | | |

**Promote Y/N:** *(fill after harvest)*

## Do-not

- Push to Tiger14n / PR #120  
- Wire turbo / T4  
- Modify tip `55949274`  
- Claim 500 / merge to main  
- Silent capture latch  
- Exceed ≤2 concurrent GPU (Ada W1 cells are PD only)

## Ruling

T2 is the fastest path to user-felt full-map wins on tiger-base. Promote only on measured ms/map-token improvement vs like-with-like baseline on A5000; report main_tps + cold_start alongside.
