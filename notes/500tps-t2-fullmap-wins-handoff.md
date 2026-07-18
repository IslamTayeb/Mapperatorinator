# T2 Certain Full-Map Wins — handoff

**Status:** **MEASURED — PROMOTE Y (partial; −23% not sealed)** (2026-07-18)  
**Package:** Pivot execution **T2**  
**Branch / WT:** `codex/t2-fullmap-wins` @ **`f171f0b3`**  
**Local WT:** `/work/projects/Mapperatorinator-worktrees/t2-fullmap-wins`  
**DCC WT:** `/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/t2-fullmap-wins`  
**Base:** tiger PR #120 mirror `d01cdd27`  
**Optimized tip FROZEN:** `55949274` / FP16 **366.11** — **do not modify**; **no merge**; **no push to PR #120**  
**T4 turbo:** **PARKED** — not wired  
**Sibling T1:** `s59-t1-rails` `50194265` PD (separate); this track is clean tiger branch

## Intent

Cut full-map **ms/map-token** on tiger-base before compile (T3):

1. **Hoist warmup/captures** (`session_warmup_captures`) — encoder + CUDA-graph buckets after load (cold_start).
2. **Encoder-precompute dedupe** (`encoder_precompute_dedupe`) — process cache by `(model id, frames fingerprint, cond)`.
3. **Timing-stride tune** (`timing_lookback=0.3`, `timing_lookahead=0.3`) — fewer timing windows.

**Gates:** real full-map wall improvement; **no silent fast-path latch** (capture raises unless `MAPPERATORINATOR_ALLOW_CAPTURE_FALLBACK=1`; jobs force SDPA).

## Jobs (authoritative A5000 fp16)

| Variant | Job | State | Artifact |
| --- | --- | --- | --- |
| baseline | **`50194534`** | COMPLETED 0:0 00:01:00 | `/work/imt11/Mapperatorinator/runs/t2-fullmap-baseline-fp16-50194534/` |
| t2 | **`50194535`** | COMPLETED 0:0 00:01:01 | `/work/imt11/Mapperatorinator/runs/t2-fullmap-t2-fp16-50194535/` |

Infra fails (ffprobe PATH): `50194016`/`50194017`, `50194393`/`50194394` — fixed in `f171f0b3` (`PATH="$(dirname "$PYTHON"):$PATH"`). Cancelled dupes `50194407`/`50194408`.

## Results (like-with-like @ `f171f0b3`, A5000, SALVALAI)

| Variant | ms/map-token | exclude wall (s) | main_tps | cold_start (s) | map_tok | timing |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **baseline** (T2 flags off) | **3.741** | 28.15 | **344.7** | 21.40 | 7526 | 3.76 s / 821 tok / 87 win |
| **t2** (all three on) | **3.296** | 26.87 | **356.7** | 24.00 | 8151 | 2.38 s / 777 tok / **23** win |
| Δ vs baseline | **−11.9%** | **−4.6%** | +3.5% | +2.60 (warmup) | +625 | stride engaged |
| W1 ref `50181770` | **4.10** | 30.85 | ~349.9 | 142.2 | 7526 | — |

Stdout evidence (t2): session warmup on map+timing models; `Timing-stride tune: … 23 timing windows (map uses 87)`. Encoder **dedupe did not hit** (separate timing vs gamemode models + different window counts).

## Gate check

| Gate | Result |
| --- | --- |
| Real full-map wall improvement | **PASS** (−4.6% exclude wall) |
| No silent fast-path latch | **PASS** (SDPA; fallback unset; both completed with graphs) |
| −~23% package claim | **MISS** (−11.9% ms/map-token like-with-like; −19.5% vs cold W1 only) |

## Promote

**Y (partial).** Land opt-in T2 flags on tiger-base mirror; do **not** advertise −23% sealed.

- **Ship:** `session_warmup_captures` (cold_start tax ~+2.6 s; measured wall win).
- **Ship opt-in with T5 follow-up:** `timing_lookback/lookahead` — changes timing → map token count (7526→8151); needs quality/token-match gate before default-on.
- **Keep:** `encoder_precompute_dedupe` for same-model same-stride; no win on W1-style dual-model path.

## Do-not

- Push to Tiger14n / PR #120  
- Wire turbo / T4  
- Modify tip `55949274`  
- Claim 500 / merge to main / claim −23% sealed  
- Silent capture latch  

## Ruling

T2 delivers real full-map wall + ms/map-token gains on A5000 without silent latch. Package −23% not sealed; next = tighten timing-stride under T5, then T3 compile-then-capture.
