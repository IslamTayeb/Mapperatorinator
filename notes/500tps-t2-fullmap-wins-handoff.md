# T2 Certain Full-Map Wins — handoff

**Status:** **WARMUP-HOIST SEALED — PROMOTE CLEAN Y** (2026-07-18)  
**Package:** Pivot execution **T2**  
**Branch / WT:** `codex/t2-fullmap-wins` @ **`f7681342`**  
**Local WT:** `/work/projects/Mapperatorinator-worktrees/t2-fullmap-wins`  
**DCC WT:** `/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/t2-fullmap-wins`  
**Base:** tiger PR #120 mirror `d01cdd27`  
**Optimized tip FROZEN:** `55949274` / FP16 **366.11** — **do not modify**; **no merge**; **no push to PR #120**  
**T4 turbo:** **PARKED** — not wired  
**Sibling:** T1 `50194265` PD (2080); T3 A5000 baseline `50194985` OK / compile `50194986` FAILED (tmpdir race)

## Intent

Cut full-map **ms/map-token** on tiger-base before compile (T3):

1. **Hoist warmup/captures** (`session_warmup_captures`) — encoder + CUDA-graph buckets after load (cold_start).
2. **Encoder-precompute dedupe** (`encoder_precompute_dedupe`) — process cache by `(model id, frames fingerprint, cond)`.
3. **Timing-stride tune** (`timing_lookback=0.3`, `timing_lookahead=0.3`) — fewer timing windows — **OPT-IN only**.

**Gates:** real full-map wall improvement; **no silent fast-path latch**; T5 greedy PASS for warmup-hoist clean promote.

## Jobs

### Bundle (all three levers) — A5000 fp16 @ `f171f0b3`

| Variant | Job | State | Artifact |
| --- | --- | --- | --- |
| baseline | **`50194534`** | COMPLETED 0:0 00:01:00 | `/work/imt11/Mapperatorinator/runs/t2-fullmap-baseline-fp16-50194534/` |
| t2 (all three) | **`50194535`** | COMPLETED 0:0 00:01:01 | `/work/imt11/Mapperatorinator/runs/t2-fullmap-t2-fp16-50194535/` |

### Warmup-hoist isolate + T5 greedy — A5000 fp16 @ `f7681342` (1 GPU sequential)

| Cell | Job | State | Artifact |
| --- | --- | --- | --- |
| baseline+t2_warmup measure + greedy seal | **`50195943`** | COMPLETED 0:0 00:03:46 | `/work/imt11/Mapperatorinator/runs/t2-warmup-isolate-fp16-50195943/` |

Harness: `scripts/dcc/t2_warmup_isolate.sbatch` + `t2_warmup_isolate_cell.py` (variant `t2_warmup` = `session_warmup_captures` only).

## Results

### Bundle (timing-stride ON) — prior

| Variant | ms/map-token | exclude wall (s) | main_tps | cold_start (s) | map_tok | timing |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **baseline** | **3.741** | 28.15 | **344.7** | 21.40 | 7526 | 3.76 s / 821 tok / 87 win |
| **t2** (all three) | **3.296** | 26.87 | **356.7** | 24.00 | **8151** | 2.38 s / 777 tok / **23** win |
| Δ | **−11.9%** | **−4.6%** | +3.5% | +2.60 | **+625** | stride engaged |

### Warmup-hoist only (timing-stride OFF) — seal @ `50195943`

| Variant | ms/map-token | exclude wall (s) | main_tps | cold_start (s) | map_tok |
| --- | ---: | ---: | ---: | ---: | ---: |
| **baseline** (measure) | **3.638** | 27.38 | **351.0** | 23.16 | **7526** |
| **t2_warmup** (measure) | **3.618** | 27.23 | **348.5** | 20.94 | **7526** |
| Δ vs baseline | **−0.55%** | **−0.55%** | −0.7% | −2.22 | **0** |

| T5 gate | Result |
| --- | --- |
| greedy `.osu` bytes (`do_sample=false`) | **PASS** (31418 B; map_tok 8394=8394) |
| ks_parity | **SKIP** (warmup isolate; stride held) |
| `t5_quality_gates_overall` | **PASS** |
| `promote_clean` | **Y** |

## Gate check

| Gate | Result |
| --- | --- |
| No silent fast-path latch | **PASS** (SDPA; fallback unset) |
| map_tok preserved (warmup-only) | **PASS** (7526→7526 measure; 8394→8394 greedy) |
| T5 greedy overall | **PASS** |
| Bundle −~23% package claim | **MISS** (−11.9% with stride; warmup-alone ≈noise) |
| Warmup-hoist clean promote | **PASS** |

## Promote

**Y (clean) for `session_warmup_captures` only** — T5 greedy PASS @ `50195943`; map_tok preserved.

- **Ship default-on:** `session_warmup_captures` (exactness sealed; wall Δ on this single song is noise-scale −0.55%; prior bundle wall win was mostly timing-stride).
- **Keep OPT-IN:** `timing_lookback/lookahead` — known map_tok drift 7526→8151; needs T5 KS PASS before default-on.
- **Keep:** `encoder_precompute_dedupe` for same-model same-stride; no win on W1-style dual-model path.

## Sibling harvest (≤2 GPU this follow-up: **1** A5000 used)

| Track | Job(s) | State | Note |
| --- | --- | --- | --- |
| T1 rails | `50194265` | PENDING (2080) | Priority queue |
| T3 baseline A5000 | `50194985` | COMPLETED | ms/map-token **3.721** / main_tps 347.3 |
| T3 compile A5000 | `50194986` | FAILED | Inductor tempfile `ENOTEMPTY` during capture warm |
| T3 retry | `50196040` (comp A5000 RUNNING); `50196041`/`50196042` (2080 PD); `50196043` (greedy PD) | live | prior `50195633–35` CANCELLED |

## Do-not

- Push to Tiger14n / PR #120  
- Wire turbo / T4  
- Modify tip `55949274`  
- Claim 500 / merge to main / claim −23% sealed  
- Default-on timing-stride without T5 KS  
- Silent capture latch  

## Ruling

Warmup-hoist alone is **exactness-clean** (T5 greedy PASS) with **no token drift**. Prior −11.9% was timing-stride–dominated and must stay opt-in. Next = T3 compile-then-capture (fix `50194986` tmpdir flake) with T5 greedy vs uncompiled.
