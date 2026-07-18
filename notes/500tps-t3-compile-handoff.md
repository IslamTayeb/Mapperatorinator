# T3 TORCH.COMPILE — handoff (PIVOT EXECUTION PACKAGE)

**Status:** **WIRED — MEASURE PENDING** (2026-07-18)  
**Package:** Pivot **T3** compile-then-capture  
**Branch / WT:** `codex/t3-compile-then-capture` @ *(fill after commit)*  
**Local WT:** `/work/projects/Mapperatorinator-worktrees/t3-compile-then-capture`  
**DCC WT:** `/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/t3-compile-then-capture`  
**Base:** `codex/turbo-on-tiger-pr120` @ `b96c3e38` (tiger PR #120 `d01cdd27` + §58/§59 rails)  
**Frozen tip:** `55949274` / FP16 **366.11** — **regression reference only**; **no merge**; **no push to PR #120**  
**T4 turbo:** **PARKED** — do not wire speculative

## Binding pattern

Compile-then-capture decoder step **per bucket**:

| Knob | Value |
| --- | --- |
| `fullgraph` | `True` |
| `dynamic` | `False` |
| `mode` | `max-autotune-no-cudagraphs` |
| Warm | **EVERY** bucket (incl. any future turbo `q_len`) **before** its capture — §22 Inductor-in-capture |
| Sampling tail | shape-static mono hoist + uniform temperature (`TemperatureLogitsWarper`) |
| Forbidden | `reduce-overhead` near manual CUDAGraph capture |

Opt-in env:

- `MAPPERATORINATOR_COMPILE_DECODE=1` — Inductor compile of decode step + sampling tail
- `MAPPERATORINATOR_WARM_ALL_BUCKETS=1` — session warm all buckets in cold_start (also auto when compile env set)

Cold start:

- Pin `TORCHINDUCTOR_CACHE_DIR` per job/install (sbatch sets unique dir)
- Unique `TMPDIR` + `TORCH_EXTENSIONS_DIR` per job (prefer `$SLURM_TMPDIR` when present)
- Optional Mega-Cache: reuse `TORCHINDUCTOR_CACHE_DIR` across jobs on same arch/driver only — **do not ship** arch/driver-bound artifacts
- Regional / `default` compile fallback if fullgraph/`max-autotune-no-cudagraphs` fails; then plain capture; never silent latch

### Windows ladder (document)

1. **triton-windows + torch≥2.10** → compile-then-capture (this path)
2. **no triton** → plain CUDA graphs (current PR #120 default)
3. **no capture** → eager / stock generate (loud failure unless `MAPPERATORINATOR_ALLOW_CAPTURE_FALLBACK=1`)

## Gates

| Gate | Criterion |
| --- | --- |
| A5000 | main-gen **+≥10%** vs like-with-like uncompiled fast path |
| 2080 Ti | **no-regression** (main_tps / ms/map-token) |
| Exactness | greedy token-match (`.osu` bytes) vs uncompiled fast path |
| Report | full-map **ms/map-token** + **main_tps** + **cold_start** |

## Baseline reference (W1 A5000 tiger fp16 `50181770`)

| Metric | Value |
| --- | ---: |
| ms/map-token | **4.10** |
| main | ~**349.9 TPS** |
| cold_start proxy | 142.2 s |

## Jobs

| Cell | Job | GPU | State | Artifact |
| --- | --- | --- | --- | --- |
| baseline A5000 | *(pending)* | a5000 | — | `/work/imt11/Mapperatorinator/runs/t3-compile-baseline-fp16-<job>/` |
| compile A5000 | *(pending)* | a5000 | — | `/work/imt11/Mapperatorinator/runs/t3-compile-compile-fp16-<job>/` |
| baseline 2080 | *(pending)* | 2080 | — | … |
| compile 2080 | *(pending)* | 2080 | — | … |
| greedy match | *(pending)* | a5000 | — | `/work/imt11/Mapperatorinator/runs/t3-greedy-match-<job>/` |

Harness: `scripts/dcc/t3_compile_cell.py` + `t3_compile.sbatch` · `t3_greedy_match_cell.py` + `t3_greedy_match.sbatch`

## Results

| GPU | Variant | ms/map-token | main_tps | cold_start_s | Δ main_tps |
| --- | --- | ---: | ---: | ---: | ---: |
| A5000 | baseline | — | — | — | — |
| A5000 | compile | — | — | — | — |
| 2080 | baseline | — | — | — | — |
| 2080 | compile | — | — | — | — |

**Promote Y/N:** **pending** (need sealed A5000 +≥10% AND 2080 no-regress AND greedy PASS)

## Do-not

- Push to Tiger14n / PR #120  
- Wire T4 turbo / speculative  
- Modify tip `55949274`  
- Claim 500 / merge to main  
- Use `reduce-overhead` near manual capture  
- Exceed ≤2 concurrent GPU with live T1/T2  

## Ruling

T3 is the settled compile-then-capture integration (vLLM + §3/§22 + Tiger experiment +30–45% prior). Promote only on sealed gates above.
