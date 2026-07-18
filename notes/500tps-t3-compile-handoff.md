# T3 TORCH.COMPILE — handoff (PIVOT EXECUTION PACKAGE)

**Status:** **SEALED PROMOTE N** (2026-07-18 harvest 2)  
**Package:** Pivot **T3** compile-then-capture  
**Branch / WT:** `codex/t3-compile-then-capture` @ `eb85f4b3` (eager-tail fix; handoff tip may lag)  
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
| Sampling tail | shape-static mono hoist + uniform temperature (**eager** — not Inductor) |
| Forbidden | `reduce-overhead` near manual CUDAGraph capture |

Opt-in env:

- `MAPPERATORINATOR_COMPILE_DECODE=1` — Inductor compile of **decode step only** (tail stays eager)
- `MAPPERATORINATOR_WARM_ALL_BUCKETS=1` — session warm all buckets in cold_start (also auto when compile env set)

Cold start:

- Pin `TORCHINDUCTOR_CACHE_DIR` per job/install (sbatch sets unique dir)
- Unique **node-local** `TMPDIR`/`TEMP`/`TMP` (`$SLURM_TMPDIR` or `/tmp/$USER-…`) + unique `TORCH_EXTENSIONS_DIR`
- Optional Mega-Cache: reuse `TORCHINDUCTOR_CACHE_DIR` across jobs on same arch/driver only — **do not ship** arch/driver-bound artifacts
- Regional / `default` compile fallback if fullgraph/`max-autotune-no-cudagraphs` fails; warmup failure → eager capture; never silent latch

### Windows ladder (document)

1. **triton-windows + torch≥2.10** → compile-then-capture (this path)
2. **no triton** → plain CUDA graphs (current PR #120 default)
3. **no capture** → eager / stock generate (loud failure unless `MAPPERATORINATOR_ALLOW_CAPTURE_FALLBACK=1`)

## Gates

| Gate | Criterion | Result |
| --- | --- | --- |
| A5000 | main-gen **+≥10%** vs like-with-like uncompiled fast path | **PASS** (+28.8% main_tps) |
| 2080 Ti | **no-regression** (main_tps / ms/map-token) | **N/A** (cancelled after greedy seal N) |
| Exactness | greedy token-match (`.osu` bytes) vs uncompiled fast path | **FAIL** (31418 vs 26464) |
| Report | full-map **ms/map-token** + **main_tps** + **cold_start** | sealed below |

## Fix history

### Harvest 1 FAIL @ `28ae22c6`

**Root cause:** Inductor-compiled `_tail` specialized on warm `(B,V)` strides (`stride0≈V`). Production B=1 logits views keep million-scale `stride0` where `.contiguous()` is a no-op → Dynamo `recompile_limit` thrash → −46% main_tps + greedy drift.

### Harvest 2 fix @ `eb85f4b3`

**Change:** drop Inductor compile of sampling `_tail`; keep decode compile-then-capture + eager mono+temp hoist.

**Outcome:** stride/recompile storm **cleared** (no Dynamo stride warnings; A5000 main_tps **+28.8%**). Greedy still **FAIL** with **identical** byte split to harvest 1 (31418 vs 26464) → remaining blocker is **decode-compile exactness**, not `_tail` stride thrash. **Not** kill-same-root-cause (perf root fixed). Promote still **N** on exactness.

**Revisit:** isolate greedy token divergence under decode Inductor (numerical / hoist / warm-all-buckets) without reintroducing compiled `_tail`. Kill only if next FAIL is again `_tail` stride/recompile storm.

## Jobs

### Harvest 1 (sealed PROMOTE N @ `28ae22c6`)

| Cell | Job | GPU | State | Artifact |
| --- | --- | ---: | --- | --- |
| baseline A5000 | **50194985** | a5000 | **COMPLETED 0:0** | `/work/imt11/Mapperatorinator/runs/t3-compile-baseline-fp16-50194985/` |
| compile A5000 | **50196040** | a5000 | **COMPLETED 0:0** | `/work/imt11/Mapperatorinator/runs/t3-compile-compile-fp16-50196040/` |
| greedy match | **50196043** | a5000 | **FAILED 1:0** | `/work/imt11/Mapperatorinator/runs/t3-greedy-match-50196043/` |

| GPU | Variant | Job | ms/map-token | main_tps | cold_start_s | Δ main_tps |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A5000 | baseline | 50194985 | **3.721** | **347.30** | **23.40** | — |
| A5000 | compile | 50196040 | **6.851** | **186.23** | **436.85** | **−46.4%** |

### Harvest 2 (eager-tail fix @ `eb85f4b3`)

| Cell | Job | GPU | State | Artifact |
| --- | --- | ---: | --- | --- |
| baseline A5000 | **50196882** | a5000 | **COMPLETED 0:0** | `/work/imt11/Mapperatorinator/runs/t3-compile-baseline-fp16-50196882/` |
| compile A5000 | **50196883** | a5000 | **COMPLETED 0:0** | `/work/imt11/Mapperatorinator/runs/t3-compile-compile-fp16-50196883/` |
| greedy match | **50196884** | a5000 | **FAILED 1:0** (exactness) | `/work/imt11/Mapperatorinator/runs/t3-greedy-match-50196884/` |
| baseline/comp 2080 | — | 2080 | **not submitted** | after seal N |

| GPU | Variant | Job | ms/map-token | main_tps | cold_start_s | Δ main_tps |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A5000 | baseline | 50196882 | **3.809** | **343.30** | **26.03** | — |
| A5000 | compile | 50196883 | **3.147** | **442.20** | **417.37** | **+28.8%** |

**Greedy (50196884):** **FAIL** — `.osu` bytes unequal (baseline 31418 vs compile 26464); same split as 50196043.

Local pulls: `notes/t3-artifacts/summary-50196882.json`, `summary-50196883.json`, `match-50196884.json`  
Remote: `islamtayeb/codex/t3-compile-then-capture` only — **no tiger14n / PR #120**

## Do-not

- Push to Tiger14n / PR #120  
- Wire T4 turbo / speculative  
- Modify tip `55949274`  
- Claim 500 / merge to main  
- Use `reduce-overhead` near manual capture  
- Exceed ≤2 concurrent GPU with live T1/T2  
- Put Triton `TMPDIR` on NFS `/work`  
- Recompile sampling `_tail` without fixed-stride staging outside Dynamo  

## Ruling

**Promote N.** Harvest 2 clears the `_tail` stride/recompile storm and passes A5000 +≥10% main-gen (**+28.8%**), but greedy `.osu` exactness still fails (same 31418/26464 split). 2080 not run. T4 stays PARKED. No PR #120 push.
