# T3 TORCH.COMPILE — handoff (PIVOT EXECUTION PACKAGE)

**Status:** **FIX WIRED — A5000 RE-MEASURE QUEUED** (2026-07-18)  
**Package:** Pivot **T3** compile-then-capture  
**Branch / WT:** `codex/t3-compile-then-capture` @ `eb85f4b3`  
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
| A5000 | main-gen **+≥10%** vs like-with-like uncompiled fast path | pending re-measure |
| 2080 Ti | **no-regression** (main_tps / ms/map-token) | pending (after A5000+greedy) |
| Exactness | greedy token-match (`.osu` bytes) vs uncompiled fast path | pending re-measure |
| Report | full-map **ms/map-token** + **main_tps** + **cold_start** | pending |

## Fix (after sealed FAIL @ `28ae22c6`)

**Root cause (harvest 1):** Inductor-compiled `_tail` specialized on warm `(B,V)` strides (`stride0≈V`). Production B=1 logits views keep million-scale `stride0` where `.contiguous()` is a no-op → Dynamo `recompile_limit` thrash → −46% main_tps + greedy `.osu` drift.

**Fix:** drop Inductor compile of sampling `_tail`; keep decode compile-then-capture + eager mono+temp hoist. Do not reintroduce compiled `_tail` without a fixed-stride staging buffer *outside* Dynamo. Kill if second FAIL is the same stride/recompile storm.

## Jobs

### Harvest 1 (sealed PROMOTE N @ `28ae22c6`)

| Cell | Job | GPU | State | Artifact |
| --- | --- | ---: | --- | --- |
| baseline A5000 | **50194985** | a5000 | **COMPLETED 0:0** | `/work/imt11/Mapperatorinator/runs/t3-compile-baseline-fp16-50194985/` |
| compile A5000 | **50194986** | a5000 | **FAILED** NFS Triton tempfile | superseded |
| compile A5000 (bad retry) | **50195632** | a5000 | **FAILED 1:0** | NFS/`ENOTEMPTY`-class |
| baseline/comp/greedy (v1 deps) | **50195633–35** | — | **CANCELLED** | after 50195632 fail |
| compile A5000 | **50196040** | a5000 | **COMPLETED 0:0** | `/work/imt11/Mapperatorinator/runs/t3-compile-compile-fp16-50196040/` |
| baseline 2080 | **50196041** | 2080 | **CANCELLED** | after seal N |
| compile 2080 | **50196042** | 2080 | **CANCELLED** | after seal N |
| greedy match | **50196043** | a5000 | **FAILED 1:0** (exactness) | `/work/imt11/Mapperatorinator/runs/t3-greedy-match-50196043/` |

| GPU | Variant | Job | ms/map-token | main_tps | cold_start_s | Δ main_tps |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A5000 | baseline | 50194985 | **3.721** | **347.30** | **23.40** | — |
| A5000 | compile | 50196040 | **6.851** | **186.23** | **436.85** | **−46.4%** |

**Greedy (50196043):** **FAIL** — baseline 31418 vs compile 26464 bytes. **Promote N.**

### Harvest 2 (eager-tail fix @ `eb85f4b3`)

| Cell | Job | GPU | State | Artifact |
| --- | --- | ---: | --- | --- |
| baseline A5000 | **50196882** | a5000 | PENDING | `/work/imt11/Mapperatorinator/runs/t3-compile-baseline-fp16-50196882/` |
| compile A5000 | **50196883** | a5000 | PENDING | `/work/imt11/Mapperatorinator/runs/t3-compile-compile-fp16-50196883/` |
| greedy match | **50196884** | a5000 | Dependency afterok:50196883 | `/work/imt11/Mapperatorinator/runs/t3-greedy-match-50196884/` |
| baseline/comp 2080 | *(after A5000+greedy PASS)* | 2080 | — | … |

Local pulls: `notes/t3-artifacts/`  
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

**Promote N** until harvest 2 seals A5000 +≥10% and greedy PASS. T4 stays PARKED. No PR #120 push.
