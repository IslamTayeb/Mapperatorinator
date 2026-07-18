# T3 TORCH.COMPILE — handoff (PIVOT EXECUTION PACKAGE)

**Status:** **HARVEST 4 IN PROGRESS** — owned sub-op compile revisit (2026-07-18 relaunch)  
**Prior sealed:** Harvest 3 **PROMOTE N / STOP** @ `3e0aacb7` / `502d0a65` (full-step Inductor greedy FAIL)  
**Package:** Pivot **T3** compile-then-capture  
**Branch / WT:** `codex/t3-compile-then-capture` @ `e407b255`  
**Local WT:** `/work/projects/Mapperatorinator-worktrees/t3-compile-then-capture`  
**DCC WT:** `/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/t3-compile-then-capture`  
**Base:** `codex/turbo-on-tiger-pr120` @ `b96c3e38` (tiger PR #120 `d01cdd27` + §58/§59 rails)  
**Frozen tip:** `55949274` / FP16 **366.11** — **regression reference only**; **no merge**; **no push to PR #120**  
**T4 turbo:** **PARKED** — do not wire speculative  
**T1 smoke:** PASS (`50215788`) — rails green; T3 may proceed

## Harvest 4 hypothesis (NEW — not bare full-step retry)

Full-step Inductor rewrite of shape-static `forward_only` flips fp16 near-tie greedy tokens (harvest 1–3 sealed). Revisit compiles **only tip-owned Linear / post-attn FFN / `proj_out` modules**, then captures the still-eager outer decode step (compile-then-capture discipline preserved). Attention / SDPA / cache path stay eager.

| Knob | Harvest 4 value |
| --- | --- |
| Outer step | **eager** `forward_only` (never Inductor full-step) |
| Sub-ops | `MAPPERATORINATOR_COMPILE_SUBOPS=proj_out,ffn` (default when decode compile on) |
| `fullgraph` / `dynamic` / `mode` | `True` / `False` / **`default`** |
| Warm | EVERY bucket before capture; warm compiled modules outside graph |
| Forbidden | full-step compile; `reduce-overhead`; recompile `_tail` |

Opt-in env:

- `MAPPERATORINATOR_COMPILE_DECODE=1` — enable T3 compile path (now **sub-op**, not full-step)
- `MAPPERATORINATOR_COMPILE_SUBOPS` — comma list: `proj_out`, `ffn`/`post_attn`, `linear` (=both). Default `proj_out,ffn`
- `MAPPERATORINATOR_COMPILE_FULL_STEP=1` — **forbidden for package gate** (harvest 1–3 STOP); loud refuse unless explicitly documenting drift
- `MAPPERATORINATOR_WARM_ALL_BUCKETS=1` — session warm all buckets
- `MAPPERATORINATOR_COMPILE_MODE` — default `default`

## Progress (relaunch)

| Time (UTC) | Note |
| --- | --- |
| 2026-07-18 ~19:20 | Relaunch after orphaned agent; tip frozen; T4 PARKED; T1 smoke PASS |
| 2026-07-18 ~20:48 | Code @ `1e3fc081` pushed to `islamtayeb/codex/t3-compile-then-capture` (owned `proj_out,ffn`; refuse full-step) |
| 2026-07-18 ~20:49 | DCC pull FF; submitted A5000 pair (queue Priority) |
| 2026-07-18 ~20:50 | Greedy **50228096** submitted with `afterany:50228030:50228031` (≤2 GPU while pair runs). Dupes 50228032/33 cancelled. |
| — | Waiting on Priority queue: **50228030** baseline, **50228031** compile, **50228096** greedy |

## Gates (harvest 4)

| Gate | Criterion | Result |
| --- | --- | --- |
| A5000 | main-gen **+≥10%** vs like-with-like uncompiled fast path | **PENDING** |
| Exactness | greedy token-match (`.osu` bytes) vs uncompiled fast path | **PENDING** |
| 2080 Ti | **no-regression** | **N/A until greedy PASS** |

## Binding pattern (harvest 1–3, sealed)

Compile-then-capture decoder step **per bucket** — harvest 3 used **full** Inductor `forward_only` → STOP.

| Knob | Value |
| --- | --- |
| `fullgraph` | `True` |
| `dynamic` | `False` |
| `mode` | **`default`** (harvest 3); override `MAPPERATORINATOR_COMPILE_MODE` (never `reduce-overhead`) |
| Warm | **EVERY** bucket before its capture — §22 Inductor-in-capture |
| Sampling tail | shape-static mono hoist + uniform temperature (**eager**, both compile on/off) |
| Forbidden | `reduce-overhead` near manual CUDAGraph capture |

Cold start:

- Pin `TORCHINDUCTOR_CACHE_DIR` per job/install (sbatch sets unique dir)
- Unique **node-local** `TMPDIR`/`TEMP`/`TMP` (`$SLURM_TMPDIR` or `/tmp/$USER-…`) + unique `TORCH_EXTENSIONS_DIR`
- Optional Mega-Cache: reuse `TORCHINDUCTOR_CACHE_DIR` across jobs on same arch/driver only — **do not ship** arch/driver-bound artifacts
- Regional / `default` compile fallback if fullgraph fails; warmup failure → eager capture; never silent latch

### Windows ladder (document)

1. **triton-windows + torch≥2.10** → compile-then-capture (this path)
2. **no triton** → plain CUDA graphs (current PR #120 default)
3. **no capture** → eager / stock generate (loud failure unless `MAPPERATORINATOR_ALLOW_CAPTURE_FALLBACK=1`)

## Root cause (harvest 3) — sealed

**Decode-step Inductor numerics under compile-then-capture**, not RNG / warm-all / `_tail` / compile-only hoist:

| Ruled out | Evidence |
| --- | --- |
| RNG order | greedy `do_sample=false` → `argmax` |
| Warm-all-buckets alone | T2 sealed greedy PASS for warmup-hoist |
| Compiled `_tail` stride thrash | harvest 2 cleared; same 31418/26464 split remained under max-autotune |
| Compile-only temp hoist | harvest 3 ungates hoist on both paths; baseline still 31418 |
| Missing warm bucket | both variants warm; coherent maps (not garbage) |

**In:** Inductor rewrite of the shape-static decode forward (fp16 GEMM/epilogue/fusion) flips near-tie greedy tokens; cascades into divergent `.osu`. First byte diffs land in TimingPoints/SV lines (harvest 2 byte 1426; harvest 3 byte 2080).

- **max-autotune** (harvest 1/2): stderr showed `addmm`/`bias_addmm` autotune + “Not enough SMs for max_autotune_gemm”; compile `.osu` **26464** vs baseline **31418**.
- **default mode** (harvest 3): removes most autotune shopping; compile `.osu` moves to **31747** (closer, still ≠ 31418). Perf retained **+22.7%**.

## Fix history

### Harvest 1 FAIL @ `28ae22c6`

**Root cause:** Inductor-compiled `_tail` specialized on warm `(B,V)` strides (`stride0≈V`). Production B=1 logits views keep million-scale `stride0` where `.contiguous()` is a no-op → Dynamo `recompile_limit` thrash → −46% main_tps + greedy drift.

### Harvest 2 fix @ `eb85f4b3`

**Change:** drop Inductor compile of sampling `_tail`; keep decode compile-then-capture + eager mono+temp hoist.

**Outcome:** stride/recompile storm **cleared**; A5000 main_tps **+28.8%**. Greedy still **FAIL** 31418 vs 26464.

### Harvest 3 fix @ `3e0aacb7` — **STOP**

**Change:**

1. Default `torch.compile` mode → **`default`** (env override for max-autotune retained but not package-default).
2. Pin `allow_fp16/bf16_reduced_precision_reduction=False` at compile.
3. Hoist mono+temp on **both** compile on/off paths (exactness diffs isolate decode Inductor).

**Outcome:** A5000 perf still **+22.7%** (gate PASS). Greedy still **FAIL** 31418 vs 31747. **One corrective scout consumed → STOP.**

**Revisit (new hypothesis required — do not bare-retry full decode-step compile):**

- Owned tip-exact sub-op compile (e.g. decode-only `proj_out` / attn GEMV) before outer capture — not full `forward_only`. ← **harvest 4**
- Or teacher-forced first-N-token logit dump (eager graph vs compiled graph) to localize the diverging op.
- Or documented opt-in compile with **declared** greedy drift (outside package promote gate).
- Kill if next attempt reintroduces `_tail` stride thrash or max-autotune without a new exactness plan.

### Harvest 4 (owned sub-op) — **IN PROGRESS**

**Change (WIP):** when `MAPPERATORINATOR_COMPILE_DECODE=1`, Inductor-compile only owned modules (`proj_out`, decoder `fc1`/`fc2` FFN linears); outer CUDAGraph captures eager `forward_only`. Refuse `MAPPERATORINATOR_COMPILE_FULL_STEP=1` for package runs.

**Jobs:** baseline **50228030** + compile **50228031** queued (Priority); greedy after ≤1 free GPU.

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

| GPU | Variant | Job | ms/map-token | main_tps | cold_start_s | Δ main_tps |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A5000 | baseline | 50196882 | **3.809** | **343.30** | **26.03** | — |
| A5000 | compile | 50196883 | **3.147** | **442.20** | **417.37** | **+28.8%** |

**Greedy (50196884):** **FAIL** — baseline 31418 vs compile 26464.

### Harvest 3 (default-mode @ `3e0aacb7`) — **STOP**

| Cell | Job | GPU | State | Artifact |
| --- | --- | ---: | --- | --- |
| greedy match | **50203099** | a5000 | **FAILED 1:0** (exactness) | `/work/imt11/Mapperatorinator/runs/t3-greedy-match-50203099/` |
| compile A5000 | **50203100** | a5000 | **COMPLETED 0:0** | `/work/imt11/Mapperatorinator/runs/t3-compile-compile-fp16-50203100/` |
| baseline A5000 | **50203101** | a5000 | **COMPLETED 0:0** | `/work/imt11/Mapperatorinator/runs/t3-compile-baseline-fp16-50203101/` |

| GPU | Variant | Job | ms/map-token | main_tps | cold_start_s | Δ main_tps |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A5000 | baseline | 50203101 | **3.673** | **348.61** | **22.98** | — |
| A5000 | compile | 50203100 | **3.253** | **427.71** | **235.24** | **+22.7%** |

**Greedy (50203099):** **FAIL** — baseline 31418 vs compile 31747 (first_diff_byte 2080, TimingPoints/SV).

Local pulls: `notes/t3-artifacts/match-50203099.json`, `summary-50203100.json`, `summary-50203101.json`  
Remote: `islamtayeb/codex/t3-compile-then-capture` only — **no tiger14n / PR #120**

### Harvest 4 (owned sub-op) — **queued** @ `1e3fc081`

| Cell | Job | GPU | State | Artifact |
| --- | --- | ---: | --- | --- |
| baseline A5000 | **50228030** | a5000 | **PENDING** (Priority) | `/work/imt11/Mapperatorinator/runs/t3-compile-baseline-fp16-50228030/` |
| compile A5000 | **50228031** | a5000 | **PENDING** (Priority) | `/work/imt11/Mapperatorinator/runs/t3-compile-compile-fp16-50228031/` |
| greedy match | **50228096** | a5000 | **PENDING** (afterany:50228030:50228031) | `/work/imt11/Mapperatorinator/runs/t3-greedy-match-50228096/` |

## Do-not

- Push to Tiger14n / PR #120  
- Wire T4 turbo / speculative  
- Modify tip `55949274`  
- Claim 500 / merge to main  
- Use `reduce-overhead` near manual capture  
- Exceed ≤2 concurrent GPU with live T1/T2  
- Put Triton `TMPDIR` on NFS `/work`  
- Recompile sampling `_tail` without fixed-stride staging outside Dynamo  
- Bare-retry full decode-step Inductor without a **new** exactness hypothesis  
- Re-default `max-autotune-no-cudagraphs` for package greedy gate  

## Ruling

**Harvest 3: Promote N. STOP** (full-step).  
**Harvest 4: IN PROGRESS** — owned `proj_out`+FFN sub-op compile; promote only if A5000 +≥10% **and** greedy PASS. Else STOP.
