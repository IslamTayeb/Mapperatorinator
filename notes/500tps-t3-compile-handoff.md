# T3 TORCH.COMPILE — handoff (PIVOT EXECUTION PACKAGE)

**Status:** **H4 SUB-OP STOPPED** · **T3 EXACTNESS RELAXATION BINDING** → replacement lands **full decode-step** + reseals A5000/2080  
**This agent:** EXIT (2026-07-18) — cancelled H4 queue; no further sub-op grind  
**Prior sealed:** Harvest 3 **PROMOTE N / STOP** under *bit-identical* greedy gate @ `3e0aacb7` / `502d0a65`  
**Package:** Pivot **T3** compile-then-capture  
**Branch / WT:** `codex/t3-compile-then-capture` @ `d0488151` (**sub-op code present — restore before reseal**)  
**Local WT:** `/work/projects/Mapperatorinator-worktrees/t3-compile-then-capture`  
**DCC WT:** `/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/t3-compile-then-capture`  
**Base:** `codex/turbo-on-tiger-pr120` @ `b96c3e38` (tiger PR #120 `d01cdd27` + §58/§59 rails)  
**Frozen tip:** `55949274` / FP16 **366.11** — **regression reference only**; **no merge**; **no push to PR #120**  
**T4 turbo:** **PARKED** — §34 turbo unchanged; do not wire speculative  
**T1 smoke:** PASS (`50215788`) — rails green  
**Do not abandon torch.compile.**

## STOP — harvest-4 sub-op (this agent, 2026-07-18)

| Item | Action |
| --- | --- |
| Sub-op path | **STOP grinding** — not the promote candidate under user ruling |
| Jobs cancelled | **50228030** baseline, **50228031** compile, **50228096** greedy — all `CANCELLED by 1512210` before start |
| Tip code | `d0488151` still has owned-subop `install_compiled_subops` + refuses `COMPILE_FULL_STEP` — **replacement must restore full-step** |
| Speed evidence | Harvest 2/3 A5000 **+28.8% / +22.7%** **STAND** (do not re-prove from zero unless wiring changed) |

## T3 EXACTNESS RELAXATION (user 2026-07-18) — BINDING

**Scope: T3 only.** Does **not** change §34 turbo, T4 PARK, tip freeze, or other tracks.

| Field | Ruling |
| --- | --- |
| Exactness bar | **NOT** bit-identical `.osu` / greedy token-match |
| Quality bar | **“Mostly good” / coherent maps** + **T5 KS pack** (distribution / metric parity) |
| Promote candidate | **Full decode-step** compile-then-capture — eager `_tail`, `mode=default` (harvest 2/3) |
| Harvest 2/3 speed | **STAND** — A5000 **+28.8%** (h2) / **+22.7%** (h3) |
| Harvest 4 (sub-op) | **STOPPED / fallback-not-required** |
| Tip / upstream | Tip `55949274` **FROZEN**; **no merge**; **no PR #120 push** |
| §34 / T4 | **Unchanged** |

### Promote gates (post-relaxation)

| Gate | Criterion |
| --- | --- |
| A5000 | main-gen **+≥10%** vs like-with-like uncompiled fast path |
| 2080 Ti | **no-regression** (now in scope — was N/A under old greedy-first rule) |
| Exactness | **Relaxed:** coherent / “mostly good” + **T5 KS pack** — **not** greedy `.osu` byte-match |
| Forbidden still | `reduce-overhead` near manual capture; recompile sampling `_tail`; tip grind; PR #120 push |

## REPLACEMENT CHECKLIST (owns GPU reseal)

1. **Restore full-step Inductor** of shape-static `forward_only` before CUDAGraph capture (eager `_tail`). Prefer resurrect harvest-3 capture path from `3e0aacb7` / `502d0a65` — undo `bec70299` sub-op default (or gate sub-ops behind non-default env only).
2. **Remove / soften** loud refuse of `MAPPERATORINATOR_COMPILE_FULL_STEP` — full-step is again the package default when `MAPPERATORINATOR_COMPILE_DECODE=1`.
3. Keep knobs: `fullgraph=True`, `dynamic=False`, `mode=default`, warm-all-buckets, unique node-local `TMPDIR`, pin Inductor cache, reduced-precision reductions off.
4. **Reseal A5000** baseline vs compile (like-with-like): confirm **+≥10%** main_tps (h3 **+22.7%** is prior evidence; re-run if code restore differs).
5. **Run 2080 Ti** no-regression cell (was skipped under old greedy-first STOP).
6. **Quality:** coherent maps + **T5 KS pack** (`notes/500tps-t5-quality-gates-handoff.md`, `utils/t5_ks_parity.py`) — update T5 track rule so T3 **does not require greedy PASS**; KS / mostly-good is the bar. Document Inductor fp16 drift as declared T3 drift.
7. ≤2 concurrent GPU; unique TMPDIR; **no** PR #120 / tip / T4 wire.
8. Update this handoff with new job IDs + **Promote Y/N** under relaxed gates.

**Reference speed seals (STAND):**

| Harvest | Commit | A5000 Δ main_tps | Notes |
| --- | ---: | ---: | --- |
| 2 | `eb85f4b3` | **+28.8%** | eager `_tail`; max-autotune era |
| 3 | `3e0aacb7` | **+22.7%** | `mode=default` + shared hoist — **preferred restore tip** |

## Binding pattern (promote candidate)

| Knob | Value |
| --- | --- |
| Outer step | **Inductor** `forward_only` (full decode-step) |
| Sampling `_tail` | **eager** (never Inductor — harvest 1 stride thrash) |
| `fullgraph` / `dynamic` / `mode` | `True` / `False` / **`default`** |
| Warm | **EVERY** bucket before capture — §22 |
| Forbidden | `reduce-overhead` near manual CUDAGraph capture |

Opt-in env:

- `MAPPERATORINATOR_COMPILE_DECODE=1` — enable T3 full-step compile-then-capture
- `MAPPERATORINATOR_COMPILE_MODE` — default `default`
- `MAPPERATORINATOR_WARM_ALL_BUCKETS=1`
- `MAPPERATORINATOR_COMPILE_SUBOPS` — harvest-4 only; **not** package default

Cold start: unique node-local `TMPDIR`/`TEMP`/`TMP` + `TORCH_EXTENSIONS_DIR` + per-job `TORCHINDUCTOR_CACHE_DIR`. Never put Triton TMPDIR on NFS `/work`.

### Windows ladder

1. **triton-windows + torch≥2.10** → compile-then-capture
2. **no triton** → plain CUDA graphs (PR #120 default)
3. **no capture** → eager / stock (loud unless `MAPPERATORINATOR_ALLOW_CAPTURE_FALLBACK=1`)

## Root cause (harvest 3) — old bit-identical gate only

Inductor rewrite of shape-static decode forward (fp16) flips near-tie greedy tokens → divergent `.osu` (h3: 31418 vs 31747). Maps remained coherent. Under relaxation → **documented T3 drift**, not automatic STOP, if T5 KS PASSes.

## Fix history (abbrev)

| Harvest | Commit | Result |
| --- | ---: | --- |
| 1 | `28ae22c6` | `_tail` compile → stride thrash −46% + drift |
| 2 | `eb85f4b3` | eager `_tail`; **+28.8%** A5000; old greedy FAIL |
| 3 | `3e0aacb7` | `mode=default`; **+22.7%**; old greedy FAIL → old STOP |
| 4 | `bec70299`…`d0488151` | owned `proj_out,ffn` sub-op — **STOPPED** before harvest; jobs **CANCELLED** |

## Jobs

### Harvest 2 speed STAND @ `eb85f4b3`

| Variant | Job | main_tps | Δ |
| --- | ---: | ---: | ---: |
| baseline | 50196882 | 343.30 | — |
| compile | 50196883 | 442.20 | **+28.8%** |

### Harvest 3 speed STAND @ `3e0aacb7`

| Variant | Job | main_tps | Δ |
| --- | ---: | ---: | ---: |
| baseline | 50203101 | 348.61 | — |
| compile | 50203100 | 427.71 | **+22.7%** |

Artifacts: `notes/t3-artifacts/summary-50203100.json`, `summary-50203101.json`, `match-50203099.json`

### Harvest 4 — **CANCELLED** (no harvest)

| Cell | Job | State |
| --- | ---: | --- |
| baseline | 50228030 | **CANCELLED** 0:0 |
| compile | 50228031 | **CANCELLED** 0:0 |
| greedy | 50228096 | **CANCELLED** 0:0 |

## Do-not

- Push to Tiger14n / PR #120  
- Wire T4 / modify tip `55949274` / claim 500 / merge  
- `reduce-overhead` near manual capture; Inductor `_tail` without fixed-stride staging  
- Grind harvest-4 sub-op as the promote path  
- Require bit-identical greedy for T3 promote  
- Fold T3 relaxed exactness into §34 turbo  

## Ruling

**H4 STOPPED. EXIT this agent.**  
**Promote candidate = full decode-step** compile-then-capture (eager `_tail`, `mode=default`).  
**Exactness = relaxed** (mostly good + T5 KS) — **not** bit-identical.  
Harvest 2/3 speeds **stand**. **Replacement owns** code restore + A5000/2080 reseal under new gates.  
T4 PARKED. Tip frozen. No PR #120 push. **Do not abandon torch.compile.**
