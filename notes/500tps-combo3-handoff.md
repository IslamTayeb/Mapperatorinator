# 500 TPS dual-track handoff (combo 3 → Cursor Multitask)

Local coordinator note (do not commit generated beatmaps/profiles). Plan source: `fp16_combo_handoff_b42769c3.plan.md`.

**Per-improvement ledger (one section per lever):** `notes/500tps-fp16-fp32-improvements.md` — update that file for every separate change.

## Goal (user-clarified 2026-07-15) — binding

**Primary success metric:** demonstrate sustainable **≥500 TPS on RTX 2080 Ti** for **FP16 and FP32** inference paths — **not** for lower-than-FP16 stacks.

**Counts toward 500:** FP32 and FP16 precision runs (weights/compute/activations in FP16 or FP32 as appropriate to that path).

**Does NOT count toward 500:** INT8 / DP4A / w8a8 / other sub-FP16 quant paths, including the hybrid “selected arena” stack that uses **INT8 MLP** (and related lower-precision mix). Hitting ~494–500 there is **not** success.

### Active tracks

1. **Track FP32 — primary:** full-FP32 (no TF32 cheat) optimized-single toward 500.
2. **Track FP16 — primary:** FP16 path toward 500 with its own reciprocal gates (exact or explicitly labeled documented-drift; never call INT8 “FP16”).
3. **Track hybrid/lower-precision — demoted:** INT8/mixed selected-stack (compiled-cross, DP4A/FlashDecode/CUTLASS, ContiguousKv cross-KV, ~494 arena) is **architectural evidence only**. Do not spend serial confirmation budget composing toward hybrid-500.

Stop only with demonstrated sustainable **FP16 and/or FP32** 500 TPS **or** measured evidence no viable FP16/FP32 path remains. When TPS and complete-song wall disagree, **song wall wins**. Improvements that help **both FP16 and FP32** are preferred. **No merge to main** without explicit approval.

## Agent / session settings

- Model: **Grok 4.5 High** (non-fast) for campaign workers/coordinators; do **not** use `cursor-grok-4.5-high-fast`
- CWD: `/work/projects/Mapperatorinator`
- Mode: Multitask coordinator + parallel workers; serialize graduation only
- Prefer this handoff + DCC artifacts over `notes/inference-status.md`

## Current FP16 / FP32 frontier (authoritative)

Fixed work for 500: **8294 tokens → 16.588 s** main-model (500 TPS). FP16 SALVALAI main length is **7809** → **15.618 s** at 500 TPS.

### Campaign tip — **GRADUATED** exact shared-RoPE + device sequence state (`55949274`)

Gpu-common authoritative reciprocals **COMPLETED** (wake harvest 2026-07-16):

| Precision | Job | Tip | Node | cand main_tps | cand main_model_s | tokens | vs shared-rope baseline | Exact |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| **FP32** | `49963835` | `55949274` | z25-20 | **313.05** | **26.494** | 8294 | 287.03 → 313.05 (+26.0 TPS / −2.40 s main) | **PASS** (IDs/stopping/`.osu`) |
| **FP16** | `49964133` | `55949274` | z25-20 | **366.11** | **21.330** | 7809 | 331.67 → 366.11 (+34.4 TPS / −2.21 s main) | **PASS** (IDs/stopping/`.osu`) |

**Decision: GRADUATE persistent tip `55949274`** (`codex/exact-shared-rope-device-state`). Song-wall also improved (FP16 complete-request −2.61 s; FP32 wall median improved but range was large — main-model is the stable claim).

### Re-baselined gap to 500 (from graduated tip)

| Precision | Auth tip TPS | main_model_s | Need @500 | Gap | Speedup needed |
| --- | ---: | ---: | ---: | --- | ---: |
| **FP16** | **366.11** | **21.330** | 15.618 s | **−133.9 TPS / −5.71 s** | **1.366×** |
| **FP32** | **313.05** | **26.494** | 16.588 s | **−187.0 TPS / −9.91 s** | **1.596×** |

Best measured FP16 main-TPS on campaign tip: **366.11** (auth). Scavenger signal `49905348` was 368.62 — consistent.

### Sealed packaging pin (still not merged; absolute FP32 floor note)

| Precision | Sealed package | Job | Tip | main_tps | main_model_s | tokens | Gap to 500 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| **FP32** | exact shared-runtime | `49906335` | `d41981ae` | **317.46** | **26.126** | 8294 | −182.5 TPS / −9.54 s (need 1.575×) |
| **FP16** | exact shared-runtime | `49906334` | `d41981ae` | **313.54** | **24.913** | 7809 | superseded by device-state tip |

Packaging note: `notes/exact-shared-runtime-packaging.md`. Exactness sealed. **Not merged.** FP32 sealed absolute TPS (317.46) still slightly above graduated device-state auth (313.05); campaign continues from `55949274` because FP16 wins big and both precisions share the composition. Do not regress FP16 by reverting tip.

## Wake harvest notes (auth infra)

- `49963287` / `49963833` FP16 auth earlier **FAILED** `2:0` (parallel guard).
- `49963835` FP32 auth **COMPLETED** `0:0` 00:05:27 on z25-20.
- `49964133` FP16 auth **COMPLETED** `0:0` 00:03:44 on z25-20 (`afterany` after FP32).
- Artifacts: `/work/imt11/Mapperatorinator/runs/exact-rope-device-state-fp32-auth-49963835/`, `.../exact-rope-device-state-fp16-auth-49964133/`.

**Parallel-guard lesson:** with `ALLOW_PARALLEL_RECIPROCAL=0`, do **not** pre-queue a dependent sibling — PENDING GPU jobs trip the guard.

## §12 Native q1 RoPE/cache head-group CTA scheduling — **OPEN**

- Branch / tip: `codex/exact-q1-rope-cache-headgroup` @ **`9d034259`** (base `55949274`).
- Lever: packs two independent q1 RoPE/cache attention heads in each 128×2 CTA while retaining each head’s reduction order; it does not replace Wo, proj_out, or RMSNorm math.
- Reciprocal jobs: FP16 **`49998739`**; FP32 **`49998740`** — submitted.
- Superseded preflight-only attempts: `49998160`/`49998162` failed before execution because the branch remote ref advanced; replacements use this immutable pushed tip.
- Run roots: `/work/imt11/Mapperatorinator/runs/q1-headgroup-fp16-49998739/` and `/work/imt11/Mapperatorinator/runs/q1-headgroup-fp32-49998740/`.
- Next action: harvest exactness and untraced `main_model` reciprocal evidence; promote only on an exact ≥5% result.

## Branches / worktrees

| Track | Branch | Tip | Role |
| --- | --- | --- | --- |
| FP32/FP16 | `codex/exact-shared-rope-device-state` | **`55949274`** | **GRADUATED campaign tip** (auth FP16 366.11 / FP32 313.05) |
| FP32/FP16 | `codex/exact-self-out-residual-fusion` | **`4e3477a2`** | full-song last look `49978677`/`678` **FAILED** — **STOP_NO_PROMOTE** / DROP (unused residual) |
| FP32/FP16 | `codex/exact-self-out-residual-fuse` | **`57fa6612`** | Live r2 full-song Wo+residual (jobs `49976417`/`451`) |
| FP32/FP16 | `codex/exact-self-norm-wqkv` | **`fd126612`** | r3 **STOP_NO_PROMOTE** (`49982390`/`391`); do not bare-retry |
| FP32/FP16 | `codex/exact-proj-out-fuse` | **`585ffc90`** | §10 final-norm+proj_out — **STOP_NO_PROMOTE** (`49989856`/`857`) |
| FP32/FP16 | `codex/exact-self-wo-linear` | **`c48f5b8b`** | §11 Wo one-token linear — **STOP_NO_PROMOTE** (`49993296`/`297` exactness collapse) |
| FP32/FP16 | `codex/exact-q1-rope-cache-headgroup` | **`9d034259`** | §12 q1 RoPE/cache head-group CTA scheduling — **OPEN** (FP16 `49998739` / FP32 `49998740` submitted) |
| FP32/FP16 | `codex/exact-self-rmsnorm-wqkv` | `55949274` | empty stub WT — superseded by `exact-self-norm-wqkv` |
| FP32/FP16 | `codex/exact-decode-cast-elim` | **`a354624f`** | **DROP** (exact; main −19 TPS on `49974095`) |
| FP32/FP16 | `codex/exact-decode-cast-copy` | **`11766d07`** | **DROP** r2 cast/copy (exact main regress `49976415`/`416`) |
| FP32/FP16 | `codex/exact-shared-runtime-promotion` | `d41981ae` | Sealed packaging pin (FP32 absolute 317.46) |
| FP32/FP16 | `codex/exact-compiled-cross-bmm` | `25d8e469` | Owned compile-before-capture on tip — **STOP_NO_PROMOTE** (see below) |
| FP16 | `codex/shared-split-kv-dtype` | `4b0adc10` | FP16 split-KV — prior **STOP_NO_GAIN**; do not blind-retry |
| FP32 | `codex/strict-fp32-split-kv` | `7861e5bf` | Strict FP32 split-KV (parked; reformulate only) |
| hybrid↓ | `codex/500tps-arena-compiled-cross-last-mile` | `0dbab9e5` | INT8-hybrid — **demoted** |
| hybrid↓ | ContiguousKv / DP4A / FlashDecode / CUTLASS | (various) | **DROP** / park |

Hard infra: reciprocal wrappers must **not** share temp/native-build dirs across concurrent jobs.

## Exact compiled-cross (post-device-state) — STOP_NO_PROMOTE

Port of hybrid compile-before-capture onto exact FP16/FP32 tip:

| Job | Prec | Tip | Slurm | Result |
| --- | --- | ---: | --- | --- |
| `49966195` / `49966208` | FP16 | `b33d9e6c` | FAILED 1:0 | RuntimeError: compiled cross requires native cross+MLP tail (timing windows) |
| `49966196` / `49966209` | FP32 | `b33d9e6c` | FAILED 1:0 | same |
| `49968303` | FP16 | `25d8e469` | FAILED analysis | Inference ran; **exactness FAIL** (7809→8402 tokens; `.osu`/RNG diverge). Apparent ~409 TPS **invalid**. |
| `49968305` | FP32 | `25d8e469` | FAILED analysis | Byte-exact; main −0.14 s / ~318 TPS (~1%) — **below 5% gate**; unused compiled_cross allowlist |

**Decision: do not promote; do not grind.** Cross BMM is only ~2% of node-path GPU time (nsight). Fixing analyzer/exactness would not close the 5–10 s gap.

## Nsight lever map (FP16 device-state, job `49966210`)

Tip `55949274`, run `exact-device-state-fp16-nsight-49966210` (COMPLETED; `ncu` permission denied — NVTX/kernel families only).

Top families on `fp16_smoke_node` main_generation kernel time:

| Family | Share | Multi-second relevance |
| --- | ---: | --- |
| gemm_gemv_projection | ~26% | Remaining Q/out/cross linears — compile/fuse without INT8 |
| native_q1_self_rope_cache | ~19% | Largest single native kernel — fusion/scheduling / reformulated split-KV |
| elementwise + memory | ~30% | Cast/copy traffic — eliminate redundant copies (best multi-second bet) |
| fused MLP (fc1+fc2) | ~15% | Already fused; incremental unless fused with neighbors |
| sampling radix | ~7% | Secondary |
| fmha_cross_attention | ~2% | Already small on node path — matches compiled-cross weak e2e |

## Hybrid demotion / stopped work (unchanged)

- Hybrid INT8-arena ~494–500 **does not count**.
- No further serial hybrid confirmation unless explicitly re-authorized as non-goal research.
- Encoder precompute `49903861`: **STOP_ENCODER_PRECOMPUTE** (b1 wall regress).

## Self-out residual component harvest (`49973580`/`49973581`) — **FIX** (not infra)

| Job | Prec | Tip | Node | Slurm | Exit | Root cause |
| --- | --- | --- | --- | ---: | ---: | --- |
| `49973580` | FP16 | `a6fc00d9` | z25-21 | FAILED | 1:0 ~32s | **code:** `ValueError: Audio file not found: /Users/islamtayeb/Downloads/SALVALAI ...mp3` |
| `49973581` | FP32 | `a6fc00d9` | z25-20 | FAILED | 1:0 ~30s | same |

`utils/profile_self_out_residual_component.py` called `compile_args` → `compile_paths` against `configs/inference/profile_salvalai.yaml`'s Mac-local `audio_path`. Preflight + GPU OK; run dirs only have `preflight.txt` (no `component.json`). **Not DROP** (hypothesis still live); **not bare RETRY** (same tip would fail again).

**FIX chain (tip now `4e3477a2`):**
1. `01e37834` — override audio via `MAPPERATORINATOR_AUDIO` / DCC `salvalai.mp3` (+ sbatch check).
2. `f8b42e49` — unwrap `InferenceEngineBinding` so Wo weight extract sees a real `nn.Module` (post-audio retry `49974790`/`791` hit `TypeError`).
3. `4e3477a2` — reciprocal allowlist for expected dispatch/graph deltas.

**Resubmitted component (ALLOW_PARALLEL=1, exclude h36-9, unique RUN labels):** **`49978315`** FP16 / **`49978316`** FP32 @ `4e3477a2` (RUNNING on h36-6). Prior mid-chain: FP32 `49976698` PASSED component; FP16 `49976696` FAILED correctness (drift 0.015625 > 1e-3).

## Primary scout harvest `49974091`–`96` (2026-07-16 ~03:36Z)

Tip baseline: auth **`55949274`** FP16 **366.11** / FP32 **313.05**. **No ≥5% / multi-second PASS → no serial confirmation from this set.**

| Job | Prec | Tip | Node | Slurm | main_tps / wall | Exact | Decision |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| `49974091` | FP16 | `baf05d95` | z25-21 | FAILED 1:0 5:59 | n/a (no analysis.json) | n/a | **DROP** — unused expected delta `*decode_cast_copy*` (path never fired) |
| `49974092` | FP16 | `e1c286e1` | z25-20 | FAILED 1:0 5:05 | salvage cand **629** tok / 2082 B | **FAIL** | **DROP** — exactness collapse (also undeclared cuda_graphs) |
| `49974093` | FP32 | `e1c286e1` | h36-5 | FAILED 1:0 6:23 | salvage cand **562** tok / 1853 B | **FAIL** | **DROP** — exactness collapse |
| `49974094` | FP16 | `b6135df0` | h36-5 | FAILED 1:0 4:27 | incomplete (1st pair only) | **FAIL** | **DROP** sibling — CUDA assert `probability … inf/nan/<0` on candidate main |
| **`49974095`** | FP16 | `a354624f` | h36-6 | **COMPLETED** 0:0 5:55 | cand **337.18** / 23.160 s vs recip base 356.25 (**−19.1 TPS / +1.23 s**); vs tip 366.11 **−28.9 TPS** | **PASS** IDs/stopping/`.osu` (7809) | **DROP** — exact but main regress (cold wall ≠ claim) |
| `49974096` | FP32 | `a354624f` | h36-6 | FAILED 1:0 6:19 | n/a (4 legs; analysis crashed) | n/a | **RETRY** (park) — `baseline_first` vs `baseline_second` timing_context token stream differ |

Artifacts: `/work/imt11/Mapperatorinator/runs/{exact-decode-cast-copy-fp16-49974091,self-out-residual-fp{16,32}-4997409{2,3},exact-self-out-residual-fp16-49974094,decode-cast-elim-fp{16,32}-4997409{5,6}}/`. Only `49974095` wrote `analysis.json`. Components `49974790`/`791` FAILED `TypeError` Module unwrap → FIX `f8b42e49`.

## r2 harvest (siblings / reformulations) — cast/copy + self-out still **NO GRADUATE**

| Job | Prec | Tip | Slurm | Claim | Decision |
| --- | --- | ---: | --- | --- | --- |
| **`49976415`** | FP16 | `11766d07` | **COMPLETED** | Exact PASS; main_tps 371.5→348.9 (**−22.6 TPS**); main_model +1.38 s | **STOP_NO_PROMOTE** |
| **`49976416`** | FP32 | `11766d07` | **COMPLETED** | Exact PASS; main_tps 319.1→313.8 (**−5.3 TPS**); main_model +0.44 s | **STOP_NO_PROMOTE** |
| `49976417` | FP16 | `57fa6612` | **FAILED** | Incomplete / crash (baseline 645 HO) | **STOP_NO_PROMOTE** |
| `49976451` | FP32 | `57fa6612` | **FAILED** | Exactness collapse HO 637→3 + undeclared deltas | **STOP_NO_PROMOTE** |
| `49976696`/`698` | component | `4e3477a2` | mixed | projection only — not production TPS | sizing only |

**Frontier unchanged:** tip `55949274` auth FP16 **366.11** / FP32 **313.05**. Cast family = exact-but-slower (**DROP**). Do not grind cast allowlists.

## Live / recent DCC jobs (FP16/FP32 focus)

| Job | Node | Candidate | Slurm | Decision |
| --- | --- | --- | --- | --- |
| **`49998739`** | — | §12 q1 headgroup FP16 @ `9d034259` | **PENDING** | reciprocal scout submitted; exact/measured pending |
| **`49998740`** | — | §12 q1 headgroup FP32 @ `9d034259` | **PENDING** | reciprocal scout submitted; exact/measured pending |
| **`49993296`** | — | §11 self-Wo linear FP16 @ `c48f5b8b` | **FAILED** | exactness collapse 7809→518 — **STOP_NO_PROMOTE** |
| **`49993297`** | — | §11 self-Wo linear FP32 @ `c48f5b8b` | **FAILED** | exactness collapse 8294→562 — **STOP_NO_PROMOTE** |
| **`49989856`** | z25-20 | **proj-out FP16** @ `585ffc90` | **FAILED** | exactness FAIL 7809→7869 / HO 645→588 — **STOP_NO_PROMOTE** |
| **`49989857`** | z25-21 | **proj-out FP32** @ `585ffc90` | **FAILED** | flat/regress main; unused allowlist — **STOP_NO_PROMOTE** |
| **`49982390`** | z25-20 | **self norm+Wqkv FP16** r3 @ `fd126612` | **FAILED** 7:14 | fuse fired (174 calls); **exactness FAIL** 7809→8483 tok / HO 645→651 + undeclared dispatch — **STOP_NO_PROMOTE** |
| **`49982391`** | z25-21 | **self norm+Wqkv FP32** r3 @ `fd126612` | **FAILED** 6:15 | 174 calls; tok/HO stable; unused allowlist pattern; salvage ~+1.4% TPS (&lt;5%) — **STOP_NO_PROMOTE** |
| `49982320` | — | self norm+Wqkv FP16 @ `fd126612` | **FAILED** 1:0 ~6s | preflight/env miss (no run dir); superseded by `49982390`/`391` |
| `49980564` | z25-20 | self norm+Wqkv FP16 @ `63869511` | **FAILED** 1:0 ~4:29 | `(1,1)` scope insufficient — TIMING CUDA-graph capture still armed fuse without q1; **not bare-retry** |
| `49980565` | z25-21 | self norm+Wqkv FP32 @ `63869511` | **FAILED** 1:0 ~3:33 | same RuntimeError on candidate_first (sibling) |
| `49979210` | z25-21 | self norm+Wqkv FP16 @ `e9ad0259` | **FAILED** 1:0 ~3:39 | refuse-fallback without q1 — superseded |
| `49979211` | z25-21 | self norm+Wqkv FP32 @ `e9ad0259` | **FAILED** 1:0 ~3:43 | same (sibling); superseded |
| `49978677` | h36-5 | self-out FP16 FIX @ `4e3477a2` | **FAILED** 1:0 ~5:52 | unused `*native_one_token_linear_residual*` + undeclared dispatch — fusion not engaged; **STOP_NO_PROMOTE** |
| `49978678` | h36-5 | self-out FP32 FIX @ `4e3477a2` | **FAILED** 1:0 ~6:04 | same CandidateAnalysisError (sibling); **STOP_NO_PROMOTE** |
| `49978311`/`312` | z25-21 | duplicate self-out @ `4e3477a2` (20m) | **FAILED** | superseded by `49978677`/`678` (do not re-submit) |
| `49978313`/`315` | — | self-out **component** FP16 @ `4e3477a2` | **FAILED** | ~2 s projected — not production |
| `49978314`/`316` | — | self-out **component** FP32 @ `4e3477a2` | **COMPLETED** | any_component_pass; promotion_pass False |
| `49976415`/`416` | z25-21 | cast-copy r2 @ `11766d07` | **COMPLETED** | **DROP** (exact main regress) |
| `49976417`/`451` | h36-5 | self-out r2 @ `57fa6612` | **FAILED** | **DROP** (exactness/crash) |
| **`49974095`** | h36-6 | cast-elim @ `a354624f` | **COMPLETED** | **DROP** (exact; −19 TPS) |
| `49974091`–`96` | — | primary scouts | **done** | see primary harvest table |
| `49963835`/`133` | z25-20 | device-state auth | **COMPLETED** | **GRADUATE** 313.05 / **366.11** |
| `49906335` | z25-20 | sealed packaging FP32 | **COMPLETED** | **SEALED** 317.46 |

## Path to close FP16 −5.71 s (need 1.366×)

1. ~~cast-elim / cast-copy~~ **DROP** (`49974095` −19 TPS; r2 −22.6/−5.3).
2. ~~self-out Wo+residual~~ **STOP_NO_PROMOTE** / DROP — last look `49978677`/`678` FAILED (unused `*native_one_token_linear_residual*` + undeclared dispatch; fusion not engaged).
3. ~~self norm+Wqkv r3 `fd126612`~~ **STOP_NO_PROMOTE** (`49982390` exactness FAIL; `49982391` &lt;5%).
4. ~~Owned final-norm+proj_out~~ **STOP_NO_PROMOTE** (ledger §10).
5. **§11 STOP_NO_PROMOTE** (`49993296`/`297` exactness collapse). **Harvest §12** q1 RoPE/cache headgroup reciprocal jobs `49998739`/`49998740` @ `9d034259` (not another Wo/proj_out fuse).
6. q1 self-attn only with new sizing vs prior STOP_NO_GAIN.
7. Serialize graduation after independent ≥5% exact PASS.

## Self RMSNorm+Wqkv (tip `55949274`) — **STOP_NO_PROMOTE** (r3 harvested)

**Root cause (r1+r2):** tip sets `native_q1_rope_cache_self_attention=False` for `ContextType.TIMING`, but the scout still armed `fuse_self_norm_wqkv`. Timing CUDA-graph capture is shape `(1,1)`, so decoder skipped RMSNorm / bound `_scout_self_norm_weight`, then `q1_rope_cache` hook was absent → refuse RuntimeError on candidate_first. `63869511` only gated on shape/`_scout_*` bind — insufficient.

| Item | Value |
| --- | --- |
| Branch / WT | `codex/exact-self-norm-wqkv` / DCC `exact-self-norm-wqkv` (`codex/exact-self-norm-wqkv-dcc`) |
| Base tip | `55949274` |
| Scout → FIX chain | `e9ad0259` → `63869511` (insufficient) → **`5816fa04`/`fd126612`** (arm fuse only when native q1 live; RMSNorm restore on non-q1 fallback; runtime_context refuse mismatch) |
| Opt-in | `self_norm_wqkv_fusion_candidate_context` → `native_self_norm_wqkv` / `fuse_self_norm_wqkv` **and** `native_q1_rope_cache_self_attention` (V32 cold default; TIMING no-op) |
| Kernel | `native_one_token_rmsnorm_linear` (output_dim=3×640) |
| r1 | `49979210`/`211` @ `e9ad0259` — **FAILED** refuse-fallback |
| r2 | `49980564`/`565` @ `63869511` — **FAILED** same error on TIMING capture (do not bare-retry) |
| r3 (FIX) | **`49982390` FP16** / **`49982391` FP32** @ `fd126612` — **FAILED** (see ledger §7c) |
| Run roots | `/work/imt11/Mapperatorinator/runs/self-norm-wqkv-fp{16,32}-fix-fd126612-4998239{0,1}/` |
| Decision | **STOP_NO_PROMOTE** — FP16 tok/HO diverge; FP32 no ≥5% main. Detail: `notes/500tps-fp16-fp32-improvements.md` §7 |

## Next gates (ordered — FP16/FP32 only)

1. ~~Auth / graduate device-state~~ **DONE**.
2. ~~Compiled-cross / cast-elim / cast-copy~~ **DROP / STOP_NO_PROMOTE**.
3. ~~Harvest norm+Wqkv r3 / proj_out~~ **STOP_NO_PROMOTE**.
4. ~~Harvest §11 self-wo linear~~ **STOP_NO_PROMOTE** (exactness collapse). **Harvest §12** q1 RoPE/cache headgroup jobs `49998739`/`49998740` @ `9d034259`.
5. No hybrid ContiguousKv / INT8 / DP4A / CUTLASS for the 500 claim.
6. No merge without approval.

## Standing orders

- Prefer **Grok 4.5 High** (non-fast); do **not** use `cursor-grok-4.5-high-fast`.
- One candidate per worktree/node; serialize graduation.
- Never present projections as production TPS.
- V32 cold default; optimized under `osuT5/.../inference/optimized/`.
- Wake 2026-07-16: §11 Wo-linear **STOP_NO_PROMOTE** (exactness collapse). §12 q1 RoPE/cache headgroup is **OPEN**: FP16 `49998739` / FP32 `49998740` submitted at `9d034259`; harvest `/work/imt11/Mapperatorinator/runs/q1-headgroup-fp{16,32}-<jobid>/`. Tip still FP16 **366.11** / −5.71 s. Ledger: `notes/500tps-fp16-fp32-improvements.md`. No merge. No hybrid-500.

## Outside-research ranking (FP16/FP32)

1. Exact shared-RoPE + device sequence state (`55949274`) — **GRADUATED** auth FP16 **366.11** / FP32 **313.05**
2. Exact shared-runtime packaging (`d41981ae`) — sealed FP32 absolute pin 317.46
3. Self rmsnorm+Wqkv fuse — **STOP_NO_PROMOTE** (r3 exactness/&lt;5%; see improvements ledger §7)
4. Decode cast-copy / cast-elim — **DROP** (exact main regress)
5. Self-out Wo+residual — **STOP_NO_PROMOTE** / DROP (`49978677`/`678` unused residual; prior exactness collapse)
6. Exact compiled-cross BMM — **STOP_NO_PROMOTE**
7. Split-KV on FP16 — prior **STOP_NO_GAIN**; park / reformulate only
8. Hybrid INT8 arena — **demoted**; not 500

## Track hybrid archive (evidence only — not 500)

Compiled-cross `49955143` GRADUATE persistent on INT8-hybrid stack (−0.192 s main) remains useful architecture evidence. Campaign arithmetic to ~500 on that stack is **out of scope** for the clarified goal.
