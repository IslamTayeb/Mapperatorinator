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

## §12 Native q1 RoPE/cache head-group CTA scheduling — **STOP_NO_PROMOTE**

- FIX tip `dd5d8e58` / tag `q1-rope-cache-headgroup-scout-dd5d8e58-r4` (prior `64372cde` jobs were infra-only unbound `profile`).
- Jobs FP16 **`50024784`** / FP32 **`50024785`** FAILED analysis after full reciprocal (all four legs ran; headgroup 174 calls).
- Exact **PASS** (tok 7809/8294 + HO 645/637 + `.osu` sha match). Main regress: FP16 **310.50→268.20 (−42.3 TPS)**; FP32 **321.28→290.19 (−31.1 TPS)**. Second legs same direction.
- Analyzer unused `*optimized_cuda_graphs*` — do not grind allowlist for a slower lever.
- Decision: **STOP_NO_PROMOTE** (wake harvest confirmed 2026-07-16).

## §13 Owned compile-before-capture self Wqkv + Wo — **STOP_NO_PROMOTE**

- Tip `3164875a` (`codex/exact-compiled-self-proj` / DCC `exact-compiled-self-proj-dcc`).
- Jobs FP16 **`50001853`** / FP32 **`50001854`** FAILED 1:0 (~4–5 min). Both have `baseline_first` + `candidate_first`.
- Candidate crash: `FailOnRecompileLimitHit` on `_linear_region` (`fullgraph=True`, `dynamic=False`) — seq-len mismatch (expected 1024, actual 564/617). No reciprocal exactness / `main_tps`.
- Baseline-only (not a claim): FP16 ~319.93 TPS / 24.408 s; FP32 ~262.04 / 31.651 s.
- Decision: **STOP_NO_PROMOTE** (lever design fail). Do not bare-retry. §14 is the shape-confined revisit.

## §14 Decode-only `(1,1)` compiled self Wqkv + Wo — **STOP_NO_PROMOTE**

- Tip **`43fea0c2`** / branch `codex/exact-compiled-self-proj-decode-only`.
- Jobs FP16 **`50031125`** / FP32 **`50031126`** FAILED analysis after full reciprocal (all four legs; compile 120+120 hits).
- Exact **PASS** (tok 7809/8294 + HO 645/637 + `.osu` sha match + token IDs equal).
- Main: FP16 **308.61→306.34 (−2.27 TPS / −0.73%)**; FP32 **308.36→315.98 (+7.62 / +2.47%)** — under 5% gate.
- Analyzer allowlist mismatch (`compiled_self_*` undeclared / unused globs) — do not grind for a non-graduate.
- Decision: **STOP_NO_PROMOTE**. Do not bare-retry §13/§14.

## §15 Decode-only `(1,1)` compiled tip-exact `proj_out` — **STOP_NO_PROMOTE**

- Tip **`5aa6eb9f`** / branch `codex/exact-compiled-proj-out-decode-only` (from tip `55949274`; **not** §10 native fuse).
- Jobs FP16 **`50036330`** FAILED 3:23 / FP32 **`50036325`** FAILED 4:22 — candidate crashed in timing gen (`compiled proj_out requires model.proj_out with rank-2 weight`); baseline-only profiles present; no reciprocal.
- Exact **n/a**; no candidate `main_tps`.
- Decision: **STOP_NO_PROMOTE**. Revisit only with timing/rank-2 gate — not bare retry.

## §16 Tip-exact CUDA `expandable_segments` allocator — **STOP_NO_PROMOTE**

- Tip **`fde4f0f2`** / branch `codex/exact-cuda-alloc-expandable` (from tip `55949274`).
- Jobs FP16 **`50039777`** COMPLETED 5:36 (z25-21) / FP32 **`50039776`** COMPLETED 7:25 (z25-20); both PASS reciprocal.
- Exact **PASS** (tok 7809/8294 + HO 645/637 + `.osu` sha match).
- Main: FP16 reciprocal 350.80→373.46 (+6.5%) but **vs tip 366.11→373.46 (+2.0% / −0.42 s)** — under tip ≥5% gate (≥384.4). FP32 276.65→270.63 (**−2.2% regress**).
- Decision: **STOP_NO_PROMOTE**. Do not bare-retry expandable_segments.

## §17 Tip-exact CUDA `cudaMallocAsync` allocator — **STOP_NO_PROMOTE**

- Tip **`b9b60cd5`** / branch `codex/exact-cuda-alloc-malloc-async` (from tip `55949274`).
- Auth jobs FP16 **`50050003`** / FP32 **`50050002`** COMPLETED PASS (dups `50050004`/`50050001` same direction).
- Exact **PASS** (tok/HO/`.osu`). Main **regress**: FP16 363.13→351.55 (−3.2% recip; **−4.0% vs tip**); FP32 305.34→288.65 (−5.5%; **−7.8% vs tip**).
- Decision: **STOP_NO_PROMOTE**. Do not bare-retry allocator env confs (§16/§17).

## §18 Tip-exact decode logits workspace + compiled softmax — **STOP_NO_PROMOTE**

- Tip **`b47a3fe2`** / branch `codex/exact-compiled-decode-logits-finalize` (from tip `55949274`).
- Jobs FP16 **`50107685`** / FP32 **`50107686`** FAILED analysis after full reciprocal (h36-5).
- Exact **PASS** (tok 7809/8294 + HO 645/637 + `.osu` sha match). Main: FP16 flat/−0.4% recip (under tip 5% gate); FP32 regress (−1.0%/−2.6%).
- Analyzer unused `*compiled_softmax*` / `*optimized_cuda_graphs*` — do not grind allowlist.
- Decision: **STOP_NO_PROMOTE**. Do not bare-retry §18.

## §19 Tip-exact native q1 attn-out reshape — **STOP_NO_PROMOTE**

- Tip **`5f4b9989`** / branch `codex/exact-q1-out-reshape`.
- Jobs FP16 **`50129056`** / FP32 **`50129057`** FAILED analysis (unused `*optimized_cuda_graphs*`).
- Exact **PASS**. Main vs tip FP16 +0.55%/+1.06% (under 5%); FP32 no gain / regress.
- Decision: **STOP_NO_PROMOTE**. Do not bare-retry reshape.

## §20 Gated decode-only compiled proj_out (§15 FIX) — **STOP_NO_PROMOTE**

- Tip **`91977df2`** / branch `codex/exact-compiled-proj-out-gated`.
- Jobs FP16 **`50129762`** / FP32 **`50129763`** FAILED analysis (unused `*compiled_proj_out*` / `*optimized_cuda_graphs*`).
- Exact **PASS**. Policy `compiled_proj_out` **prepare_failed** / hits **0**. FP16 vs tip −14–20%; FP32 under 5%.
- Decision: **STOP_NO_PROMOTE**. Do not bare-retry §20 gate.

## §20b Prepare try/except rank-2 skip (§20 FIX) — **STOP_NO_PROMOTE** @ `3215d14b`

- Tip **`3215d14b`** / branch `codex/exact-compiled-proj-out-gated` (try/except rank-2; no aggressive pre-gate).
- Follow-up harden **`8bd44130`** pushed (unwrap + dtype skip) but **not** this job tip.
- Hypothesis: exact + hits&gt;0 + ≥5% main vs tip (FP16 ≤20.264 s / ≥384.4 TPS).
- Jobs FP16 **`50131926`** / FP32 **`50131927`** FAILED analysis — still **hits=0** @ `3215d14b`. Branch has unpromoted **`8bd44130`** engage harden (not submitted).

 rank-2 skip (§20 FIX) — **STOP_NO_PROMOTE**

- Tip **`3215d14b`**. Jobs FP16 **`50131926`** / FP32 **`50131927`** FAILED analysis.
- Exact **PASS**. Still `prepare_failed` / hits **0**. No ≥5% vs tip.
- Decision: **STOP_NO_PROMOTE**.

## §21 Compiled proj_out unwrap + dtype-skip engage FIX — **STOP_NO_PROMOTE**

- Tip **`8bd44130`** / branch `codex/exact-compiled-proj-out-gated`.
- Jobs FP16 **`50132354`** / FP32 **`50132355`** FAILED analysis after full reciprocal.
- Exact **PASS** (tok/HO/`.osu`/RNG). Compile **not engaged**: hits **0**, `disabled_reason=missing_rank2_proj_out_weight` on all 87 main windows.
- Main vs tip: FP16 **−15%**; FP32 under 5% (+0.9–2.1%).
- Decision: **STOP_NO_PROMOTE**. Leave compiled-proj_out family (§15/§20/§20b/§21).

## §22 Decode-only compiled self Wo + residual (gemm) — **STOP_NO_PROMOTE**

- Tip **`a5ea705d`** / branch `codex/exact-compiled-self-wo-residual`.
- Jobs FP16 **`50133112`** FAILED 4:49 (z25-20) / FP32 **`50133113`** FAILED 4:05 (z25-21).
- Exact **n/a** — candidate crashed in CUDA-graph capture: Inductor synced during capture (`operation not permitted when stream is capturing`) despite zeros warmup.
- Compile engage **FAIL**; no candidate `main_tps`. Baseline-only ~296.7 / ~321.4 TPS (not a claim).
- Decision: **STOP_NO_PROMOTE**. Leave compiled self Wo+residual; revisit only with real-weight warm + Inductor finished before `torch.cuda.graph`.

## §23 Tip-exact q1 float32 mask workspace — **STOP_NO_PROMOTE**

- Tip **`8759db3a`** / branch `codex/exact-q1-mask-workspace`.
- Jobs FP16 **`50138023`** FAILED 7:29 / FP32 **`50138024`** FAILED 6:35 (analyzer unused `*optimized_cuda_graphs*`).
- Exact **PASS** (tok/HO/`.osu`/RNG). Engage hits **0** (`q1_mask_workspace` capture counter).
- Main vs tip: FP16 **312.05/310.26 (−14.8%/−15.3%)**; FP32 **309.87/308.02 (−1.0%/−1.6%)**. Recip 2nd FP16 flat.
- Decision: **STOP_NO_PROMOTE**. Do not grind allowlist; revisit only with proven hot-path hits&gt;0.

## §24 Tip-exact active-prefix decode bucket 256 — **STOP_NO_PROMOTE**

- Tip **`1076bfd3`** / branch `codex/exact-active-prefix-bucket` (CUDA-graph scheduling; tip bucket 64→256).
- Jobs FP16 **`50138950`** FAILED 5:57 / FP32 **`50138951`** FAILED 6:17 (analyzer unused `*active_prefix_bucket*`; all four legs present).
- Exact **PASS** (tok 7809/8294 + HO 645/637 + `.osu` sha + RNG). Engage size **256**; graph delta **12→4**.
- Main vs tip: FP16 **347.14/347.95 (−5.18%/−4.96%)**; FP32 **312.88/311.51 (−0.05%/−0.49%)**. Pad tax > capture savings.
- Decision: **STOP_NO_PROMOTE**. Do not grind allowlist; inverse exact-length is §25.

## §25 Tip-exact active-prefix exact-length (bucket=1) — **STOP_NO_PROMOTE**

- Tip **`25376173`** / branch `codex/exact-active-prefix-exact-length` (inverse §24: no pad, more captures).
- Jobs FP16 **`50139609`** FAILED 4:35 / FP32 **`50139610`** FAILED 6:45 (analyzer unused `*active_prefix_exact_length*`).
- Engage size **1**; FP32 graph delta **~50** (capture explosion). FP16 cand **OOM** in graph capture (CUDA-graph pools).
- Exact: FP16 **n/a**; FP32 **PASS** (tok 8294 / HO 637 / `.osu` sha + RNG).
- Main vs tip: FP32 **254.12/250.61 (−18.8%/−20.0%)**. Leave bucket/pad family.
- Decision: **STOP_NO_PROMOTE**. Do not grind allowlist; do not bare-retry §24/§25.

## §26 Tip-exact native cross+MLP `outputs_per_block` 8→4 — **STOP_NO_PROMOTE**

- Tip **`b1736b0b`** / branch `codex/exact-native-mlp-opb`.
- Jobs FP16 **`50140036`** / FP32 **`50140037`** FAILED analysis (unused `*optimized_cuda_graphs*`) after full reciprocal on z25-21.
- Exact **PASS** (tok 7809/8294 + HO 645/637 + `.osu` sha + RNG). Engage opb=**4**, decode calls 174.
- Main vs tip: FP16 **370.25 / 338.11 (+1.13% / −7.65%)**; FP32 **317.75 / 317.28 (+1.50% / +1.35%)** — under ≥5% gate.
- Decision: **STOP_NO_PROMOTE**. Do not grind allowlist; do not bare-retune opb→2.

## §27 Tip-exact native q1 attention `block_size` 128→256 — **STOP_NO_PROMOTE**

- Tip **`10df9bc0`** / branch `codex/exact-q1-block-size`.
- Jobs FP16 **`50140877`** FAILED 7:19 (z25-20; analyzer undeclared cuda_graphs after full reciprocal) / FP32 **`50140878`** COMPLETED PASS 6:24 (z25-21).
- Engage block_size=**256**, decode calls 174.
- Exact: FP16 **FAIL** (tok 7809→8011 / HO 645→618 / `.osu` mismatch); FP32 **PASS** (8294/637).
- Main vs tip: FP16 invalid; FP32 **308.76 / 309.93 (−1.37% / −1.00%)**.
- Decision: **STOP_NO_PROMOTE**. Leave q1 block_size family; do not bare-retune →64.

## §28 Tip-exact owned rectangular self-Wqkv one-token linear — **STOP_NO_PROMOTE**

- Tip **`f5c6b901`** / branch `codex/exact-self-wqkv-linear` (from tip `55949274`; no RMSNorm fuse; no Wo).
- Jobs FP16 **`50141327`** FAILED 6:23 (h36-5; analyzer undeclared cuda_graphs) / FP32 **`50141328`** COMPLETED PASS 6:50 (h36-5). Already finished when strategy wake ran — **no cancel needed**.
- Engage: `self_wqkv_linear_enabled=true` opb **8**, decode calls 174 (FP32 dispatch hits 120/12/12; capture-hit counter quirk 0).
- Exact: FP16 **FAIL** (tok 7809→7981 / HO 645→597 / `.osu` mismatch); FP32 **PASS** (8294/637).
- Main vs tip: FP16 invalid; FP32 **319.85 / 319.40 (+2.17% / +2.03%)** — under ≥5% gate.
- Decision: **STOP_NO_PROMOTE / DROP for strategy pivot**. Revisit native Wqkv **only after** structure levers (§29/§30). Do not bare-retry; do not grind allowlist.

## Strategy (re-plan 2026-07-17 + scope ruling) — binding

**Track C is primary for 500** under user-approved §34 (verbatim in ledger + `docs/inference_evidence_packs.md`): non-bit-exact engines ARE in scope if quality-equivalent behind `inference_engine=v32|optimized|turbo` (one immutable preset per value; `turbo` = distribution-equivalent). Bit-exact `optimized` remains valid; tip still `55949274` / FP16 **366.11**. Server V32-only. **No merge without approval.**

### Track C primary ladder (§34–§38)

| Ledger § | Content | Status |
| --- | --- | --- |
| **§34** | SCOPE RULING — `inference_engine=v32\|optimized\|turbo`; evidence packs TIER1/2/3 | **RECORDED** (verbatim + `docs/inference_evidence_packs.md`) |
| **§35** | Layer-skip acceptance probe (cheap; FIRST) | **DONE** — E[acc]=**1.014** → **GO_SECTION_37** (`50145885`) |
| **§36** | Turbo speculative runtime (tiny-draft draft+verify) | **OPEN** — scout **`50147054`** harvested main_tps **11.94** (directional); TIER1a → §40 |
| **§37** | Tiny draft (2-layer CE/KL) → turbo runtime | **OPEN** — tip `3bfd7bdb`; train **`50146289`** E=**2.921**; speculative generate_window wired |
| **§38** | Tier-2 relaxed fused decoder step (7-kernel; fp32 acc) | **OPEN** — `codex/turbo-tier2-fused-step` / rung-1 logit probe |
| **§44** | TIER1 evidence pack harness (30×3 + KS + canary) | **HARNESS READY** — `codex/turbo-tier1-harness` / `scripts/run_tier1_evidence_pack.sh` (GPU execute deferred; no 500 claim; §39 not duplicated) |
| **§40** | TIER1a canary fix (mismatch@110) | **STOP_ESCALATE** — cross-engine FP16 numerics (`50147276`); → **§41** |
| **§41** | Verify fastpath + graph-aligned teacher (W2) | **PARTIAL** — tip `7033d62f`; c_verify **1.686×** MISS (`50147970`); canary@110 **PASS** (`50148210`) |

### Track A continues (bit-exact structure; still valid)

Tip is **launch / structure bound**, not single-kernel bound:

| Fact | Value |
| --- | --- |
| Launches / token (node path) | **~400** |
| Bandwidth roofline | **~22%** of peak — not BW-limited |
| Gap to 500 (FP16) | **≥0.6 ms/token** (5.71 s / 7809 ≈ **0.73 ms/token**) |
| Stop doing | More single-kernel / native-fuse / allocator / Inductor-inside-capture scouts (§6/§7/§10/§11/§16–§18/§28 family) |
| Prefer (Track A) | New ≥0.15 ms/token measured evidence only; §33 OPEN instrumentation (nsys `49966210`; ncu blocked) |

**Per-token budget (FP16 tip → 500):** need ~0.73 ms/token cut on main. **Probe ladder (Track A):** claim **≥0.15 ms/token** from *measured* data (nsys/NVTX/capture smoke) **before** full-song DCC reciprocal.

**Explicitly deprioritized:** megakernels, allocator envs (§16/§17), Inductor-inside-capture (§13/§14/§22), more single-GEMV/native-fuse scouts (§6/§7/§10/§11/§28). INT8 still does not count. No merge.

### Research → ledger § mapping (do **not** reuse §24–§28)

| Research name | Ledger § | Content | Status |
| --- | --- | --- | --- |
| Research §24 whole-token-step CUDA graph | **§29** | Forward+logits/processors in graph; sample/append eager at tip `e11fb7ab` | **STOP_NO_PROMOTE** (exact; main −8.7% vs tip) |
| Research §25 elementwise/copy fusion | **§30** | Fuse elementwise-only chains; never matmul reductions | **STOP_NO_PROMOTE** (`50144615` 0.05 ms/tok) |
| Research §26 batch-invariant speculative | **§31** | After free CPU n-gram acceptance probe | **PARKED** (max ~1.06 accepted/step &lt;1.3) |
| Research §27 CUDA graph WHILE/conditional | **§32** | After whole-token graph | **STOP_NO_PROMOTE** (exact; **STOP budget miss** −0.20≪0.15 ms/tok) |
| Research §28 q1 occupancy | **§33** | Tip FP16 nsys node trace + ncu when unblocked | **OPEN** (instrumentation; `49966210`) |
| Track C (scope §34) | **§35–§38** | layer-skip → self-spec turbo → tiny draft → Tier-2 | **PRIMARY for 500**; §35 DONE; **§37/§36** @ `3bfd7bdb` (smoke retest `50147161`, scout `50147054`) |

### N-gram / prompt-lookup acceptance probe (CPU-only)

Source: tip auth SALVALAI profile dumps `49964133` / `49963835`. Script: `/work/imt11/Mapperatorinator/tmp/ngram_acceptance_probe.py`.

| Prec | n-gram | max_draft | **accepted/step** | match_frac | gt1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP16 | 2 | 3–8 | **1.047** | 0.342 | 0.044 |
| FP16 | 3 | 3–8 | 1.003 | 0.171 | 0.003 |
| FP32 | 2 | 3–8 | **1.059** | 0.364 | 0.053 |
| FP32 | 3 | 3–8 | 1.006 | 0.187 | 0.006 |

Gate: &lt;1.3 → park speculative; ≥1.5 → go after graph work. **Result: PARK §31.**

## §29 Whole-token-step CUDA graph (research §24) — **STOP_NO_PROMOTE** (eager + §29b Philox)

- Base tip **`55949274`** / branch `codex/exact-whole-token-cuda-graph` @ **`e11fb7ab`** (eager-sample STOP).
- Smoke **`50143692` PASS**; full-song FP16 **`50143838`** exact PASS but main **−8.73%** vs tip (local graph_cache tax).
- Decision (eager tip): **STOP_NO_PROMOTE**. Tip unchanged `55949274` / FP16 **366.11**.

### §29b Philox-safe sample graph — **STOP_NO_PROMOTE** (Track A closed)

- Diagnosis: **`50143356`** @ `e168b81b` failed at **generated idx 1** (4081→4083); CUDA RNG diverge; capture-as-first-sample Philox desync. Eager control **`50143692`** PASS.
- Fix tip **`e3f555f8`**: `register_generator_state` + RNG-restore capture warmup + always replay; shared-cache amortize; eager `index_copy_` append.
- Smoke FP16 **`50145494` PASS** (z25-21, 3:12): tok 1322; RNG equal; hits 1302.
- Budget **MISS**: **−0.043 ms/token** vs smoke baseline (need ≥0.15). Reciprocal **not submitted**.
- Decision: **STOP_NO_PROMOTE** — Philox-in-graph exactness proven; no bit-exact headroom. Tip still `55949274` / **366.11**.
- Track A bit-exact structure closed here; Track C (§34–§38) is primary for 500 via separate `turbo` flag. §30/§32 STOP. §31 PARKED. §33 **OPEN** instrumentation (`49966210`; ncu blocked).

## §30 Elementwise/copy fusion (research §25) — **STOP_NO_PROMOTE**

- Branch **`codex/exact-elementwise-copy-fusion`** @ **`30ea745d`** (from tip `55949274`).
- Rung 1: top-5 from nsight **`49966210`** — `direct_copy_cast`, `float16_copy`, float `mul`, `add<Half>`, RoPE `cos`/`sin`.
- Rung 2: opt-in fuse landed (RoPE epilogue kernel + q1 pack reshape).
- Rung 3: component **`50144615`** COMPLETED — exact **PASS**; projected **0.050 ms/token** (need ≥0.15) → budget miss.
- Smoke/reciprocal **not submitted**. Tip unchanged `55949274` / **366.11**.
- Decision: **STOP_NO_PROMOTE**. Next Track A: **§32** WHILE/conditional. §31 PARKED. No merge.

## §32 CUDA graph WHILE/conditional (research §27) — **STOP_NO_PROMOTE**

- Base tip **`55949274`** / branch **`codex/exact-cuda-graph-while-conditional`** @ **`6bc01810`**.
- Rung 1 toy WHILE **`50144761` PASS** (z25-20; CUDA 12.8 / 2080 Ti).
- Rung 2 step wrap **`50145010` PASS** exact (forced stops 1/3/7; no post-stop waste) but budget **MISS**: best WHILE−k1 **−0.199 ms/token** (need ≥0.15); fixed 8-step −6.6% vs k1.
- Smoke/reciprocal **not submitted**. Tip unchanged `55949274` / **366.11**.
- Decision: **STOP_NO_PROMOTE**. Revisit only with full-window host-launch probe ≥0.15 ms/token — not bare retry. §33 **OPEN** instrumentation (`49966210`; ncu blocked). §31 PARKED. No merge. No INT8.

## §33 q1 occupancy (research §28) — **OPEN (instrumentation)**

- Documented tip FP16 nsys node job **`49966210`** @ `55949274` (`--cuda-graph-trace=node`, smoke_node **1158** tok).
- q1 kernel **18.79%** stage GPU (~402 µs/tok; avg 33.8 µs). Host: GraphLaunch~42% / LaunchKernel~38% of stage API time.
- **`ncu` BLOCKED** `ERR_NVGPUCTRPERM` — admin request in `notes/500tps-section33-q1-occupancy-instrumentation.md`.
- Decision: **OPEN**. Revisit = microbench only after counters exist. No WHILE bare-retry. No single-kernel math-replace. No turbo under this §.

## Branches / worktrees

| Track | Branch | Tip | Role |
| --- | --- | --- | --- |
| FP32/FP16 | `codex/exact-shared-rope-device-state` | **`55949274`** | **GRADUATED campaign tip** (auth FP16 366.11 / FP32 313.05) |
| FP32/FP16 | `codex/exact-self-out-residual-fusion` | **`4e3477a2`** | full-song last look `49978677`/`678` **FAILED** — **STOP_NO_PROMOTE** / DROP (unused residual) |
| FP32/FP16 | `codex/exact-self-out-residual-fuse` | **`57fa6612`** | Live r2 full-song Wo+residual (jobs `49976417`/`451`) |
| FP32/FP16 | `codex/exact-self-norm-wqkv` | **`fd126612`** | r3 **STOP_NO_PROMOTE** (`49982390`/`391`); do not bare-retry |
| FP32/FP16 | `codex/exact-proj-out-fuse` | **`585ffc90`** | §10 final-norm+proj_out — **STOP_NO_PROMOTE** (`49989856`/`857`) |
| FP32/FP16 | `codex/exact-self-wo-linear` | **`c48f5b8b`** | §11 Wo one-token linear — **STOP_NO_PROMOTE** (`49993296`/`297` exactness collapse) |
| FP32/FP16 | `codex/exact-q1-rope-cache-headgroup` | **`dd5d8e58`** | §12 q1 headgroup — **STOP_NO_PROMOTE** (exact; main −42/−31 TPS) |
| FP32/FP16 | `codex/exact-compiled-self-proj` | **`3164875a`** | §13 compiled self Wqkv/Wo — **STOP_NO_PROMOTE** (`50001853`/`854`) |
| FP32/FP16 | `codex/exact-compiled-self-proj-decode-only` | **`43fea0c2`** | §14 decode-only Wqkv/Wo — **STOP_NO_PROMOTE** (`50031125`/`126`) |
| FP32/FP16 | `codex/exact-compiled-proj-out-decode-only` | **`5aa6eb9f`** | §15 decode-only `proj_out` compile — **STOP_NO_PROMOTE** (`50036330`/`325`) |
| FP32/FP16 | `codex/exact-cuda-alloc-expandable` | **`fde4f0f2`** | §16 CUDA expandable_segments — **STOP_NO_PROMOTE** (`50039777`/`776`) |
| FP32/FP16 | `codex/exact-cuda-alloc-malloc-async` | **`b9b60cd5`** | §17 CUDA cudaMallocAsync — **STOP_NO_PROMOTE** (`50050003`/`002`) |
| FP32/FP16 | `codex/exact-compiled-decode-logits-finalize` | **`b47a3fe2`** | §18 decode logits+softmax — **STOP_NO_PROMOTE** (`50107685`/`686`) |
| FP32/FP16 | `codex/exact-q1-out-reshape` | **`5f4b9989`** | §19 q1 attn-out reshape — **STOP_NO_PROMOTE** (`50129056`/`057`) |
| FP32/FP16 | `codex/exact-compiled-proj-out-gated` | **`8bd44130`** | §21 unwrap — **STOP_NO_PROMOTE** (`50132354`/`355`; hits=0) |
| FP32/FP16 | `codex/exact-compiled-self-wo-residual` | **`a5ea705d`** | §22 compiled Wo+residual — **STOP_NO_PROMOTE** (`50133112`/`113`) |
| FP32/FP16 | `codex/exact-q1-mask-workspace` | **`8759db3a`** | §23 q1 mask workspace — **STOP_NO_PROMOTE** (`50138023`/`024`) |
| FP32/FP16 | `codex/exact-active-prefix-bucket` | **`1076bfd3`** | §24 active-prefix bucket 256 — **STOP_NO_PROMOTE** (`50138950`/`951`) |
| FP32/FP16 | `codex/exact-active-prefix-exact-length` | **`25376173`** | §25 active-prefix exact-length bucket=1 — **STOP_NO_PROMOTE** (`50139609`/`610`) |
| FP32/FP16 | `codex/exact-native-mlp-opb` | **`b1736b0b`** | §26 native MLP outputs_per_block 8→4 — **STOP_NO_PROMOTE** (`50140036`/`037`) |
| FP32/FP16 | `codex/exact-q1-block-size` | **`10df9bc0`** | §27 native q1 block_size 128→256 — **STOP_NO_PROMOTE** (`50140877`/`878`) |
| FP32/FP16 | `codex/exact-self-wqkv-linear` | **`f5c6b901`** | §28 owned self-Wqkv linear — **STOP_NO_PROMOTE** (`50141327`/`328`; native-fuse pivot) |
| FP32/FP16 | `codex/exact-whole-token-cuda-graph` | **`e3f555f8`** | §29b Philox sample graph — **STOP** (`50145494` exact; −0.043≪0.15 ms/tok) |
| FP32/FP16 | `codex/exact-elementwise-copy-fusion` | **`30ea745d`** | §30 elementwise/copy fusion — **STOP_NO_PROMOTE** (`50144615` 0.05≪0.15) |
| FP32/FP16 | `codex/exact-cuda-graph-while-conditional` | **`6bc01810`** | §32 CUDA graph WHILE — **STOP_NO_PROMOTE** (`50144761`/`50145010`; budget miss) |
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
| **`50143838`** | z25-21 | §29 whole-token graph FP16 @ `e11fb7ab` | **FAILED** 1:0 5:59 | exact PASS hits 8456; main **334.15** (−8.7% vs tip) — **STOP_NO_PROMOTE** |
| **`50143692`** | z25-21 | §29 capture smoke FP16 @ `e11fb7ab` | **COMPLETED** | tok/RNG PASS hits 1302 — smoke gate green |
| `50142058`…`50143356` | — | §29 smoke FIX ladder | **FAILED** | sample-graph / hits / tok drift — superseded by `e11fb7ab` |
| **`50141327`** | h36-5 | §28 self-wqkv-linear FP16 @ `f5c6b901` | **FAILED** 6:23 | exact FAIL tok/HO — **STOP_NO_PROMOTE** |
| **`50141328`** | h36-5 | §28 self-wqkv-linear FP32 @ `f5c6b901` | **COMPLETED** | exact PASS; vs tip +2.2%/+2.0% — **STOP_NO_PROMOTE** |
| **`50140877`** | z25-20 | §27 q1-block-size FP16 @ `10df9bc0` | **FAILED** 7:19 | exact FAIL tok/HO — **STOP_NO_PROMOTE** |
| **`50140878`** | z25-21 | §27 q1-block-size FP32 @ `10df9bc0` | **COMPLETED** | exact PASS; vs tip −1.4%/−1.0% — **STOP_NO_PROMOTE** |
| **`50140036`** | z25-21 | §26 native-mlp-opb FP16 @ `b1736b0b` | **FAILED** | exact PASS; under tip 5% — **STOP_NO_PROMOTE** |
| **`50140037`** | z25-21 | §26 native-mlp-opb FP32 @ `b1736b0b` | **FAILED** | exact PASS; under tip 5% — **STOP_NO_PROMOTE** |
| **`50139609`** | z25-20 | §25 active-prefix-exact-length FP16 @ `25376173` | **FAILED** | cand OOM (graph pools) — **STOP_NO_PROMOTE** |
| **`50139610`** | z25-21 | §25 active-prefix-exact-length FP32 @ `25376173` | **FAILED** | exact PASS; −19%/−20% vs tip; gdelta~50 — **STOP_NO_PROMOTE** |
| **`50138950`** | z25-21 | §24 active-prefix-bucket FP16 @ `1076bfd3` | **FAILED** | exact PASS; −5.2% vs tip; gdelta 12→4 — **STOP_NO_PROMOTE** |
| **`50138951`** | z25-21 | §24 active-prefix-bucket FP32 @ `1076bfd3` | **FAILED** | exact PASS; flat vs tip — **STOP_NO_PROMOTE** |
| **`50138023`** | z25-20 | §23 q1-mask-workspace FP16 @ `8759db3a` | **FAILED** | exact PASS; hits=0; −14.8% vs tip — **STOP_NO_PROMOTE** |
| **`50138024`** | z25-21 | §23 q1-mask-workspace FP32 @ `8759db3a` | **FAILED** | exact PASS; −1.0% vs tip — **STOP_NO_PROMOTE** |
| **`50133112`** | z25-20 | §22 compiled-self-wo-residual FP16 @ `a5ea705d` | **FAILED** | cand capture Inductor sync — **STOP_NO_PROMOTE** |
| **`50133113`** | z25-21 | §22 compiled-self-wo-residual FP32 @ `a5ea705d` | **FAILED** | same — **STOP_NO_PROMOTE** |
| **`50132354`** | z25-20 | §21 gated-proj-out r3 FP16 @ `8bd44130` | **FAILED** | exact PASS; hits=0 — **STOP_NO_PROMOTE** |
| **`50132355`** | z25-21 | §21 gated-proj-out r3 FP32 @ `8bd44130` | **FAILED** | exact PASS; hits=0 — **STOP_NO_PROMOTE** |
| **`50129056`** | — | §19 q1-out-reshape FP16 @ `5f4b9989` | **FAILED** | exact PASS; +0.6–1.1% vs tip — **STOP_NO_PROMOTE** |
| **`50129057`** | — | §19 q1-out-reshape FP32 @ `5f4b9989` | **FAILED** | exact PASS; no tip ≥5% — **STOP_NO_PROMOTE** |
| **`50107685`** | h36-5 | §18 compiled-logits-finalize FP16 @ `b47a3fe2` | **FAILED** 6:48 | exact PASS; flat/−0.4% — **STOP_NO_PROMOTE** |
| **`50107686`** | h36-5 | §18 compiled-logits-finalize FP32 @ `b47a3fe2` | **FAILED** 6:48 | exact PASS; FP32 regress — **STOP_NO_PROMOTE** |
| **`50050003`** | z25-21 | §17 cudaMallocAsync FP16 @ `b9b60cd5` | **COMPLETED** | exact PASS; −4.0% vs tip — **STOP_NO_PROMOTE** |
| **`50050002`** | z25-21 | §17 cudaMallocAsync FP32 @ `b9b60cd5` | **COMPLETED** | exact PASS; −7.8% vs tip — **STOP_NO_PROMOTE** |
| `50050004`/`001` | — | §17 dups (accidental double-submit) | **COMPLETED** | same direction; ignore for claim |
| **`50039777`** | z25-21 | §16 expandable_segments FP16 @ `fde4f0f2` | **COMPLETED** | exact PASS; +2.0% vs tip — **STOP_NO_PROMOTE** |
| **`50039776`** | z25-20 | §16 expandable_segments FP32 @ `fde4f0f2` | **COMPLETED** | exact PASS; −2.2% — **STOP_NO_PROMOTE** |
| **`50036330`** | z25-21 | §15 decode-only compiled-proj-out FP16 @ `5aa6eb9f` | **FAILED** 3:23 | timing proj_out rank-2 crash — **STOP_NO_PROMOTE** |
| **`50036325`** | z25-20 | §15 decode-only compiled-proj-out FP32 @ `5aa6eb9f` | **FAILED** 4:22 | same — **STOP_NO_PROMOTE** |
| **`50031125`** | z25-20 | §14 decode-only compiled-self FP16 @ `43fea0c2` | **FAILED** 7:20 | exact PASS; FP16 −0.7% — **STOP_NO_PROMOTE** |
| **`50031126`** | z25-21 | §14 decode-only compiled-self FP32 @ `43fea0c2` | **FAILED** 6:28 | exact PASS; FP32 +2.5% — **STOP_NO_PROMOTE** |
| **`50024784`** | z25-20 | §12 q1 headgroup FP16 FIX @ `dd5d8e58` | **FAILED** 7:24 | exact PASS; main −42 TPS — **STOP_NO_PROMOTE** |
| **`50024785`** | z25-21 | §12 q1 headgroup FP32 FIX @ `dd5d8e58` | **FAILED** 6:27 | exact PASS; main −31 TPS — **STOP_NO_PROMOTE** |
| **`50000634`/`635`** | z25-20 | §12 q1 headgroup @ `64372cde` | **FAILED** 1:0 | infra `profile: unbound variable` after baseline — not STOP |
| **`50002697`/`698`** | z25-20 | §12 q1 headgroup retry @ `64372cde` | **FAILED** 1:0 | same unbound `profile` — not STOP |
| **`50001853`** | z25-20 | §13 compiled-self-proj FP16 @ `3164875a` | **FAILED** 1:0 | `FailOnRecompileLimitHit` — **STOP_NO_PROMOTE** |
| **`50001854`** | z25-20 | §13 compiled-self-proj FP32 @ `3164875a` | **FAILED** 1:0 | same — **STOP_NO_PROMOTE** |
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
5. ~~§11–§18~~ **STOP_NO_PROMOTE** (allocator + logits-softmax closed).
6. ~~§19–§27~~ STOP (incl. q1 block_size FP16 exactness FAIL; MLP opb &lt;5%).
7. ~~Harvest §28 owned self-Wqkv linear~~ **STOP_NO_PROMOTE** (`50141327` FP16 exact FAIL; `50141328` FP32 &lt;5%). Native-fuse pivot → sibling whole-token CUDA-graph plan.

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
4. ~~Harvest §11 self-wo linear~~ **STOP_NO_PROMOTE** (exactness collapse).
5. ~~Harvest §13 compiled self Wqkv/Wo~~ **STOP_NO_PROMOTE** (`50001853`/`854` FailOnRecompileLimitHit).
6. ~~Harvest §12 q1 headgroup~~ **STOP_NO_PROMOTE** (`50024784`/`785` exact but main −42/−31 TPS).
7. ~~Harvest §14~~ **STOP_NO_PROMOTE** (`50031125`/`126` exact; FP16 −0.7%, FP32 +2.5%).
8. ~~Harvest §15~~ **STOP_NO_PROMOTE** (`50036330`/`325` timing proj_out crash).
9. ~~Harvest §16/§17 allocators~~ **STOP_NO_PROMOTE**.
10. ~~Harvest §18–§28~~ **STOP_NO_PROMOTE**. Tip still 366.11.
11. ~~§29 whole-token-step CUDA graph (eager tip)~~ **STOP_NO_PROMOTE** (`50143838` exact; main −8.7% vs tip).
12. ~~§29b Philox-safe sample graph~~ **STOP_NO_PROMOTE** @ `e3f555f8` (`50145494` exact; −0.043≪0.15 ms/tok; no recip). Track A closed.
13. ~~§30 elementwise/copy fusion~~ **STOP_NO_PROMOTE** @ `30ea745d` (`50144615` exact PASS, 0.05≪0.15 ms/tok).
14. ~~§32 CUDA graph WHILE/conditional~~ **STOP_NO_PROMOTE** @ `6bc01810` (`50144761`/`50145010` exact; −0.20≪0.15 ms/tok / **STOP budget miss**). Do **not** bare-retry WHILE.
15. **§33 q1 occupancy OPEN (instrumentation)** — documented tip nsys node job **`49966210`** (FP16 smoke_node, `--cuda-graph-trace=node`; q1 **18.79%** / ~402 µs/tok). **`ncu` BLOCKED** `ERR_NVGPUCTRPERM`. Artifact: `notes/500tps-section33-q1-occupancy-instrumentation.md`. Revisit = microbench only after counters exist. No single-kernel math-replace scouts.
16. **§34 SCOPE RULING recorded** — `v32\|optimized\|turbo`; TIER1/2/3 in `docs/inference_evidence_packs.md`.
17. **§35 layer-skip acceptance probe DONE** — job **`50145885`**: E[accepted/step]=**1.0138** → **GO_SECTION_37**. §36 **SKIPPED**.
18. **§37/§36 turbo speculative WIRED** — scout **`50147054`** @ `f0b6565a` COMPLETED: main_tps **11.94** / main_model **467.34 s** (directional; ≪366.11; **no 500 claim**). TIER1a ownership → **§40** / `codex/turbo-canary-fix`. Campaign tip still `55949274` / **366.11**. No merge.
19. **§39 hybrid-arena TIER3 FAIL** — 90 maps/engine × 3 songs; exact `50145980` @ `55949274` FP16 vs hybrid `50145981`/`982`/`983` @ `0dbab9e5` INT8-hybrid (not FP16). Audit **`50147476`**: KS **slider_lengths FAIL** (p=2.1e-17); HO/density/type/timeshift PASS; classifier-FID **0.384** (weak, advisory); rcomplexion PASS; MaiMod incomplete (`rich` missing). **No `turbo_mixed`.** No campaign-close ruling (FAIL path). Report: `/work/imt11/Mapperatorinator/runs/s39-hybrid-tier3-audit-50147476/tier3_report.json`.
20. **§40 TIER1a STOP_ESCALATE** — logit dump **`50147276`**: eager=batched=**2236**, optimized=**2213** (Δ≈0.008). Not rejection-rule; not K-batch bug. Escalated into **§41** W2. Handoff: `notes/500tps-section40-canary-handoff.md`. No 500 claim.
21. **§41 W2 PARTIAL** — tip `7033d62f` / auth canary tip `b3a0b27e`; c_verify **1.686×** MISS (`50147970`); canary@110 **PASS** 2213 (`50148210`). Handoff: `notes/500tps-section41-verify-fastpath-handoff.md`.

22. **§45 combined turbo STOP_DEAD_END** — scout **`50148770`** main_tps **44.52** / sustained ~**23 ms/tok**; TIER1a PASS `50148575`. Diagnose: crop-to-L+rebuild doubles teacher; structural ceiling ~**310 TPS** (no rebuild) at c_verify 1.686× + c_draft 0.36× + E≈1.97 ≪ tip/≥384. **No §46**; no §44; no tip graduate. Handoff: `notes/500tps-section45-handoff.md`. Tip still `55949274` / **366.11**.
23. **Track C speculative closed** — next **§38 TIER2 fused step** (or reopen spec only after c_verify≤1.2× and c_draft≤0.15× with TIER1a-safe keep-prefix KV).

## Standing orders

- Prefer **Grok 4.5 High** / auto (non-fast); do **not** use `cursor-grok-4.5-high-fast`.
- One candidate per worktree/node; serialize graduation.
- Never present projections as production TPS.
- V32 cold default; `optimized` under `osuT5/.../inference/optimized/` (bit-exact, default-off unchanged); future `turbo` is a separate immutable preset.
- **Track C primary for 500:** speculative **STOP_DEAD_END** (§45). **§38 TIER2 fused step OPEN** (`codex/turbo-tier2-fused-step`). Tip still `55949274` / **366.11**. No merge. No INT8-as-FP16. No hybrid-500 / no `turbo_mixed`. No 500 claim without TIER2(+TIER1c).
- Wake: **§38 OPEN** rung-1 teacher-forced logit agreement. Tip 55949274/366.11. No 500 claim. No merge.

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
