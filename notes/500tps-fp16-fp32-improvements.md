# 500 TPS FP16/FP32 — per-improvement ledger

Separate entry for **each** lever / packaging change / FIX attempt.
Do not fold multiple improvements into one line. Projections are never production TPS.

**Binding goal:** ≥500 TPS on RTX 2080 Ti for **FP16 and/or FP32 only** (INT8/hybrid does not count).
**Authoritative tip (graduated):** `55949274` — FP16 **366.11** TPS / 21.330 s (7809 tok); FP32 **313.05** / 26.494 s (8294).
**Gap to 500:** FP16 −133.9 TPS / −5.71 s (need 1.366×); FP32 −187.0 / −9.91 s (need 1.596×).
**Coordinator handoff:** `notes/500tps-combo3-handoff.md`.

---

## 1. Exact shared-RoPE + device sequence state

| Field | Value |
| --- | --- |
| What | Persist shared RoPE + device sequence state on optimized single |
| Branch / tip | `codex/exact-shared-rope-device-state` / **`55949274`** |
| Auth jobs | FP32 `49963835`; FP16 `49964133` |
| Exact | **PASS** (token IDs / stopping / `.osu`) |
| Measured | FP16 **366.11** TPS (+34.4 vs shared-rope base); FP32 **313.05** (+26.0) |
| Wall | FP16 main −2.21 s; song-wall also improved |
| Decision | **GRADUATE** — campaign tip |
| Why it matters | Only graduated multi-second FP16/FP32 win in this campaign so far |

---

## 2. Exact shared-runtime packaging pin

| Field | Value |
| --- | --- |
| What | Sealed packaging of shared-runtime composition (not tip merge) |
| Branch / tip | `codex/exact-shared-runtime-promotion` / `d41981ae` |
| Auth jobs | FP32 `49906335`; FP16 `49906334` |
| Exact | Sealed |
| Measured | FP32 **317.46** TPS (absolute pin); FP16 313.54 (superseded by tip) |
| Decision | **SEALED** packaging pin — **not merged**; do not revert tip to chase FP32 +4 TPS |
| Note | `notes/exact-shared-runtime-packaging.md` |

---

## 3. Exact compiled-cross BMM (on tip)

| Field | Value |
| --- | --- |
| What | Port hybrid compile-before-capture cross BMM onto exact tip |
| Branch / tip | `codex/exact-compiled-cross-bmm` / `25d8e469` (earlier `b33d9e6c`) |
| Jobs | `49966195`/`196`/`208`/`209`; r2 `49968303`/`305` |
| Exact | FP16 **FAIL** (7809→8402); FP32 byte-exact |
| Measured | FP16 ~409 TPS **invalid**; FP32 ~318 TPS (~1%, &lt;5%) |
| Nsight | Cross BMM ~2% node-path GPU time |
| Decision | **STOP_NO_PROMOTE** — do not grind |

---

## 4. Decode cast/copy elimination (`cast-elim`)

| Field | Value |
| --- | --- |
| What | Logits workspace + zero-copy attn flatten to cut cast/copy traffic |
| Branch / tip | `codex/exact-decode-cast-elim` / `a354624f` |
| Jobs | FP16 `49974095` (COMPLETED); FP32 `49974096` (FAILED baseline token flake) |
| Exact | FP16 **PASS** |
| Measured | FP16 main_tps **356.25 → 337.18 (−19.1 TPS / +1.23 s)**; vs tip −28.9 TPS |
| Cold wall | Looks “faster” but is cold/compile noise — **not** the claim |
| Decision | **DROP** — exact but main regress |

---

## 5. Decode cast/copy sibling (`cast-copy`)

| Field | Value |
| --- | --- |
| What | Parallel-agent cast/copy variant + allowlist r2 |
| Branch / tip | `codex/exact-decode-cast-copy` / `baf05d95` → allowlist `11766d07` |
| Jobs | r1 `49974091` (unused `*decode_cast_copy*`); r2 FP16 `49976415` / FP32 `49976416` |
| Exact | r2 **PASS** |
| Measured | FP16 371.5→348.9 (**−22.6 TPS**); FP32 319.1→313.8 (**−5.3 TPS**) |
| Decision | **DROP / STOP_NO_PROMOTE** — exact but main regress; do not grind allowlists |

---

## 6. Self-out + residual fuse (`native_one_token_linear_residual` on Wo)

| Field | Value |
| --- | --- |
| What | Fuse Wo linear + residual via owned `linear_residual` |
| Branch / tip | `codex/exact-self-out-residual-fusion` / `e1c286e1` → FIX chain `4e3477a2`; sibling `b6135df0`/`57fa6612` |
| Component jobs | Mac-audio FAIL `49973580`/`81` → FIX audio `01e37834` → binding unwrap `f8b42e49` → allowlist `4e3477a2`; r3 `49978315`/`316` |
| Component result | FP32 sizing PASS (~2.0–2.4 s **projected**); FP16 drift &gt;1e-3 — **not production TPS** |
| Full-song jobs | r1 `49974092`/`93`; sibling `49974094`; r2 `49976417`/`451`; last look `49978677`/`678` |
| Exact | **FAIL** — HO collapse (e.g. 645→3 / 637→1); tokens 7809→629 / 8294→562 |
| Last look | unused `*native_one_token_linear_residual*` (+ undeclared dispatch) — fusion not engaged |
| Decision | **STOP_NO_PROMOTE / DROP** — exactness collapse; component projection ≠ e2e |

### 6a. Component harness FIX (audio path) — separate infra/code FIX

| Field | Value |
| --- | --- |
| What | DCC audio override for component profiler |
| Tip | `01e37834` |
| Why | `profile_salvalai.yaml` Mac path broke `49973580`/`81` |
| Decision | **FIX applied** (enables sizing; does not graduate Wo fuse) |

### 6b. Component harness FIX (InferenceEngineBinding unwrap) — separate

| Field | Value |
| --- | --- |
| What | Unwrap binding so Wo extract sees `nn.Module` |
| Tip | `f8b42e49` |
| Why | Post-audio `49974790`/`91` `TypeError: optimized engine did not expose a torch.nn.Module` |
| Decision | **FIX applied** |

### 6c. Reciprocal allowlist FIX (dispatch/graph deltas) — separate

| Field | Value |
| --- | --- |
| What | Allow expected CUDA-graph / dispatch metadata deltas |
| Tip | `4e3477a2` |
| Why | Analyzer aborted on undeclared `optimized_cuda_graphs*` |
| Decision | **FIX applied** for analyzer; **did not** fix exactness / unused residual |

---

## 7. Self RMSNorm + Wqkv fuse (`native_one_token_rmsnorm_linear`, 3×640)

| Field | Value |
| --- | --- |
| What | Fuse self RMSNorm + Wqkv on decode path |
| Branch / WT | `codex/exact-self-norm-wqkv` / DCC `exact-self-norm-wqkv` |
| Base tip | `55949274` |
| Opt-in | `self_norm_wqkv_fusion_candidate_context` (V32 cold default) |

### 7a. Scout tip `e9ad0259` (r1)

| Field | Value |
| --- | --- |
| Jobs | FP16 `49979210`; FP32 `49979211` |
| Failure | `RuntimeError: fuse_self_norm_wqkv requires native q1 RoPE/cache self-attention; refusing unnormalized Wqkv fallback` on candidate_first |
| Decision | **FIX needed** (not bare retry) |

### 7b. Scope FIX `63869511` (r2) — separate attempt

| Field | Value |
| --- | --- |
| What | Scope RMSNorm attach/skip to decoder decode shape `(1,1)` |
| Jobs | FP16 `49980564`; FP32 `49980565` |
| Failure | **Same RuntimeError** — TIMING CUDA-graph capture is `(1,1)` but tip disables native q1 for `ContextType.TIMING` |
| Decision | **Insufficient FIX** — do not bare-retry |

### 7c. Native-q1 gate FIX `fd126612` (r3) — separate attempt

| Field | Value |
| --- | --- |
| What | Arm fuse only when native q1 live; restore RMSNorm on non-q1 fallback; TIMING no-op |
| Jobs | FP16 `49982390`; FP32 `49982391` (infra miss `49982320` superseded) |
| Evidence | Both show `self_norm_wqkv_enabled=True`, `self_norm_wqkv_calls=174` |
| FP16 result | Analyzer FAIL undeclared dispatch; **exactness FAIL** tok **7809→8483**, HO **645→651**; TPS not comparable |
| FP32 result | Analyzer FAIL unused `*one_token_rmsnorm_linear*` / `*optimized_cuda_graphs*` (pattern mismatch vs 174 calls); tok/HO stable 8294/637; salvage main_tps ~312.7→~317.1 (**~+1.4%**, &lt;5%) |
| Decision | **STOP_NO_PROMOTE** for promotion claim — FP16 exactness collapse; FP32 no ≥5% even if allowlist fixed. Revisit only with a **new** exactness+gain hypothesis (not bare retry). |

---

## 8. Nsight lever map (measurement only — not an improvement)

| Field | Value |
| --- | --- |
| Job | `49966210` on tip `55949274` |
| Families | elementwise+memory ~30%; gemm/projection ~26%; native q1 self ~19%; fused MLP ~15%; sampling ~7%; cross FMHA ~2% |
| Decision | Evidence for ordering levers — not a graduated change |

---

## 9. Explicitly demoted / out of 500 scope (separate archive)

| Lever | Decision | Note |
| --- | --- | --- |
| Hybrid INT8 arena / compiled-cross last-mile | **Demoted** | ~494 does **not** count as 500 |
| DP4A / FlashDecode / CUTLASS-without-headers | **DROP** | — |
| ContiguousKv hybrid last-mile | **DROP** | — |
| Encoder precompute `49903861` | **STOP_ENCODER_PRECOMPUTE** | wall regress |
| FP16 split-KV `4b0adc10` | **STOP_NO_GAIN** | reformulate only |
| Strict FP32 split-KV | Parked | — |

---

## Running scoreboard (graduated / sealed only)

| Rank | Improvement (#) | FP16 main_tps | FP32 main_tps | Status |
| --- | ---: | ---: | ---: | --- |
| 1 | #1 shared-RoPE + device state | **366.11** | 313.05 | **GRADUATED tip** |
| 2 | #2 shared-runtime packaging | 313.54 | **317.46** | **SEALED** pin only |
| — | #3–#7c, #10–#15 | — | — | no graduate |
| — | #16 CUDA expandable_segments | exact PASS | FP16 +2.0% vs tip; FP32 −2.2% | **STOP_NO_PROMOTE** `50039777`/`776` @ `fde4f0f2` |
| — | #17 CUDA cudaMallocAsync | exact PASS | FP16 −4.0% / FP32 −7.8% vs tip | **STOP_NO_PROMOTE** `50050003`/`002` @ `b9b60cd5` |
| — | #18 decode logits workspace + compiled softmax | exact PASS | FP16 flat/−0.4%; FP32 regress | **STOP_NO_PROMOTE** `50107685`/`686` @ `b47a3fe2` |
| — | #19 q1 attn-out reshape (skip contiguous) | exact PASS | vs tip +0.6–1.1% (noisy recip) | **STOP_NO_PROMOTE** `50129056`/`057` @ `5f4b9989` |
| — | #20 gated decode-only compiled proj_out (§15 FIX) | exact PASS | hits=0 / prepare_failed; vs tip FP16 −14–20% | **STOP_NO_PROMOTE** `50129762`/`763` @ `91977df2` |
| — | #20b prepare try/except rank-2 skip (§20 FIX) | exact PASS | hits=0 @ `3215d14b` | **STOP_NO_PROMOTE** `50131926`/`927` |
| — | #21 compiled proj_out unwrap+dtype engage | exact PASS | hits=0 / `missing_rank2_proj_out_weight`; vs tip FP16 −15% | **STOP_NO_PROMOTE** `50132354`/`355` @ `8bd44130` |
| — | #22 decode-only compiled self Wo+residual | n/a (cand crash) | n/a | **STOP_NO_PROMOTE** `50133112`/`113` @ `a5ea705d` |
| — | #23 tip-exact q1 float32 mask workspace | exact PASS | vs tip FP16 −14.8%/−15.3%; FP32 −1.0%/−1.6%; capture hits=0 | **STOP_NO_PROMOTE** `50138023`/`024` @ `8759db3a` |
| — | #24 tip-exact active-prefix bucket 256 | exact PASS | vs tip FP16 −5.0%/−5.2%; FP32 flat/−0.5%; gdelta 12→4 | **STOP_NO_PROMOTE** `50138950`/`951` @ `1076bfd3` |
| — | #25 tip-exact active-prefix exact-length (bucket=1) | FP16 OOM; FP32 exact | FP32 vs tip ~−19%/−20%; gdelta ~50 | **STOP_NO_PROMOTE** `50139609`/`610` @ `25376173` |
| — | #26 tip-exact native MLP outputs_per_block 8→4 | exact PASS | vs tip FP16 +1.1%/−7.7%; FP32 +1.5%/+1.4% | **STOP_NO_PROMOTE** `50140036`/`037` @ `b1736b0b` |
| — | #27 tip-exact native q1 block_size 128→256 | FAIL exact FP16; FP32 exact ~−1% | under tip 5% | **STOP_NO_PROMOTE** `50140877`/`878` @ `10df9bc0` |
| — | #28 tip-exact owned self-Wqkv linear | FAIL exact FP16; FP32 exact ~+2% | under tip 5%; native-fuse pivot | **STOP_NO_PROMOTE** `50141327`/`328` @ `f5c6b901` |
| — | #29 whole-token-step CUDA graph | exact PASS; hits 8456; main **−8.7%** vs tip | `50143692` smoke / `50143838` recip @ `e11fb7ab` | **STOP_NO_PROMOTE** |
| — | #29b Philox sample graph | smoke exact PASS; budget **−0.043≪0.15** ms/tok | `50145494` @ `e3f555f8` | **STOP_NO_PROMOTE** |
| — | #32 CUDA graph WHILE | exact PASS; budget **−0.20≪0.15** ms/tok | `50144761`/`50145010` @ `6bc01810` | **STOP_NO_PROMOTE** |
| — | #33 q1 occupancy | nsys node documented; q1 **18.79%**; ncu blocked | `49966210` | **OPEN** (instrumentation) |
| — | #34 SCOPE RULING | `v32\|optimized\|turbo` + TIER1/2/3 packs | — | **RECORDED** |
| — | #35 layer-skip acceptance | E[acc]=**1.014** (α=0.0137) @ 0.9/0.9 | `50145885` | **GO_SECTION_37** (§36 skipped) |

**Still short of 500:** FP16 needs ≤15.618 s main (−5.71 s from tip). Track C next = **§37 tiny draft**.

---

## Documentation rule (standing)

When any new lever, FIX tip, or scout job lands:

1. Add or update **one dedicated subsection** in this file (do not merge into a sibling lever).
2. Record: what / tip / jobs / exact / measured / decision / revisit condition.
3. Mirror the decision line into `notes/500tps-combo3-handoff.md`.
4. Never present component projections or invalid-token TPS as production throughput.

---

## 10. Owned proj_out / LM-head one-token fuse (final-norm + proj_out) — **STOP_NO_PROMOTE**

| Field | Value |
| --- | --- |
| What | Fuse decoder `layer_norm` + `proj_out` via owned `native_one_token_rmsnorm_linear` on one-token decode |
| Hypothesis | Exact reciprocal ≥5% main_model vs tip `55949274` on SALVALAI (FP16 primary; FP32 sibling) |
| Base tip | `55949274` |
| Branch / WT | `codex/exact-proj-out-fuse` / DCC `exact-proj-out-fuse` |
| Local/remote tip | `47ceae76` (pushed `islamtayeb/codex/exact-proj-out-fuse`) |
| DCC evidence tip | **`585ffc90`** (same tree content; re-committed on DCC after rsync) |
| Opt-in | `proj_out_fusion_candidate_context` → `native_proj_out` / `fuse_final_norm_proj_out` (V32 cold default) |
| Kernel | `native_one_token_rmsnorm_linear` (output_dim=vocab) |
| Jobs | FP16 **`49989856`** FAILED 7:12; FP32 **`49989857`** FAILED 6:18 — both analyzer unused `*proj_out_fuse*`/`*one_token_rmsnorm_linear*` (calls=174 — pattern mismatch) |
| Run roots | `/work/imt11/Mapperatorinator/runs/proj-out-fuse-fp{16,32}-4998985{6,7}/` |
| Exact | FP16 **FAIL**; FP32 salvage tok/HO match (analyzer unused-delta FAIL) |
| Measured | FP16 invalid (token mismatch). FP32 salvage main_tps ~317.2→312.9 (no gain) |
| Decision | **STOP_NO_PROMOTE** — FP16 exactness FAIL (7809→7869 / HO 645→588) with 174 fuse calls; FP32 no ≥5% (salvage ~317→313) |
| Revisit | Only with a new exactness+gain hypothesis (not allowlist-only retry) |
| Ledger rule | Keep this section for proj_out only; do not merge with Wo residual (§6) or RMSNorm+Wqkv (§7) |


## 11. Owned self-attn Wo one-token linear (no residual) — **STOP_NO_PROMOTE**

| Field | Value |
| --- | --- |
| What | Replace self-attn attention-out `Wo` with owned `native_one_token_linear`; residual stays a **separate** add (not §6 residual fuse) |
| Hypothesis | Exact reciprocal ≥5% main_model vs tip `55949274` on SALVALAI from remaining gemm/projection Wo linears (nsight `49966210` ~26% family) |
| Base tip | `55949274` |
| Branch / WT | `codex/exact-self-wo-linear` / DCC `exact-self-wo-linear` |
| Tip / commit | **`c48f5b8b`** |
| Opt-in | `self_wo_linear_candidate_context` → `native_self_wo_linear` + `skip_out_proj` (V32 cold default) |
| Kernel | `native_one_token_linear` (new; no residual; FP16/FP32 only) |
| Jobs | FP16 **`49993296`** FAILED 6:37; FP32 **`49993297`** FAILED 5:44 |
| Run roots | `/work/imt11/Mapperatorinator/runs/self-wo-linear-fp{16,32}-4999329{6,7}/` |
| Exact | **FAIL** — FP16 tok 7809→518 / HO 645→3; FP32 8294→562 / HO 637→1 (174 calls) |
| Measured | Invalid early-stop (not a TPS win) |
| Decision | **STOP_NO_PROMOTE** — exactness collapse (same family as §6) |
| Revisit | Only with component-exact kernel first; no bare retry |
| Not this lever | §6 Wo+residual; §7 RMSNorm+Wqkv; §10 final-norm+proj_out; INT8; bare split-KV |
| Ledger rule | Own section only; do not fold into §6/§7/§10 |

## 12. Native q1 RoPE/cache head-group CTA scheduling — **STOP_NO_PROMOTE**

| Field | Value |
| --- | --- |
| What | Pack 2 heads/CTA on already-exact native q1 RoPE/cache path (identical per-head reduction order; no Wo/proj_out/RMSNorm math replace) |
| Hypothesis | Exact reciprocal ≥5% main_model vs tip `55949274` from q1 scheduling / fewer underfilled CTAs (nsight ~19%) |
| Base tip | `55949274` |
| Branch / WT | `codex/exact-q1-rope-cache-headgroup` / local+DCC `exact-q1-rope-cache-headgroup` |
| Execution tip / immutable ref | **`dd5d8e58`** / `q1-rope-cache-headgroup-scout-dd5d8e58-r4` (wrapper FIX; was `64372cde`) |
| Opt-in | `q1_rope_cache_headgroup_candidate_context` → `native_q1_rope_cache_headgroup` (V32 cold default) |
| Kernel | `q1_rope_cache_attention_headgroup` (HEADS_PER_CTA=2, block 128×2) |
| Prior infra fails | `50000634`/`635` + `50002697`/`698` — `profile: unbound variable` after baseline (candidate never ran). FIX: split locals (`dd5d8e58`). |
| Jobs (r4 FIX) | FP16 **`50024784`** FAILED 1:0 00:07:24 (z25-20); FP32 **`50024785`** FAILED 1:0 00:06:27 (z25-21) — all four reciprocal legs present |
| Run roots | `/work/imt11/Mapperatorinator/runs/q1-headgroup-fp{16,32}-r4-dd5d8e58-5002478{4,5}/` |
| Exact | **PASS** — FP16/FP32 tok match (7809 / 8294); `.osu` sha match; headgroup engaged (174 calls both legs) |
| Measured | FP16 main_tps **310.50 → 268.20 (−42.3 / −13.6%)**; FP32 **321.28 → 290.19 (−31.1 / −9.7%)**. Second legs same direction (FP16 313.7→267.2; FP32 317.0→288.5). |
| Analyzer | FAILED unused expected delta `*optimized_cuda_graphs*` (allowlist mismatch) — does not change the exact-but-regress claim |
| Decision | **STOP_NO_PROMOTE** — exact but main regress; do not grind allowlist for a slower lever |
| Revisit | Only with a new scheduling hypothesis that shows component ≥5% headroom without e2e regress |
| Not this lever | §6/§7/§10/§11 math replaces; INT8; bare split-KV; compiled-cross |
| Ledger rule | Own section only |

## 13. Owned compile-before-capture self Wqkv + Wo GEMVs — **STOP_NO_PROMOTE**

| Field | Value |
| --- | --- |
| What | Owned `torch.compile` of tip-exact one-token self-attn `Wqkv` + `Wo` via `F.linear` (prepare before outer CUDA graph). **Not** `native_one_token_linear` / §11 |
| Hypothesis | Exact reciprocal ≥5% main_model vs tip `55949274` on SALVALAI from remaining eager gemm/projection (~26% nsight family): FP16 primary ≤20.264 s / ≥384.4 TPS (from 21.330 s / 366.11) |
| Base tip | `55949274` |
| Branch / WT | `codex/exact-compiled-self-proj` / DCC `exact-compiled-self-proj` (`codex/exact-compiled-self-proj-dcc`) |
| Tip / commit | **`3164875a`** |
| Opt-in | `compiled_self_proj_candidate_context` → `compiled_self_wqkv` + `compiled_out_proj` (V32 cold default) |
| Compile | `torch.compile(F.linear region, fullgraph=True, dynamic=False, mode="default")`; SM75; bitwise warmup gate (raise if not equal) — no max-autotune |
| Jobs | FP16 **`50001853`** FAILED 1:0 00:04:13 (z25-20); FP32 **`50001854`** FAILED 1:0 00:04:45 (z25-20) |
| Run roots | `/work/imt11/Mapperatorinator/runs/compiled-self-proj-fp{16,32}-5000185{3,4}/` (both have `baseline_first` + `candidate_first`) |
| Exact | **n/a** — candidate crashed mid-song before `.osu`/profile; no analyzer reciprocal |
| Measured | No candidate `main_tps`. Baseline-only (not a claim): FP16 **319.93** TPS / 24.408 s (7809); FP32 **262.04** / 31.651 s (8294). Candidate died ~window 3/87 after Dynamo recompiles. |
| Analyzer / root cause | `torch._dynamo.exc.FailOnRecompileLimitHit` on `_linear_region` (`compiled_self_proj.py:66`): `fullgraph=True` + `dynamic=False` hit `recompile_limit` (8) because tensor `x` size at index 1 varied (FP16 expected 1024 actual 564; FP32 expected 1024 actual 617). Lever design fail — not wrapper infra. |
| Decision | **STOP_NO_PROMOTE** — candidate unused for reciprocal (crash); not FIX+resubmit (compile/dynamic-shape failure, not unbound-var infra) |
| Revisit | Only with a new exactness hypothesis that owns dynamic seq lengths (or confines compile to fixed `(1,1)` decode GEMV shapes without timing-prefix pollution) |
| Not this lever | §3 compiled-cross BMM; §6/§7/§10/§11 native math replaces; §12 q1 headgroup; INT8 |
| Ledger rule | Own section only; do not fold into §3/§11/§12 |

---

## 14. Decode-only `(1,1)` owned compile-before-capture self Wqkv + Wo — **STOP_NO_PROMOTE**

| Field | Value |
| --- | --- |
| What | §13 revisit with **shape confinement**: `torch.compile` tip-exact one-token `F.linear` Wqkv/Wo **only** when `hidden.shape[:2]==(1,1)`; prefill stays eager. Prepare before outer CUDA-graph capture. **Not** bare §13 Dynamo retry; **not** `native_one_token_linear` |
| Hypothesis | Exact reciprocal ≥5% `main_model` vs tip `55949274` on SALVALAI (FP16 primary ≤20.264 s / ≥384.4 TPS from 21.330 s / 366.11) from remaining gemm/projection decode GEMVs without variable-seq recompiles |
| Base tip | `55949274` (via §13 scout `3164875a`) |
| Branch / WT | `codex/exact-compiled-self-proj-decode-only` / local+DCC `exact-compiled-self-proj-decode-only` |
| Tip / commit | **`43fea0c2`** |
| Opt-in | `compiled_self_proj_decode_only_candidate_context` → same flags as §13 + decode-only Wo gate |
| Delta vs §13 | `modeling_varwhisper` uses compiled Wo only at `(1,1,*)`; engine wrapper refuses other shapes (`FailOnRecompileLimitHit` root cause) |
| Jobs | FP16 **`50031125`** FAILED 1:0 00:07:20 (z25-20); FP32 **`50031126`** FAILED 1:0 00:06:28 (z25-21) — all four reciprocal legs present |
| Run roots | `/work/imt11/Mapperatorinator/runs/compiled-self-proj-decode-only-fp{16,32}-5003112{5,6}/` |
| Exact | **PASS** — FP16/FP32 tok match (7809 / 8294); HO 645 / 637; `.osu` sha match across all legs; token IDs equal |
| Measured | FP16 main_tps **308.61 → 306.34 (−2.27 / −0.73%)**; FP32 **308.36 → 315.98 (+7.62 / +2.47%)** — under 5% gate. Second legs same direction (FP16 306.30; FP32 315.07). Compile engaged (120 wqkv + 120 wo hits both candidates). |
| Analyzer | FAILED undeclared `compiled_self_wqkv`/`compiled_self_wo` capture-hits + unused allowlist globs `*compiled_wqkv*`/`*compiled_wo*` — does not change exact-but-&lt;5%/regress claim |
| Decision | **STOP_NO_PROMOTE** — exact but no ≥5% main (FP16 slight regress; FP32 +2.5%) |
| Revisit | Only with a new compile/scheduling hypothesis (not bare §13/§14 retry; not allowlist-only) |
| Not this lever | Bare §13 retry; §6/§7/§10/§11 native math; §12 headgroup; INT8 |
| Ledger rule | Own section only; do not fold into §13 |

## 15. Decode-only `(1,1)` owned compile-before-capture tip-exact `proj_out` — **STOP_NO_PROMOTE**

| Field | Value |
| --- | --- |
| What | Owned `torch.compile` of tip-exact LM-head `proj_out` `F.linear` **only** at decode shape `(1,1,*)`; final RMSNorm stays eager. Prepare before outer CUDA-graph capture. **Not** §10 native `rmsnorm+proj_out` fuse; **not** §14 Wqkv/Wo |
| Hypothesis | Exact reciprocal ≥5% `main_model` vs tip `55949274` on SALVALAI from remaining large vocab GEMV (gemm/projection family) without changing math |
| Base tip | `55949274` |
| Branch / WT | `codex/exact-compiled-proj-out-decode-only` / local+DCC `exact-compiled-proj-out-decode-only` |
| Tip / commit | **`5aa6eb9f`** |
| Opt-in | `compiled_proj_out_decode_only_candidate_context` → `compiled_proj_out` (V32 cold default) |
| Jobs | FP16 **`50036330`** FAILED 1:0 00:03:23 (z25-21); FP32 **`50036325`** FAILED 1:0 00:04:22 (z25-20) |
| Run roots | `/work/imt11/Mapperatorinator/runs/compiled-proj-out-decode-only-fp{16,32}-500363{30,25}/` |
| Exact | **n/a** — candidate crashed in timing generation before `.osu`/profile; no analyzer reciprocal |
| Measured | No candidate `main_tps`. Baseline-only (not a claim): FP16 ~377.7 TPS / 20.676 s (7809); FP32 ~277.6 / 29.881 s (8294). |
| Analyzer / root cause | `RuntimeError: compiled proj_out requires model.proj_out with rank-2 weight` in `prepare_compiled_proj_out` during timing window 0/87 — timing model lacks usable rank-2 `proj_out`; prepare was not gated. Lever design fail — not wrapper infra. |
| Decision | **STOP_NO_PROMOTE** — candidate unused for reciprocal (crash); not bare FIX+resubmit without a new gated hypothesis |
| Revisit | Only with a timing/rank-2 gate (skip prepare when weight missing/non-rank-2 or TIMING context) plus falsifiable ≥5% claim — do not bare-retry |
| Not this lever | §10 native fuse; bare §13/§14 retry; §6/§7/§11/§12; INT8 |
| Ledger rule | Own section only; do not fold into §10/§14 |

## 16. Tip-exact CUDA `expandable_segments` allocator scheduling — **STOP_NO_PROMOTE**

| Field | Value |
| --- | --- |
| What | Non-mutating memory-scheduling: candidate sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before CUDA init; baseline leaves it unset. Tip composition (shared-RoPE + device sequence state) always on. **No** kernel/math replace. |
| Hypothesis | Exact reciprocal ≥5% `main_model` vs tip `55949274` on SALVALAI from allocator scheduling cutting alloc/fragmentation overhead inside the ~30% elementwise+memory nsight family |
| Base tip | `55949274` |
| Branch / WT | `codex/exact-cuda-alloc-expandable` / local+DCC `exact-cuda-alloc-expandable` |
| Tip / commit | **`fde4f0f2`** |
| Opt-in | env-only on candidate legs (`PYTORCH_CUDA_ALLOC_CONF`); V32 cold default unchanged |
| Jobs | FP16 **`50039777`** COMPLETED 0:0 00:05:36 (z25-21); FP32 **`50039776`** COMPLETED 0:0 00:07:25 (z25-20); both `cuda_alloc_expandable_reciprocal=PASS` |
| Run roots | `/work/imt11/Mapperatorinator/runs/cuda-alloc-expandable-fp{16,32}-500397{77,76}/` |
| Exact | **PASS** — FP16 tok 7809 / HO 645 / `.osu` sha match; FP32 tok 8294 / HO 637 / `.osu` sha match; token streams equal; stopping equal |
| Measured | FP16 main_tps **350.80 → 373.46 (+22.7 / +6.5% reciprocal)** / main_model 22.263→20.910 (−1.35 s); **vs tip 366.11 → 373.46 (+2.0% / −0.42 s)** — under tip 5% gate (≥384.4 TPS / ≤20.264 s). FP32 main_tps **276.65 → 270.63 (−6.0 / −2.2%)** / main_model 29.980→30.660 (**regress**). Cold walls noisy (not the claim). |
| Decision | **STOP_NO_PROMOTE** — exact but vs tip &lt;5% (FP16 +2.0%); FP32 regress; do not bare-retry expandable_segments |
| Revisit | Only with a **different** allocator/memory-scheduling hypothesis (not expandable_segments re-tune) |
| Not this lever | §6/§7/§10/§11 native math; bare §12–§15 retry; Wo/proj_out native fuse; INT8 |
| Ledger rule | Own section only |

## 17. Tip-exact CUDA `cudaMallocAsync` allocator scheduling — **STOP_NO_PROMOTE**

| Field | Value |
| --- | --- |
| What | Non-mutating memory-scheduling sibling to §16: candidate sets `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` before CUDA init; baseline unset. Tip composition always on. **No** kernel/math replace. **Not** bare §16 expandable_segments retry. |
| Hypothesis | Exact reciprocal ≥5% `main_model` vs tip `55949274` on SALVALAI (FP16 primary ≤20.264 s / ≥384.4 TPS from 21.330 s / 366.11) from async pooling cutting alloc/sync overhead in the ~30% elementwise+memory family |
| Base tip | `55949274` |
| Branch / WT | `codex/exact-cuda-alloc-malloc-async` / local+DCC `exact-cuda-alloc-malloc-async` |
| Tip / commit | **`b9b60cd5`** |
| Opt-in | env-only on candidate legs (`PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`); V32 cold default unchanged |
| Jobs | Auth FP16 **`50050003`** COMPLETED 0:0 00:06:11; FP32 **`50050002`** COMPLETED 0:0 00:06:29. Dup pair FP16 `50050004` / FP32 `50050001` (same tip; accidental double-submit — same direction). All four `cuda_alloc_malloc_async_reciprocal=PASS`. |
| Run roots | `/work/imt11/Mapperatorinator/runs/cuda-alloc-malloc-async-fp{16,32}-500500{03,02}/` (+ dups `04`/`01`) |
| Exact | **PASS** — FP16 tok 7809 / HO 645 / `.osu` sha match; FP32 tok 8294 / HO 637 / `.osu` sha match |
| Measured | Auth FP16 main_tps **363.13 → 351.55 (−3.2% recip; vs tip 366.11 → 351.55 −4.0%)**; FP32 **305.34 → 288.65 (−5.5%; vs tip −7.8%)**. Dups same direction (FP16 −6.7% recip / −7.2% vs tip). |
| Decision | **STOP_NO_PROMOTE** — exact but main regress vs tip; do not bare-retry cudaMallocAsync / expandable_segments |
| Revisit | Only with a **new** memory/scheduling / owned-compile-of-already-exact hypothesis (not allocator env retune) |
| Not this lever | §16 expandable_segments; bare §12–§15; Wo/proj_out native math; INT8 |
| Ledger rule | Own section only |

## 18. Tip-exact decode logits workspace + compiled float32 softmax — **STOP_NO_PROMOTE**

| Field | Value |
| --- | --- |
| What | Sampling / elementwise-memory lever: reuse one static float32 vocab buffer for last-token logits + `torch.compile` softmax at fixed `(1,V)`. Multinomial/argmax stay eager for RNG exactness. Tip composition always on. |
| Hypothesis | Exact reciprocal ≥5% `main_model` vs tip `55949274` on SALVALAI (FP16 primary ≤20.264 s / ≥384.4 TPS from 21.330 s / 366.11) by cutting per-token logits alloc + compiling still-eager softmax in the sampling/~7% + elementwise/memory families |
| Base tip | `55949274` |
| Branch / WT | `codex/exact-compiled-decode-logits-finalize` / local+DCC `exact-compiled-decode-logits-finalize` |
| Tip / commit | **`b47a3fe2`** |
| Opt-in | `compiled_decode_logits_finalize_candidate_context` → workspace + compiled softmax (V32 cold default) |
| Jobs | FP16 **`50107685`** FAILED 1:0 00:06:48 (h36-5); FP32 **`50107686`** FAILED 1:0 00:06:48 (h36-5) — all four reciprocal legs present |
| Run roots | `/work/imt11/Mapperatorinator/runs/compiled-decode-logits-finalize-fp{16,32}-501076{85,86}/` |
| Exact | **PASS** — FP16 tok 7809 / HO 645 / `.osu` sha match; FP32 tok 8294 / HO 637 / `.osu` sha match; RNG after-inference match |
| Measured | FP16 first 351.79→350.26 (−0.43%); second 360.55→360.73 (+0.05%); **vs tip 366.11 → ~350–361 (under ≥384.4 / 5% gate)**. FP32 first 316.06→312.97 (−1.0%); second 313.51→305.29 (−2.6%) — regress. |
| Analyzer | FAILED unused expected deltas `*compiled_softmax*` / `*optimized_cuda_graphs*` — does not change exact-but-no-gain claim |
| Decision | **STOP_NO_PROMOTE** — exact but no ≥5% main; do not bare-retry softmax/workspace |
| Revisit | Only with a new sampling / non-allocator memory / owned-compile hypothesis (not §18 retune) |
| Not this lever | §16/§17 allocator env; bare §13–§15 compile; Wo/proj_out/Wqkv native math; INT8; cast-elim flatten |
| Ledger rule | Own section only |

## 19. Tip-exact native q1 attn-out reshape (skip transpose+contiguous) — **STOP_NO_PROMOTE**

| Field | Value |
| --- | --- |
| What | Memory/scheduling lever on already-exact native q1 RoPE/cache path: replace `transpose(1,2).contiguous().view(B,S,H*D)` with `reshape(B,S,H*D)` when output is contiguous `[B,H,1,D]` (same head-major bytes; skip D2D copy). Tip composition always on. |
| Hypothesis | Exact reciprocal ≥5% `main_model` vs tip `55949274` on SALVALAI (FP16 primary ≤20.264 s / ≥384.4 TPS from 21.330 s / 366.11) by cutting per-layer-per-token contiguous copies in the ~30% elementwise+memory nsight family |
| Base tip | `55949274` |
| Branch / WT | `codex/exact-q1-out-reshape` / local+DCC `exact-q1-out-reshape` |
| Tip / commit | **`5f4b9989`** |
| Opt-in | `q1_out_reshape_candidate_context` → reshape path (V32 cold default) |
| Jobs | FP16 **`50129056`** FAILED 1:0 00:05:56; FP32 **`50129057`** FAILED 1:0 00:07:37 — all four reciprocal legs present |
| Run roots | `/work/imt11/Mapperatorinator/runs/q1-out-reshape-fp{16,32}-501290{56,57}/` |
| Exact | **PASS** — FP16 tok 7809 / HO 645 / `.osu` sha match; FP32 tok 8294 / HO 637 / `.osu` sha match |
| Measured | FP16 first 369.67→368.12 (−0.42%); second 342.01→369.99 (noisy base); **vs tip 366.11 → 368.12 / 369.99 (+0.55% / +1.06%)** — under ≥384.4 / 5% gate. FP32 first +0.25%; second −4.47% (regress). |
| Analyzer | FAILED unused `*optimized_cuda_graphs*` — does not change exact-but-no-gain claim |
| Decision | **STOP_NO_PROMOTE** — exact but no ≥5% vs tip; do not bare-retry reshape / grind allowlist |
| Revisit | Only with a new q1-layout / memory hypothesis (not bare reshape retune; not §12 headgroup) |
| Not this lever | §12 headgroup CTA; §16/§17 allocator env; bare §13–§15/§18; Wo/proj_out/Wqkv native math; INT8 |
| Ledger rule | Own section only |

## 20. Gated decode-only `(1,1)` compiled tip-exact `proj_out` (§15 FIX) — **STOP_NO_PROMOTE**

| Field | Value |
| --- | --- |
| What | §15 revisit with **timing/rank-2 prepare gate**: owned `torch.compile` of tip-exact LM-head `F.linear` only when `context_type != TIMING` and `model.proj_out.weight` is rank-2 matching dtype; otherwise skip prepare (eager). Decode-only `(1,1,*)`. **Not** bare §15 retry; **not** §10 native fuse. |
| Hypothesis | Exact reciprocal ≥5% `main_model` vs tip `55949274` on SALVALAI (FP16 primary ≤20.264 s / ≥384.4 TPS) from large vocab GEMV once timing windows stop crashing prepare |
| Base tip | `55949274` via §15 scout `5aa6eb9f` |
| Branch / WT | `codex/exact-compiled-proj-out-gated` / local+DCC `exact-compiled-proj-out-gated` |
| Tip / commit | **`91977df2`** |
| Opt-in | same `compiled_proj_out_decode_only_candidate_context` as §15 (V32 cold default) |
| Jobs | FP16 **`50129762`** FAILED 1:0 00:07:16 (z25-20); FP32 **`50129763`** FAILED 1:0 00:06:31 (z25-21) — all four reciprocal legs present |
| Run roots | `/work/imt11/Mapperatorinator/runs/compiled-proj-out-gated-fp{16,32}-501297{62,63}/` |
| Exact | **PASS** — FP16 tok 7809 / HO 645 / `.osu` sha match + token IDs; FP32 tok 8294 / HO 637 / `.osu` sha match + token IDs |
| Measured | Policy `compiled_proj_out.enabled=false` / `disabled_reason=prepare_failed`; capture hits **0** on all candidate legs (lever never engaged). FP16 first 311.40→291.32 (−6.5% recip; **vs tip 366.11 → 291.32 −20.4%**); second 307.07→313.23 (+2.0%; vs tip −14.4%). FP32 first −0.7% recip / vs tip +2.2%; second +0.6% / vs tip +1.9% — under 5% tip gate. |
| Analyzer | FAILED unused `*compiled_proj_out*` / `*optimized_cuda_graphs*` — matches unused path; do not grind allowlist |
| Decision | **STOP_NO_PROMOTE** — exact but unused (prepare_failed); no ≥5% vs tip; do not bare-retry §20 gate |
| Revisit | Only with a prepare-engage FIX that yields `compiled_proj_out` hits &gt; 0 (§20b) |
| Not this lever | Bare §15 crash-retry; §10 native; §13/§14; §16–§19; INT8 |
| Ledger rule | Own section only |

## 20b. Prepare try/except rank-2 skip — compiled `proj_out` engagement FIX (§20 FIX) — **STOP_NO_PROMOTE**

| Field | Value |
| --- | --- |
| What | §20 FIX: try `prepare_compiled_proj_out` without aggressive context/dtype pre-gate; skip only on missing rank-2 weight (timing). Goal is **hits &gt; 0** then ≥5% main vs tip. **Not** bare §20 retune; **not** §10 native. Follow-up harden `8bd44130` (unwrap + dtype skip + true enabled stats) pushed but **not** this job tip. |
| Hypothesis | Exact reciprocal ≥5% `main_model` vs tip `55949274` (FP16 ≤20.264 s / ≥384.4 TPS) once vocab GEMV compile actually runs on MAP decode |
| Base tip | `55949274` via §20 `91977df2` |
| Branch / WT | `codex/exact-compiled-proj-out-gated` / local+DCC `exact-compiled-proj-out-gated` |
| Tip / commit | **`3215d14b`** (jobs); local harden tip `8bd44130` ready if still unused |
| Opt-in | same `compiled_proj_out_decode_only_candidate_context` |
| Jobs | FP16 **`50131926`** / FP32 **`50131927`** FAILED analysis 1:0 (`ALLOW_PARALLEL=1`; `RUN_LABEL=compiled-proj-out-gated-r2-fp{16,32}`) |
| Run roots | `/work/imt11/Mapperatorinator/runs/compiled-proj-out-gated-r2-fp{16,32}-501319{26,27}/` |
| Exact | **PASS** reciprocal legs; analyzer unused `*compiled_proj_out*` — **hits still 0** at `3215d14b` |
| Measured | FP16 first 310.08→312.41 (+0.8% recip; **vs tip 366.11 −14.7%**); second noisy base. FP32 first −0.4% recip / vs tip +1.7%; second −3.7% / vs tip −2.5%. Policy still `prepare_failed`. |
| Decision | **STOP_NO_PROMOTE** at `3215d14b`; reopen as §21 with unwrap harden `8bd44130` |
| Revisit | Only with engagement proof (`compiled_proj_out` hits &gt; 0) then ≥5% tip gate |
| Not this lever | Bare §20; §10 native; §13/§14 Wqkv/Wo; §16–§19; INT8 |
| Ledger rule | Own section only |

## 21. Compiled proj_out unwrap + dtype-skip engage FIX — **STOP_NO_PROMOTE**

| Field | Value |
| --- | --- |
| What | §20b follow-up: unwrap `InferenceEngineBinding` before prepare, catch skippable rank-2/**dtype** refusals, and report true `enabled` in stats/policy. Addresses still-unused hits at `3215d14b` (`prepare_failed`). **Not** bare §20/§20b retune; **not** §10 native. |
| Hypothesis | Exact reciprocal ≥5% `main_model` vs tip `55949274` (FP16 ≤20.264 s / ≥384.4 TPS) once vocab GEMV compile actually runs on MAP decode |
| Base tip | `55949274` via §20b `3215d14b` |
| Branch / WT | `codex/exact-compiled-proj-out-gated` / local+DCC `exact-compiled-proj-out-gated` |
| Tip / commit | **`8bd44130`** |
| Opt-in | same `compiled_proj_out_decode_only_candidate_context` |
| Jobs | FP16 **`50132354`** FAILED 1:0 00:07:05 (z25-20); FP32 **`50132355`** FAILED 1:0 00:06:09 (z25-21) — all four reciprocal legs present |
| Run roots | `/work/imt11/Mapperatorinator/runs/compiled-proj-out-gated-r3-fp{16,32}-501323{54,55}/` |
| Exact | **PASS** — FP16 tok 7809 / HO 645 / `.osu` sha + token IDs + RNG; FP32 tok 8294 / HO 637 / `.osu` sha + token IDs + RNG |
| Compile engage | **NO** — `compiled_proj_out` hits **0** on all candidate main windows; policy `requested=true` / `enabled=false` / `disabled_reason=missing_rank2_proj_out_weight` (87/87 main_generation). Analyzer unused `*compiled_proj_out*` / `*optimized_cuda_graphs*`. |
| Measured | FP16 first 311.39→310.48 (−0.3% recip; **vs tip 366.11 → 310.48 −15.2%**); second 313.52→311.10 (−0.8%; vs tip −15.0%). FP32 first 321.51→319.56 (−0.6% recip; vs tip +2.1%); second 318.67→315.94 (−0.9%; vs tip +0.9%) — under ≥5% tip gate. |
| Decision | **STOP_NO_PROMOTE** — exact but compile never engaged; leave entire compiled-proj_out family (§15/§20/§20b/§21) |
| Revisit | Only with a **different** nsight-family hypothesis (not another proj_out prepare/unwrap tweak) |
| Not this lever | Bare §20/§20b; §10 native; §13/§14; §16–§19; INT8 |
| Ledger rule | Own section only |

## 22. Decode-only compiled self Wo + residual (gemm family) — **STOP_NO_PROMOTE**

| Field | Value |
| --- | --- |
| What | Owned `torch.compile` of tip-exact self-attn `F.linear(Wo) + residual` at decode `(1,1,*)` before outer CUDA-graph capture; skip in-attn Wo via `skip_out_proj`. **Not** native §6/`native_one_token_linear_residual`; **not** §14 Wqkv+Wo; **not** LM proj_out / reshape / allocator. |
| Hypothesis | Exact reciprocal ≥5% `main_model` vs tip `55949274` (FP16 ≤20.264 s / ≥384.4 TPS) from gemm_gemv_projection (~26% nsight) by fusing Wo GEMV epilogue with residual add |
| Base tip | `55949274` |
| Branch / WT | `codex/exact-compiled-self-wo-residual` / local+DCC `exact-compiled-self-wo-residual` |
| Tip / commit | **`a5ea705d`** |
| Opt-in | `compiled_self_wo_residual_candidate_context` → prepare compiled Wo+residual when native q1 RoPE/cache live (V32 cold default) |
| Jobs | FP16 **`50133112`** FAILED 1:0 00:04:49 (z25-20); FP32 **`50133113`** FAILED 1:0 00:04:05 (z25-21) — baseline_first only; candidate_first crashed mid main |
| Run roots | `/work/imt11/Mapperatorinator/runs/compiled-self-wo-residual-fp{16,32}-501331{12,13}/` |
| Exact | **n/a** — candidate never finished (no candidate profile / `.osu`) |
| Compile engage | **FAIL** — prepare warmed zeros outside capture, but Inductor still compiled/synced **during** CUDA-graph capture on first real Wo weights (`InductorError: operation not permitted when stream is capturing` → capture invalidated). hits n/a |
| Measured | Baseline-only (not a claim): FP16 main ~296.7 TPS / 26.316 s (7809); FP32 ~321.4 / 25.802 s (8294). No candidate `main_tps`. |
| Decision | **STOP_NO_PROMOTE** — lever design fail (compile-during-capture); leave compiled self Wo+residual family |
| Revisit | Only with proven compile-before-capture that warms **real** Wo weights **and** finishes Inductor before `torch.cuda.graph` (plus capture warmup≥1); not bare retry of zeros-warmup path |
| Not this lever | §6 native residual; §11 native Wo; §13/§14; §15–§21 proj_out; §16–§19; INT8 |
| Ledger rule | Own section only |

## 23. Tip-exact q1 float32 mask workspace (elementwise/memory) — **STOP_NO_PROMOTE**

| Field | Value |
| --- | --- |
| What | Request-local contiguous float32 buffer reuse for native q1 `_native_mask` materialization (copy into workspace instead of per-call `.to(float32).contiguous()` alloc). |
| Hypothesis | Exact reciprocal ≥5% `main_model` vs tip `55949274` (FP16 ≤20.264 s / ≥384.4 TPS) by cutting per-token mask alloc/cast traffic in the ~30% elementwise+memory / q1-adjacent nsight family |
| Base tip | `55949274` |
| Branch / WT | `codex/exact-q1-mask-workspace` / local+DCC `exact-q1-mask-workspace` |
| Tip / commit | **`8759db3a`** |
| Opt-in | `q1_mask_workspace_candidate_context` → prepare workspace when native q1 RoPE/cache live (V32 cold default) |
| Jobs | FP16 **`50138023`** FAILED 1:0 00:07:29 (z25-20); FP32 **`50138024`** FAILED 1:0 00:06:35 (z25-21) — full reciprocal; analyzer unused `*optimized_cuda_graphs*` |
| Run roots | `/work/imt11/Mapperatorinator/runs/q1-mask-workspace-fp{16,32}-501380{23,24}/` |
| Exact | **PASS** (tok 7809/8294 + HO 645/637 + `.osu` sha match + RNG after_inf match) |
| Engage | Policy `q1_mask_workspace` enabled on map windows (87); but `optimized_dispatch_capture_hits.q1_mask_workspace=0` (workspace `materialize` never counted — likely mask path not hit on captured decode) |
| Measured | FP16 cand main_tps **312.05 / 310.26** (main_s 25.025 / 25.170) vs tip **366.11** (**−14.8% / −15.3%**); recip 2nd flat (+0.07%). FP32 cand **309.87 / 308.02** vs tip **313.05** (**−1.0% / −1.6%**). |
| Decision | **STOP_NO_PROMOTE** — exact but no tip ≥5%; engage hits=0; do not grind allowlist |
| Revisit | Only with proven hot-path mask materialization (hits&gt;0) + new ≥5% hypothesis; not bare retry / allowlist-only |
| Not this lever | §12–§22 stopped families; INT8 |
| Ledger rule | Own section only |

## 24. Tip-exact active-prefix decode bucket 256 (CUDA-graph scheduling) — **STOP_NO_PROMOTE**

| Field | Value |
| --- | --- |
| What | Opt-in raise `active_prefix_decode_bucket_size` tip **64 → 256** to cut unique CUDA-graph captures (amortized capture tax) vs more padded KV attention per step. |
| Hypothesis | Exact reciprocal ≥5% `main_model` vs tip `55949274` (FP16 ≤20.264 s / ≥384.4 TPS) from CUDA-graph scheduling family (not reshape/allocator/proj_out/Wo-residual/mask-workspace) |
| Base tip | `55949274` |
| Branch / WT | `codex/exact-active-prefix-bucket` / DCC `exact-active-prefix-bucket` |
| Tip / commit | **`1076bfd3`** (evidence-assert FIX on `3123426b`) |
| Opt-in | `active_prefix_bucket_candidate_context` → effective bucket 256 (V32 cold default stays 64) |
| Jobs | FP16 **`50138950`** FAILED 5:57 / FP32 **`50138951`** FAILED 6:17 (analyzer unused `*active_prefix_bucket*` + undeclared dispatch hits; all four reciprocal legs present) |
| Run roots | `/work/imt11/Mapperatorinator/runs/active-prefix-bucket-fp{16,32}-501389{50,51}/` |
| Engage | Evidence `active_prefix_bucket_enabled=True` / size **256**; decode_graph_count_delta **12→4** (capture tax cut); metadata `optimized_effective_config` still reports tip 64 (reporting quirk — graphs prove engage) |
| Exact | **PASS** — FP16 tok 7809 / HO 645 / `.osu` sha match + RNG match; FP32 tok 8294 / HO 637 / `.osu` sha match + RNG match |
| Measured | FP16 cand **347.14 / 347.95** TPS (22.495 / 22.443 s) — **vs tip 366.11 −5.18% / −4.96%** (need ≥384.4); recip −2.1% / −3.7%. FP32 cand **312.88 / 311.51** — vs tip 313.05 **−0.05% / −0.49%**; recip mixed +3.6% / −1.4%. Capture savings ~0.09 s; pad tax dominated. |
| Decision | **STOP_NO_PROMOTE** — exact + engaged but no tip ≥5%; FP16 regress. Do not grind allowlist. |
| Revisit | Inverse exact-length (§25) or other-family only — not same-direction larger bucket; not §12–§23 |
| Not this lever | §12–§23; INT8 |
| Ledger rule | Own section only |

## 25. Tip-exact active-prefix exact-length CUDA graphs (bucket=1) — **STOP_NO_PROMOTE**

| Field | Value |
| --- | --- |
| What | Opt-in force `active_prefix_decode_bucket_size=1` so each decode step graphs the **true** prefix length (no pad). Inverse of §24: §24 showed fewer captures (12→4) but main regress from pad FLOPs. |
| Hypothesis | Exact reciprocal ≥5% `main_model` vs tip `55949274` (FP16 ≤20.264 s / ≥384.4 TPS) by removing pad KV attention; accept more CUDA-graph captures. Not same-direction bucket retune; not reshape/allocator/proj_out/Wo-residual/mask-workspace. |
| Base tip | `55949274` |
| Branch / WT | `codex/exact-active-prefix-exact-length` / DCC `exact-active-prefix-exact-length` |
| Tip / commit | **`25376173`** |
| Opt-in | `active_prefix_exact_length_candidate_context` → effective bucket **1** (V32 cold default stays 64) |
| Jobs | FP16 **`50139609`** FAILED 4:35 (z25-20) / FP32 **`50139610`** FAILED 6:45 (z25-21); analyzer unused `*active_prefix_exact_length*` + undeclared tip dispatch hits |
| Run roots | `/work/imt11/Mapperatorinator/runs/active-prefix-exact-length-fp{16,32}-501396{09,10}/` |
| Engage | FP32 evidence `active_prefix_exact_length_enabled=True` / size **1**; decode_graph_count_delta **~50** (vs tip baseline ~2) — capture explosion. FP16 candidate_first **OOM** during graph capture (`8.31 GiB` CUDA-graph private pools). |
| Exact | FP16 **n/a** (cand crash, no `.osu`). FP32 **PASS** — tok 8294 / HO 637 / `.osu` sha match + RNG match (all four legs). |
| Measured | FP32 cand **254.12 / 250.61** TPS (32.638 / 33.095 s) — **vs tip 313.05 −18.8% / −20.0%**; recip −19.4% / −19.1%. Capture tax ~0.85 s first window. FP16 baseline-only ~313.22 TPS (not a claim). |
| Decision | **STOP_NO_PROMOTE** — FP16 OOM; FP32 exact but large main regress. Leave entire active-prefix bucket/pad family (§24/§25). Do not grind allowlist. |
| Revisit | Only with a **non-bucket** family hypothesis; not bare §24/§25 retry; not intermediate bucket retune |
| Not this lever | §12–§24; INT8 |
| Ledger rule | Own section only |

## 26. Tip-exact native cross+MLP `outputs_per_block` 8→4 — **STOP_NO_PROMOTE**

| Field | Value |
| --- | --- |
| What | Opt-in force native cross Q / Wo / MLP residual `outputs_per_block` tip **8 → 4** (allowed 2/4/8) for more CTAs / SM75 occupancy on already-fused kernels (~15% fused-MLP nsight family). Math unchanged. |
| Hypothesis | Exact reciprocal ≥5% `main_model` vs tip `55949274` (FP16 ≤20.264 s / ≥384.4 TPS) from native-kernel occupancy scheduling (not q1 headgroup §12; not bucket/pad; not proj_out-compile; not allocator-env). |
| Base tip | `55949274` |
| Branch / WT | `codex/exact-native-mlp-opb` / DCC `exact-native-mlp-opb` |
| Tip / commit | **`b1736b0b`** |
| Opt-in | `native_mlp_outputs_per_block_candidate_context` → effective **4** (V32 cold default stays 8) |
| Jobs | FP16 **`50140036`** FAILED 6:06 (z25-21) / FP32 **`50140037`** FAILED 6:36 (z25-21) — analysis unused `*optimized_cuda_graphs*` after full reciprocal |
| Run roots | `/work/imt11/Mapperatorinator/runs/native-mlp-opb-fp{16,32}-501400{36,37}/` |
| Exact | **PASS** (tok 7809/8294 + HO 645/637 + `.osu` sha + RNG match both pairs) |
| Engage | Candidate `native_mlp_outputs_per_block_enabled=true` size **4**; decode_loop_calls **174** |
| Measured | FP16 cand **370.25 / 338.11** TPS (21.091 / 23.096 s) — vs tip **+1.13% / −7.65%** (gate ≥384.42); recip −0.27% / −3.48%. FP32 cand **317.75 / 317.28** — vs tip **+1.50% / +1.35%** (gate ≥328.70); recip −0.51% / +4.82%. |
| Decision | **STOP_NO_PROMOTE** — exact + opb=4 engaged but no tip ≥5% main. Do not grind allowlist; do not bare-retune opb→2. |
| Revisit | Only with a **different** nsight-family hypothesis (not same-direction opb retune) |
| Not this lever | §12–§25; INT8 |
| Ledger rule | Own section only |

## 27. Tip-exact native q1 attention `block_size` 128→256 — **STOP_NO_PROMOTE**

| Field | Value |
| --- | --- |
| What | Opt-in force native q1 self / rope-cache attention CUDA `block_size` tip **128 → 256** (allowed 64/128/256) to widen per-head reduction CTAs on the ~19% native_q1 nsight family. Math unchanged. |
| Hypothesis | Exact reciprocal ≥5% `main_model` vs tip `55949274` (FP16 ≤20.264 s / ≥384.4 TPS) from q1 launch-geometry / reduction-width scheduling — **not** §12 headgroup grid, not §26 MLP opb, not bucket/pad, not proj_out-compile, not allocator-env, not Wo-residual-compile. |
| Base tip | `55949274` |
| Branch / WT | `codex/exact-q1-block-size` / DCC `exact-q1-block-size` |
| Tip / commit | **`10df9bc0`** (evidence-field FIX on `180ac084`) |
| Opt-in | `q1_block_size_candidate_context` → effective **256** (V32 cold default stays 128) |
| Jobs | FP16 **`50140877`** FAILED 7:19 (z25-20; analyzer undeclared `*optimized_cuda_graphs*` after full reciprocal) / FP32 **`50140878`** COMPLETED PASS 6:24 (z25-21) |
| Run roots | `/work/imt11/Mapperatorinator/runs/q1-block-size-fp{16,32}-501408{77,78}/` |
| Engage | Candidate `q1_block_size_enabled=true` size **256**; decode_loop_calls **174**; FP32 analysis `q1_block_size` capture hits &gt;0 |
| Exact | FP16 **FAIL** — tok **7809→8011**, HO **645→618**, `.osu` sha mismatch + CUDA RNG diverge (reduction-order drift). FP32 **PASS** — tok 8294 / HO 637 / `.osu` sha + RNG match |
| Measured | FP16 cand main_tps **296.00 / 314.04** (invalid exactness; tok≠7809). FP32 cand **308.76 / 309.93** — vs tip 313.05 **−1.37% / −1.00%** (gate ≥328.70); recip ~flat/−0.2% |
| Decision | **STOP_NO_PROMOTE** — FP16 exactness fail under “math-unchanged” launch geometry; FP32 exact but no tip ≥5%. Leave q1 `block_size` family. Do not grind allowlist; do not bare-retune →64 |
| Revisit | Only with a **non-block_size** family hypothesis |
| Not this lever | §12–§26; INT8 |
| Ledger rule | Own section only |

## 28. Tip-exact owned rectangular self-Wqkv one-token linear — **STOP_NO_PROMOTE**

| Field | Value |
| --- | --- |
| What | Opt-in replace tip still-eager decode `module.Wqkv` with owned warp-group rectangular `one_token_linear_rect` (`output_dim=3H`) after tip-eager RMSNorm — **no** norm fuse, **no** Wo touch. Targets ~26% gemm_gemv_projection nsight family. |
| Hypothesis | Exact reciprocal ≥5% `main_model` vs tip `55949274` (FP16 ≤20.264 s / ≥384.4 TPS) from owned CUDA GEMV on self-QKV — not §7 RMSNorm+Wqkv fuse; not §11 Wo linear; not §12–§27 stopped families |
| Base tip | `55949274` |
| Branch / WT | `codex/exact-self-wqkv-linear` / DCC `exact-self-wqkv-linear` |
| Tip / commit | **`f5c6b901`** |
| Opt-in | `self_wqkv_linear_candidate_context` → opb **8** (V32 cold default stays eager `nn.Linear`) |
| Jobs | FP16 **`50141327`** FAILED 6:23 (h36-5; analyzer undeclared `*optimized_cuda_graphs*` after full reciprocal) / FP32 **`50141328`** COMPLETED PASS 6:50 (h36-5) |
| Run roots | `/work/imt11/Mapperatorinator/runs/self-wqkv-linear-fp{16,32}-501413{27,28}/` |
| Engage | Candidate `self_wqkv_linear_enabled=true` opb **8**; decode_loop_calls **174**. Capture-hit counter `self_wqkv_linear=0` (path still armed via candidate context; FP16 tok/HO drift proves candidate path ran). |
| Exact | FP16 **FAIL** — tok **7809→7981**, HO **645→597**, `.osu` sha mismatch. FP32 **PASS** — tok 8294 / HO 637 / `.osu` sha match (both pairs). |
| Measured | FP16 cand main_tps **367.15 / 365.63** invalid (tok≠7809). FP32 cand **319.85 / 319.40** — vs tip 313.05 **+2.17% / +2.03%** (gate ≥328.70); recip main_tps median +2.19 TPS (~+0.7%). |
| Decision | **STOP_NO_PROMOTE** — FP16 exactness collapse (same native-Wqkv family pattern as §6/§7/§11); FP32 exact but under tip ≥5%. **Strategy pivot / DROP:** deprioritize further native Wqkv/Wo/proj_out math replaces. Do not bare-retry. |
| Revisit | **Only after** structure levers (§29 whole-token graph, §30 elementwise fusion) show measured ≥0.15 ms/token headroom |
| Not this lever | §12–§27; INT8; further native Wqkv/Wo/proj_out |
| Ledger rule | Own section only |

---

## Research → ledger renumber (2026-07-17)

Existing campaign §§24–28 already used (bucket / exact-length / opb / block_size / Wqkv). Research plan §§24–28 map to **new** ledger numbers:

| Research | Ledger | Status |
| --- | --- | --- |
| §24 whole-token-step CUDA graph | **§29** / **§29b** | eager tip **STOP**; Philox sample-graph **STOP** (exact smoke; −0.043≪0.15 ms/tok) |
| §25 elementwise/copy fusion | **§30** | **STOP_NO_PROMOTE** — exact; 0.050≪0.15 ms/tok |
| §26 batch-invariant speculative | **§31** | **PARKED** (n-gram accepted/step ≤1.059) |
| §27 CUDA graph WHILE/conditional | **§32** | **STOP_NO_PROMOTE** — r1/r2 exact PASS; budget −0.20≪0.15 ms/tok (**STOP budget miss**) |
| §28 q1 occupancy | **§33** | **OPEN** (instrumentation; nsys `49966210`; ncu blocked) |
| **Track C** (scope ruling §34) | **§35–§38** | layer-skip probe → self-spec turbo → tiny draft → optional Tier-2 |

## 29. Whole-token-step CUDA graph (research §24) — **STOP_NO_PROMOTE**

| Field | Value |
| --- | --- |
| What | Structure lever: expand decode CUDA-graph beyond tip forward-only. **Harvest tip `e11fb7ab`:** graph owns forward + logits workspace + processors; **sample/`index_copy_` stay eager** (CUDA-graph multinomial exact-fails at first decode token). |
| Hypothesis | Exact reciprocal ≥5% `main_model` vs tip `55949274` (FP16 ≤20.264 s / ≥384.4 TPS) by cutting launch/gap/eager-tail overhead |
| Base tip | `55949274` |
| Branch / WT | `codex/exact-whole-token-cuda-graph` / local+DCC `exact-whole-token-cuda-graph` |
| Tip / commit | **`e11fb7ab`** |
| Opt-in | `whole_token_step_cuda_graph_candidate_context` (requires device sequence state; V32 cold default) |
| Budget line | `structure_launches_gaps_eager_tail` |

### 29a. Capture smoke ladder

| Job | Tip | Result | Note |
| --- | --- | --- | --- |
| `50142058`…`50143356` | pre-`e11fb7ab` FIXes | **FAILED** | sample-in-graph / zero-hits / tok drift (e.g. `50143356` 1322→1243) |
| **`50143692`** | **`e11fb7ab`** | **PASS** | eager-sample control; tok **1322**=1322; RNG equal; hits **1302**; z25-21 3:06 |

### Eager tip full-song reciprocal (`50143838` @ `e11fb7ab`) — **STOP_NO_PROMOTE**

| Field | Value |
| --- | --- |
| Job / node | FP16 **`50143838`** FAILED analysis `1:0` 00:05:59 on **z25-21** @ `e11fb7ab` (all four legs ran) |
| Run root | `/work/imt11/Mapperatorinator/runs/whole-token-step-cuda-graph-fp16-50143838/` |
| Exact | **PASS** — tok **7809**=7809; HO **645**=645; `.osu` sha equal; RNG match |
| Main vs tip **366.11** | cand **334.15** (−8.73%); local per-window `graph_cache={}` capture tax |
| Decision | **STOP_NO_PROMOTE** — do not bare-retry eager tip; Philox revisit = **§29b** |

### 29b. Philox-safe sample graph + amortizing capture — **STOP_NO_PROMOTE**

| Field | Value |
| --- | --- |
| What | Bit-exact Philox-in-graph revisit (Track A). Not turbo (§34). Not layer-skip (§35). |
| Diagnosis (`50143356` @ `e168b81b` vs `50143692` @ `e11fb7ab`) | Sample-in-graph **FAILED** tok 1322→1243, RNG CUDA diverge, hits 1240; **first_mismatch_idx=1** (prefill matched; first graph sample 4081→4083). Eager control **PASS**. Root cause: **capture-as-first-sample** full-vocab multinomial Philox desync (small-vocab probe `49231654`/`667` insufficient). Secondary: local `graph_cache` → `50143838` −8.73% main. |
| Fix tip | **`e3f555f8`** — `register_generator_state(default_generator)`; RNG-restore capture warmup; **always `replay()`**; session-static scores/next_tokens; eager prealloc `index_copy_`; shared forward+sample cache |
| Branch / WT | `codex/exact-whole-token-cuda-graph` / local+DCC `exact-whole-token-cuda-graph` |
| Rung 1 smoke | FP16 **`50145494` COMPLETED** 0:0 00:03:12 on **z25-21** — **PASS** tok **1322**; RNG equal; hits **1302** |
| Run root | `/work/imt11/Mapperatorinator/runs/whole-token-step-cuda-graph-smoke-fp16-50145494/` |
| Rung 2 budget | 1158 tok main_generation: baseline **3.046 s / 380.18 TPS** → cand **3.095 s / 374.11 TPS**; **−0.049 s / −0.043 ms/token** (need ≥0.15) → **MISS** |
| Rung 3 reciprocal | **not submitted** (budget miss) |
| Decision | **STOP_NO_PROMOTE** — Philox-in-graph exactness **proven possible**; headroom **−0.043 ≪ 0.15 ms/token**. Tip stays `55949274` / **366.11**. Track A §29b closed. |
| Revisit | Only with new measured ≥0.15 ms/token evidence — not another sample-graph shape tweak; not turbo. |
| Not this lever | bare-retry `e11fb7ab`; Inductor-in-capture; INT8; §31 unpark; §33/§34/§35 |
| Ledger rule | Own subsection under §29 |

## 30. Elementwise/copy fusion (research §25) — **STOP_NO_PROMOTE**

| Field | Value |
| --- | --- |
| What | Fuse elementwise-ONLY chains (residual add, scale, cast, RoPE cache write); never matmul reductions |
| Hypothesis | Exact reciprocal ≥5% `main_model` vs tip `55949274` by cutting ~254 glue/copy launches/token toward &lt;30; claim ≥0.15 ms/token before full-song |
| Base tip | `55949274` |
| Branch / WT | `codex/exact-elementwise-copy-fusion` / local+DCC `exact-elementwise-copy-fusion` |
| Tip / commit | **`30ea745d`** |
| Opt-in | `elementwise_fusion_candidate_context` (V32 cold default); packs q1 attn-out + fused RoPE emb epilogue kernel |
| Budget line | elementwise ~345 µs/token + copies ~295 µs/token (~0.64 ms family from nsight `49966210`) |
| Nsight source | `49966210` fp16_smoke_node main (1158 tok; tip commit, plain `inference.py` — shared-RoPE wrapper not armed) |

### 30a. Rung 1 — top-5 glue chains by call count (`49966210` elem+mem)

| Rank | Chain | Calls | Total ms | ≈/token | Fuse plan |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | `direct_copy_cast` | 42726 | 189.74 | 36.9 | Fold RoPE `.to(dtype)` into epilogue kernel |
| 2 | `float16_copy` | 41818 | 67.10 | 36.1 | q1 attn-out pack via reshape (no D2D) |
| 3 | float unary `mul` (scale) | 41808 | 63.74 | 36.1 | Fold RoPE `attention_scaling` into epilogue |
| 4 | `add<Half>` residual | 14136 | 26.92 | 12.2 | Standalone EW add only (never into GEMM) |
| 5 | `cos` / `sin` (+ cat) | 14016×2 | ~59 | 12.1 | Single `rope_epilogue` kernel from freqs |

Family totals: elementwise 178921 calls / 398.71 ms; memory 118039 / 341.78 ms → **~256 glue launches/token**.

### 30b. Rungs / jobs

| Rung | Job | Result |
| --- | --- | --- |
| 2 Implement opt-in | tip `30ea745d` | landed |
| 3 Component bitwise | **`50144615`** COMPLETED 0:0 1:23 on **z25-21** | **exact PASS** (rope fp16/fp32 + pack + residual); projected saving **0.050 ms/token** (`clears_budget=false`) |
| 4 Smoke | — | **not submitted** (budget miss) |
| 5 FP16+FP32 reciprocal | — | **not submitted** |

| Run root | `/work/imt11/Mapperatorinator/runs/elemwise-fusion-component-50144615/` |
| Decision | **STOP_NO_PROMOTE** — component bitwise green but measured headroom **0.050 ≪ 0.15 ms/token**; tip stays `55949274` / **366.11**. Do not bare-retry; do not open single-kernel native math replaces. |
| Why | On tip composition, shared-RoPE collapses per-layer RoPE glue to ~1 compute/token; attn-out pack is already near-free at decode shape (`[1,12,1,64]`). Remaining cast-copy mass is outside this EW-only fuse set (§23 mask workspace already STOP). |
| Revisit | Only with a **new** measured EW-only chain list (node/nsight on tip+shared-RoPE path) showing ≥0.15 ms/token before smoke — not bare retry of `30ea745d`. Next Track A: **§32** CUDA graph WHILE/conditional. |
| Not this lever | matmul residual fuse (§6/§11); RMSNorm+Wqkv (§7); §29 bare retry; INT8; allowlist grind |
| Ledger rule | Own section only |

## 31. Batch-invariant speculative (research §26) — **PARK**

| Field | Value |
| --- | --- |
| What | Speculative decode with batch-invariant acceptance |
| Probe | CPU n-gram/prompt-lookup on tip SALVALAI dumps — max accepted/step **1.059** (&lt;1.3 park gate; ≥1.5 would unlock) |
| Decision | **PARK** until acceptance improves or a different draft source is justified |
| Ledger rule | Own section only |

## 32. CUDA graph WHILE/conditional (research §27) — **STOP_NO_PROMOTE**

| Field | Value |
| --- | --- |
| What | Device-side window decode loop via CUDA graph conditional WHILE nodes: whole window = one graph launch; EOS clears handle device-side; **no Inductor** |
| Hypothesis | Exact reciprocal ≥5% `main_model` vs tip `55949274` (FP16 ≤20.264 s / ≥384.4 TPS) by cutting host launch/decision overhead across ~87 windows; claim ≥0.15 ms/token (or window-level ≥0.15 ms×avg tokens/window) from measured/probe data before full reciprocal |
| Base tip | `55949274` |
| Branch / WT | `codex/exact-cuda-graph-while-conditional` / local+DCC `exact-cuda-graph-while-conditional` |
| Tip / commit | **`6bc01810`** (rung 1 `9caec88c`) |
| Opt-in | `conditional_while_scout` + `while_child_graphs` cold by default (absent from production imports); V32 cold default |
| Budget line | host launch/decision + per-token EOS sync across ~87 windows (structure family; not §29 eager-sample tip; not Inductor-in-capture) |
| Prior scout | `49818762` @ `5fafbaf5` — Turing WHILE capability PASS (cited). §32 re-verified on tip composition. |
| Not this lever | §29 bare-retry; Inductor-in-capture; INT8; single-kernel scouts; §31 unpark; smoke/reciprocal after budget miss |

### 32a. Rungs

| Rung | Gate | Status |
| --- | --- | --- |
| 1 Toy WHILE on 2080 Ti | Standalone counter-probe; CUDA 12.8 / torch 2.10+cu128 | **PASS** **`50144761`** COMPLETED 0:0 1:29 on **z25-20** @ `9caec88c` |
| 2 Wrap decode step / EOS | keep_graph model + argmax EOS tail in WHILE; forced stops 1/3/7 | **PASS** exact **`50145010`** COMPLETED 0:0 4:56 on **z25-20** @ `6bc01810` |
| 3 Capture smoke | ~200 tok IDs+RNG | **not submitted** (budget miss) |
| 4 FP16+FP32 reciprocal | smoke + ≥0.15 ms/token | **not submitted** |

### 32b. Rung 1 harvest (`50144761`)

| Field | Value |
| --- | --- |
| Node / GPU | z25-20 / RTX 2080 Ti |
| Runtime | torch 2.10.0+cu128; CUDA runtime 12080; driver 13020 |
| Result | **PASS** — limits 1/3/10 counters exact; memory_stable; median cuda_ms 0.059 / 0.102 / 0.265 |
| Artifacts | `/work/imt11/Mapperatorinator/runs/while-toy-r1-50144761/` |

### 32c. Rung 2 harvest (`50145010`)

| Field | Value |
| --- | --- |
| Node / GPU | z25-20 / RTX 2080 Ti |
| Exact | **PASS** — fixed-work + forced stops 1/3/7; WHILE no post-stop waste; visible/logical cache parity vs k1 |
| Engage | WHILE parent + keep_graph model/tail children; greedy argmax component control only |
| Fixed 8-step | WHILE 23.31 ms vs k1 21.86 (**−6.6%**); vs k4/k8 also negative |
| Forced-stop vs k1 | best Δ **−0.199 ms/token** (stops 1/3/7 all ~−0.20); vs padded k4/k8 early-stop wins (not the budget claim) |
| Budget | need ≥**0.15** ms/token → **MISS** (`clears_token_budget=false`); window-level −17.9 ms vs +13.5 ms gate |
| Artifacts | `/work/imt11/Mapperatorinator/runs/while-wrap-r2-50145010/` (`budget.json`, `real-prefix.json`) |
| Smoke/reciprocal | **not submitted** |

| Decision | **STOP_NO_PROMOTE** — Turing WHILE + tip step-wrap exact green, but measured headroom **−0.20 ≪ 0.15 ms/token** vs equal-work k1; tip stays `55949274` / **366.11** |
| Why | Conditional WHILE machinery adds ~7% vs per-step child-graph replay; host-launch amortization does not clear the structure budget on tip composition |
| Revisit | Only with a **new** measured host-launch/decision probe at full-window scale (≥~90 tok) showing ≥0.15 ms/token **before** smoke — not bare retry of `6bc01810`. §33 OPEN instrumentation (nsys documented; ncu blocked). |
| Ledger rule | Own section only |

## 33. q1 occupancy (research §28) — **OPEN (instrumentation)**

| Field | Value |
| --- | --- |
| What | q1 occupancy instrumentation from tip FP16 `nsys --cuda-graph-trace=node` (+ `ncu` when unblocked) |
| Base tip | `55949274` |
| Pass | **Documented existing** job **`49966210`** (not re-queued; smoke_node already node-trace) |
| Run root | `/work/imt11/Mapperatorinator/runs/exact-device-state-fp16-nsight-49966210/` |
| Slice | FP16 `smoke_node` main_generation **1158** tok (`profile_salvalai_smoke15`; preferred 200–500 exceeded — reuse over re-queue) |
| Artifact | `notes/500tps-section33-q1-occupancy-instrumentation.md` |
| Top node | `q1_rope_cache_attention_kernel<__half,128>` **18.79%** stage GPU time (465.67 ms / 13776 calls / **~33.8 µs** avg / **~402 µs/token**) |
| q1 family share | **18.79%** (`native_q1_self_rope_cache`); next families gemm ~26%, elem+mem ~30% |
| Host gaps | Stage API: `cudaGraphLaunch` ~42% / `cudaLaunchKernel` ~38%; stage sync ~50 ms total (~43 µs/tok); one &gt;500 ms idle is inter-stage |
| `ncu` | **BLOCKED** — `ERR_NVGPUCTRPERM` (probe + `analysis.json`); admin request text in artifact |
| Decision | **OPEN (instrumentation)** — structure levers §29/§30/§32 already STOP; counters feed later occupancy work |
| Revisit | **Microbench only after** usable `ncu` (or equivalent) occupancy/SOL counters exist **and** show a falsifiable ≥0.15 ms/token hypothesis — not bare WHILE retry; not turbo |
| Not this lever | §32 bare-retry; single-kernel math-replace scouts; §35/§36 turbo |
| Ledger rule | Own section only |

---

## 34. SCOPE RULING (2026-07-17, user-approved) — binding

SCOPE RULING (2026-07-17, user-approved — add as ledger §34, verbatim): non-bit-exact engines ARE in scope if provably quality-equivalent and shipped behind a separate flag. Flag naming is by GUARANTEE, not quality: inference_engine=v32 (legacy) | optimized (bit-exact, current tip, unchanged) | turbo (provably distribution-equivalent; token stream differs per seed; requires passing evidence pack). One immutable preset per value, no combinable knobs. Server stays V32-only.

| Field | Value |
| --- | --- |
| What | User-approved scope expansion for quality-equivalent non-bit-exact engines toward 500 TPS |
| Tip still | `55949274` / FP16 **366.11** / FP32 **313.05** (unchanged) |
| Flag | `inference_engine=v32\|optimized\|turbo` — one immutable preset per value |
| `turbo` | distribution-equivalent (rejection-sampling); requires passing evidence pack (see TIER1) |
| `optimized` | stays bit-exact; default-off; nothing changes |
| Server | V32-only |
| Merge | **No merge without approval** |
| Decision | **SCOPE RULING recorded** — opens Track C ladder §35–§38 |
| Track A continues | §32 **STOP** (budget miss); §33 q1 occupancy **OPEN** (nsys `49966210` documented; `ncu` `ERR_NVGPUCTRPERM`); §31 n-gram **PARKED** |
| Evidence packs | `docs/inference_evidence_packs.md` (TIER1/TIER2/TIER3) |
| Ledger rule | Own section only; verbatim ruling above is authoritative |

---

## Evidence packs (TIER1 / TIER2 / TIER3) — definitions

Authoritative full text: `docs/inference_evidence_packs.md`. Reference per lever; do not redefine inline.

| Tier | Applies to | Zero-drift guarantee |
| --- | --- | --- |
| **TIER1** | `turbo` / speculation | Theorem + greedy canary + rejection-rule tests + ≥30×3 KS parity |
| **TIER2** | relaxed-numerics fusion | teacher-forced logit/top-1 gates + TIER1(c) + gallery |
| **TIER3** | quantized / output-distribution | classifier-FID + KS + MaiMod + rcomplexion |

---

## Track C ladder (queued under §34)

Primary path to 500 under the scope ruling. Bit-exact Track A remains valid but structure-bound pending new ≥0.15 ms/token evidence.

## 35. Layer-skip acceptance probe — **GO_SECTION_37** (E&lt;1.3)

| Field | Value |
| --- | --- |
| What | Cheap acceptance probe for layer-skip draft quality (Track C rung 1): teacher-force full 12-layer vs 4-of-12 `[0,3,6,9]` over tip SALVALAI map dumps; E[accepted/step] at temp 0.9 / top-p 0.9 |
| Tip | `55949274` / FP16 dump `49964133` |
| Jobs | `50145811` FAILED 0:30 (dump-token `generate_timing` → 0 timing points); FIX **`50145885` COMPLETED** 0:0 00:00:29 on **z25-21** (tip `.osu` TimingPoints) |
| Script / artifact | `/work/imt11/Mapperatorinator/tmp/layer_skip_acceptance_probe.py` → `runs/s35-layer-skip-acceptance-50145885/acceptance.json` |
| Setup | FP16; temp **0.9** / top-p **0.9**; draft layers `[0,3,6,9]`; γ_primary=5; 87 map windows / **7809** positions |
| Measured | mean α **0.01365**; **E[accepted/step]=1.0138** (γ=3…8 all ≈1.014) |
| Gate | ≥1.8 → §36; 1.3–1.8 → §36 tree/§37; **&lt;1.3 → §37** |
| Decision | **GO_SECTION_37** — layer-skip self-draft acceptance too low for §36; skip self-spec runtime; next = **§37 tiny draft** |
| Not this lever | full §36 turbo runtime; §39 hybrid TIER3 (parallel sibling); INT8-as-FP16 |
| Ledger rule | Own section only |

## 36. Turbo speculative runtime (tiny-draft) — **OPEN (scout harvested)**

| Field | Value |
| --- | --- |
| What | `inference_engine=turbo` speculative runtime: 2-layer draft K=5 + batched teacher verify + Leviathan reject |
| Status | **OPEN** — speculative `generate_window` wired; layer-skip self-spec still **SKIPPED** (§35) |
| Flag | immutable preset `turbo` (not combinable with `optimized` tuning flags) |
| Exactness | **not** bit-exact; requires TIER1 evidence pack (greedy canary in smoke) |
| Runtime | `osuT5/.../inference/turbo/speculate.py` + `engine.py`; draft via `MAPPERATORINATOR_TURBO_DRAFT_CKPT` |
| Branch / turbo tip | `codex/turbo-tiny-draft` (scout @ **`f0b6565a`**; canary ownership → §40 / `codex/turbo-canary-fix`) |
| Smoke `50146929` @ `600f4f14` | COMPLETED — e2e OK (accepted/verify≈**2.13**); **TIER1a FAIL** first_mismatch=**110** |
| FP16 scout | **`50147054`** COMPLETED 0:0 00:09:49 @ `f0b6565a` z25-21 — **directional only** |
| Scout numbers | main_tps **11.94**; main_model **467.34 s** / 5581 tok; main_wall **467.93 s** (≪ tip **366.11** — not a 500 claim; TIER1 incomplete) |
| Artifact | `/work/imt11/Mapperatorinator/runs/s36-turbo-fp16-scout-50147054/summary.json` |
| Optimized path | **untouched** (bit-exact tip still `55949274` / **366.11**) |
| TIER1a | owned by sibling §40 / `codex/turbo-canary-fix` — do not duplicate here |
| Evidence | scout harvested; **full TIER1 pack not started**; no 500 claim |
| Campaign tip | still `55949274` / FP16 **366.11** — do **not** tip-graduate |
| Not this lever | §39 sibling; INT8-as-FP16; §40 canary |
| Ledger rule | Own section only |

## 37. Tiny draft model — **OPEN (runtime wired)**

| Field | Value |
| --- | --- |
| What | 2-layer same-width distilled decoder + Leviathan rejection; EAGLE deferred |
| Status | **OPEN** — train E=**2.921** → speculative generate_window wired @ **`f0b6565a`** |
| Plan | `notes/500tps-section37-tiny-draft-plan.md` |
| Branch / WT / tip | `codex/turbo-tiny-draft` / `turbo-tiny-draft` / **`3bfd7bdb`** (base `55949274`) |
| Scripts | `utils/s37_*.py`, `utils/s36_turbo_speculative_smoke.py`, `jobs/s36-turbo-*.sbatch`, `jobs/s37-tiny-draft-*.sbatch` |
| Runtime | `osuT5/.../inference/turbo/` (`inference_engine=turbo`); draft via `MAPPERATORINATOR_TURBO_DRAFT_CKPT` |
| Data | tip FP16 dump `49964133` (87 win / 7809 tok); tip `.osu` timing |
| Smoke `50146182` @ `12da78ea` | COMPLETED — plumbing PASS; baseline E=**1.0629**; fp16 train **NaN** |
| FIX `50146230` @ `e4c4e662` | COMPLETED — E=**1.750** (1.3–1.8) |
| Train `50146289` @ `818002cd` | **COMPLETED** 0:0 00:03:07 h36-5 — resume smoke; CE+KL; 87 windows; 2000 steps; after **E=2.921** (α=0.697); gate **GO_SECTION_36_TURBO_SCAFFOLD** |
| Artifact (smoke) | `/work/imt11/Mapperatorinator/runs/s37-tiny-draft-smoke-50146230/smoke.json` |
| Artifact (train) | `/work/imt11/Mapperatorinator/runs/s37-tiny-draft-train-50146289/train.json` |
| Draft ckpt | `/work/imt11/Mapperatorinator/runs/s37-tiny-draft-train-50146289/draft_train.pt` |
| Next | §40 canary ownership; TIER1b/c after TIER1a PASS; scout `50147054` harvested (directional **11.94** main_tps) |
| Evidence | **TIER1** required — do **not** claim 500 |
| Campaign tip | still `55949274` / FP16 **366.11** |
| Not this lever | §39 sibling; INT8-as-FP16 |
| Ledger rule | Own section only |

## 38. Tier-2 relaxed fused decoder step — **PARKED** (was STOP_NO_PROMOTE)

| Field | Value |
| --- | --- |
| What | TIER2 relaxed-numerics fused decoder step (7-kernel layer: norm+Wqkv, q1, Wo+res, cross block, fc1, fc2+res, glue; fp32 reductions) behind `inference_engine=turbo` |
| Status | **PARKED** — strategy shift: turbo deep-research package supersedes §38; was STOP_NO_PROMOTE (quality PASS; perf MISS) |
| Branch / WT | `codex/turbo-tier2-fused-step` / tip `03f8a494` (fusion `24ee13bd`; base `55949274`) |
| Rung 1a | Blanket fp32 Linear wrap — `50149339`/`50149340` @ `c77aab55`: top1 PASS; max_rel **FAIL** (superseded) |
| Rung 1b | True 7-stage — `50149619`/`50149620` @ `24ee13bd`: max_rel **0** / top1 **1.0** / n=7809 |
| Rung 1c quality | `50149733` FP16: n=**102055**; max_rel **0**; top1 **1.0** → **tier2_quality_gate_pass=true** |
| Microbench | `50149734` FP16 proxy: tip 0.1005 → turbo 0.0936 ms/tok; saved **0.0069 ≪ 0.15** → **MISS** |
| Decision | **STOP_NO_PROMOTE** — do not reciprocal; tip stays `55949274` / **366.11** |
| Revisit | Deeper one-token CUDA launch-collapse with measured ≥0.15 ms/tok — not bare-retry Python rearrange |
| Optimized default | **unchanged** (bit-exact) |
| Campaign tip | still `55949274` / FP16 **366.11** — **no 500 claim**; **no merge** |
| Handoff | `notes/500tps-section38-handoff.md` |
| Not this lever | §39 hybrid / turbo_mixed; speculative re-open; INT8-as-FP16 |
| Ledger rule | Own section only |

## 39. Hybrid-arena TIER3 quality audit — **FAIL**

| Field | Value |
| --- | --- |
| What | Quality audit of sealed INT8-hybrid selected-arena+compiled-cross (~502–504 TPS) vs exact tip optimized FP16 |
| Status | **FAIL** — does **not** unlock `turbo_mixed`; calibrates quality bar |
| Exact engine | tip `55949274` / `inference_engine=optimized` / **precision=fp16** / `attn_implementation=sdpa` |
| Hybrid engine | tip `0dbab9e5` / sealed stack (INT8 MLP + FP16-packed cross + compiled-cross + shared arena); **outer precision=fp32** — **not FP16** |
| Sealed perf refs | jobs `49965070` / `49959842` (h36-5; ~502–504 main TPS) |
| Songs × seeds | salvalai, lambada, pegasus × 30 seeds = **90 maps/engine** |
| Gen jobs | exact `50145980`; hybrid `50145981` (salvalai) / `50145982` (lambada) / `50145983` (pegasus) |
| Audit job | `50147476` (retry after `50145984` density hist bug, `50147284` MaiMod/`rich` miss) |
| Maps root | `/work/imt11/Mapperatorinator/runs/s39-hybrid-tier3-maps/` |
| Report | `/work/imt11/Mapperatorinator/runs/s39-hybrid-tier3-audit-50147476/tier3_report.json` |
| Scripts | `/work/imt11/Mapperatorinator/tmp/s39-hybrid-tier3/{s39_batch_generate.py,s39_tier3_audit.py}`; `jobs/s39_*.sbatch` |

### TIER3 summary (α=0.01; KS Bonferroni m=5 → α′=0.002)

| Metric | Result | Detail |
| --- | --- | --- |
| classifier-FID | **advisory PASS** | FID=**0.384** (1950 feats/side); weak signal per `classifier/README.md` |
| KS HO count | **PASS** | p=0.989 |
| KS density curve | **PASS** | p=0.914 |
| KS type histogram | **PASS** | p≈1.0 |
| KS timeshift | **PASS** | p=0.059 (≥ α′) |
| KS slider lengths | **FAIL** | p=**2.1e-17** (stat=0.042); n≈21.7k / 22.0k |
| MaiMod issue-count | **INCOMPLETE** | env missing `rich` on compute (`ModuleNotFoundError`); not decisive |
| rcomplexion | **PASS** | p=0.164; mean exact **2.496** vs hybrid **1.843** (KS non-reject @ 0.01) |

| Field | Value |
| --- | --- |
| Decision | **FAIL** — hybrid slider-length distribution differs from exact tip |
| Degraded | **slider_lengths** (hard); MaiMod incomplete (infra) |
| Recommendation | Do **not** ship `turbo_mixed` / mixed-precision preset from this stack. Keep hybrid as architectural/perf evidence only (~502 TPS sealed). Track C continues via turbo (§37/§40+). |
| Campaign close | **No** — §39-pass alone would need user ruling; **§39-fail does not close** and does **not** request a close ruling |
| Merge | **No merge** |
| Not this lever | §35 (sibling); bare-retry §6–§32 exact scouts; calling INT8 “FP16” |
| Ledger rule | Own section only |

---

## 40. TIER1a greedy canary fix (turbo vs optimized) — **STOP_ESCALATE**

| Field | Value |
| --- | --- |
| What | Diagnose/fix TIER1a first_mismatch=**110** (smoke `50146929` / retest `50147161`) |
| Status | **STOP_ESCALATE** — not a rejection-rule bug; not K-batch vs one-token inside turbo |
| Branch / WT | `codex/turbo-canary-fix` / `turbo-canary-fix` (base `3bfd7bdb`) |
| Mismatch window | turbo `[12,1648,2236,…]` vs optimized `[12,1648,2213,…]` @ abs=110 |
| Logit dump `50147276` | eager argmax **2236**; batched-K (forced) argmax **2236**; optimized **2213**; top2 Δlogit≈**0.008** |
| Classification | **cross-engine FP16 numerics** (HF DynamicCache teacher vs optimized cuda-graph). Speculation indexing OK. |
| Fix tip | `e9d3d58a` (probe/handoff only; base `3bfd7bdb`) — STOP (no Leviathan/KV patch closes optimized canary) |
| Canary gate | **FAIL** — ≥500×3 seeds vs optimized not attempted after STOP |
| Artifact | `/work/imt11/Mapperatorinator/runs/s40-tier1a-logit-dump-50147276/logit_dump.json` |
| Handoff | `notes/500tps-section40-canary-handoff.md` |
| Escalate | **§41** exact-shared / graph-aligned teacher verify for TIER1a (batch-invariant kernels alone insufficient — eager already matches batched) |
| Optimized path | **untouched** |
| Campaign tip | still `55949274` / FP16 **366.11** — no 500 claim |
| Not this lever | §39; scout `50147054` harvest; INT8-as-FP16 |
| Ledger rule | Own section only |

## 41. Verify fastpath + graph-aligned teacher (W2) — **PARTIAL (canary PASS; c_verify MISS)**

| Field | Value |
| --- | --- |
| What | (A) K-token teacher verify on StaticCache/active-prefix (c_verify ≤1.2× Q=1). (B) **Aligned sequential Q=1 teacher** (StaticCache + native hooks) so turbo greedy matches optimized at §40 mismatch@110 |
| Status | **PARTIAL** — canary@110 **PASS**; c_verify gate **MISS** (best 1.69× > 1.2) |
| Branch / WT / tip | `codex/turbo-verify-fastpath` / `turbo-verify-fastpath` / **`7033d62f`** (canary auth `b3a0b27e`; base `3bfd7bdb`) |
| Runtime | `osuT5/.../inference/turbo/verify_fastpath.py` + speculate wiring |
| Modes | greedy → sequential Q=1 eager+native (CUDA-graph Q=1 unsafe — zero logits `50148138`); sample → multi-token K-forward (+ optional K-graphs) |
| Auth microbench | **`50147970`** @ `951bf53b` — Q1 **1.853 ms** (cuda_graph); K=5 cudagraph verify **3.125 ms** → ratio **1.686×**; eager K≈**7.07×** |
| Gate A (≤1.2×) | **MISS** — best **1.686×** (K=5 cudagraph); revisit with tighter K-graph / shared arenas |
| Gate B (canary@110) | **PASS** — job **`50148210`** @ `b3a0b27e`: aligned argmax **2213** == optimized **2213** (was turbo eager 2236) |
| Artifacts | `/work/imt11/Mapperatorinator/runs/s41-verify-fastpath-50147970/`, `.../s41-canary-probe-50148210/` |
| Handoff | `notes/500tps-section41-verify-fastpath-handoff.md` |
| §40 | **STOP_ESCALATE** absorbed — no rejection-rule retries |
| Optimized path | **untouched** |
| Campaign tip | still `55949274` / FP16 **366.11** — no 500 claim; full ≥500×3 TIER1a still required |
| Next | cut c_verify toward ≤1.2×; keep greedy on eager-native Q=1; optional fix Q=1 CUDA graphs |
| Not this lever | §39; §40 Leviathan patches; §43 train |
| Ledger rule | Own section only |

## 43. Draft quality held-out + sweeps (W4) — **DONE**

| Field | Value |
| --- | --- |
| What | Offline draft-quality: held-out E, multi-song distill, K/γ/temp + tree/cheap-draft sweeps → recommend (K, draft) for perf build |
| Status | **DONE** — job **`50147299`** COMPLETED 00:09:40 z25-20 |
| Branch / WT | `codex/turbo-draft-quality` / `turbo-draft-quality` |
| Base draft | tip train **`50146289`** E=**2.921** SALVALAI-only |
| Held-out (a) | tip draft: ela **1.980** (nan-filtered), nube **2.087** → mean **2.034**; SALVALAI ref **2.922** |
| Multi-song (b) | salvalai+pegasus+lambada CE/KL 4000 → held-out mean E **2.438** |
| Cheap (d) | 1-layer held-out mean E **2.099**; half-width = cost sim (α proxy=1-layer) |
| Sweeps (c) | T∈{0.7,0.9,1.1}, γ∈3…8, K∈{1,2,4}; tree K>1 loses E/cost |
| **Recommend (perf)** | **1-layer · K=1 · γ=3 · temp=0.9** (`draft_1layer.pt`) |
| **Recommend (quality)** | **multi-2layer · K=1 · γ=5 · temp=0.9** (`draft_multsong.pt`) |
| Artifacts | `/work/imt11/Mapperatorinator/runs/s43-draft-quality-50147299/` (`acceptance_table_v2.json`) |
| Docs | `notes/500tps-section43-draft-quality.md`, `notes/500tps-section43-handoff.md` |
| Not this lever | generate_window / canary / §39 wiring; no 500 claim |
| Ledger rule | Own section only |

## 44. TIER1 evidence pack harness — **HARNESS READY**

| Field | Value |
| --- | --- |
| What | Automate TIER1 evidence pack (W5): 30×3 generate driver + KS parity + greedy-canary runner + one-command orchestrator |
| Status | **HARNESS READY** — CPU dry-run + unit tests PASS; full GPU pack deferred until perf tip + §40 TIER1a PASS |
| Branch / WT | `codex/turbo-tier1-harness` / `turbo-tier1-harness` (base `3bfd7bdb`) |
| Entrypoint | `scripts/run_tier1_evidence_pack.sh` → `utils/s44_tier1_evidence_pack.py` |
| Docs | `docs/inference_evidence_packs.md` (TIER1); handoff `notes/500tps-tier1-harness-handoff.md` |
| Automated | generate manifest (180 jobs); KS (HO count / density / type / timeshift / slider length); canary plan; TIER1b pytest |
| Opt-in GPU | `--execute-generate`, `--execute-canary`; DCC `jobs/s44-tier1-evidence-pack.sbatch` |
| Stubbed | optional classifier-FID / MaiMod / rcomplexion (§39 reuse plan only — do **not** duplicate live §39) |
| Campaign tip | still `55949274` / FP16 **366.11** — **no 500 claim** |
| Decision | Harness scaffolding only; record pack paths/job IDs when executed after integration |
| Not this lever | §39 audit execution; tip graduation; INT8-as-FP16; waiting on perf |
| Ledger rule | Own section only |

## 45. Combined turbo perf integrator — **EXPLAINED (not dead-end)** — TURBO DEEP-RESEARCH PACKAGE

| Field | Value |
| --- | --- |
| What | Integrate §41 canary-aligned teacher verify + §43 perf draft **1-layer K=1 γ=3 temp=0.9**; TIER1a ≥500×3; one FP16 SALVALAI scout |
| Status | **EXPLAINED / REOPENED** — 44.52 TPS was an unoptimized sampled-path measurement, not a structural dead-end |
| Package | TURBO DEEP-RESEARCH PACKAGE (2026-07-17): cycle dissection + machinery map + production survey |
| Branch / WT / tip | `codex/turbo-integrator` / scout **`70599188`**; diagnose instrumentation on integrator tip |
| Preset | `turbo-integrator-s45-1layer-g3-v1` · `PRIMARY_GAMMA=3` · tree K=1 |
| Canary | `50148575` **PASS** (3×500) |
| Scout | **`50148770`** main_tps **44.52** / main_model **155.36 s** / **6917** tok; sustained median ~**23 ms/tok** (~45 ms/cycle) |
| Why 44.5 (path hits off) | (1) **verify graphs hard-off for K>1** (`verify_fastpath.py` gate) while `_VERIFY_CUDA_GRAPH=1` still set; (2) **TEACHER_ALIGNED inert under `do_sample`**; (3) **crop-to-L KV rebuild** recomputing accepted KV (~12–16 ms/cycle); (4) **eager draft ×4** (3rd discarded); (5) **~10 `.item()` host syncs + 7 vocab sorts** |
| Measured constants | tip step **1.853 ms**; graphed K=5 verify **3.125 ms**; §42 draft **1.08 ms/tok**; E held-out ≈**1.97–2.44** |
| Ceilings (projections, not prod TPS) | **~570 TPS** if W-KV+W-VG+W-DG all land (~1.7 ms/tok); **~310 TPS** if any one misses (below tip); **baseline-glue ~540 TPS** with no speculation (1/1.853 ms) if 0.88 ms/tok loop glue dies |
| Keep-KV ruling | See **RULING** below — sampled keep-KV is DOCUMENTED DRIFT; do not re-block on bit-parity |
| Handoff | `notes/500tps-section45-handoff.md` |
| Campaign tip | still `55949274` / FP16 **366.11** — **no 500 claim**; **no tip graduate**; **no merge** |
| Decision | **OPEN** parallel workers §46–§50; §38 TIER2 **PARKED** behind them; first path to ≥450 scout carries |
| Not this lever | §39 / turbo_mixed; tip graduation; merge to main; §44 until ≥384 scout |
| Ledger rule | Own section only |

### RULING (binding) — keep-accepted-KV fp16 ULP

| Field | Value |
| --- | --- |
| Finding | keep-accepted-KV fp16 ULP divergence (verify-written KV rows vs fused-kernel rows) forced the `600f4f14` revert / crop-to-L rebuild |
| Sampled turbo | **DOCUMENTED DRIFT** under §34 — TIER1 KS parity covers distribution equivalence |
| Greedy TIER1a | keeps **crop-rebuild / aligned-Q1** mode — certification unaffected |
| Binding | **Do not block sampled-path keep-KV on bit-parity again** |

### Process changes (binding, from package)

1. **Path-hit logs:** every scout must assert and LOG which paths actually executed (graph hit counters per phase: draft/verify/rebuild) + accepted/verify + per-phase NVTX ms. Env flags ≠ active paths (the 44.5 scout had `_VERIFY_CUDA_GRAPH=1` while eager ran).
2. **Recompute ceiling** from measured constants after every rung; if ceiling &lt;420 after W-KV+W-VG+W-DG land → STOP turbo and shift budget to W-BASE + EAGLE-head.
3. **Kill criteria:** verify stuck &gt;1.35× after graph-native attempt, or draft chain &gt;2 ms, or E(real runtime) &lt;1.7 → that worker STOPs with its number recorded; no grinding.
4. **No §44 / no 500 claim** until a scout ≥**384** sustained; song wall wins; tip stays `55949274`/366.11 until then.
5. **Persistent caches across windows** everywhere — never rebuild verify_fp/graph caches per window (`speculate.py` per-window teardown class of bug).

## 46. Baseline glue-elimination v2 (W-BASE) — **OPEN STUB**

| Field | Value |
| --- | --- |
| What | Independent no-speculation path: cut ~0.88 ms/tok non-model glue on optimized engine |
| Worker | **W-BASE** — own branch; possibly bit-exact |
| Targets | persistent sampling-tail graph (temp+top-p sort+softmax+multinomial; §29c Philox PASS); device token buffer vs `torch.cat`; pinned-flag EOS vs per-token `.max()==0` sync |
| Not | bare §29 retry — v2 mandates persistent cross-window caches + separate small tail graph |
| Ceiling | **~540 TPS** (1/1.853 ms) — projection |
| Gate | e2e scout ≥**450**; if bitwise holds, can graduate EXACT tip itself |
| Status | **OPEN** |
| Campaign tip | `55949274` / **366.11** — no 500 claim yet |
| Ledger rule | Own section only |

## 47. Keep-accepted-KV + O(1) rollback (W-KV) — **OPEN STUB**

| Field | Value |
| --- | --- |
| What | keep-accepted-KV under sampled DOCUMENTED DRIFT; O(1) `cache_position` rewind + in-graph stale masking |
| Worker | **W-KV** — branch `codex/turbo-keep-accepted-kv` |
| Rules | accepted tokens NEVER re-forwarded (teacher or draft); q1 bucket from rolled-back length; greedy TIER1a stays crop-rebuild/aligned-Q1 |
| Gates | teacher forwards/cycle == **1**; **−10 ms/cycle** vs §45; TIER1a still PASS in canary mode |
| Kill | cannot get teacher forwards/cycle==1 without breaking canary-mode separation → STOP with number |
| Status | **OPEN** |
| Campaign tip | `55949274` / **366.11** |
| Ledger rule | Own section only |

## 48. Graph-native K=γ verify (W-VG) — **OPEN (code + in-loop probe)**

| Field | Value |
| --- | --- |
| What | Lift k>1 graph gate; graph-native verify with static `{ids[1,γ], cache_position[γ]}`; mask in-graph; **no HF prepare_inputs** (device-scalar sync) |
| Worker | **W-VG** — branch `codex/turbo-graph-native-verify` |
| Pattern | production capture + side-stream warmup (§41 zero-logits bug) |
| Wiring | `verify_fastpath.py` graph-native path; persistent `TurboRuntime.verify_fastpath`; `utils/s48_graph_native_verify_inloop.py` |
| Gate | in-loop c_verify ≤**1.2×** (≤2.22 ms) measured inside the real cycle, not a harness |
| Kill | verify stuck &gt;1.35× after graph-native attempt → STOP |
| Status | **OPEN** — measuring in-loop (see handoff jobs) |
| Handoff | `notes/500tps-section48-handoff.md` |
| Campaign tip | `55949274` / **366.11** — **no 500 claim** |
| Ledger rule | Own section only |

## 49. Graphed draft chain (W-DG) — **OPEN STUB**

| Field | Value |
| --- | --- |
| What | Merge §42; draft on StaticCache; chain γ=3 steps in **ONE** graph incl. in-graph sampling (Philox graph-safe per §29c) + embedding feedback |
| Worker | **W-DG** — own branch |
| Note | with keep-KV the discarded 3rd forward becomes next cycle's first |
| Gate | 3 drafts ≤**1.2 ms** total |
| Kill | draft chain &gt;2 ms → STOP |
| Status | **OPEN** |
| Campaign tip | `55949274` / **366.11** |
| Ledger rule | Own section only |

## 50. Config / margin sweep (W-ACC) — **QUEUED** (after cycle &lt;10 ms)

| Field | Value |
| --- | --- |
| What | Multi-song 2-layer draft (E 2.44) vs 1-layer on REAL runtime; γ∈{3,4,5} e2e sweep; pick max e2e TPS not max E |
| Worker | **W-ACC** — queued until speculative cycle &lt;**10 ms** |
| Stretch | EAGLE-style head (c_d≈0.05–0.1×) as its own § only if needed |
| Status | **QUEUED** |
| Campaign tip | `55949274` / **366.11** |
| Ledger rule | Own section only |

### Parked behind §46–§50

| Ledger § | Status |
| --- | --- |
| **§38** TIER2 fused decoder step | **PARKED** — STOP_NO_PROMOTE (quality PASS / microbench MISS); reopen only after turbo/baseline workers settle or deeper CUDA collapse ≥0.15 ms/tok |
| **§44** TIER1 evidence pack | **HARNESS READY** — fire full pack only on ≥**384** scout (W-CERT) |
