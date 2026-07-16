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
| — | #12 q1 RoPE/cache head-group CTA scheduling | pending | pending | **OPEN** — FP16 `49998739`; FP32 `49998740` submitted |
| 1 | #1 shared-RoPE + device state | **366.11** | 313.05 | **GRADUATED tip** |
| 2 | #2 shared-runtime packaging | 313.54 | **317.46** | **SEALED** pin only |
| — | #3–#7c, #10 above | — | — | no graduate |
| — | #11 self Wo linear (no residual) | — | — | **STOP_NO_PROMOTE** |

**Still short of 500:** FP16 needs ≤15.618 s main (−5.71 s from tip).

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

## 12. Native q1 RoPE/cache head-group CTA scheduling — **OPEN**

| Field | Value |
| --- | --- |
| What | Pack 2 heads/CTA on already-exact native q1 RoPE/cache path (identical per-head reduction order; no Wo/proj_out/RMSNorm math replace) |
| Hypothesis | Exact reciprocal ≥5% main_model vs tip `55949274` from q1 scheduling / fewer underfilled CTAs (nsight ~19%) |
| Base tip | `55949274` |
| Branch / WT | `codex/exact-q1-rope-cache-headgroup` / local `exact-q1-rope-cache-headgroup` |
| Tip / commit | **`9d034259`** |
| Opt-in | `q1_rope_cache_headgroup_candidate_context` → `native_q1_rope_cache_headgroup` (V32 cold default) |
| Kernel | `q1_rope_cache_attention_headgroup` (HEADS_PER_CTA=2, block 128×2) |
| Jobs | FP16 **`49998739`**; FP32 **`49998740`** (submitted) |
| Run roots | `/work/imt11/Mapperatorinator/runs/q1-headgroup-fp{16,32}-<jobid>/` (match RUN_LABEL) |
| Exact | pending |
| Measured | pending |
| Decision | **OPEN** — scout submitted |
| Not this lever | §6/§7/§10/§11 math replaces; INT8; bare split-KV; compiled-cross |
| Ledger rule | Own section only |
