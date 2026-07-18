# §58 Turbo-on-Tiger (PRIMARY TRACK) — handoff

**Status:** **C_VERIFY GATE PASS** (2026-07-18) — ratio **≪1.3×** (better than expect band)  
**Strategy pivot:** turbo stacks on Tiger14n PR #120 `feat/compiled-decode`, **not** on optimized tip.  
**Branch / WT:** `codex/turbo-on-tiger-pr120` @ **`3641f39e`** (scaffold `447df085`)  
**Local WT:** `/work/projects/Mapperatorinator-worktrees/turbo-on-tiger-pr120`  
**DCC WT:** `/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-on-tiger-pr120`  
**Base:** `w1/tiger14n-compiled-decode` / `d01cdd2799891f78c871ff5889cfbb6a661a9e46` (verified)  
**Optimized tip FROZEN:** `55949274` / FP16 **366.11** — **do not modify its internals**; **no merge**; **no 500 claim**  
**Remote:** `islamtayeb` only — **no push to Tiger14n / PR #120**

## Intent

Port strict rejection-sampling turbo (§34) + keep-accepted-KV (`cache_position` rewind on bucketed StaticCache) + graphed draft chain onto tiger's decode loop. Teacher verify = **tiger `CUDAGraphDecoder` at q_len=K** (uniform HF path). **Do not** port §54 Stage B fused-kernel grind.

Upstream posture: additive so this can later stack as a PR on #120.

## First gate — c_verify ratio — **PASS**

| Field | Value |
| --- | --- |
| Expect | **~1.1–1.3×** vs tiger q_len=1 (kill ≫1.3 without capture bug) |
| Scout | `utils/s58_tiger_c_verify_scout.py` |
| Job | **`50192694`** COMPLETED 00:00:31 on **RTX 2080 Ti** @ `3641f39e` |
| Metric | CUDA-event avg replay ms |
| Artifacts | `/work/imt11/Mapperatorinator/runs/s58-tiger-cverify-50192694/` · local `notes/s58-artifacts/` |

**Measured (2080 Ti, fp16, cache_len=768, prefix=256):**

| K | c_verify ms | q1 ms | ratio | vs expect |
| --- | ---: | ---: | ---: | --- |
| **3** | **3.372** | **3.984** | **0.846×** | **PASS** (below 1.1 — better) |
| 5 | 3.386 | 3.984 | 0.850× | **PASS** |

Interpretation: on tiger's uniform StaticCache path, K-token verify is **not** the §48/§54 tax (1.67× / 1.41× on optimized fused stack). Cache-length attention dominates; widening q_len to γ is nearly free. Duplicate scout `50192778` canceled.

## Scaffold landed

| Piece | Path | Notes |
| --- | --- | --- |
| q_len=K graph | `compiled_decode.CUDAGraphDecoder(q_len=…)` + `get_k_decoder` | q_len=1 production unchanged |
| tiger verify helper | `turbo/tiger_verify.py` | persistent session; rewind after probe |
| rejection | `turbo/rejection.py` | §34 Leviathan |
| keep-KV | `turbo/kv_rollback.py` | O(γ) StaticCache zero |
| engine scaffold | `turbo/engine.py` | preset; full window loop **not wired yet** |
| draft chain scaffold | `turbo/draft_chain_graph.py` | pending tiger-native capture |

## Next gate

1. **Wire speculative window** onto tiger decode: rejection + keep-KV + graphed draft chain (still no fused verify).  
2. **In-loop E** (accepted/verify) on tiger teacher + draft.  
3. Full-map `ms_per_map_token` + `cold_start_seconds` + `main_tps` (metric ruling).  
4. Prefer A5000/Ada for song walls once §57b A5000 dumps drain (Turing q1 ~4 ms is not the production target).

## GPU coordination

§57b may hold A5000 — c_verify used **2080**. Keep ≤2 concurrent GPU with §57b.

## Do-not

- Modify optimized tip `55949274` internals  
- Port §54 fused mrow / Stage B verify grind onto tiger  
- Claim 500 / merge to main from this track yet  
- Fold relaxed acceptance into `turbo` (§34)

## Ruling

Primary track = **turbo-on-tiger**. c_verify gate **cleared** at **0.85×**. Tip remains frozen evidence, not the integration base.
