# §58 Turbo-on-Tiger (PRIMARY TRACK) — handoff

**Status:** **C_VERIFY PASS (better than expect)** (2026-07-18)  
**Strategy pivot:** turbo stacks on Tiger14n PR #120 `feat/compiled-decode`, **not** on optimized tip.  
**Branch / WT:** `codex/turbo-on-tiger-pr120` @ **`0eb51365`**  
**Local WT:** `/work/projects/Mapperatorinator-worktrees/turbo-on-tiger-pr120`  
**DCC WT:** `/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-on-tiger-pr120`  
**Base:** `w1/tiger14n-compiled-decode` / `d01cdd2799891f78c871ff5889cfbb6a661a9e46` (verified)  
**Optimized tip FROZEN:** `55949274` / FP16 **366.11** — **do not modify its internals**; **no merge**; **no 500 claim**  
**Remote:** `islamtayeb` only — **no push to Tiger14n / PR #120**

## Intent

Port strict rejection-sampling turbo (§34) + keep-accepted-KV (`cache_position` rewind on bucketed StaticCache) + graphed draft chain onto tiger's decode loop. Teacher verify = **tiger `CUDAGraphDecoder` at q_len=K** (uniform HF path). **Do not** port §54 Stage B fused-kernel grind.

Upstream posture: additive so this can later stack as a PR on #120 (when user asks).

## First gate — c_verify ratio — **SEALED**

| Field | Value |
| --- | --- |
| Expect | **~1.1–1.3×** vs tiger q_len=1 (kill >1.35× class) |
| Scout | `utils/s58_tiger_c_verify_scout.py` |
| Job | **`50192694`** COMPLETED 00:00:31 · `gpu:2080:1` · RTX 2080 Ti |
| Commit | `3641f39e` |
| Artifact | `/work/imt11/Mapperatorinator/runs/s58-tiger-cverify-50192694/` |

**Measured (CUDA-event avg, prefix=256, cache_len=768, fp16):**

| K | c_verify ms | q1 ms | ratio | vs expect |
| --- | ---: | ---: | ---: | --- |
| 3 | **3.372** | **3.984** | **0.846×** | **PASS** (≪1.3; below 1.1 band = cheaper verify) |
| 5 | **3.386** | **3.984** | **0.850×** | **PASS** |

**Verdict:** tiger uniform path has **no fused-kernel forfeiture** — K≈Q1 (slightly under). Gate clears; do not grind. Absolute Q1 ~4.0 ms matches W1 tiger-2080 ~229 TPS ballpark.

## Scaffold landed

| Piece | Path | Notes |
| --- | --- | --- |
| q_len=K graph | `compiled_decode.CUDAGraphDecoder(q_len=…)` + `get_k_decoder` | q_len=1 production unchanged |
| tiger verify | `turbo/tiger_verify.py` | persistent session; rewind after probe |
| rejection | `turbo/rejection.py` | §34 Leviathan |
| keep-KV | `turbo/kv_rollback.py` | O(γ) StaticCache zero |
| draft chain scaffold | `turbo/draft_chain_graph.py` | tiger q_len=1 × γ + rewind; in-graph sample next |
| engine scaffold | `turbo/engine.py` | preset; speculative window **not wired yet** |

## Next

1. **Wire speculative window** on tiger: draft propose → tiger `verify_k` → rejection → keep-KV rewind  
2. Measure **in-loop E** + path-hit counters  
3. Full-map `ms_per_map_token` + `cold_start_seconds` + `main_tps` (metric ruling)  
4. Optional: A5000/Ada c_verify confirm (not blocking — 2080 already clears)

## GPU coordination

§57b on A5000 + this scout on 2080 = 2 GPU (at limit). Prefer 2080 for tiger microbenches while dumps run.

## Do-not

- Modify optimized tip `55949274` internals  
- Port §54 fused mrow / Stage B verify grind onto tiger  
- Claim 500 / merge to main from this track yet  
- Fold relaxed acceptance into `turbo` (§34)  
- Push to Tiger14n / PR #120 without explicit user request  

## Ruling

Primary track = **turbo-on-tiger**. Tip remains frozen evidence, not the integration base. c_verify gate **PASS** @ **0.85×**.
