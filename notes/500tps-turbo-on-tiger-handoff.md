# §58 Turbo-on-Tiger (PRIMARY TRACK) — handoff

**Status:** **STEP 2 WIRED — SPEC WINDOW SCOUT QUEUED** (2026-07-18)  
**Strategy pivot:** turbo stacks on Tiger14n PR #120 `feat/compiled-decode`, **not** on optimized tip.  
**Branch / WT:** `codex/turbo-on-tiger-pr120` @ *(see job preflight)*  
**Local WT:** `/work/projects/Mapperatorinator-worktrees/turbo-on-tiger-pr120`  
**DCC WT:** `/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-on-tiger-pr120`  
**Base:** `w1/tiger14n-compiled-decode` / `d01cdd2799891f78c871ff5889cfbb6a661a9e46` (verified)  
**Optimized tip FROZEN:** `55949274` / FP16 **366.11** — **do not modify its internals**; **no merge**; **no 500 claim**  
**Remote:** `islamtayeb` only — **no push to Tiger14n / PR #120**

## Intent

Port strict rejection-sampling turbo (§34) + keep-accepted-KV (`cache_position` rewind on bucketed StaticCache) + graphed draft chain onto tiger's decode loop. Teacher verify = **tiger `CUDAGraphDecoder` at q_len=K** (uniform HF path). **Do not** port §54 Stage B fused-kernel grind.

Upstream posture: additive so this can later stack as a PR on #120 (when user asks).

## First gate — c_verify ratio — **SEALED PASS**

| Field | Value |
| --- | --- |
| Expect | **~1.1–1.3×** vs tiger q_len=1 (kill >1.35× class) |
| Job | **`50192694`** COMPLETED · `gpu:2080:1` · RTX 2080 Ti |
| Commit | `3641f39e` |
| Artifact | `/work/imt11/Mapperatorinator/runs/s58-tiger-cverify-50192694/` |

| K | c_verify ms | q1 ms | ratio | vs expect |
| --- | ---: | ---: | ---: | --- |
| 3 | **3.372** | **3.984** | **0.846×** | **PASS** (≪1.3) |
| 5 | **3.386** | **3.984** | **0.850×** | **PASS** |

## STEP 2 — speculative window (this turn)

| Piece | Path | Notes |
| --- | --- | --- |
| Leviathan loop | `turbo/speculate.py` | draft → tiger `verify_k` → reject → keep-KV |
| engine | `turbo/engine.py` | `speculative_generate_wired=True`; draft attach |
| tiger verify | `turbo/tiger_verify.py` | default **leave** KV at prefix+K for keep-KV |
| draft chain | `turbo/draft_chain_graph.py` | γ×q1 + `logits_after` for keep path |
| processor hook | `processor.py` | `MAPPERATORINATOR_TURBO=1` opt-in; skips incompatible teachers |
| scout | `utils/s58_tiger_spec_window_scout.py` | E + ms/map-token + cold_start + main_tps |
| job | `jobs/s58-tiger-spec-window.sbatch` | 2080 Ti; unique TMPDIR/TORCH_EXTENSIONS |

**Preset:** strict rejection-sampling only (§34). γ default scout **3** (`MAPPERATORINATOR_TURBO_GAMMA`). Draft: §43 `draft_1layer.pt`. Structural processors default **off** (§43 align).

**Kill gates:** E_median < 1.05 · negative wall · no turbo windows.

**Measured:** *(fill from `runs/s58-tiger-specwin-<job>/summary.json`)*

| Metric | Value |
| --- | --- |
| job | — |
| E_median / E_mean | — |
| ms_per_map_token | — |
| main_tps | — |
| cold_start_seconds | — |
| decision | — |

## Next (after STEP 2 harvest)

1. If E/wall PASS → path-hit counters + optional γ=5 confirm  
2. A5000/Ada song wall when §57b drains (Turing ~4 ms q1 not production target)  
3. Stackable PR hygiene vs #120 (user-requested only)

## GPU coordination

Prefer **2080** for parity with c_verify. Keep ≤2 concurrent GPU with §57b. Pending W1 Ada cells are PD (not running).

## Do-not

- Modify optimized tip `55949274` internals  
- Port §54 fused mrow / Stage B verify grind onto tiger  
- Claim 500 / merge to main from this track yet  
- Fold relaxed acceptance into `turbo` (§34)  
- Push to Tiger14n / PR #120 without explicit user request  

## Ruling

Primary track = **turbo-on-tiger**. Tip remains frozen evidence. c_verify **PASS** @ **0.85×**. STEP 2 window wired; awaiting scout.
