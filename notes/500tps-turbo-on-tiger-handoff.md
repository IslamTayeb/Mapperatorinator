# §58 Turbo-on-Tiger (PRIMARY TRACK) — handoff

**Status:** **SCAFFOLD + C_VERIFY SCOUT QUEUED** (2026-07-18)  
**Strategy pivot:** turbo stacks on Tiger14n PR #120 `feat/compiled-decode`, **not** on optimized tip.  
**Branch / WT:** `codex/turbo-on-tiger-pr120` @ **`447df085`**  
**Local WT:** `/work/projects/Mapperatorinator-worktrees/turbo-on-tiger-pr120`  
**DCC WT:** `/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-on-tiger-pr120`  
**Base:** `w1/tiger14n-compiled-decode` / `d01cdd2799891f78c871ff5889cfbb6a661a9e46` (verified)  
**Optimized tip FROZEN:** `55949274` / FP16 **366.11** — **do not modify its internals**; **no merge**; **no 500 claim**  
**Remote:** `islamtayeb` only — **no push to Tiger14n / PR #120**

## Intent

Port strict rejection-sampling turbo (§34) + keep-accepted-KV (`cache_position` rewind on bucketed StaticCache) + graphed draft chain onto tiger's decode loop. Teacher verify = **tiger `CUDAGraphDecoder` at q_len=K** (uniform HF path). **Do not** port §54 Stage B fused-kernel grind.

Upstream posture: additive so this can later stack as a PR on #120.

## First gate — c_verify ratio

| Field | Value |
| --- | --- |
| Expect | **~1.1–1.3×** vs tiger q_len=1 |
| Scout | `utils/s58_tiger_c_verify_scout.py` |
| Job script | `jobs/s58-tiger-c-verify-scout.sbatch` (2080; unique TMPDIR/TORCH_EXTENSIONS) |
| Job | **`50192694`** PENDING (Priority) `gpu:2080:1` — coord with §57b A5000 |
| Metric | CUDA-event avg replay ms |

**Measured:** *(fill from `/work/imt11/Mapperatorinator/runs/s58-tiger-cverify-50192694/summary.json`)*

| K | c_verify ms | q1 ms | ratio | in 1.1–1.3? |
| --- | ---: | ---: | ---: | --- |
| 3 | — | — | — | — |
| 5 | — | — | — | — |

## Scaffold landed (this WT)

| Piece | Path | Notes |
| --- | --- | --- |
| q_len=K graph | `compiled_decode.CUDAGraphDecoder(q_len=…)` + `get_k_decoder` | q_len=1 production path unchanged |
| tiger verify helper | `turbo/tiger_verify.py` | persistent session; rewind after probe |
| rejection | `turbo/rejection.py` | §34 Leviathan |
| keep-KV | `turbo/kv_rollback.py` | O(γ) StaticCache zero |
| engine scaffold | `turbo/engine.py` | preset; full window loop **not wired yet** |
| draft chain scaffold | `turbo/draft_chain_graph.py` | tiger q_len=1 × γ + keep-KV rewind; in-graph sample next |

## Later gates (after c_verify)

1. Harvest **`50192694`** — seal ratio; STOP if ≫1.3× without clear capture bug  
2. Wire speculative window (rejection + keep-KV + draft chain) onto tiger decode  
3. **In-loop E** on tiger teacher + draft  
4. Full-map `ms_per_map_token` + `cold_start_seconds` + `main_tps` (metric ruling)

## GPU coordination

§57b may hold A5000 dump slots — this scout prefers **2080**. Keep ≤2 concurrent GPU jobs with §57b.

## Do-not

- Modify optimized tip `55949274` internals  
- Port §54 fused mrow / Stage B verify grind onto tiger  
- Claim 500 / merge to main from this track yet  
- Fold relaxed acceptance into `turbo` (§34)

## Ruling

Primary track = **turbo-on-tiger**. Tip remains frozen evidence, not the integration base.
