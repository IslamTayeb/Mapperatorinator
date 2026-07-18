# §58 Turbo-on-Tiger (PRIMARY TRACK) — handoff

**Status:** **SCAFFOLD + C_VERIFY SCOUT QUEUED/IN FLIGHT** (2026-07-18)  
**Strategy pivot:** turbo stacks on Tiger14n PR #120 `feat/compiled-decode`, **not** on optimized tip.  
**Branch / WT:** `codex/turbo-on-tiger-pr120` @ *(see job preflight)*  
**Base:** `w1/tiger14n-compiled-decode` / `d01cdd2799891f78c871ff5889cfbb6a661a9e46` (verified)  
**Optimized tip FROZEN:** `55949274` / FP16 **366.11** — **do not modify its internals**; **no merge**; **no 500 claim**

## Intent

Port strict rejection-sampling turbo (§34) + keep-accepted-KV (`cache_position` rewind on bucketed StaticCache) + graphed draft chain onto tiger's decode loop. Teacher verify = **tiger `CUDAGraphDecoder` at q_len=K** (uniform HF path). **Do not** port §54 Stage B fused-kernel grind.

Upstream posture: additive so this can later stack as a PR on #120.

## First gate — c_verify ratio

| Field | Value |
| --- | --- |
| Expect | **~1.1–1.3×** vs tiger q_len=1 |
| Scout | `utils/s58_tiger_c_verify_scout.py` |
| Job script | `jobs/s58-tiger-c-verify-scout.sbatch` (2080 Ti; unique TMPDIR/TORCH_EXTENSIONS) |
| Metric | CUDA-event avg replay ms |

**Measured:** *(fill from `runs/s58-tiger-cverify-<job>/summary.json`)*

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

## Later gates (after c_verify)

1. **In-loop E** (acceptance) on tiger teacher + draft  
2. Full-map `ms_per_map_token` + `cold_start_seconds` + `main_tps` (metric ruling)  
3. Wire graphed draft chain onto tiger (no optimized q1 kernel reuse)  
4. Stackable PR hygiene vs #120

## GPU coordination

§57b may hold A5000 dump slots — this scout prefers **2080**. Keep ≤2 concurrent GPU jobs with §57b.

## Do-not

- Modify optimized tip `55949274` internals  
- Port §54 fused mrow / Stage B verify grind onto tiger  
- Claim 500 / merge to main from this track yet  
- Fold relaxed acceptance into `turbo` (§34)

## Ruling

Primary track = **turbo-on-tiger**. Tip remains frozen evidence, not the integration base.
