# §41 verify fastpath + graph-aligned teacher — handoff

**Status:** **PARTIAL** (2026-07-17) — canary@110 **PASS**; c_verify ≤1.2× **MISS**  
**Branch / WT:** `codex/turbo-verify-fastpath`  
**Tip:** **`b3a0b27e`** (base `3bfd7bdb`). Campaign tip unchanged: `55949274` / **366.11**.  
**Absorbs §40 STOP_ESCALATE** — do not bare-retry rejection-rule / KV-index patches.

## Measured

| Gate | Result | Evidence |
| --- | --- | --- |
| A: c_verify / Q1 ≤ 1.2 | **MISS** | `50147970`: Q1=**1.853 ms**; best K=5 cudagraph verify=**3.125 ms** → **1.686×**; eager K≈**7.07×** |
| B: canary@110 argmax match | **PASS** | `50148210`: aligned teacher **2213** == optimized **2213** (ref §40 turbo-eager was 2236) |

## Wiring

| Piece | Path |
| --- | --- |
| Fastpath + aligned Q=1 | `osuT5/osuT5/inference/turbo/verify_fastpath.py` |
| Speculate integration | `osuT5/osuT5/inference/turbo/speculate.py` |
| Microbench | `utils/s41_verify_fastpath_microbench.py` |
| Canary@110 probe | `utils/s41_graph_aligned_canary_probe.py` |

**Greedy teacher:** sequential Q=1 **eager + native hooks + StaticCache** (matches optimized argmax).  
**Q=1 CUDA graphs:** disabled for sequential (zero-logits on `50148138`).  
**K-token CUDA graphs:** OK for perf microbench (`--cuda-graph-verify`).

## Jobs

| Job | Role | Result |
| --- | --- | --- |
| `50147416` | microbench r1 | FAILED Preprocessor API |
| `50147547` | microbench+canary | FAILED capture/encoder |
| `50147812` | microbench FIX | FAILED encoder_outputs stash |
| `50147970` | microbench auth | **ratios** — ratio 1.686× / 7.07× |
| `50148055`/`50148138` | canary | FAIL (graph zeros / inputs key) |
| `50148210` | canary eager-native | **PASS** @2213 |

## Next

1. Drive c_verify ≤1.2× (shared arenas / tighter K-graphs) — budget-gate before full reciprocal.  
2. Keep greedy on eager-native Q=1 until sequential Q=1 CUDA graphs proven.  
3. Full ≥500×3 TIER1a after integration with W1/W3 — not claimed here.  
4. No 500 / tip graduation.
