# §45 Combined turbo perf integrator — handoff

**Status:** TIER1a PASS; FP16 scout PENDING (2026-07-17)  
**Branch / WT:** `codex/turbo-integrator` @ `/work/projects/Mapperatorinator-worktrees/turbo-integrator`  
**Integration tip:** `530cafe1` (CM fix `d7ed9f73`; base `7033d62f`)  
**Campaign tip unchanged:** `55949274` / FP16 **366.11** — **no 500 claim** without full TIER1.

## Absorbs

| Source | What |
| --- | --- |
| §41 | Eager-native aligned teacher verify (`speculate.py` / `verify_fastpath.py`); one outer teacher runtime CM |
| §43 | Perf draft **1-layer · K=1 · γ=3 · temp=0.9** · `draft_1layer.pt` |
| §42 | GATE MISS c_draft **1.08 ms/tok** (0.36× main) @ `e6bf3bb2`/`c11c62c7` job `50148590` — **merge path OK**, not blocking; **not** in this scout tip |

## Runtime pin

| Knob | Value |
| --- | --- |
| Preset | `turbo-integrator-s45-1layer-g3-v1` |
| `PRIMARY_GAMMA` | **3** |
| Tree K | **1** |
| Draft ckpt | `/work/imt11/Mapperatorinator/runs/s43-draft-quality-50147299/draft_1layer.pt` |
| Teacher verify | §41 StaticCache + eager-native Q=1 greedy |

## Gates

| Gate | Job | Result |
| --- | --- | --- |
| TIER1a ≥500×3 | `50148420` | FAIL (teacher_ctx one-shot CM reuse) |
| TIER1a ≥500×3 | `50148575` @ `d7ed9f73` | **PASS** — seeds 12345/23456/34567; 500 gen toks; match=true |
| FP16 SALVALAI scout | `50148770` | **PENDING** |
| Full TIER1 | §44 harness | next if scout promising — not claimed |

## Canary detail (`50148575`)

All three seeds: `seed_ok=true`, `turbo_generated_tokens=500`, `turbo_teacher_aligned_greedy=true`, γ=3, init_layers=`[0]`.  
Turbo wall ~20s / seed vs optimized ~1.4s (greedy aligned Q=1 rebuild — expected before draft fastpath compose).

## Next

1. Harvest scout `50148770` (directional TPS only).  
2. If promising → run §44 TIER1 harness.  
3. Optional: merge §42 draft-fastpath (`e6bf3bb2`) then re-canary + re-scout.  
4. No tip graduation; no 500 claim; no merge to main.

## Not this lever

§39 hybrid / turbo_mixed; INT8-as-FP16.
