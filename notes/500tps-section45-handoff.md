# §45 Combined turbo perf integrator — handoff

**Status:** INTEGRATING (2026-07-17)  
**Branch / WT:** `codex/turbo-integrator` @ `/work/projects/Mapperatorinator-worktrees/turbo-integrator`  
**Base tip:** `7033d62f` (`codex/turbo-verify-fastpath` — §41 canary@110 PASS / c_verify 1.686× MISS)  
**Campaign tip unchanged:** `55949274` / FP16 **366.11** — **no 500 claim** without full TIER1.

## Absorbs

| Source | What |
| --- | --- |
| §41 | Eager-native aligned teacher verify in `speculate.py` / `verify_fastpath.py` |
| §43 | Perf draft **1-layer · K=1 · γ=3 · temp=0.9** · ckpt `draft_1layer.pt` |
| §42 | Still running (`50148311`) at submit time — **not** merged; prefer c_draft later if PASS |

## Runtime pin

| Knob | Value |
| --- | --- |
| Preset | `turbo-integrator-s45-1layer-g3-v1` |
| `PRIMARY_GAMMA` | **3** |
| Tree K | **1** (no tree) |
| Draft ckpt | `/work/imt11/Mapperatorinator/runs/s43-draft-quality-50147299/draft_1layer.pt` |
| Teacher verify | §41 StaticCache + eager-native Q=1 greedy (CUDA-graph Q=1 off) |
| Quality alt (not default) | multi-2layer K=1 γ=5 `draft_multsong.pt` |

## Gates

1. **TIER1a canary** — `jobs/s45-turbo-tier1a-canary.sbatch` · ≥500 tok × 3 seeds · hard-fail  
2. **If canary PASS** — one FP16 SALVALAI scout `jobs/s45-turbo-fp16-perf-scout.sbatch`  
3. **Next if scout promising** — §44 TIER1 harness (`codex/turbo-tier1-harness`)  
4. Do **not** claim 500 from scout alone.

## Jobs

| Job | Role | Status |
| --- | --- | --- |
| (pending) | TIER1a canary | submit after §42 or `afterany:50148311` |
| (pending) | FP16 perf scout | only after canary PASS |

## Not this lever

§39 hybrid / turbo_mixed; tip graduation; merge to main; INT8-as-FP16.
