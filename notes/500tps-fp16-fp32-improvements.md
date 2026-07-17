
## 45. Combined turbo perf integrator (§41 verify + §43 1-layer γ=3) — **OPEN**

| Field | Value |
| --- | --- |
| What | Integrate §41 canary-aligned teacher verify into turbo speculate path; wire §43 perf draft **1-layer K=1 γ=3 temp=0.9** (`draft_1layer.pt`); TIER1a ≥500×3 then one FP16 SALVALAI scout |
| Status | **OPEN** — branch wired; TIER1a canary pending; scout deferred until canary PASS |
| Branch / WT | `codex/turbo-integrator` / `turbo-integrator` (base **`7033d62f`**) |
| Preset | `turbo-integrator-s45-1layer-g3-v1` · `PRIMARY_GAMMA=3` · tree K=1 |
| Draft ckpt | `/work/imt11/Mapperatorinator/runs/s43-draft-quality-50147299/draft_1layer.pt` |
| §42 | `50148311` still running at integrate time — **not** merged; prefer c_draft later if available |
| Canary | `jobs/s45-turbo-tier1a-canary.sbatch` / `utils/s45_turbo_tier1a_canary.py` |
| Scout | `jobs/s45-turbo-fp16-perf-scout.sbatch` (unique TMPDIR/EXT; after canary PASS only) |
| Handoff | `notes/500tps-section45-handoff.md` |
| Campaign tip | still `55949274` / FP16 **366.11** — **no 500 claim**; full TIER1 (§44) before turbo ship |
| Next | canary → scout → if promising run §44 harness |
| Not this lever | §39 / turbo_mixed; tip graduation; merge to main |
| Ledger rule | Own section only |
