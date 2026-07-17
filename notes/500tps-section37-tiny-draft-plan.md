# §37 Tiny draft — plan (Track C)

**Status:** OPEN — train `50146289` E[acc]=**2.921**; §43 held-out **DONE** (`50147299`, tip mean E≈2.03 / multi≈2.44; prefer K=1 — `notes/500tps-section43-handoff.md`). Turbo speculative wired separately; TIER1 before ship.
**Tip (campaign auth):** `55949274` / FP16 **366.11** (unchanged). **No merge.** Do not claim tip graduate / 500 until TIER1+perf.
**Not this work:** §39 hybrid TIER3 (sibling); INT8-as-FP16.

## Choice this rung

| Option | Feasibility | Decision |
| --- | --- | --- |
| **2-layer same-width distilled decoder** | Reuses VarWhisper layers / tip dumps / §35 acceptance math | **PRIMARY** |
| EAGLE-style AR head | Needs new head + hidden dumps + train loop | Deferred until 2-layer fails E≥1.3 after serious train |

Draft: `decoder_layers=2`, `d_model=768`, vocab 4097; share / freeze teacher encoder; init from teacher layers `[0,1]` then CE (hard labels from tip dumps) ± optional KL later.

## Pipeline

1. **Shards** — tip SALVALAI map `generated_token_ids` from `49964133` (+ optional FP32 `49963835`).
2. **Smoke train** — short CE on teacher-forced windows (same prep as §35; tip `.osu` timing).
3. **Acceptance gate** — α=Σ min(p,q); E=(1−α^(γ+1))/(1−α); temp 0.9 / top-p 0.9; γ=5.
4. **Promote bar** — E[acc] ≥ **1.3** before turbo runtime; ≥ **1.8** preferred for simple K-draft. Then TIER1 pack (`docs/inference_evidence_packs.md`) before 500 claim.

## Rungs

| Rung | Job pattern | Gate |
| --- | --- | --- |
| 0 Smoke | init `[0,1]` + ≤few hundred CE steps + acceptance | plumbing; FIX `50146230` E=**1.750** |
| 1 Train | longer CE/KL on all tip map dumps (± resume smoke ckpt) | E≥**1.8** preferred; ≥1.3 floor |
| 1b Tree-draft | if CE/KL saturates in 1.3–1.8: multi-candidate / tree verify (Medusa-style) | E≥1.8 effective |
| 2 Turbo scaffold | `inference_engine=turbo` draft+verify (rejection sampling) | TIER1 + TPS rungs |
| 3 EAGLE revisit | only if 2-layer saturates &lt;1.3 after serious train | new hypothesis |

## Artifacts

- Branch / WT: `codex/turbo-tiny-draft` / `turbo-tiny-draft`
- Scripts: `utils/s37_build_distill_shards.py`, `utils/s37_tiny_draft_train_smoke.py`, `utils/s36_turbo_speculative_smoke.py`
- Sbatch: `jobs/s37-tiny-draft-smoke.sbatch`, `jobs/s37-tiny-draft-train.sbatch`, `jobs/s36-turbo-speculative-smoke.sbatch`, `jobs/s36-turbo-fp16-perf-scout.sbatch`
- Smoke: `/work/imt11/Mapperatorinator/runs/s37-tiny-draft-smoke-50146230/`
- Train: `/work/imt11/Mapperatorinator/runs/s37-tiny-draft-train-50146289/` (E=2.921)
- Draft ckpt: `/work/imt11/Mapperatorinator/runs/s37-tiny-draft-train-50146289/draft_train.pt`
- Turbo: `osuT5/osuT5/inference/turbo/` (`inference_engine=turbo`)
  - `speculate.py`: draft K + batched teacher verify + Leviathan reject + KV crop
  - Env: `MAPPERATORINATOR_TURBO_DRAFT_CKPT`
