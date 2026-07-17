# §37 Tiny draft — plan (Track C)

**Status:** OPEN after §35 E[acc]=**1.014** → GO_SECTION_37 (§36 layer-skip self-spec skipped).
**Tip:** `55949274` / FP16 **366.11** (unchanged). **No merge.** TIER1 required before any 500 / `turbo` claim.
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
| 0 Smoke | init `[0,1]` + ≤few hundred CE steps + acceptance | baseline E; prove plumbing |
| 1 Train | longer CE/KL on tip dumps (± more songs) | E≥1.3 |
| 2 Turbo scaffold | `inference_engine=turbo` draft+verify | TIER1 + TPS rungs |
| 3 EAGLE revisit | only if 2-layer saturates &lt;1.3 | new hypothesis |

## Artifacts

- Branch / WT: `codex/turbo-tiny-draft` / `turbo-tiny-draft`
- Scripts: `utils/s37_build_distill_shards.py`, `utils/s37_tiny_draft_train_smoke.py`
- Sbatch: `jobs/s37-tiny-draft-smoke.sbatch` (also mirrored under DCC `/work/imt11/.../jobs/`)
