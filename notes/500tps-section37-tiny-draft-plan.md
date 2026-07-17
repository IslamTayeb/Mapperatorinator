# §37 Tiny draft — plan (Track C)

**Status:** OPEN — FIX smoke `50146230` E[acc]=**1.750** (1.3–1.8) → longer CE/KL train toward **≥1.8**.
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
| 0 Smoke | init `[0,1]` + ≤few hundred CE steps + acceptance | plumbing; FIX `50146230` E=**1.750** |
| 1 Train | longer CE/KL on all tip map dumps (± resume smoke ckpt) | E≥**1.8** preferred; ≥1.3 floor |
| 1b Tree-draft | if CE/KL saturates in 1.3–1.8: multi-candidate / tree verify (Medusa-style) | E≥1.8 effective |
| 2 Turbo scaffold | `inference_engine=turbo` draft+verify (rejection sampling) | TIER1 + TPS rungs |
| 3 EAGLE revisit | only if 2-layer saturates &lt;1.3 after serious train | new hypothesis |

### Tree-draft plan (rung 1b — only if train &lt;1.8)

If longer CE/KL lands **1.3≤E&lt;1.8**, do **not** open full turbo ship yet. Next:

1. Keep the 2-layer draft; at each step sample **K∈{2,4}** draft continuations (γ=5) under temp/top-p 0.9.
2. Verify with tip teacher via standard rejection sampling; take first fully accepted chain or longest prefix.
3. Report effective E[accepted/step] = mean accepted tokens / verify-step (same formula as α-probe, but measured).
4. If effective E≥1.8 → open §36-style turbo scaffold with tree verify; else EAGLE / more data.

## Artifacts

- Branch / WT: `codex/turbo-tiny-draft` / `turbo-tiny-draft`
- Scripts: `utils/s37_build_distill_shards.py`, `utils/s37_tiny_draft_train_smoke.py`
- Sbatch: `jobs/s37-tiny-draft-smoke.sbatch`, `jobs/s37-tiny-draft-train.sbatch`
- Smoke: `/work/imt11/Mapperatorinator/runs/s37-tiny-draft-smoke-50146230/`
