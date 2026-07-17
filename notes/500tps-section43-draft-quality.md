# §43 Draft quality (W4) — held-out E, multi-song distill, K/γ/temp sweeps

**Status:** RUNNING / OPEN  
**Branch / WT:** `codex/turbo-draft-quality` / `turbo-draft-quality`  
**Base tip draft:** train `50146289` E[acc]=**2.921** on SALVALAI (in-domain).  
**Constraint:** fully offline suite preferred — **no** `generate_window` / canary / §39 wiring.

## Goal

Acceptance table + recommended `(K, draft config)` for the turbo perf build.

## Protocol

| Phase | What |
| --- | --- |
| (a) | Tip SALVALAI-only draft → E on held-out **ela-ke-leitada**, **nube-negra** (+ SALVALAI ref) |
| (b) | Multi-song distill shards (salvalai+pegasus+lambada) + longer CE/KL (4000 steps) |
| (c) | Temp ∈ {0.7,0.9,1.1}, γ∈3…8, K∈{1,2,4} from logit dumps → acceptance-vs-cost |
| (d) | Tree (K>1 MC) + 1-layer train + half-width **cost sim** (α proxy = 1-layer) |

**Cost model:** `cost = K·γ·(n_draft_layers/12) + 1` verify unit; half-width multiplies draft term by 0.5.  
**Recommend:** among held-out aggregate, prefer E≥1.8 then max E/cost; else E≥1.3; else best E/cost.

## Scripts / jobs

- `utils/s43_draft_quality_suite.py`
- `utils/s43_offline_sweep_from_logits.py` (CPU re-sweep)
- `jobs/s43-draft-quality.sbatch`

## Artifacts (fill after job)

| Item | Path |
| --- | --- |
| Run root | `/work/imt11/Mapperatorinator/runs/s43-draft-quality-<JOB>/` |
| Acceptance table | `…/acceptance_table.json` |
| Sweeps | `…/phase_c_sweeps.json` |
| Logits | `…/logit_dumps/*.pt` |
| Multi-song ckpt | `…/draft_multsong.pt` |
| 1-layer ckpt | `…/draft_1layer.pt` |

## Results

_Pending job harvest._

## Decision (for perf build)

_Pending._
