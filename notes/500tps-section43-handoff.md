# §43 W4 handoff — draft quality → turbo perf build

**Date:** 2026-07-17  
**Owner workstream:** W4 / ledger §43  
**Branch:** `codex/turbo-draft-quality`  
**Do not:** wire `generate_window` / canary / §39 from this branch without a separate plan.

## Verdict

Held-out E clears the simple-K promote bar. **Ship K=1.**  

| Preset | Draft ckpt | K | γ | temp | Held-out E (γ=5 / or noted) | Role |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| **Perf** | `draft_1layer.pt` | 1 | **3** | 0.9 | E(γ=3)≈**1.97** (E/cost best) | default turbo draft |
| **Quality** | `draft_multsong.pt` | 1 | **5** | 0.9 | mean **2.44** | higher acceptance headroom |
| Baseline ref | tip `50146289` `draft_train.pt` | 1 | 5 | 0.9 | mean **2.03** | SALVALAI-only; already OK |

Tree K>1 and half-width architecture train: **not** recommended for v1 (no E/cost win; half-width was cost-sim only).

## Job / artifacts

- Job **`50147299`** COMPLETED 00:09:40 on z25-20  
- Root: `/work/imt11/Mapperatorinator/runs/s43-draft-quality-50147299/`  
- Authoritative table: `acceptance_table_v2.json` (nan-filtered ela)  
- Detail: `notes/500tps-section43-draft-quality.md`

## Caveats for next agent

1. ela five-song dumps × tip teacher → NaN teacher logits on ~42% of positions; filter finite rows (suite now does). Prefer fresh tip dumps for future held-out.
2. Cost model is analytic (`K·γ·layers/12 + 1`); confirm with timed draft+verify before 500 claim.
3. TIER1 evidence pack still required before `turbo` / 500 ship.
4. GPU concurrency: ≤2 jobs; this workstream is offline-complete — no further GPU needed unless regenerating tip held-out dumps.

## Next (not this workstream)

- Perf build: load recommended ckpt into turbo draft path; measure real TPS with K=1 γ∈{3,5}.  
- Optional: tip-FP16 dumps for ela/nube to drop nan filter caveat.  
- Still no merge without approval.
