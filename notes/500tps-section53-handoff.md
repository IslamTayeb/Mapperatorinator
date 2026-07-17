# §53 EAGLE draft head — handoff

**Status:** **STOP_KILL** — cheap smoke miss (budget + E)  
**Branch / WT:** `codex/turbo-eagle-draft-head` @ `0178e9e0`  
**Tip:** campaign `55949274` / FP16 **366.11** (unchanged)  
**Plan:** `notes/500tps-section53-eagle-draft-head.md`

## Renumber

- Ledger **§51 VACATED**. Verify kernels = **§54**; bankable scout = **§55**.
- This lever is **§53 EAGLE head** (endgame Track 3).
- `s51_*` → `s53_*`.

## Jobs

| Job | Role | State |
| --- | --- | --- |
| **50151156** | teacher hidden dump (SALVALAI 48w + nube 23w) | COMPLETED 0:0 **00:00:31** |
| **50151157** | smoke_head 400 CE + mandatory in-loop | COMPLETED 0:0 **00:00:32** |

Artifacts:

- Dump: `/work/imt11/Mapperatorinator/runs/s53-dump-teacher-hiddens-50151156/`
- Probe: `/work/imt11/Mapperatorinator/runs/s53-eagle-acceptance-probe-50151157/probe.json`
- Ckpt: `…/eagle_head_smoke.pt` (not promoted)

## Gate numbers (γ=3, temp/top-p 0.9)

| Metric | Value | Bar | Pass? |
| --- | ---: | ---: | --- |
| Held-out in-loop E (α→E) | **1.163** | ≥2.4 | **NO** |
| Runtime / empirical in-loop E | **0.353** | ≥2.2 | **NO** |
| Held-out TF E | 1.484 | (info) | — |
| Train TF E | 1.733 | (info) | — |
| c_d median (γ-chain) | **0.806 ms** | 0.093–0.185 ms (≤0.278 w/ 1.5× slack) | **NO** |
| Ceiling @ E=2.4 w/ measured c_d | 618.5* | ≥420 | budget path fails first |

\*Ceiling formula at E=2.4 with measured c_d still ≥420 arithmetically, but **c_d ≫ 0.1× tip step** → `STOP_BUDGET`. Feature-roll in-loop E collapses vs TF (same §52 lesson).

**Decision:** `STOP_BUDGET` (also would `STOP_HELD_OUT_E` / `STOP_IN_LOOP_E`).

## Kill

- Cheap 400-step MLP head does **not** reach held-out E≥2.4.
- Measured draft chain **~0.81 ms** ≈ **0.44×** tip step — outside 0.05–0.1× budget.
- Do **not** wire runtime. Do **not** heavy-train grind without a new architecture hypothesis (e.g. feature-regressed EAGLE-1 / shared emb) that can hit both bars.
- No §52 verify-path re-grind. No §44. No merge. No 500 claim.

## Revisit

Only if a draft head shows (offline) held-out in-loop E≥2.4 **and** measured c_d≤0.185 ms on 2080 Ti before any integrator wire.
