# §53 EAGLE-style draft head — plan (Track C endgame)

**Status:** **OPEN** — renumbered from former §51 scaffold (ledger §51 reclaimed for verify kernels)  
**Branch / WT:** `codex/turbo-eagle-draft-head` @ `/work/projects/Mapperatorinator-worktrees/turbo-eagle-draft-head`  
**Base:** integrator `44ab1f3e` (campaign tip still `55949274` / **366.11**)  
**Do not:** merge to main; claim 500; open §44 until scout ≥384; grind §48/§52 verify path.

## Why now

§52 path-hit scout landed DG + graph-native + keep-KV but **E_runtime≈1.06 ≪ wire bars**. Authorizing ceilings that assumed E∈{2.0,2.4} are falsified. Endgame Track 3: EAGLE-style head with **c_d ≈ 0.05–0.1×** tip step.

## Gates

| Gate | Bar |
| --- | --- |
| Held-out E (wire) | offline/in-loop held-out **E ≥ 2.4** before runtime wire |
| Runtime E | scout / in-loop **E ≥ 2.2** |
| Draft cost | c_d ∈ **0.05–0.1×** tip step (~0.09–0.19 ms @ 1.85 ms/tok) |
| Ceiling | keep-KV + c_verify 3.075 + measured c_d → TPS ≥ **420** at E=2.4 |

## Rungs

| Rung | Work | Kill |
| --- | --- | --- |
| 0 Scaffold | module + probe + budget table | — |
| 1 Hidden dump | tip SALVALAI + held-out `output_hidden_states` packs | dump cost blow-up |
| 2 Cheap probe | `linear_fit` / `smoke_head` with **mandatory in-loop** | held-out E≪2.4 **and** no linear headroom |
| 3 Wire + scout | turbo preset opt-in; one FP16 SALVALAI scout | runtime E&lt;2.2 or ceiling&lt;420 |

## §52 binding (do not repeat)

- Offline §43 TF tip-dump E≈1.97 **≠** in-loop map weighted E≈1.25 / median≈1.08.
- **No §52b** re-scout. Do not re-grind §52 verify path.

## Artifacts

| Path | Role |
| --- | --- |
| `osuT5/.../turbo/eagle_draft.py` | head + budget helpers |
| `utils/s53_dump_teacher_hiddens.py` | teacher hidden dumps |
| `utils/s53_eagle_acceptance_probe.py` | cheap probe (in-loop mandatory) |
| `jobs/s53-*.sbatch` | DCC wrappers |
| `notes/500tps-section53-handoff.md` | status |

## Ledger note

Former §51 EAGLE OPEN/SCAFFOLD content lives here as **§53**. Sibling reclaim **§51** for verify kernels.
