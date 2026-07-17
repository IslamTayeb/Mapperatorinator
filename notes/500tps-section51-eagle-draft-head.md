# §51 EAGLE-style draft head — plan

**Status:** **OPEN / SCAFFOLD** (eligible after §52 A falsified)  
**Branch / WT:** `codex/turbo-eagle-draft-head` @ `/work/projects/Mapperatorinator-worktrees/turbo-eagle-draft-head`  
**Base:** integrator `44ab1f3e` (tip campaign still `55949274` / **366.11**)  
**Do not:** merge to main; claim 500; open §44 until scout ≥384; grind §48/§52.

## Why now

§52 path-hit scout landed DG + graph-native + keep-KV but **E_runtime≈1.06 ≪ 1.7**. Authorizing ceilings that assumed E∈{2.0,2.4} are falsified. Post-rung decision (B) is live: shift draft quality to an EAGLE-style head with **c_d ≈ 0.05–0.1×** tip step so E≥1.7 restores ceiling ≥420.

## Goal / gates

| Gate | Bar |
| --- | --- |
| Draft cost | c_d ∈ **0.05–0.1×** tip step (~0.09–0.19 ms @ 1.85 ms/tok) for γ-chain |
| Acceptance | offline **and** runtime E[acc] ≥ **1.7** (Leviathan; temp/top-p 0.9) |
| Ceiling | with keep-KV + c_verify 3.075 + measured c_d → TPS ≥ **420** at E=1.7 |
| Promote | component → short in-loop → SALVALAI scout; stop on first E/budget miss |

## Architecture (v0)

1. Teacher decode produces hidden `h_t` (last decoder state).
2. **EagleDraftHead** maps `h_{t-1} → logits_t` (MLP + lm_head; optional later 1-layer attn).
3. Speculative loop: draft γ tokens from head (features from teacher verify path / cached h), teacher verifies K=1 Leviathan.
4. Share teacher encoder + KV verify path from integrator; do **not** revive N-layer same-width as primary.

## Rungs

| Rung | Work | Kill |
| --- | --- | --- |
| 0 Scaffold | module + probe + budget table | — |
| 1 Cheap probe | `dry_run` → `linear_fit` / `smoke_head` on hidden dumps | TF E≪1.7 **and** no linear headroom |
| 2 Hidden dump | tip SALVALAI `output_hidden_states` packs | dump cost blow-up |
| 3 Smoke train | ≤few hundred CE on head | TF E&lt;1.7 |
| 4 In-loop probe | short speculative windows (mandatory; §52) | in-loop E&lt;1.7 |
| 5 Wire + scout | turbo preset opt-in; one FP16 SALVALAI scout | ceiling&lt;420 or E&lt;1.7 |

## §52 binding (do not repeat)

- Offline §43 TF tip-dump E≈1.97 **≠** in-loop map weighted E≈1.25 / median≈1.08.
- Eager (§47) and graphed-chain (§52) both collapse → not a 1-layer vs 2-layer ckpt bug; ckpt was `init_layers=[0]`.
- Chain missing top-p/processors is real but secondary (~1.06 vs ~1.09).
- **No §52b** re-scout unless a proven ≥1.7 wiring fix appears (none found).

## Artifacts

| Path | Role |
| --- | --- |
| `osuT5/.../turbo/eagle_draft.py` | head + budget helpers |
| `utils/s51_eagle_acceptance_probe.py` | cheap probe |
| `jobs/s51-eagle-acceptance-probe.sbatch` | DCC wrapper |
| `notes/500tps-section51-handoff.md` | status |

## Not this lever

N-layer distill grind; §48 VG grind; §44; main merge; 500 claim.
