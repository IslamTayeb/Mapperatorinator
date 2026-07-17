# §51 EAGLE draft head — handoff

**Status:** **OPEN / SCAFFOLD** — branch live; dry_run probe ready; no heavy train; no GPU jobs yet  
**Branch / WT:** `codex/turbo-eagle-draft-head` @ `/work/projects/Mapperatorinator-worktrees/turbo-eagle-draft-head`  
**Tip:** base integrator `44ab1f3e`; campaign tip unchanged `55949274` / FP16 **366.11**  
**Plan:** `notes/500tps-section51-eagle-draft-head.md`

## Done this open

- Branch from integrator tip after §52 STOP.
- Scaffold `EagleDraftHead` + budget/ceiling helpers (`eagle_draft.py`).
- Cheap probe CLI: `dry_run` | `linear_fit` | `smoke_head` (`utils/s51_eagle_acceptance_probe.py`).
- Sbatch wrapper `jobs/s51-eagle-acceptance-probe.sbatch`.
- Documented §52 E-collapse root cause (ledger + §52 addendum); **no §52b**.

## Next agent

1. Sync WT to DCC; run `S51_MODE=dry_run` smoke (CPU/GPU ok).
2. Build tip SALVALAI hidden-state packs (`output_hidden_states`) — required for `linear_fit` / `smoke_head`.
3. Cheap probe: need TF E≥1.7 **and** budget; then **mandatory in-loop** E≥1.7 before wire.
4. Only then heavy train / integrator wire / one scout.
5. Gate: E≥1.7 with c_d that restores ceiling ≥420. Else STOP.

## Standing

No merge to main. No §44. No 500 claim. No §48 grind. Tip stays `55949274` / **366.11**.
