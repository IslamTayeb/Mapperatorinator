# §53 EAGLE draft head — handoff

**Status:** **OPEN** — renumbered from §51; dump+smoke+in-loop next  
**Branch / WT:** `codex/turbo-eagle-draft-head` @ `/work/projects/Mapperatorinator-worktrees/turbo-eagle-draft-head`  
**Tip:** campaign `55949274` / FP16 **366.11**; branch tip TBD after §53 commit  
**Plan:** `notes/500tps-section53-eagle-draft-head.md`

## Renumber

- Ledger **§51 VACATED** (do not reopen). Verify kernels = **§54**; bankable scout = **§55**.
- This lever is **§53 EAGLE head** (endgame Track 3).
- Old `s51_*` scaffold paths superseded by `s53_*`.

## Gates

- Held-out E ≥ **2.4** before runtime wire
- Runtime / in-loop E ≥ **2.2**
- c_d ∈ 0.05–0.1× tip step → ceiling ≥420 at E=2.4

## Done

- Scaffold `EagleDraftHead` + gate helpers (`eagle_draft.py`) with §53 bars
- `utils/s53_dump_teacher_hiddens.py` (SALVALAI + nube-negra)
- `utils/s53_eagle_acceptance_probe.py` (`dry_run`/`linear_fit`/`smoke_head`/`in_loop`; in-loop mandatory on smoke)
- Sbatch: `jobs/s53-dump-teacher-hiddens.sbatch`, `jobs/s53-eagle-acceptance-probe.sbatch`

## Next

1. Sync WT → DCC; submit dump job (unique TMPDIR/TORCH_EXTENSIONS).
2. `S53_MODE=smoke_head` on `hidden_pack_combined.pt` (mandatory in-loop).
3. Gate on numbers; STOP if held-out E&lt;2.4 or budget miss.
4. Only then runtime wire / scout. No §52 grind. No §44 until ≥384.

## Standing

No merge to main. No §44. No 500 claim. Tip stays `55949274` / **366.11**. Keep nohup loop.
