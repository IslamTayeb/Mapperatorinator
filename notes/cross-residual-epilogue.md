# Cross out_proj + residual epilogue fusion (Track A hedge)

Parent tip: `0dbab9e5` (`codex/500tps-arena-compiled-cross-last-mile`).
Branch: `codex/500tps-cross-residual-epilogue`.
Do **not** compose with compiled-cross. Do **not** merge to main.

## Hypothesis

Selected stack already fuses Wo + bias + residual (`weight_only_linear_residual`).
Remaining bridge tax is `transpose+contiguous` before Wo. Contiguous
`[1,H,1,D]` BMM output already matches `view(1,1,H*D)` memory order, so
layout fusion may eliminate a pure copy. Research band was ~0.05–0.15s;
component gate requires projected main save **≥0.08s** before any full song.

## Stop

- Component projected save &lt;0.08s → **DROP**, no full song.
- Correctness/drift fail → **DROP**.
- Full song only if component `any_component_pass`; song &lt;0.05s or capture break → **DROP**.

## Isolation

- Unique `TMPDIR=/tmp/imt11/cross-residual-epilogue-$JOBID` (node-local) or
  `/work/imt11/Mapperatorinator/tmp/cross-residual-epilogue-$JOBID`
- Unique `TORCH_EXTENSIONS_DIR=.../torch_extensions/reciprocal-$JOBID/$COMMIT`
- Exclude `dcc-core-gpu-ferc-s-h36-5` (compile-cross remeasure) and `h36-9`
- Opt-in runtime flag (off by default): `MAPPERATORINATOR_CROSS_OUT_LAYOUT_FUSED=1`

## Full-song reciprocal

- Control: `utils/run_k4_shared_rope_fp16_cross_shared_arena.py`
- Candidate: `utils/run_k4_shared_rope_fp16_cross_shared_arena_layout_fused.py`
  (`heads_view_fused`; no compiled-cross)
- Wrapper: `scripts/dcc/verify_cross_residual_epilogue_full_song_reciprocal.sbatch`
- Submit: `scripts/dcc/submit_cross_residual_epilogue_full_song.sh`

## Component evidence

- Job `49960894` @ `5e271814`: `heads_view_fused` PASS, projected ≈0.419s,
  `any_component_pass=true`, `promotion_pass=false` (needs full song).
