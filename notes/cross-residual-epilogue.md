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
- Full song only if component `promotion_pass`; song &lt;0.05s or capture break → **DROP**.

## Isolation

- Unique `TMPDIR=/work/imt11/Mapperatorinator/tmp/cross-residual-epilogue-$JOBID`
- Unique `TORCH_EXTENSIONS_DIR=.../torch_extensions/cross-residual-epilogue-$COMMIT-$JOBID`
- Exclude `dcc-core-gpu-ferc-s-h36-9`
- Opt-in runtime flag (off by default): `MAPPERATORINATOR_CROSS_OUT_LAYOUT_FUSED=1`
