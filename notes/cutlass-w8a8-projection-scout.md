# CUTLASS-seeking SM75 w8a8 projection GEMM scout

Track A / research ranking #3 after compiled-cross (~0.208s measured win).
Targets the largest Nsight hotspot (~2.44s projection GEMMs). Independent of
the compiled-cross confirmation job; parent tip `0dbab9e5`.

## Exactness / claims

- Class: `component_scout_documented_drift` (relaxed scout).
- Not production wiring. Not a 500 TPS claim.
- Do not present projections as production throughput.

## CUTLASS status (DCC check 2026-07-15)

- No CUTLASS headers at standard roots; no `cutlass` Python module.
- Scout records `cutlass_available=false` and measures the bounded SM75
  signed-DP4A w8a8 fallback (`self_norm_qkv` + `cross_norm_q` only).
- Revisit: vendor/install CUTLASS (`MAPPERATORINATOR_CUTLASS_HOME` with
  `include/cutlass/gemm/device/gemm.h`), wire SM75 w8a8 GEMM templates, then
  re-gate vs this DP4A fallback on a distinct 2080 Ti.

## DP4A full-stack note

- DP4A self-QKV full-stack (job 49908852) remains DROP.
- This job is component/scout only.

## Harvest metric

`summary.conservative_fixed_main_saving_seconds` (hot∩cold) vs gate `0.3s`,
plus `metadata.cutlass_available` / `metadata.scout_backend`.

## Submit (after commit+push+DCC sync)

```bash
ssh dcc 'squeue -u imt11; \
REPO=/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/500tps-cutlass-w8a8-projection-scout; \
cd "$REPO" && git fetch origin && git checkout codex/500tps-cutlass-w8a8-projection-scout && \
git reset --hard origin/codex/500tps-cutlass-w8a8-projection-scout && \
COMMIT=$(git rev-parse HEAD) && \
sbatch \
  --export=ALL,MAPPERATORINATOR_REPO="$REPO",MAPPERATORINATOR_COMMIT="$COMMIT",MAPPERATORINATOR_BRANCH=codex/500tps-cutlass-w8a8-projection-scout,MAPPERATORINATOR_REMOTE=origin \
  scripts/dcc/profile_cutlass_w8a8_projection_component.sbatch'
```

Exclude keeps confirmation node `dcc-core-gpu-ferc-s-h36-9` free.
TMPDIR / TORCH_EXTENSIONS_DIR are job-id isolated.
