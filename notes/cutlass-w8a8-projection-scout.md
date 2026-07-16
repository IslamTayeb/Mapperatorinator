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
- This scout is component/region only (not full-song reciprocal).
- Parked: do not touch DP4A self-QKV branch.

## Harvest metric

`summary.conservative_fixed_main_saving_seconds` (hot∩cold) vs gate `0.3s`,
plus `metadata.cutlass_available` / `metadata.scout_backend`.

## Decision: DROP (2026-07-15)

Do **not** submit full-song reciprocal. Two independent component gates on
`b4e5c774` both miss the 0.3s gate on the SM75 signed-DP4A fallback:

| Job | Node | Tip | conservative_s | component_pass |
| --- | --- | --- | --- | --- |
| `49953314` | z25-20 | `b4e5c774` | `0.1498` | false |
| `49954011` | z25-21 | `b4e5c774` | `0.1499` | false |

- `cutlass_available=false`, backend=`sm75_signed_dp4a_w8a8_fallback`.
- `invariants_pass=true`; hot L2 ~0.15s (self_qkv helps; cross_q ~flat).
- Excluded confirmation node `dcc-core-gpu-ferc-s-h36-9` (also confirmed
  `49952708` FAILED earlier; not reused).
- Prior tip `dcfee9ac` jobs failed on missing `entry_snapshotter` (fixed in
  `b4e5c774`). Duplicate/cancel noise: `49953350`, `49953565`.

Run artifacts: `/work/imt11/Mapperatorinator/runs/cutlass-w8a8-projection-component-{49953314,49954011}/`

## Submit recipe (only after true CUTLASS wire)

```bash
ssh dcc 'REPO=/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/500tps-cutlass-w8a8-projection-scout; \
cd "$REPO" && COMMIT=$(git rev-parse HEAD) && \
sbatch \
  --export=ALL,MAPPERATORINATOR_REPO="$REPO",MAPPERATORINATOR_COMMIT="$COMMIT",MAPPERATORINATOR_BRANCH=codex/500tps-cutlass-w8a8-projection-scout,MAPPERATORINATOR_REMOTE=origin \
  scripts/dcc/profile_cutlass_w8a8_projection_component.sbatch'
```

TMPDIR / TORCH_EXTENSIONS_DIR are job-id isolated; sbatch excludes h36-9.
