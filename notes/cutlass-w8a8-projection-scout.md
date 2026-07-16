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
- This job is component/scout only (real-tensor region gate via
  `utils/profile_cutlass_w8a8_projection_component.py`; not full-song
  authoritative reciprocal).

## Harvest metric

`summary.conservative_fixed_main_saving_seconds` (hot∩cold) vs gate `0.3s`,
plus `metadata.cutlass_available` / `metadata.scout_backend`.

## DCC sync + jobs (2026-07-15)

- Worktree: `/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/500tps-cutlass-w8a8-projection-scout`
- Tip: `b4e5c774` (fixes `entry_snapshotter` TypeError on `dcfee9ac`)

| Job | Node | Tip | Slurm | Note |
| --- | --- | --- | --- | --- |
| `49952996` / `49953134` | z25-20 / h36-5 | `dcfee9ac` | FAILED | `install_k8_candidate(... entry_snapshotter=...)` unexpected kwarg |
| `49953314` | `dcc-core-ferc-s-z25-20` | `b4e5c774` | RUNNING (active) | excludes h36-9; component scout |
| `49953350` | h36-5 | `b4e5c774` | (check sacct) | parallel resubmit; avoid duplicating |
| `49953565` | h36-5 | `b4e5c774` | cancelled | duplicate after 49953314 already running |

Run path (active): `/work/imt11/Mapperatorinator/runs/cutlass-w8a8-projection-component-49953314`  
Logs: `/work/imt11/Mapperatorinator/logs/cutlass-w8a8-proj-49953314.{out,err}`

## Submit recipe (if needed again)

```bash
ssh dcc 'REPO=/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/500tps-cutlass-w8a8-projection-scout; \
cd "$REPO" && COMMIT=$(git rev-parse HEAD) && \
sbatch \
  --exclude=dcc-core-gpu-ferc-s-h36-9,dcc-core-gpu-ferc-s-h36-6 \
  --export=ALL,MAPPERATORINATOR_REPO="$REPO",MAPPERATORINATOR_COMMIT="$COMMIT",MAPPERATORINATOR_BRANCH=codex/500tps-cutlass-w8a8-projection-scout,MAPPERATORINATOR_REMOTE=origin \
  scripts/dcc/profile_cutlass_w8a8_projection_component.sbatch'
```

TMPDIR / TORCH_EXTENSIONS_DIR are job-id isolated in the sbatch.
