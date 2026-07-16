# Split-KV num_splits sweep (Track A)

Parent tip: `0dbab9e5` (`codex/500tps-arena-compiled-cross-last-mile`).
Branch: `codex/500tps-split-kv-num-splits-sweep`.
Not DP4A.

## Hypothesis

Fixed `_SPLIT_KV_Q1_SPLITS = 8` may under/over-split on SM75 (68 SMs).
Sweep `{4,6,8,12,16}` via env `MAPPERATORINATOR_SPLIT_KV_SPLITS`, pick the
fastest non-8 from an attention microbench, then one sealed reciprocal vs
selected shared-arena control (splits=8).

## Stop

Promote only if sealed wall ≤ baseline−0.05s with OK tokens; else **DROP**.
One sealed reciprocal — no endless sweeps.

## Job history

| Field | Value |
| --- | --- |
| First job | `49953714` |
| Node | `dcc-core-gpu-ferc-s-h36-5` (excludes h36-9) |
| Component | chose `num_splits=16` |
| Reciprocal | `/work/imt11/Mapperatorinator/runs/split-kv-num-splits-16-49953714/` |
| Status | **FAILED at analyzer only** — undeclared `optimized_effective_config.native_q1_rope_cache_split_kv_split_count`; unused default main_generation hit patterns |
| Fix | declare split_count delta; optional selected-stack capture-hit patterns (DP4A pattern) |
| Reseal job | `49955271` (`split-kv-num-splits-16`, tip `178a966c`) |
| Reseal script | `verify_split_kv_num_splits_reciprocal.sbatch` with `MAPPERATORINATOR_SPLIT_KV_SPLITS=16` |

## Paths

- Local worktree: `/work/projects/Mapperatorinator-worktrees/500tps-split-kv-num-splits-sweep`
- DCC worktree: `/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/500tps-split-kv-num-splits-sweep`
- Component run: `/work/imt11/Mapperatorinator/runs/split-kv-num-splits-sweep-49953714/`

## Job isolation

- Exclude `dcc-core-gpu-ferc-s-h36-9`
- Reciprocal: `TMPDIR=/work/imt11/Mapperatorinator/tmp/reciprocal-$JOBID`
- Reciprocal: `TORCH_EXTENSIONS_DIR=.../torch_extensions/reciprocal-$JOBID/...`
