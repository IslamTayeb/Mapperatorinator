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

## Job isolation

- Exclude `dcc-core-gpu-ferc-s-h36-9`
- `TMPDIR=/work/imt11/Mapperatorinator/tmp/split-kv-num-splits-$JOBID`
- `TORCH_EXTENSIONS_DIR=.../split-kv-num-splits-$COMMIT-$JOBID`
- Reciprocal uses job-local extension cache under `reciprocal-$JOBID`

## Submit

```bash
REPO=/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/500tps-split-kv-num-splits-sweep
COMMIT=$(git -C "$REPO" rev-parse HEAD)
sbatch --export=ALL,\
CANDIDATE_REPO="$REPO",\
CANDIDATE_COMMIT="$COMMIT",\
CANDIDATE_BRANCH=codex/500tps-split-kv-num-splits-sweep,\
CANDIDATE_REMOTE=islamtayeb,\
CANDIDATE_REMOTE_BRANCH=codex/500tps-split-kv-num-splits-sweep \
  "$REPO/scripts/dcc/verify_split_kv_num_splits_sweep.sbatch"
```
