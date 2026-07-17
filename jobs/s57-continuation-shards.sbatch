#!/usr/bin/env bash
#SBATCH --job-name=s57-cont-shards
#SBATCH --partition=common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

# CPU-only shard prep for §57. No GPU.
set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-continuation-distill}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/s57-continuation-shards-$JOB"

mkdir -p "$RUN_ROOT" "$WORK/logs"
export PATH="$ENV/bin:$PATH"
export PROBE_REPO="$REPO"
export PYTHONUNBUFFERED=1

cd "$REPO"
echo "host=$(hostname)"
echo "job_id=$JOB"
echo "commit=$(git rev-parse HEAD)"
echo "branch=$(git branch --show-current)"

"$ENV/bin/python" "$REPO/utils/s57_build_continuation_shards.py" \
  --out-dir "$RUN_ROOT" \
  --include-five-song-timing

# Stable pointer for train job
ln -sfn "$RUN_ROOT" "$WORK/runs/s57-continuation-shards"
echo "STABLE $WORK/runs/s57-continuation-shards -> $RUN_ROOT"
echo DONE
