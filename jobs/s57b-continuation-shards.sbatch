#!/usr/bin/env bash
#SBATCH --job-name=s57b-cont-shards
#SBATCH --partition=common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

# §57b CPU shard rebuild: s55 salvalai + multi-song turbo short-rest dumps.
# No GPU. Tip 55949274/366.11. No wire.

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-continuation-distill}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/s57b-continuation-shards-$JOB"

mkdir -p "$RUN_ROOT" "$WORK/logs"
export PATH="$ENV/bin:$PATH"
export PROBE_REPO="$REPO"
export PYTHONUNBUFFERED=1

cd "$REPO"
echo "host=$(hostname)"
echo "job_id=$JOB"
echo "commit=$(git rev-parse HEAD)"
echo "branch=$(git branch --show-current)"

DUMP_ARGS=()
for song in pegasus lambada ela-ke-leitada nube-negra; do
  link="$WORK/runs/s57b-turbo-dump-${song}"
  if [[ -e "$link" ]]; then
    DUMP_ARGS+=(--turbo-dump "${song}=${link}")
    echo "include_turbo_dump ${song}=${link}"
  else
    echo "skip_missing_turbo_dump ${song}"
  fi
done

if [[ ${#DUMP_ARGS[@]} -lt 1 ]]; then
  echo "ERROR: no s57b turbo dumps found under $WORK/runs/s57b-turbo-dump-*" >&2
  exit 2
fi

"$ENV/bin/python" "$REPO/utils/s57_build_continuation_shards.py" \
  --out-dir "$RUN_ROOT" \
  --include-five-song-timing \
  "${DUMP_ARGS[@]}"

ln -sfn "$RUN_ROOT" "$WORK/runs/s57b-continuation-shards"
ln -sfn "$RUN_ROOT" "$WORK/runs/s57-continuation-shards"
echo "STABLE $WORK/runs/s57b-continuation-shards -> $RUN_ROOT"
echo "STABLE $WORK/runs/s57-continuation-shards -> $RUN_ROOT (train default)"
echo DONE
