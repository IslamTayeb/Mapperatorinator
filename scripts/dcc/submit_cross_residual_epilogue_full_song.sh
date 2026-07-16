#!/usr/bin/env bash
# Submit full-song reciprocal: shared-arena control vs layout-fused (heads_view).
# Does NOT compose with compiled-cross. Excludes h36-5 (compile-cross remeasure)
# and h36-9 (confirmation pin) by default.
set -euo pipefail

REPO=$(git rev-parse --show-toplevel)
BRANCH=$(git -C "$REPO" branch --show-current)
COMMIT=$(git -C "$REPO" rev-parse HEAD)
REMOTE=${MAPPERATORINATOR_REMOTE:-islamtayeb}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
EXCLUDE=${MAPPERATORINATOR_EXCLUDE_NODES:-dcc-core-gpu-ferc-s-h36-5,dcc-core-gpu-ferc-s-h36-9}
DCC_REPO=${MAPPERATORINATOR_DCC_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/500tps-cross-residual-epilogue}

if [[ -z "$BRANCH" ]]; then
  echo "submit requires a named experiment branch" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "refuse dirty worktree" >&2
  git -C "$REPO" status --short >&2
  exit 2
fi
REMOTE_REF="refs/remotes/$REMOTE/$BRANCH"
if ! git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF" \
  || [[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" != "$COMMIT" ]]; then
  echo "push $REMOTE/$BRANCH@$COMMIT before submit" >&2
  exit 2
fi

echo "Checking squeue..."
squeue -u "${USER:-imt11}" -o "%.18i %.9P %.40j %.2t %.10M %N" || true
if squeue -u "${USER:-imt11}" -h -o "%N %j" | grep -E "h36-5" >/dev/null 2>&1; then
  echo "WARNING: activity on h36-5; submit still excludes h36-5" >&2
fi

mkdir -p "$WORK/logs" "$WORK/runs" "$WORK/tmp"
JOB_ID=$(sbatch --parsable \
  --export=ALL,BASELINE_REPO="$DCC_REPO",CANDIDATE_REPO="$DCC_REPO",BASELINE_COMMIT="$COMMIT",CANDIDATE_COMMIT="$COMMIT",BASELINE_BRANCH="$BRANCH",CANDIDATE_BRANCH="$BRANCH",CANDIDATE_REMOTE=origin,CANDIDATE_REMOTE_BRANCH="$BRANCH",MAPPERATORINATOR_WORK="$WORK" \
  --exclude="$EXCLUDE" \
  "$DCC_REPO/scripts/dcc/verify_cross_residual_epilogue_full_song_reciprocal.sbatch")
echo "submitted job_id=$JOB_ID branch=$BRANCH commit=$COMMIT exclude=$EXCLUDE"
echo "logs: $WORK/logs/cross-resid-epilogue-full-${JOB_ID}.out"
echo "run:  $WORK/runs/cross-residual-epilogue-full-song-${JOB_ID}/"
