#!/usr/bin/env bash
# Submit the cross residual epilogue component scout on a free 2080 Ti.
# Usage (from the clean experiment worktree on DCC):
#   MAPPERATORINATOR_REMOTE=islamtayeb ./scripts/dcc/submit_cross_residual_epilogue_component.sh
set -euo pipefail

REPO=$(git rev-parse --show-toplevel)
BRANCH=$(git -C "$REPO" branch --show-current)
COMMIT=$(git -C "$REPO" rev-parse HEAD)
REMOTE=${MAPPERATORINATOR_REMOTE:-islamtayeb}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
EXCLUDE=${MAPPERATORINATOR_EXCLUDE_NODES:-dcc-core-gpu-ferc-s-h36-9}

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

echo "Checking squeue for overlapping confirmation work..."
squeue -u "${USER:-imt11}" -o "%.18i %.9P %.40j %.2t %.10M %N" || true

mkdir -p "$WORK/logs" "$WORK/runs" "$WORK/tmp"
JOB_ID=$(sbatch --parsable \
  --export=ALL,MAPPERATORINATOR_REPO="$REPO",MAPPERATORINATOR_COMMIT="$COMMIT",MAPPERATORINATOR_BRANCH="$BRANCH",MAPPERATORINATOR_REMOTE="$REMOTE" \
  --exclude="$EXCLUDE" \
  "$REPO/scripts/dcc/profile_cross_residual_epilogue_component.sbatch")
echo "submitted job_id=$JOB_ID branch=$BRANCH commit=$COMMIT exclude=$EXCLUDE"
echo "logs: $WORK/logs/cross-resid-epilogue-${JOB_ID}.out"
echo "run:  $WORK/runs/cross-residual-epilogue-component-${JOB_ID}/component.json"
