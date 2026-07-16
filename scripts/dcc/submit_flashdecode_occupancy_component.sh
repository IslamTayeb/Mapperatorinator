#!/usr/bin/env bash
# Submit the FlashDecode occupancy component scout on a free 2080 Ti.
# Usage (from the clean experiment worktree on DCC):
#   MAPPERATORINATOR_REMOTE=islamtayeb ./scripts/dcc/submit_flashdecode_occupancy_component.sh
set -euo pipefail

REPO=$(git rev-parse --show-toplevel)
BRANCH=$(git -C "$REPO" branch --show-current)
COMMIT=$(git -C "$REPO" rev-parse HEAD)
REMOTE=${MAPPERATORINATOR_REMOTE:-islamtayeb}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
# Default excludes authoritative confirmation (h36-9). Override to also avoid
# other occupied GPUs (e.g. h36-6 during DROP/DP4A retries) at submit time.
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
if squeue -u "${USER:-imt11}" -h -o "%N %j" | grep -E "h36-9|confirm" >/dev/null 2>&1; then
  echo "WARNING: confirmation/h36-9 activity present; scout still excludes h36-9" >&2
fi

mkdir -p "$WORK/logs" "$WORK/runs" "$WORK/tmp"
JOB_ID=$(sbatch --parsable \
  --export=ALL,MAPPERATORINATOR_REPO="$REPO",MAPPERATORINATOR_COMMIT="$COMMIT",MAPPERATORINATOR_BRANCH="$BRANCH",MAPPERATORINATOR_REMOTE="$REMOTE" \
  --exclude="$EXCLUDE" \
  "$REPO/scripts/dcc/profile_flashdecode_occupancy_component.sbatch")
echo "submitted job_id=$JOB_ID branch=$BRANCH commit=$COMMIT exclude=$EXCLUDE"
echo "logs: $WORK/logs/flashdecode-occ-component-${JOB_ID}.out"
echo "run:  $WORK/runs/flashdecode-occupancy-component-${JOB_ID}/component.json"
