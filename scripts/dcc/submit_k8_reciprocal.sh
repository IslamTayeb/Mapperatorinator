#!/usr/bin/env bash
set -euo pipefail

REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean DCC worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT to the pushed commit}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH to the pushed branch}
REMOTE=${MAPPERATORINATOR_REMOTE:-origin}

fail() {
  echo "$*" >&2
  exit 2
}

command -v sbatch >/dev/null 2>&1 || fail "sbatch is unavailable; run this on a DCC login node"
command -v sinfo >/dev/null 2>&1 || fail "sinfo is unavailable; run this on a DCC login node"
command -v squeue >/dev/null 2>&1 || fail "squeue is unavailable; run this on a DCC login node"
[[ -z "$(squeue -h -u "$USER")" ]] || {
  squeue -u "$USER" >&2
  fail "refusing to submit while this user already has queued/running jobs"
}
sinfo -h -p gpu-common -o '%P %G' | grep -q 'gpu-common' \
  || fail "live gpu-common partition check failed"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || fail "DCC worktree is dirty"
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$COMMIT" ]] || fail "DCC worktree differs from COMMIT"
[[ "$(git -C "$REPO" branch --show-current)" == "$BRANCH" ]] || fail "DCC worktree differs from BRANCH"
REMOTE_REF="refs/remotes/$REMOTE/$BRANCH"
git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF" || fail "missing fetched remote ref $REMOTE_REF"
[[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" == "$COMMIT" ]] || fail "remote ref differs from COMMIT"

exec sbatch \
  --export=ALL,MAPPERATORINATOR_REPO="$REPO",MAPPERATORINATOR_COMMIT="$COMMIT",MAPPERATORINATOR_BRANCH="$BRANCH",MAPPERATORINATOR_REMOTE="$REMOTE" \
  "$REPO/scripts/dcc/verify_k8_reciprocal.sbatch"
