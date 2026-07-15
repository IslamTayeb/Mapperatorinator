#!/usr/bin/env bash
#SBATCH --job-name=fp32-cold-start
#SBATCH --partition=common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

die() {
  echo "fp32-cold-start: $*" >&2
  exit 2
}

SLURM_JOB_ID=${SLURM_JOB_ID:?Run this wrapper inside a Slurm allocation}
PYTHON=${MAPPERATORINATOR_PYTHON:-/hpc/group/romerolab/imt11/envs/mapperatorinator/bin/python}
BASELINE_REPO=${MAPPERATORINATOR_BASELINE_REPO:?Set the clean baseline worktree}
BASELINE_COMMIT=${MAPPERATORINATOR_BASELINE_COMMIT:?Set the baseline commit}
BASELINE_BRANCH=${MAPPERATORINATOR_BASELINE_BRANCH:?Set the baseline branch}
CANDIDATE_REPO=${MAPPERATORINATOR_CANDIDATE_REPO:?Set the clean candidate worktree}
CANDIDATE_COMMIT=${MAPPERATORINATOR_CANDIDATE_COMMIT:?Set the candidate commit}
CANDIDATE_BRANCH=${MAPPERATORINATOR_CANDIDATE_BRANCH:?Set the candidate branch}
CANDIDATE_REMOTE_REF=${MAPPERATORINATOR_CANDIDATE_REMOTE_REF:?Set the pushed candidate remote ref}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
RUN_LABEL=${RUN_LABEL:-fp32-cold-start}
RUN_ROOT="$WORK/runs/${RUN_LABEL}-${SLURM_JOB_ID}"

export NVIDIA_TF32_OVERRIDE=0

[[ -x "$PYTHON" ]] || die "missing executable Python at $PYTHON"
[[ ! -e "$RUN_ROOT" ]] || die "refusing to overwrite $RUN_ROOT"
for repo in "$BASELINE_REPO" "$CANDIDATE_REPO"; do
  [[ -d "$repo/.git" || -f "$repo/.git" ]] || die "missing worktree: $repo"
  [[ -z "$(git -C "$repo" status --porcelain)" ]] || {
    git -C "$repo" status --short >&2
    die "dirty worktree: $repo"
  }
done
[[ "$(git -C "$BASELINE_REPO" rev-parse HEAD)" == "$BASELINE_COMMIT" ]] \
  || die "baseline commit mismatch"
[[ "$(git -C "$BASELINE_REPO" branch --show-current)" == "$BASELINE_BRANCH" ]] \
  || die "baseline branch mismatch"
[[ "$(git -C "$CANDIDATE_REPO" rev-parse HEAD)" == "$CANDIDATE_COMMIT" ]] \
  || die "candidate commit mismatch"
[[ "$(git -C "$CANDIDATE_REPO" branch --show-current)" == "$CANDIDATE_BRANCH" ]] \
  || die "candidate branch mismatch"
git -C "$CANDIDATE_REPO" show-ref --verify --quiet "$CANDIDATE_REMOTE_REF" \
  || die "missing pushed candidate ref $CANDIDATE_REMOTE_REF"
[[ "$(git -C "$CANDIDATE_REPO" rev-parse "$CANDIDATE_REMOTE_REF")" == "$CANDIDATE_COMMIT" ]] \
  || die "pushed candidate ref does not match candidate commit"

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" "$RUN_ROOT"
export PATH="$(dirname "$PYTHON"):$PATH"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$WORK/tmp"
export TOKENIZERS_PARALLELISM=false

{
  echo "host=$(hostname)"
  echo "job_id=$SLURM_JOB_ID"
  echo "partition=${SLURM_JOB_PARTITION:-unknown}"
  echo "python=$PYTHON"
  echo "baseline_repo=$BASELINE_REPO"
  echo "baseline_commit=$BASELINE_COMMIT"
  echo "baseline_branch=$BASELINE_BRANCH"
  echo "candidate_repo=$CANDIDATE_REPO"
  echo "candidate_commit=$CANDIDATE_COMMIT"
  echo "candidate_branch=$CANDIDATE_BRANCH"
  echo "candidate_remote_ref=$CANDIDATE_REMOTE_REF"
  echo "NVIDIA_TF32_OVERRIDE=$NVIDIA_TF32_OVERRIDE"
  "$PYTHON" --version
  scontrol show job -o "$SLURM_JOB_ID"
  df -h "$WORK"
} | tee "$RUN_ROOT/preflight.txt"

cd "$CANDIDATE_REPO"
"$PYTHON" utils/profile_inference_cold_start.py \
  --python "$PYTHON" \
  --baseline-repo "$BASELINE_REPO" \
  --baseline-commit "$BASELINE_COMMIT" \
  --baseline-branch "$BASELINE_BRANCH" \
  --candidate-repo "$CANDIDATE_REPO" \
  --candidate-commit "$CANDIDATE_COMMIT" \
  --candidate-branch "$CANDIDATE_BRANCH" \
  --repetitions 5 \
  --minimum-improvement-pct 5 \
  --output "$RUN_ROOT/analysis.json" \
  --text-output "$RUN_ROOT/analysis.txt"

[[ -s "$RUN_ROOT/analysis.json" ]] || die "analysis JSON is missing"
[[ -s "$RUN_ROOT/analysis.txt" ]] || die "analysis text is missing"
find "$RUN_ROOT" -type f ! -name sha256sums.txt -print0 \
  | sort -z | xargs -0 sha256sum > "$RUN_ROOT/sha256sums.txt"
echo "fp32_cold_start=PASS"
