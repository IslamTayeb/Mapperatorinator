#!/usr/bin/env bash
#SBATCH --job-name=combined-startup
#SBATCH --partition=common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

die() {
  echo "combined-startup: $*" >&2
  exit 2
}

SLURM_JOB_ID=${SLURM_JOB_ID:?Run this wrapper inside Slurm}
PYTHON=${MAPPERATORINATOR_PYTHON:-/hpc/group/romerolab/imt11/envs/mapperatorinator/bin/python}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
LAZY_REPO=${MAPPERATORINATOR_LAZY_REPO:?Set the clean lazy-parent worktree}
LAZY_COMMIT=${MAPPERATORINATOR_LAZY_COMMIT:?Set the lazy-parent commit}
LAZY_BRANCH=${MAPPERATORINATOR_LAZY_BRANCH:?Set the lazy-parent branch}
AOT_REPO=${MAPPERATORINATOR_AOT_REPO:?Set the clean AOT-parent worktree}
AOT_COMMIT=${MAPPERATORINATOR_AOT_COMMIT:?Set the AOT-parent commit}
AOT_BRANCH=${MAPPERATORINATOR_AOT_BRANCH:?Set the AOT-parent branch}
COMBINED_REPO=${MAPPERATORINATOR_COMBINED_REPO:?Set the clean combined worktree}
COMBINED_COMMIT=${MAPPERATORINATOR_COMBINED_COMMIT:?Set the combined commit}
COMBINED_BRANCH=${MAPPERATORINATOR_COMBINED_BRANCH:?Set the combined branch}
COMBINED_REMOTE_REF=${MAPPERATORINATOR_COMBINED_REMOTE_REF:?Set the pushed combined ref}
MANIFEST=${MAPPERATORINATOR_NATIVE_EXTENSION_MANIFEST:?Set the validated AOT manifest}
NATIVE_CACHE=${MAPPERATORINATOR_NATIVE_EXTENSION_CACHE:?Set the populated native-extension cache}
RUN_ROOT="$WORK/runs/combined-startup-${SLURM_JOB_ID}"

[[ -x "$PYTHON" ]] || die "missing Python: $PYTHON"
[[ -s "$MANIFEST" ]] || die "missing native-extension manifest: $MANIFEST"
[[ -d "$NATIVE_CACHE" ]] || die "missing native-extension cache: $NATIVE_CACHE"
[[ ! -e "$RUN_ROOT" ]] || die "refusing to overwrite $RUN_ROOT"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  die "CPU combined-startup job received CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

verify_repo() {
  local repo=$1
  local commit=$2
  local branch=$3
  [[ -d "$repo/.git" || -f "$repo/.git" ]] || die "missing worktree: $repo"
  [[ -z "$(git -C "$repo" status --porcelain)" ]] || die "dirty worktree: $repo"
  [[ "$(git -C "$repo" rev-parse HEAD)" == "$commit" ]] \
    || die "commit mismatch: $repo"
  [[ "$(git -C "$repo" branch --show-current)" == "$branch" ]] \
    || die "branch mismatch: $repo"
}

verify_repo "$LAZY_REPO" "$LAZY_COMMIT" "$LAZY_BRANCH"
verify_repo "$AOT_REPO" "$AOT_COMMIT" "$AOT_BRANCH"
verify_repo "$COMBINED_REPO" "$COMBINED_COMMIT" "$COMBINED_BRANCH"
git -C "$COMBINED_REPO" show-ref --verify --quiet "$COMBINED_REMOTE_REF" \
  || die "missing pushed combined ref"
[[ "$(git -C "$COMBINED_REPO" rev-parse "$COMBINED_REMOTE_REF")" == "$COMBINED_COMMIT" ]] \
  || die "pushed combined ref mismatch"

mkdir -p "$WORK/logs" "$WORK/tmp" "$RUN_ROOT"
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$COMBINED_REPO"
export TORCH_EXTENSIONS_DIR="$NATIVE_CACHE"
export TORCH_CUDA_ARCH_LIST=7.5
export MAX_JOBS=${MAX_JOBS:-4}
export TMPDIR="$WORK/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
unset MAPPERATORINATOR_NATIVE_EXTENSION_MANIFEST

for tool in git sha256sum scontrol date; do
  command -v "$tool" >/dev/null 2>&1 || die "missing executable: $tool"
done

{
  echo "host=$(hostname)"
  echo "job_id=$SLURM_JOB_ID"
  echo "python=$PYTHON"
  echo "lazy_repo=$LAZY_REPO"
  echo "lazy_commit=$LAZY_COMMIT"
  echo "lazy_branch=$LAZY_BRANCH"
  echo "aot_repo=$AOT_REPO"
  echo "aot_commit=$AOT_COMMIT"
  echo "aot_branch=$AOT_BRANCH"
  echo "combined_repo=$COMBINED_REPO"
  echo "combined_commit=$COMBINED_COMMIT"
  echo "combined_branch=$COMBINED_BRANCH"
  echo "combined_remote_ref=$COMBINED_REMOTE_REF"
  echo "manifest=$MANIFEST"
  echo "manifest_sha256=$(sha256sum "$MANIFEST" | awk '{print $1}')"
  echo "native_cache=$NATIVE_CACHE"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  "$PYTHON" -c 'import torch; print(f"torch={torch.__version__}"); print(f"torch_cuda={torch.version.cuda}")'
  scontrol show job -o "$SLURM_JOB_ID"
} | tee "$RUN_ROOT/preflight.txt"

cd "$COMBINED_REPO"
"$PYTHON" utils/profile_combined_startup.py \
  --python "$PYTHON" \
  --manifest "$MANIFEST" \
  --lazy-repo "$LAZY_REPO" \
  --lazy-commit "$LAZY_COMMIT" \
  --lazy-branch "$LAZY_BRANCH" \
  --aot-repo "$AOT_REPO" \
  --aot-commit "$AOT_COMMIT" \
  --aot-branch "$AOT_BRANCH" \
  --combined-repo "$COMBINED_REPO" \
  --combined-commit "$COMBINED_COMMIT" \
  --combined-branch "$COMBINED_BRANCH" \
  --rounds 5 \
  --output "$RUN_ROOT/analysis.json" \
  --text-output "$RUN_ROOT/analysis.txt"

test -s "$RUN_ROOT/preflight.txt"
test -s "$RUN_ROOT/analysis.json"
test -s "$RUN_ROOT/analysis.txt"
find "$RUN_ROOT" -type f ! -name sha256sums.txt -print0 \
  | sort -z | xargs -0 sha256sum > "$RUN_ROOT/sha256sums.txt"
echo "COMBINED_STARTUP_PASS"
