#!/usr/bin/env bash
#SBATCH --job-name=split-kv-num-splits
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=01:30:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err
# Keep clear of confirmation pin even if that job finished.
#SBATCH --exclude=dcc-core-gpu-ferc-s-h36-9

set -euo pipefail

ENV=${MAPPERATORINATOR_ENV:-/hpc/group/romerolab/imt11/envs/mapperatorinator}
PYTHON=${MAPPERATORINATOR_PYTHON:-$ENV/bin/python}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
REPO=${CANDIDATE_REPO:?Set CANDIDATE_REPO to the clean pushed worktree}
COMMIT=${CANDIDATE_COMMIT:?Set CANDIDATE_COMMIT}
BRANCH=${CANDIDATE_BRANCH:?Set CANDIDATE_BRANCH}
REMOTE=${CANDIDATE_REMOTE:-origin}
REMOTE_BRANCH=${CANDIDATE_REMOTE_BRANCH:-$BRANCH}
RUN_ROOT="$WORK/runs/split-kv-num-splits-sweep-${SLURM_JOB_ID:-manual}"

export PATH="$(dirname "$PYTHON"):$PATH"
mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" \
  "$WORK/cache/torch_inductor" "$WORK/cache/triton" "$WORK/cache/cuda" \
  "$WORK/torch_extensions" "$RUN_ROOT"

export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$WORK/tmp/split-kv-num-splits-${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
export TORCH_EXTENSIONS_DIR="$WORK/torch_extensions/split-kv-num-splits-$COMMIT-$SLURM_JOB_ID"
mkdir -p "$TORCH_EXTENSIONS_DIR"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

if [[ ! -d "$REPO/.git" && ! -f "$REPO/.git" ]]; then
  echo "missing worktree at $REPO" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "num_splits sweep worktree is dirty" >&2
  git -C "$REPO" status --short >&2
  exit 2
fi
if [[ "$(git -C "$REPO" rev-parse HEAD)" != "$COMMIT" ]]; then
  echo "worktree differs from CANDIDATE_COMMIT" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" branch --show-current)" != "$BRANCH" ]]; then
  echo "worktree differs from CANDIDATE_BRANCH" >&2
  exit 2
fi
REMOTE_REF="refs/remotes/$REMOTE/$REMOTE_BRANCH"
if ! git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF" \
  || [[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" != "$COMMIT" ]]; then
  echo "pushed $REMOTE/$REMOTE_BRANCH does not match CANDIDATE_COMMIT" >&2
  exit 2
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
GPU_CAPABILITY=$($PYTHON -c 'import torch; print("".join(map(str, torch.cuda.get_device_capability())))')
if [[ "$GPU_NAME" != "NVIDIA GeForce RTX 2080 Ti" || "$GPU_CAPABILITY" != 75 ]]; then
  echo "split-KV num_splits sweep requires RTX 2080 Ti (SM75), got $GPU_NAME / $GPU_CAPABILITY" >&2
  exit 2
fi

echo "host=$(hostname)"
echo "job=$SLURM_JOB_ID"
echo "commit=$COMMIT"
echo "branch=$BRANCH"
echo "repo=$REPO"
echo "tmpdir=$TMPDIR"
echo "torch_extensions=$TORCH_EXTENSIONS_DIR"

cd "$REPO"
COMPONENT_JSON="$RUN_ROOT/component-num-splits.json"
set +e
"$PYTHON" utils/profile_split_kv_num_splits_component.py \
  --output-json "$COMPONENT_JSON" | tee "$RUN_ROOT/component-stdout.txt"
COMPONENT_STATUS=${PIPESTATUS[0]}
set -e
if [[ "$COMPONENT_STATUS" -ne 0 ]]; then
  echo "component sweep failed with status $COMPONENT_STATUS" >&2
  exit "$COMPONENT_STATUS"
fi
CHOSEN_SPLIT=$(
  "$PYTHON" - "$COMPONENT_JSON" <<'PY'
import json, sys
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(int(payload["chosen_split"]))
PY
)
echo "chosen_split=$CHOSEN_SPLIT" | tee "$RUN_ROOT/chosen-split.txt"
if [[ "$CHOSEN_SPLIT" == "8" ]]; then
  echo "component chose control split=8; DROP without reciprocal" >&2
  exit 3
fi

export BASELINE_REPO="$REPO"
export CANDIDATE_REPO="$REPO"
export BASELINE_COMMIT="$COMMIT"
export CANDIDATE_COMMIT="$COMMIT"
export BASELINE_BRANCH="$BRANCH"
export CANDIDATE_BRANCH="$BRANCH"
export CANDIDATE_REMOTE="$REMOTE"
export CANDIDATE_REMOTE_BRANCH="$REMOTE_BRANCH"
export BASELINE_RUNNER=utils/run_k4_shared_rope_fp16_cross_shared_arena.py
export CANDIDATE_RUNNER=utils/run_k4_shared_rope_fp16_cross_shared_arena_split_kv_num_splits.py
export REQUIRE_EXACT_TIMING=true
export REQUIRE_K4_CANDIDATE=true
export REQUIRE_SPLIT_KV_NUM_SPLITS_INCREMENTAL=true
export CROSS_CANDIDATE_MODE=fp16_packed_projections
export MAPPERATORINATOR_ALLOW_PARALLEL=1
export MAPPERATORINATOR_SPLIT_KV_SPLITS="$CHOSEN_SPLIT"
export RUN_LABEL="split-kv-num-splits-${CHOSEN_SPLIT}"

# Reciprocal owns its own RUN_ROOT via RUN_LABEL-JOBID.
source "$REPO/scripts/dcc/verify_weight_only_full_song_reciprocal.sbatch"
