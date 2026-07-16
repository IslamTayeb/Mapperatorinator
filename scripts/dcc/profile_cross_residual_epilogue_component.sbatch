#!/usr/bin/env bash
#SBATCH --job-name=cross-residual-epilogue
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:20:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err
# Keep clear of authoritative confirmation pins.
#SBATCH --exclude=dcc-core-gpu-ferc-s-h36-9

set -euo pipefail

ENV=${MAPPERATORINATOR_ENV:-/hpc/group/romerolab/imt11/envs/mapperatorinator}
PYTHON=${MAPPERATORINATOR_PYTHON:-$ENV/bin/python}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT to the pushed commit}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH to the branch}
REMOTE=${MAPPERATORINATOR_REMOTE:-origin}
RUN_ROOT="$WORK/runs/cross-residual-epilogue-component-${SLURM_JOB_ID:-manual}"

export PATH="$(dirname "$PYTHON"):$PATH"
for tool in "$PYTHON" nvidia-smi; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing required tool: $tool" >&2
    exit 2
  fi
done
if [[ ! -d "$REPO/.git" && ! -f "$REPO/.git" ]]; then
  echo "missing worktree at $REPO" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "cross residual epilogue worktree is dirty" >&2
  git -C "$REPO" status --short >&2
  exit 2
fi
if [[ "$(git -C "$REPO" rev-parse HEAD)" != "$COMMIT" ]]; then
  echo "worktree differs from MAPPERATORINATOR_COMMIT" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" branch --show-current)" != "$BRANCH" ]]; then
  echo "worktree differs from MAPPERATORINATOR_BRANCH" >&2
  exit 2
fi
REMOTE_REF="refs/remotes/$REMOTE/$BRANCH"
if ! git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF" \
  || [[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" != "$COMMIT" ]]; then
  echo "pushed $REMOTE branch does not match MAPPERATORINATOR_COMMIT" >&2
  exit 2
fi

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" \
  "$WORK/cache/torch_inductor" "$WORK/cache/triton" "$WORK/cache/cuda" \
  "$WORK/torch_extensions" "$RUN_ROOT"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$WORK/cache/torch_inductor"
export TRITON_CACHE_DIR="$WORK/cache/triton"
export CUDA_CACHE_PATH="$WORK/cache/cuda"
export TORCH_EXTENSIONS_DIR="$WORK/torch_extensions/cross-residual-epilogue-$COMMIT-$SLURM_JOB_ID"
# Unique per-job TMPDIR (node-local preferred when available).
if [[ -d /tmp ]] && [[ -w /tmp ]]; then
  export TMPDIR="/tmp/mapperatorinator-cross-residual-${SLURM_JOB_ID:-manual}-$$"
else
  export TMPDIR="$WORK/tmp/cross-residual-epilogue-${SLURM_JOB_ID:-manual}"
fi
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR"
export TOKENIZERS_PARALLELISM=false

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
GPU_CAPABILITY=$($PYTHON -c 'import torch; print("".join(map(str, torch.cuda.get_device_capability())))')
if [[ "$GPU_NAME" != "NVIDIA GeForce RTX 2080 Ti" || "$GPU_CAPABILITY" != 75 ]]; then
  echo "component requires RTX 2080 Ti SM75, got $GPU_NAME sm_$GPU_CAPABILITY" >&2
  exit 2
fi

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "commit=$COMMIT"
echo "branch=$BRANCH"
echo "remote=$REMOTE"
echo "repo=$REPO"
echo "python=$PYTHON"
echo "run_root=$RUN_ROOT"
echo "torch_extensions_dir=$TORCH_EXTENSIONS_DIR"
echo "tmpdir=$TMPDIR"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job -o "$SLURM_JOB_ID"
fi
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader

cd "$REPO"
set +e
"$PYTHON" utils/profile_cross_residual_epilogue_component.py \
  --report-path "$RUN_ROOT/component.json" \
  --warmup 100 \
  --iters 1000 \
  --seed 12345
STATUS=$?
set -e
echo "component_exit=$STATUS"
if [[ -f "$RUN_ROOT/component.json" ]]; then
  echo "wrote $RUN_ROOT/component.json"
  echo "HARVEST: projected_main_saving_seconds / any_component_pass / stop_condition"
fi
exit "$STATUS"
