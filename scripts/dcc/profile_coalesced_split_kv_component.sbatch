#!/usr/bin/env bash
#SBATCH --job-name=coal-split-kv-component
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=01:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

ENV=${MAPPERATORINATOR_ENV:-/hpc/group/romerolab/imt11/envs/mapperatorinator}
PYTHON=${MAPPERATORINATOR_PYTHON:-$ENV/bin/python}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT to the pushed commit}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH to the branch}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
RUN_ROOT="$WORK/runs/coalesced-split-kv-component-${SLURM_JOB_ID:-manual}"

export PATH="$(dirname "$PYTHON"):$PATH"
for tool in "$PYTHON" ffmpeg ffprobe nvidia-smi; do
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
  echo "coalesced split-KV worktree is dirty" >&2
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
REMOTE_REF="refs/remotes/origin/$BRANCH"
if ! git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF" \
  || [[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" != "$COMMIT" ]]; then
  echo "pushed origin branch does not match MAPPERATORINATOR_COMMIT" >&2
  exit 2
fi
if [[ ! -f "$AUDIO" ]]; then
  echo "missing SALVALAI audio at $AUDIO" >&2
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
export TORCH_EXTENSIONS_DIR="$WORK/torch_extensions/coalesced-split-$COMMIT"
export TMPDIR="$WORK/tmp"
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
echo "repo=$REPO"
echo "python=$PYTHON"
echo "ffmpeg=$(command -v ffmpeg)"
echo "ffprobe=$(command -v ffprobe)"
echo "run_root=$RUN_ROOT"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job -o "$SLURM_JOB_ID"
fi
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader

cd "$REPO"
set +e
"$PYTHON" utils/profile_coalesced_split_kv_component.py \
  --config-name profile_salvalai \
  --report-path "$RUN_ROOT/component.json" \
  --output-path "$RUN_ROOT/baseline-capture" \
  --warmup 100 \
  --iters 1000 \
  audio_path="$AUDIO" \
  device=cuda \
  precision=fp32 \
  attn_implementation=sdpa \
  inference_engine=optimized \
  use_server=false \
  parallel=false \
  cfg_scale=1.0 \
  num_beams=1 \
  seed=12345
STATUS=$?
set -e
if [[ ! -s "$RUN_ROOT/component.json" ]]; then
  echo "profiler failed before producing component.json (status=$STATUS)" >&2
  exit "$STATUS"
fi
sha256sum "$RUN_ROOT/component.json"
exit "$STATUS"
