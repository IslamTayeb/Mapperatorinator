#!/usr/bin/env bash
#SBATCH --job-name=cutlass-w8a8-proj
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
# Prefer a distinct 2080 Ti from authoritative confirmation (h36-9).
#SBATCH --exclude=dcc-core-gpu-ferc-s-h36-9
#SBATCH --time=00:30:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

ENV=${MAPPERATORINATOR_ENV:-/hpc/group/romerolab/imt11/envs/mapperatorinator}
PYTHON=${MAPPERATORINATOR_PYTHON:-$ENV/bin/python}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean CUTLASS w8a8 scout worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT to the pushed CUTLASS w8a8 scout commit}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH to the CUTLASS w8a8 scout branch}
REMOTE=${MAPPERATORINATOR_REMOTE:-origin}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
RUN_ROOT="$WORK/runs/cutlass-w8a8-projection-component-${SLURM_JOB_ID:-manual}"
TORCH_EXTENSIONS_DIR="$WORK/torch_extensions/cutlass-w8a8-projection-$COMMIT-$SLURM_JOB_ID"

if [[ ! -d "$REPO/.git" && ! -f "$REPO/.git" ]]; then
  echo "missing CUTLASS w8a8 scout worktree at $REPO" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "CUTLASS w8a8 scout worktree is dirty" >&2
  git -C "$REPO" status --short >&2
  exit 2
fi
if [[ "$(git -C "$REPO" rev-parse HEAD)" != "$COMMIT" ]]; then
  echo "CUTLASS w8a8 scout worktree commit mismatch" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" branch --show-current)" != "$BRANCH" ]]; then
  echo "CUTLASS w8a8 scout worktree branch mismatch" >&2
  exit 2
fi
REMOTE_REF="refs/remotes/$REMOTE/$BRANCH"
if ! git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF"; then
  echo "missing fetched remote ref $REMOTE_REF" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" != "$COMMIT" ]]; then
  echo "remote CUTLASS w8a8 ref does not match COMMIT" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" || ! -f "$AUDIO" ]]; then
  echo "missing Python or SALVALAI audio" >&2
  exit 2
fi

export PATH="$(dirname "$PYTHON"):$PATH"
for tool in ffmpeg ffprobe; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "missing required audio tool: $tool" >&2
    exit 2
  }
done
mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" \
  "$WORK/cache/torch_inductor" "$WORK/cache/triton" "$WORK/cache/cuda" \
  "$TORCH_EXTENSIONS_DIR" "$RUN_ROOT"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$WORK/cache/torch_inductor"
export TRITON_CACHE_DIR="$WORK/cache/triton"
export CUDA_CACHE_PATH="$WORK/cache/cuda"
export TORCH_EXTENSIONS_DIR
export TORCH_CUDA_ARCH_LIST=7.5
export TMPDIR="$WORK/tmp/cutlass-w8a8-projection-${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
CUDA_LAUNCH_BLOCKING_VALUE=${MAPPERATORINATOR_CUDA_LAUNCH_BLOCKING:-0}
if [[ "$CUDA_LAUNCH_BLOCKING_VALUE" != "0" && "$CUDA_LAUNCH_BLOCKING_VALUE" != "1" ]]; then
  echo "MAPPERATORINATOR_CUDA_LAUNCH_BLOCKING must be 0 or 1" >&2
  exit 2
fi
export CUDA_LAUNCH_BLOCKING="$CUDA_LAUNCH_BLOCKING_VALUE"
export TORCH_SHOW_CPP_STACKTRACES=1

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ "$GPU_NAME" != "NVIDIA GeForce RTX 2080 Ti" ]]; then
  echo "CUTLASS w8a8 projection component requires RTX 2080 Ti, got $GPU_NAME" >&2
  exit 2
fi

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "repo=$REPO"
echo "commit=$COMMIT"
echo "branch=$BRANCH"
echo "remote_ref=$REMOTE_REF"
echo "python=$PYTHON"
echo "audio=$AUDIO"
echo "run_root=$RUN_ROOT"
echo "ffmpeg=$(command -v ffmpeg)"
echo "ffprobe=$(command -v ffprobe)"
echo "cuda_launch_blocking=$CUDA_LAUNCH_BLOCKING"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job -o "$SLURM_JOB_ID"
fi
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
nvidia-smi \
  --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv -l 1 > "$RUN_ROOT/nvidia-smi.csv" &
SMI_PID=$!
finalize() {
  kill "$SMI_PID" 2>/dev/null || true
  if [[ -d "$RUN_ROOT" ]]; then
    find "$RUN_ROOT" -type f ! -name sha256sums.txt -print0 \
      | sort -z | xargs -0 sha256sum > "$RUN_ROOT/sha256sums.txt"
  fi
}
trap finalize EXIT

cd "$REPO"
"$PYTHON" utils/profile_cutlass_w8a8_projection_component.py \
  --config-name profile_salvalai \
  --report-path "$RUN_ROOT/component.json" \
  --csv-path "$RUN_ROOT/component.csv" \
  --text-path "$RUN_ROOT/component.txt" \
  --diagnostic-path "$RUN_ROOT/diagnostics.jsonl" \
  --output-path "$RUN_ROOT/baseline-capture" \
  --bucket-mode sentinel \
  --warmup 100 \
  --iters 1000 \
  --cold-warmup 10 \
  --cold-iters 100 \
  --eviction-bytes 67108864 \
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
