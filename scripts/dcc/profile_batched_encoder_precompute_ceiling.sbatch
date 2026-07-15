#!/usr/bin/env bash
#SBATCH --job-name=encoder-batch-ceiling
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:45:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

PYTHON=${MAPPERATORINATOR_PYTHON:-/hpc/group/romerolab/imt11/envs/mapperatorinator/bin/python}
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean ceiling worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT to the pushed ceiling commit}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH to the ceiling branch}
PRECISION=${MAPPERATORINATOR_PRECISION:?Set MAPPERATORINATOR_PRECISION to fp32 or fp16}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
RUN_LABEL=${RUN_LABEL:-encoder-batch-ceiling}
RUN_ROOT="$WORK/runs/${RUN_LABEL}-${SLURM_JOB_ID:-manual}"

case "$PRECISION" in
  fp32|fp16) ;;
  *)
    echo "MAPPERATORINATOR_PRECISION must be fp32 or fp16" >&2
    exit 2
    ;;
esac

# Must precede every Python process and CUDA context.
export NVIDIA_TF32_OVERRIDE=0

if [[ ! -x "$PYTHON" ]]; then
  echo "missing executable Python at $PYTHON" >&2
  exit 2
fi
if [[ ! -d "$REPO/.git" && ! -f "$REPO/.git" ]]; then
  echo "missing ceiling worktree at $REPO" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "ceiling worktree is dirty at $REPO" >&2
  git -C "$REPO" status --short >&2
  exit 2
fi
if [[ "$(git -C "$REPO" rev-parse HEAD)" != "$COMMIT" ]]; then
  echo "ceiling worktree does not match MAPPERATORINATOR_COMMIT" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" branch --show-current)" != "$BRANCH" ]]; then
  echo "ceiling worktree is not on MAPPERATORINATOR_BRANCH=$BRANCH" >&2
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
export TORCH_EXTENSIONS_DIR="$WORK/torch_extensions"
export TMPDIR="$WORK/tmp"
export TOKENIZERS_PARALLELISM=false
export PATH="$(dirname "$PYTHON"):$PATH"

for tool in ffmpeg ffprobe; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing required audio tool: $tool" >&2
    exit 2
  fi
done

{
  echo "host=$(hostname)"
  echo "job_id=${SLURM_JOB_ID:-manual}"
  echo "commit=$COMMIT"
  echo "branch=$BRANCH"
  echo "precision=$PRECISION"
  echo "repo=$REPO"
  echo "python=$PYTHON"
  echo "audio=$AUDIO"
  echo "ffmpeg=$(command -v ffmpeg)"
  echo "ffprobe=$(command -v ffprobe)"
  echo "run_root=$RUN_ROOT"
  git -C "$REPO" status --short --branch
  "$PYTHON" -c 'import torch, transformers; print(f"torch={torch.__version__}"); print(f"torch_cuda={torch.version.cuda}"); print(f"transformers={transformers.__version__}")'
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
  command -v nsys || echo "nsys=missing"
  command -v ncu || echo "ncu=missing"
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol show job -o "$SLURM_JOB_ID"
  fi
} | tee "$RUN_ROOT/preflight.txt"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ "$GPU_NAME" != *"2080 Ti"* ]]; then
  echo "expected RTX 2080 Ti, got $GPU_NAME" >&2
  exit 2
fi

nvidia-smi \
  --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv -l 1 > "$RUN_ROOT/nvidia-smi.csv" &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null || true' EXIT

cd "$REPO"
"$PYTHON" -m utils.profile_batched_encoder_precompute_ceiling \
  --config-name profile_salvalai \
  --report-path "$RUN_ROOT/encoder-precompute-ceiling.json" \
  --batch-sizes 1,2,4,8,16 \
  --warmup 1 \
  audio_path="$AUDIO" \
  output_path="$RUN_ROOT/unused-output" \
  device=cuda \
  precision="$PRECISION" \
  attn_implementation=sdpa \
  inference_engine=optimized \
  use_server=false \
  parallel=false \
  cfg_scale=1.0 \
  num_beams=1 \
  max_batch_size=16 \
  seed=12345

find "$RUN_ROOT" -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/sha256sums.txt"
cat "$RUN_ROOT/encoder-precompute-ceiling.json"
