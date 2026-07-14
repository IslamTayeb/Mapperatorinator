#!/usr/bin/env bash
#SBATCH --job-name=batched-decoder-regions
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=01:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

PYTHON=${MAPPERATORINATOR_PYTHON:-/hpc/group/romerolab/imt11/envs/mapperatorinator/bin/python}
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean profiler worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT to the pushed profiler commit}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH to the profiler branch}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
RUN_LABEL=${RUN_LABEL:-batched-decoder-regions}
RUN_ROOT="$WORK/runs/${RUN_LABEL}-${SLURM_JOB_ID:-manual}"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing executable Python at $PYTHON" >&2
  exit 2
fi
if [[ ! -d "$REPO/.git" && ! -f "$REPO/.git" ]]; then
  echo "missing profiler worktree at $REPO" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "profiler worktree is dirty at $REPO" >&2
  git -C "$REPO" status --short >&2
  exit 2
fi
if [[ "$(git -C "$REPO" rev-parse HEAD)" != "$COMMIT" ]]; then
  echo "profiler worktree does not match MAPPERATORINATOR_COMMIT" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" branch --show-current)" != "$BRANCH" ]]; then
  echo "profiler worktree is not on MAPPERATORINATOR_BRANCH=$BRANCH" >&2
  exit 2
fi
if [[ ! -f "$AUDIO" ]]; then
  echo "missing SALVALAI audio at $AUDIO" >&2
  exit 2
fi

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" \
  "$WORK/cache/torch_inductor" "$WORK/cache/triton" "$WORK/cache/cuda" \
  "$WORK/torch_extensions" "$RUN_ROOT" "$RUN_ROOT/traces"
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
  echo "repo=$REPO"
  echo "python=$PYTHON"
  echo "audio=$AUDIO"
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
finalize() {
  kill "$SMI_PID" 2>/dev/null || true
  if [[ -d "$RUN_ROOT" ]]; then
    find "$RUN_ROOT" -type f ! -name sha256sums.txt -print0 \
      | sort -z \
      | xargs -0 sha256sum > "$RUN_ROOT/sha256sums.txt"
  fi
}
trap finalize EXIT

cd "$REPO"
set +e
"$PYTHON" utils/profile_batched_decoder_regions.py \
  --config-name profile_salvalai \
  --report-path "$RUN_ROOT/batched-decoder-regions.json" \
  --trace-dir "$RUN_ROOT/traces" \
  --warmup 5 \
  --iterations 10 \
  audio_path="$AUDIO" \
  output_path="$RUN_ROOT/unused-output" \
  device=cuda \
  precision=fp32 \
  attn_implementation=sdpa \
  inference_engine=v32 \
  use_server=false \
  parallel=false \
  super_timing=false \
  timer_num_beams=1 \
  timer_cfg_scale=1.0 \
  timer_iterations=20 \
  max_batch_size=16 \
  seed=12345 \
  2>&1 | tee "$RUN_ROOT/profile.log"
STATUS=${PIPESTATUS[0]}
set -e

if [[ ! -f "$RUN_ROOT/batched-decoder-regions.json" ]]; then
  echo "profiler produced no report (status=$STATUS)" >&2
  exit "$STATUS"
fi
if [[ "$STATUS" -ne 0 && "$STATUS" -ne 4 ]]; then
  echo "profiler failed before the declared component gate (status=$STATUS)" >&2
  exit "$STATUS"
fi
exit "$STATUS"
