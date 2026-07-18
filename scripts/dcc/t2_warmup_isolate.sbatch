#!/usr/bin/env bash
# T2 warmup-hoist isolate — A5000 measure + T5 greedy seal (1 GPU sequential).
# Timing-stride OFF. Tip 55949274 frozen. No PR #120 push.
#SBATCH --job-name=t2-warmup-iso
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=02:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

PRECISION=${PRECISION:-fp16}
EXPECTED_GPU_SUBSTR=${EXPECTED_GPU_SUBSTR:-A5000}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
ENV=${MAPPERATORINATOR_ENV:-/hpc/group/romerolab/imt11/envs/mapperatorinator}
PYTHON=${MAPPERATORINATOR_PYTHON:-$ENV/bin/python}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
REPO=${T2_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/t2-fullmap-wins}
T5_UTILS=${T5_UTILS:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/t5-quality-gates/utils}
SEED=${SEED:-12345}

JOB_ID=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/t2-warmup-isolate-${PRECISION}-${JOB_ID}"
JOB_TMP="$WORK/tmp/t2-warmup-iso-${JOB_ID}"
JOB_EXT="$WORK/torch_extensions/t2-warmup-iso-${JOB_ID}-${PRECISION}"

mkdir -p "$WORK/logs" "$WORK/runs" "$JOB_TMP" "$JOB_EXT" "$RUN_ROOT"
export PATH="$(dirname "$PYTHON"):$PATH"
export TMPDIR="$JOB_TMP"
export TORCH_EXTENSIONS_DIR="$JOB_EXT"
export HF_HOME="${HF_HOME:-$WORK/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$WORK/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORK/cache/xdg}"
export CUDA_CACHE_PATH="$WORK/cache/cuda"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
unset MAPPERATORINATOR_ALLOW_CAPTURE_FALLBACK || true

# Expose T5 harness to the isolate cell when present on the node.
if [[ -d "$T5_UTILS" ]]; then
  export PYTHONPATH="$T5_UTILS/..${PYTHONPATH:+:$PYTHONPATH}"
fi

for tool in ffmpeg ffprobe; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing $tool on PATH=$PATH" >&2
    exit 2
  fi
done

echo "host=$(hostname)"
echo "job=$JOB_ID"
echo "precision=$PRECISION"
echo "repo=$REPO"
echo "commit=$(git -C "$REPO" rev-parse HEAD)"
echo "t5_utils=$T5_UTILS"
echo "ffmpeg=$(command -v ffmpeg)"
echo "ffprobe=$(command -v ffprobe)"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

cd "$REPO"
"$PYTHON" scripts/dcc/t2_warmup_isolate_cell.py \
  --repo "$REPO" \
  --python "$PYTHON" \
  --audio "$AUDIO" \
  --run-root "$RUN_ROOT" \
  --precision "$PRECISION" \
  --seed "$SEED" \
  --expected-gpu-substr "$EXPECTED_GPU_SUBSTR"

echo "artifact=$RUN_ROOT"
