#!/usr/bin/env bash
# T3 compile-then-capture — A5000 / 2080 full-map before/after.
#SBATCH --job-name=t3-compile
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err
# GRES set at submit time: --gres=gpu:a5000:1 or --gres=gpu:2080:1

set -euo pipefail

VARIANT=${VARIANT:?Set VARIANT=baseline|compile}
PRECISION=${PRECISION:-fp16}
EXPECTED_GPU_SUBSTR=${EXPECTED_GPU_SUBSTR:?Set EXPECTED_GPU_SUBSTR}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
ENV=${MAPPERATORINATOR_ENV:-/hpc/group/romerolab/imt11/envs/mapperatorinator}
PYTHON=${MAPPERATORINATOR_PYTHON:-$ENV/bin/python}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
REPO=${T3_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/t3-compile-then-capture}
SEED=${SEED:-12345}

JOB_ID=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/t3-compile-${VARIANT}-${PRECISION}-${JOB_ID}"
# Unique per-job caches (Inductor/Triton NFS flake lesson: prefer node-local when set).
JOB_TMP="${SLURM_TMPDIR:-$WORK/tmp}/t3-${JOB_ID}-${VARIANT}"
JOB_EXT="$WORK/torch_extensions/t3-${JOB_ID}-${VARIANT}-${PRECISION}"
INDUCTOR_CACHE="$WORK/cache/torchinductor/t3-${JOB_ID}-${VARIANT}"

mkdir -p "$WORK/logs" "$WORK/runs" "$JOB_TMP" "$JOB_EXT" "$INDUCTOR_CACHE" "$RUN_ROOT"
export PATH="$(dirname "$PYTHON"):$PATH"
export TMPDIR="$JOB_TMP"
export TORCH_EXTENSIONS_DIR="$JOB_EXT"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_CACHE"
export HF_HOME="${HF_HOME:-$WORK/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$WORK/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORK/cache/xdg}"
export CUDA_CACHE_PATH="$WORK/cache/cuda"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
unset MAPPERATORINATOR_ALLOW_CAPTURE_FALLBACK || true

for tool in ffmpeg ffprobe; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing $tool on PATH=$PATH" >&2
    exit 2
  fi
done

echo "host=$(hostname)"
echo "job=$JOB_ID"
echo "variant=$VARIANT"
echo "precision=$PRECISION"
echo "expected_gpu=$EXPECTED_GPU_SUBSTR"
echo "repo=$REPO"
echo "commit=$(git -C "$REPO" rev-parse HEAD)"
echo "TMPDIR=$TMPDIR"
echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
echo "TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR"
echo "ffmpeg=$(command -v ffmpeg)"
echo "ffprobe=$(command -v ffprobe)"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

cd "$REPO"
"$PYTHON" scripts/dcc/t3_compile_cell.py \
  --repo "$REPO" \
  --python "$PYTHON" \
  --audio "$AUDIO" \
  --run-root "$RUN_ROOT" \
  --precision "$PRECISION" \
  --seed "$SEED" \
  --variant "$VARIANT" \
  --expected-gpu-substr "$EXPECTED_GPU_SUBSTR"

echo "artifact=$RUN_ROOT"
