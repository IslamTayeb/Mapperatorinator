#!/usr/bin/env bash
# T2 certain full-map wins — A5000 tiger-base before/after.
#SBATCH --job-name=t2-fullmap
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=02:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

VARIANT=${VARIANT:?Set VARIANT=baseline|t2}
PRECISION=${PRECISION:-fp16}
EXPECTED_GPU_SUBSTR=${EXPECTED_GPU_SUBSTR:-A5000}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
ENV=${MAPPERATORINATOR_ENV:-/hpc/group/romerolab/imt11/envs/mapperatorinator}
PYTHON=${MAPPERATORINATOR_PYTHON:-$ENV/bin/python}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
REPO=${T2_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/t2-fullmap-wins}
SEED=${SEED:-12345}

JOB_ID=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/t2-fullmap-${VARIANT}-${PRECISION}-${JOB_ID}"
JOB_TMP="$WORK/tmp/t2-${JOB_ID}"
JOB_EXT="$WORK/torch_extensions/t2-${JOB_ID}-${VARIANT}-${PRECISION}"

mkdir -p "$WORK/logs" "$WORK/runs" "$JOB_TMP" "$JOB_EXT" "$RUN_ROOT"
export TMPDIR="$JOB_TMP"
export TORCH_EXTENSIONS_DIR="$JOB_EXT"
export HF_HOME="${HF_HOME:-$WORK/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$WORK/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORK/cache/xdg}"
# No silent capture fallback (T2 gate).
unset MAPPERATORINATOR_ALLOW_CAPTURE_FALLBACK || true

echo "host=$(hostname)"
echo "job=$JOB_ID"
echo "variant=$VARIANT"
echo "precision=$PRECISION"
echo "repo=$REPO"
echo "commit=$(git -C "$REPO" rev-parse HEAD)"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

cd "$REPO"
"$PYTHON" scripts/dcc/t2_fullmap_cell.py \
  --repo "$REPO" \
  --python "$PYTHON" \
  --audio "$AUDIO" \
  --run-root "$RUN_ROOT" \
  --precision "$PRECISION" \
  --seed "$SEED" \
  --variant "$VARIANT" \
  --expected-gpu-substr "$EXPECTED_GPU_SUBSTR"

echo "artifact=$RUN_ROOT"
