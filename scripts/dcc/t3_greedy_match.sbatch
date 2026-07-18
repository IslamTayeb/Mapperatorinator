#!/usr/bin/env bash
# T3 greedy token-match (compile vs uncompiled fast path).
#SBATCH --job-name=t3-greedy
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=02:30:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
ENV=${MAPPERATORINATOR_ENV:-/hpc/group/romerolab/imt11/envs/mapperatorinator}
PYTHON=${MAPPERATORINATOR_PYTHON:-$ENV/bin/python}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
REPO=${T3_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/t3-compile-then-capture}
SEED=${SEED:-12345}
JOB_ID=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/t3-greedy-match-${JOB_ID}"
if [[ -n "${SLURM_TMPDIR:-}" && -d "${SLURM_TMPDIR}" ]]; then
  JOB_TMP="${SLURM_TMPDIR}/t3-greedy-${JOB_ID}"
else
  JOB_TMP="/tmp/${USER:-imt11}-t3-greedy-${JOB_ID}"
fi
JOB_EXT="$WORK/torch_extensions/t3-greedy-${JOB_ID}"
INDUCTOR_CACHE="$WORK/cache/torchinductor/t3-greedy-${JOB_ID}"

mkdir -p "$WORK/logs" "$WORK/runs" "$JOB_TMP" "$JOB_EXT" "$INDUCTOR_CACHE" "$RUN_ROOT"
export PATH="$(dirname "$PYTHON"):$PATH"
export TMPDIR="$JOB_TMP"
export TEMP="$JOB_TMP"
export TMP="$JOB_TMP"
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

echo "host=$(hostname) job=$JOB_ID commit=$(git -C "$REPO" rev-parse HEAD)"
cd "$REPO"
"$PYTHON" scripts/dcc/t3_greedy_match_cell.py \
  --repo "$REPO" --python "$PYTHON" --audio "$AUDIO" --run-root "$RUN_ROOT" --seed "$SEED"
echo "artifact=$RUN_ROOT"
