#!/usr/bin/env bash
#SBATCH --job-name=s42-draft-diag
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:40:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-draft-fastpath}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/s42-draft-diag-$JOB"
UNIQUE="$WORK/tmp/s42-diag-$JOB"
EXT="$WORK/torch_extensions/s42-diag-$JOB"
DRAFT_CKPT=${DRAFT_CKPT:-$WORK/runs/s37-tiny-draft-train-50146289/draft_train.pt}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
OSU_HINT=${OSU_HINT:-$WORK/runs/exact-rope-device-state-fp16-auth-49964133}

mkdir -p "$RUN_ROOT" "$UNIQUE" "$EXT" "$WORK/logs"
export PATH="$ENV/bin:$PATH"
export PROBE_REPO="$REPO"
export HF_HOME="$WORK/hf_home"
export TRANSFORMERS_CACHE="$WORK/hf_home/transformers"
export HF_HUB_CACHE="$WORK/hf_home/hub"
export XDG_CACHE_HOME="$WORK/cache"
export TMPDIR="$UNIQUE"
export TORCH_EXTENSIONS_DIR="$EXT"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=1
export MAPPERATORINATOR_TURBO_DRAFT_CKPT="$DRAFT_CKPT"

cd "$REPO"
{
  echo "host=$(hostname)"; echo "job_id=$JOB"; echo "commit=$(git rev-parse HEAD)"
  nvidia-smi -L
} | tee "$RUN_ROOT/preflight.txt"

OSU_PATH=$(find "$OSU_HINT" -type f -name 'beatmap*.osu' 2>/dev/null | head -n 1 || true)
OSU_ARGS=()
[[ -n "${OSU_PATH}" ]] && OSU_ARGS+=(--osu-hint "$OSU_PATH")

"$ENV/bin/python" utils/s42_draft_fastpath_diagnose.py \
  --audio "$AUDIO" \
  --draft-ckpt "$DRAFT_CKPT" \
  --precision fp16 \
  --output "$RUN_ROOT/diagnose.json" \
  "${OSU_ARGS[@]}" \
  > "$RUN_ROOT/stdout.txt" 2> "$RUN_ROOT/stderr.txt"

echo DONE
cat "$RUN_ROOT/diagnose.json"
