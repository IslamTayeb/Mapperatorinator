#!/usr/bin/env bash
#SBATCH --job-name=s37-tiny-draft-smoke
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=01:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-tiny-draft}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/s37-tiny-draft-smoke-$JOB"
UNIQUE="$WORK/tmp/s37-$JOB"
EXT="$WORK/torch_extensions/s37-$JOB"

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

cd "$REPO"
echo "host=$(hostname)"
echo "job_id=$JOB"
echo "commit=$(git rev-parse HEAD)"
echo "branch=$(git branch --show-current)"
nvidia-smi -L

"$ENV/bin/python" "$REPO/utils/s37_build_distill_shards.py" \
  --dump-run "$WORK/runs/exact-rope-device-state-fp16-auth-49964133" \
  --out "$RUN_ROOT/distill_shards_map.json" \
  --context map

"$ENV/bin/python" "$REPO/utils/s37_tiny_draft_train_smoke.py" \
  --dump-run "$WORK/runs/exact-rope-device-state-fp16-auth-49964133" \
  --audio "$WORK/data/salvalai.mp3" \
  --out "$RUN_ROOT/smoke.json" \
  --ckpt-out "$RUN_ROOT/draft_smoke.pt" \
  --precision fp16 \
  --temperature 0.9 \
  --top-p 0.9 \
  --max-map-windows 16 \
  --train-steps 100 \
  --lr 3e-5

echo DONE
