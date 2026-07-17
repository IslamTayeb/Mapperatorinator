#!/usr/bin/env bash
#SBATCH --job-name=s40-tier1a-logit
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:45:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-canary-fix}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/s40-tier1a-logit-dump-$JOB"
UNIQUE="$WORK/tmp/s40-logit-$JOB"
EXT="$WORK/torch_extensions/s40-logit-$JOB"
DRAFT_CKPT=${DRAFT_CKPT:-$WORK/runs/s37-tiny-draft-train-50146289/draft_train.pt}

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
export MAPPERATORINATOR_TURBO_DRAFT_CKPT="$DRAFT_CKPT"

cd "$REPO"
echo "host=$(hostname)"
echo "job_id=$JOB"
echo "commit=$(git rev-parse HEAD)"
echo "branch=$(git branch --show-current)"
echo "draft_ckpt=$DRAFT_CKPT"
echo "tmpdir=$UNIQUE"
echo "torch_extensions=$EXT"
nvidia-smi -L

"$ENV/bin/python" "$REPO/utils/s40_tier1a_logit_dump.py" \
  --dump-run "$WORK/runs/exact-rope-device-state-fp16-auth-49964133" \
  --audio "$WORK/data/salvalai.mp3" \
  --draft-ckpt "$DRAFT_CKPT" \
  --out "$RUN_ROOT/logit_dump.json" \
  --precision fp16 \
  --gamma 5 \
  --mismatch-abs 110 \
  --known-opt-prefix "12,1648,2213,2749,3878"

echo DONE
cat "$RUN_ROOT/logit_dump.json"
