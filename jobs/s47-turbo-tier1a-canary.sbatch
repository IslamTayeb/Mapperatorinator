#!/usr/bin/env bash
#SBATCH --job-name=s47-turbo-tier1a-canary
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=02:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

# §47 TIER1a: greedy canary must stay on crop-rebuild/aligned-Q1 (PASS).
# Unique TMPDIR / TORCH_EXTENSIONS. No 500 claim.

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-keep-accepted-kv}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/s47-turbo-tier1a-canary-$JOB"
UNIQUE="$WORK/tmp/s47-canary-$JOB"
EXT="$WORK/torch_extensions/s47-canary-$JOB"
DRAFT_CKPT=${DRAFT_CKPT:-$WORK/runs/s43-draft-quality-50147299/draft_1layer.pt}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
DUMP=${MAPPERATORINATOR_DUMP_RUN:-$WORK/runs/exact-rope-device-state-fp16-auth-49964133}
SEEDS=${CANARY_SEEDS:-12345,23456,34567}
MAX_NEW=${CANARY_MAX_NEW_TOKENS:-500}

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
export MAPPERATORINATOR_TURBO_GAMMA=3
export MAPPERATORINATOR_TURBO_VERIFY_FASTPATH=1
export MAPPERATORINATOR_TURBO_TEACHER_ALIGNED=1
export MAPPERATORINATOR_TURBO_VERIFY_CUDA_GRAPH=0

cd "$REPO"
{
  echo "host=$(hostname)"
  echo "job_id=$JOB"
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "draft_ckpt=$DRAFT_CKPT"
  echo "seeds=$SEEDS"
  echo "max_new_tokens=$MAX_NEW"
  echo "tmpdir=$UNIQUE"
  echo "torch_extensions=$EXT"
  echo "section=47"
  nvidia-smi -L
} | tee "$RUN_ROOT/preflight.txt"

"$ENV/bin/python" "$REPO/utils/s45_turbo_tier1a_canary.py" \
  --audio "$AUDIO" \
  --dump-run "$DUMP" \
  --draft-ckpt "$DRAFT_CKPT" \
  --out "$RUN_ROOT/canary.json" \
  --precision fp16 \
  --max-new-tokens "$MAX_NEW" \
  --seeds "$SEEDS" \
  | tee "$RUN_ROOT/stdout.txt"

cp -f "$RUN_ROOT/canary.json" "$RUN_ROOT/summary.json"
echo DONE
cat "$RUN_ROOT/summary.json"
