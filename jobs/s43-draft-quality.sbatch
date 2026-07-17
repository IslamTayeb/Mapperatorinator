#!/usr/bin/env bash
#SBATCH --job-name=s43-draft-quality
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=04:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-draft-quality}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/s43-draft-quality-$JOB"
UNIQUE="$WORK/tmp/s43-$JOB"
EXT="$WORK/torch_extensions/s43-$JOB"
TIP_CKPT="$WORK/runs/s37-tiny-draft-train-50146289/draft_train.pt"

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
echo "tip_ckpt=$TIP_CKPT"
nvidia-smi -L

"$ENV/bin/python" "$REPO/utils/s43_draft_quality_suite.py" \
  --out-dir "$RUN_ROOT" \
  --draft-ckpt "$TIP_CKPT" \
  --precision fp16 \
  --temperature 0.9 \
  --top-p 0.9 \
  --held-out-songs "ela-ke-leitada,nube-negra" \
  --multi-train-songs "salvalai,pegasus,lambada" \
  --multi-train-steps 4000 \
  --cheap-train-steps 1500 \
  --force-multi-train \
  --lr 3e-5 \
  --kl-weight 0.5 \
  --kl-temp 2.0

echo DONE
python3 -c "
import json
from pathlib import Path
p=Path('$RUN_ROOT/acceptance_table.json')
d=json.loads(p.read_text())
print('held_out', d.get('phase_a_held_out'))
print('held_mean', d.get('phase_a_held_out_mean_E'))
print('rec', d.get('recommendation'))
"
