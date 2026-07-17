#!/usr/bin/env bash
#SBATCH --job-name=s37-tiny-draft-train
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=02:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-tiny-draft}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/s37-tiny-draft-train-$JOB"
UNIQUE="$WORK/tmp/s37-train-$JOB"
EXT="$WORK/torch_extensions/s37-train-$JOB"
SMOKE_CKPT="$WORK/runs/s37-tiny-draft-smoke-50146230/draft_smoke.pt"

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
echo "tmpdir=$TMPDIR"
echo "torch_extensions=$TORCH_EXTENSIONS_DIR"
nvidia-smi -L

"$ENV/bin/python" "$REPO/utils/s37_build_distill_shards.py" \
  --dump-run "$WORK/runs/exact-rope-device-state-fp16-auth-49964133" \
  --out "$RUN_ROOT/distill_shards_map.json" \
  --context map

RESUME_ARGS=()
if [[ -f "$SMOKE_CKPT" ]]; then
  RESUME_ARGS=(--resume-ckpt "$SMOKE_CKPT")
  echo "resume_ckpt=$SMOKE_CKPT"
else
  echo "WARN: smoke ckpt missing; training from layer init only"
fi

"$ENV/bin/python" "$REPO/utils/s37_tiny_draft_train_smoke.py" \
  --dump-run "$WORK/runs/exact-rope-device-state-fp16-auth-49964133" \
  --audio "$WORK/data/salvalai.mp3" \
  --out "$RUN_ROOT/train.json" \
  --ckpt-out "$RUN_ROOT/draft_train.pt" \
  --precision fp16 \
  --temperature 0.9 \
  --top-p 0.9 \
  --max-map-windows 0 \
  --train-steps 2000 \
  --lr 3e-5 \
  --kl-weight 0.5 \
  --kl-temp 2.0 \
  "${RESUME_ARGS[@]}"

echo DONE
python3 -c "
import json
from pathlib import Path
p=Path('$RUN_ROOT/train.json')
d=json.loads(p.read_text())
print('GATE', d.get('gate_decision'), 'E', d.get('E_accepted_per_step_primary'), 'alpha', d.get('mean_alpha'))
"
