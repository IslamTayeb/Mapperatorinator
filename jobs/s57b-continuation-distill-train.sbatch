#!/usr/bin/env bash
#SBATCH --job-name=s57b-cont-distill
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=04:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

# §57b GPU retrain from 50182001 on multi-song short-rest shards.
# Unique TMPDIR / TORCH_EXTENSIONS. Tip 55949274/366.11. Wire only if offline clears.

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-continuation-distill}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/s57b-continuation-distill-$JOB"
UNIQUE="$WORK/tmp/s57b-train-$JOB"
EXT="$WORK/torch_extensions/s57b-train-$JOB"
SHARDS_DIR=${S57_SHARDS_DIR:-$WORK/runs/s57b-continuation-shards}
RESUME=${S57_RESUME_CKPT:-$WORK/runs/s57-continuation-distill-50182001/draft_continuation.pt}
TRAIN_STEPS=${S57_TRAIN_STEPS:-4000}
LR=${S57_LR:-3e-5}
# empty = all songs in shards (salvalai + turbo multi-song)
SONG_FILTER=${S57_SONG_FILTER:-}

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
echo "shards_dir=$SHARDS_DIR"
echo "resume=$RESUME"
echo "train_steps=$TRAIN_STEPS"
echo "lr=$LR"
echo "song_filter=${SONG_FILTER:-<all>}"
nvidia-smi -L

if [[ ! -f "$SHARDS_DIR/continuation_shards.json" ]]; then
  echo "ERROR: shards missing at $SHARDS_DIR/continuation_shards.json" >&2
  exit 2
fi
if [[ ! -f "$RESUME" ]]; then
  echo "ERROR: resume ckpt missing at $RESUME" >&2
  exit 2
fi

FILTER_ARGS=()
if [[ -n "$SONG_FILTER" ]]; then
  FILTER_ARGS+=(--song-filter "$SONG_FILTER")
else
  FILTER_ARGS+=(--song-filter "")
fi

"$ENV/bin/python" "$REPO/utils/s57_continuation_distill_train.py" \
  --shards "$SHARDS_DIR/continuation_shards.json" \
  --out "$RUN_ROOT/train.json" \
  --ckpt-out "$RUN_ROOT/draft_continuation.pt" \
  --resume-ckpt "$RESUME" \
  --precision fp16 \
  --temperature 0.9 \
  --top-p 0.9 \
  --train-steps "$TRAIN_STEPS" \
  --lr "$LR" \
  --kl-weight 0.5 \
  --kl-temp 2.0 \
  --w-map0 1.0 \
  --w-map-rest 4.0 \
  --w-timing 3.0 \
  "${FILTER_ARGS[@]}" \
  --baseline-rest-e 1.121

echo DONE
python3 -c "
import json
from pathlib import Path
d=json.loads(Path('$RUN_ROOT/train.json').read_text())
g=d.get('gate') or {}
print('GATE', g.get('decision'), 'map_rest_E', g.get('map_rest_E'), 'timing_E', g.get('timing_E'),
      'map_wE', g.get('map_wE'), 'full_wE', g.get('full_wE'), 'wire', g.get('wire_runtime'))
"
