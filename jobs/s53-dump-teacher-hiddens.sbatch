#!/usr/bin/env bash
#SBATCH --job-name=s53-dump-teacher-hiddens
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=01:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-eagle-draft-head}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/s53-dump-teacher-hiddens-$JOB"
UNIQUE="$WORK/tmp/s53-dump-$JOB"
EXT="$WORK/torch_extensions/s53-dump-$JOB"

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
nvidia-smi -L || true

"$ENV/bin/python" "$REPO/utils/s53_dump_teacher_hiddens.py" \
  --out-dir "$RUN_ROOT" \
  --songs "${S53_SONGS:-salvalai,nube-negra}" \
  --precision fp16 \
  --max-map-windows "${S53_MAX_WINDOWS:-48}" \
  --heldout-max-windows "${S53_HELDOUT_WINDOWS:-24}"

echo DONE
echo "artifact=$RUN_ROOT/index.json"
echo "combined=$RUN_ROOT/hidden_pack_combined.pt"
