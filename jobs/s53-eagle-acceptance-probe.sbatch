#!/usr/bin/env bash
#SBATCH --job-name=s53-eagle-acceptance-probe
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:45:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-eagle-draft-head}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/s53-eagle-acceptance-probe-$JOB"
UNIQUE="$WORK/tmp/s53-probe-$JOB"
EXT="$WORK/torch_extensions/s53-probe-$JOB"
MODE=${S53_MODE:-smoke_head}
HIDDEN_PACK=${S53_HIDDEN_PACK:-}

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
echo "mode=$MODE"
nvidia-smi -L || true

EXTRA=()
if [[ "$MODE" != "dry_run" ]]; then
  if [[ -z "$HIDDEN_PACK" ]]; then
    echo "S53_HIDDEN_PACK required for mode=$MODE" >&2
    exit 2
  fi
  EXTRA+=(--hidden-pack "$HIDDEN_PACK")
fi
if [[ "$MODE" == "smoke_head" ]]; then
  EXTRA+=(--ckpt-out "$RUN_ROOT/eagle_head_smoke.pt" --train-steps "${S53_TRAIN_STEPS:-400}")
fi
if [[ "$MODE" == "in_loop" ]]; then
  if [[ -z "${S53_CKPT_IN:-}" ]]; then
    echo "S53_CKPT_IN required for mode=in_loop" >&2
    exit 2
  fi
  EXTRA+=(--ckpt-in "$S53_CKPT_IN")
fi

"$ENV/bin/python" "$REPO/utils/s53_eagle_acceptance_probe.py" \
  --mode "$MODE" \
  --out "$RUN_ROOT/probe.json" \
  --temperature 0.9 \
  --top-p 0.9 \
  --gamma 3 \
  "${EXTRA[@]}"

echo DONE
echo "artifact=$RUN_ROOT/probe.json"
