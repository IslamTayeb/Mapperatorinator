#!/usr/bin/env bash
#SBATCH --job-name=s38-tier2-microbench
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=01:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

# §38 turbo vs tip decode microbench (≥0.15 ms/token budget before reciprocal).
# Not a 500 claim. Campaign tip 55949274 / 366.11.

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-tier2-fused-step}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
PREC=${PRECISION:-fp16}
RUN_ROOT="$WORK/runs/s38-tier2-microbench-${PREC}-$JOB"
UNIQUE="$WORK/tmp/s38-micro-${PREC}-$JOB"
EXT="$WORK/torch_extensions/s38-micro-${PREC}-$JOB"
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
DUMP_RUN=${DUMP_RUN:-$WORK/runs/exact-rope-device-state-fp16-auth-49964133}

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
export MAPPERATORINATOR_TURBO_TIER2_FUSED=1

cd "$REPO"
{
  echo "host=$(hostname)"
  echo "job_id=$JOB"
  echo "commit=$(git rev-parse HEAD)"
  echo "precision=$PREC"
  echo "budget_ms_per_token=0.15"
  nvidia-smi -L
} | tee "$RUN_ROOT/preflight.txt"

set +e
"$ENV/bin/python" utils/s38_tier2_decode_microbench.py \
  --repo "$REPO" \
  --dump-run "$DUMP_RUN" \
  --audio "$AUDIO" \
  --out "$RUN_ROOT/microbench.json" \
  --precision "$PREC" \
  > "$RUN_ROOT/stdout.txt" 2> "$RUN_ROOT/stderr.txt"
rc=$?
set -e
cp -f "$RUN_ROOT/microbench.json" "$RUN_ROOT/summary.json" || true
echo "exit=$rc" | tee -a "$RUN_ROOT/preflight.txt"
echo DONE | tee -a "$RUN_ROOT/preflight.txt"
exit 0
