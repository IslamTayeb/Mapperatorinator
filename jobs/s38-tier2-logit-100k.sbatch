#!/usr/bin/env bash
#SBATCH --job-name=s38-tier2-logit-100k
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=04:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

# §38 expand teacher-forced TIER2 logit agreement to ≥100k positions.
# Uses multiple hybrid lambada seed dumps as forced token sequences (same audio).
# Tip optimized default untouched. Not a 500 claim.

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-tier2-fused-step}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
PREC=${PRECISION:-fp16}
RUN_ROOT="$WORK/runs/s38-tier2-logit-100k-${PREC}-$JOB"
UNIQUE="$WORK/tmp/s38-logit100k-${PREC}-$JOB"
EXT="$WORK/torch_extensions/s38-logit100k-${PREC}-$JOB"
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/five-song-profile/lambada.mp3}
DUMP_ROOT=${DUMP_ROOT:-$WORK/runs/s39-hybrid-tier3-maps/hybrid/lambada}
CONFIG_NAME=${CONFIG_NAME:-profile_salvalai}

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

# ~9×~12k ≈ 108k map tokens
mapfile -t DUMPS < <(ls -d "$DUMP_ROOT"/seed100{0..8} 2>/dev/null)
if [[ ${#DUMPS[@]} -lt 1 ]]; then
  echo "no dump seeds under $DUMP_ROOT" >&2
  exit 2
fi

cd "$REPO"
{
  echo "host=$(hostname)"
  echo "job_id=$JOB"
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "precision=$PREC"
  echo "audio=$AUDIO"
  echo "dumps=${#DUMPS[@]}"
  printf 'dump=%s\n' "${DUMPS[@]}"
  echo "campaign_tip=55949274"
  echo "rung=1b_expand_100k"
  nvidia-smi -L
} | tee "$RUN_ROOT/preflight.txt"

# Prefer a lambada-capable config if present; else profile_salvalai with audio override.
if [[ -f "$REPO/configs/inference/${CONFIG_NAME}.yaml" ]]; then
  CFG="$CONFIG_NAME"
else
  CFG=profile_salvalai
fi

"$ENV/bin/python" utils/s38_tier2_logit_agreement_probe.py \
  --repo "$REPO" \
  --dump-run "${DUMPS[@]}" \
  --audio "$AUDIO" \
  --out "$RUN_ROOT/logit_agreement.json" \
  --precision "$PREC" \
  --config-name "$CFG" \
  --max-positions 110000 \
  > "$RUN_ROOT/stdout.txt" 2> "$RUN_ROOT/stderr.txt"

cp -f "$RUN_ROOT/logit_agreement.json" "$RUN_ROOT/summary.json" || true
echo DONE | tee -a "$RUN_ROOT/preflight.txt"
