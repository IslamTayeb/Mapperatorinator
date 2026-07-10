#!/usr/bin/env bash
#SBATCH --job-name=spec-mini-loaders
#SBATCH --partition=common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the isolated v32-mini DCC worktree}
WORK=/work/imt11/Mapperatorinator
EXPECTED_COMMIT=${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the pushed loader-preflight commit}
if [[ ! -d "$REPO" ]]; then
  echo "Explicit DCC worktree is missing: $REPO" >&2
  exit 2
fi
COMMIT=$(git -C "$REPO" rev-parse HEAD)
SHORT_COMMIT=$(git -C "$REPO" rev-parse --short HEAD)
RUN_ROOT="$WORK/runs/spec-mini-loaders-cpu-${SLURM_JOB_ID:-manual}-${SHORT_COMMIT}"
REPORT="$RUN_ROOT/v32-mini-loader-preflight.json"

if [[ "$COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "DCC checkout is $COMMIT, expected $EXPECTED_COMMIT" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "DCC checkout is dirty; refusing loader preflight" >&2
  git -C "$REPO" status --short >&2
  exit 2
fi

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" "$RUN_ROOT"
export PATH="$ENV/bin:$PATH"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export TMPDIR="$WORK/tmp"
export TOKENIZERS_PARALLELISM=false

cd "$REPO"

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "partition=${SLURM_JOB_PARTITION:-unknown}"
echo "commit=$COMMIT"
echo "branch=$(git branch --show-current)"
echo "run_root=$RUN_ROOT"
echo "report=$REPORT"
echo "contract=CPU-loader-preflight-only-pinned-v32-and-v32-mini-gamemode0"

python -m py_compile utils/verify_speculative_mini_draft.py

python utils/verify_speculative_mini_draft.py \
  --loaders-only \
  --config-name profile_salvalai_smoke15 \
  --sequence-index 9 \
  --speculation-k 4 \
  --max-new-tokens 256 \
  --report-path "$REPORT" \
  audio_path="$WORK/data/salvalai.mp3" \
  output_path="$RUN_ROOT/output" \
  device=cpu \
  precision=fp32 \
  attn_implementation=sdpa \
  inference_engine=v32 \
  optimized_inference_mode=single \
  use_server=false \
  parallel=false \
  cfg_scale=1.0 \
  num_beams=1 \
  seed=12345 \
  inference_generation_compile=false \
  inference_active_prefix_decode_loop=false \
  inference_stateful_monotonic_logits_processor=false

echo "report=$REPORT"
echo "run_root=$RUN_ROOT"
