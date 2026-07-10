#!/usr/bin/env bash
#SBATCH --job-name=merged-batch-physics
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=01:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the dedicated DCC batch worktree}
WORK=/work/imt11/Mapperatorinator
EXPECTED_COMMIT=${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the pushed verifier commit}
MERGED_BATCH_SIZE=${MERGED_BATCH_SIZE:?Set MERGED_BATCH_SIZE to 1, 2, or 5}
PREVIOUS_GATE_REPORT=${PREVIOUS_GATE_REPORT:-}
COMMIT=$(git -C "$REPO" rev-parse HEAD)
SHORT_COMMIT=$(git -C "$REPO" rev-parse --short HEAD)
RUN_ROOT="$WORK/runs/merged-batch-physics-b${MERGED_BATCH_SIZE}-${SLURM_JOB_ID:-manual}-${SHORT_COMMIT}"
REPORT="$RUN_ROOT/merged-b${MERGED_BATCH_SIZE}.json"
TELEMETRY="$RUN_ROOT/nvidia-smi.csv"

case "$MERGED_BATCH_SIZE" in
  1)
    if [[ -n "$PREVIOUS_GATE_REPORT" ]]; then
      echo "B=1 must not receive PREVIOUS_GATE_REPORT" >&2
      exit 2
    fi
    ;;
  2|5|8)
    if [[ -z "$PREVIOUS_GATE_REPORT" || ! -f "$PREVIOUS_GATE_REPORT" ]]; then
      echo "B=$MERGED_BATCH_SIZE requires an existing PREVIOUS_GATE_REPORT" >&2
      exit 2
    fi
    ;;
  *)
    echo "MERGED_BATCH_SIZE must be 1, 2, 5, or 8" >&2
    exit 2
    ;;
esac

if [[ "$COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "DCC checkout is $COMMIT, expected $EXPECTED_COMMIT" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "DCC checkout is dirty; refusing to benchmark" >&2
  git -C "$REPO" status --short >&2
  exit 2
fi

mkdir -p \
  "$WORK/logs" \
  "$WORK/tmp" \
  "$WORK/cache" \
  "$WORK/cache/torch_inductor" \
  "$WORK/cache/triton" \
  "$WORK/cache/cuda" \
  "$WORK/torch_extensions" \
  "$RUN_ROOT"
export PATH="$ENV/bin:$PATH"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export TORCHINDUCTOR_CACHE_DIR="$WORK/cache/torch_inductor"
export TRITON_CACHE_DIR="$WORK/cache/triton"
export CUDA_CACHE_PATH="$WORK/cache/cuda"
export TMPDIR="$WORK/tmp"
export TOKENIZERS_PARALLELISM=false
export TORCH_EXTENSIONS_DIR="$WORK/torch_extensions"

cd "$REPO"

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "batch_size=$MERGED_BATCH_SIZE"
echo "commit=$COMMIT"
echo "branch=$(git branch --show-current)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "previous_gate_report=${PREVIOUS_GATE_REPORT:-none}"
nvidia-smi -L
nvidia-smi \
  --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv -l 1 > "$TELEMETRY" &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null || true' EXIT

python -m py_compile \
  osuT5/osuT5/inference/optimized/batch/merged_one_token.py \
  osuT5/osuT5/inference/optimized/benchmark/batch_physics.py \
  utils/verify_optimized_merged_one_token.py

PREVIOUS_ARGS=()
if [[ -n "$PREVIOUS_GATE_REPORT" ]]; then
  PREVIOUS_ARGS=(--previous-gate-report "$PREVIOUS_GATE_REPORT")
fi

python utils/verify_optimized_merged_one_token.py \
  --config-name profile_salvalai_smoke15 \
  --sequence-index 9 \
  --batch-size "$MERGED_BATCH_SIZE" \
  --report-path "$REPORT" \
  --warmup-repeats 5 \
  --timing-repeats 50 \
  --active-prefix-decode \
  --active-prefix-bucket-size 64 \
  --q1-bmm-cross-attention \
  "${PREVIOUS_ARGS[@]}" \
  audio_path="$WORK/data/salvalai.mp3" \
  output_path="$RUN_ROOT/output" \
  device=cuda \
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
  inference_active_prefix_decode_loop=true \
  inference_active_prefix_decode_bucket_size=64 \
  inference_stateful_monotonic_logits_processor=true

echo "report=$REPORT"
echo "telemetry=$TELEMETRY"
echo "run_root=$RUN_ROOT"
