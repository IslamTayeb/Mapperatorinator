#!/usr/bin/env bash
#SBATCH --job-name=spec-mini-onewindow
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
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the isolated v32-mini DCC worktree}
WORK=/work/imt11/Mapperatorinator
EXPECTED_COMMIT=${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the pushed v32-mini verifier commit}
if [[ ! -d "$REPO" ]]; then
  echo "Explicit DCC worktree is missing: $REPO" >&2
  exit 2
fi
COMMIT=$(git -C "$REPO" rev-parse HEAD)
SHORT_COMMIT=$(git -C "$REPO" rev-parse --short HEAD)
RUN_ROOT="$WORK/runs/spec-mini-onewindow-${SLURM_JOB_ID:-manual}-${SHORT_COMMIT}"
REPORT="$RUN_ROOT/v32-mini-feasibility.json"
TELEMETRY="$RUN_ROOT/nvidia-smi.csv"

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
echo "partition=${SLURM_JOB_PARTITION:-unknown}"
echo "commit=$COMMIT"
echo "branch=$(git branch --show-current)"
echo "run_root=$RUN_ROOT"
echo "report=$REPORT"
echo "telemetry=$TELEMETRY"
echo "contract=SALVALAI-smoke15-main-seq9-seed12345-fp32-sdpa-K4-max256"
echo "torch_extensions_dir=$TORCH_EXTENSIONS_DIR"
echo "torch_extensions_entries=$(find "$TORCH_EXTENSIONS_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
echo "inductor_cache_dir=$TORCHINDUCTOR_CACHE_DIR"
echo "inductor_cache_entries=$(find "$TORCHINDUCTOR_CACHE_DIR" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)"
nvidia-smi -L
nvidia-smi \
  --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv -l 1 > "$TELEMETRY" &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null || true' EXIT

python - <<'PY'
import torch
import transformers

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_device", torch.cuda.get_device_name() if torch.cuda.is_available() else None)
print("cuda_capability", torch.cuda.get_device_capability() if torch.cuda.is_available() else None)
print("transformers", transformers.__version__)
PY

python -m py_compile \
  osuT5/osuT5/inference/optimized/speculative/mini_feasibility.py \
  osuT5/osuT5/inference/optimized/speculative/mini_draft_gpu.py \
  utils/verify_speculative_mini_draft.py

set +e
python utils/verify_speculative_mini_draft.py \
  --config-name profile_salvalai_smoke15 \
  --sequence-index 9 \
  --speculation-k 4 \
  --max-new-tokens 256 \
  --encoder-warmup 2 \
  --encoder-timing-iters 8 \
  --steady-cache-warmup 2 \
  --steady-cache-timing-iters 8 \
  --report-path "$REPORT" \
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
  inference_stateful_monotonic_logits_processor=true
STATUS=$?
set -e

echo "report=$REPORT"
echo "telemetry=$TELEMETRY"
echo "run_root=$RUN_ROOT"
exit "$STATUS"
