#!/usr/bin/env bash
#SBATCH --job-name=inference-denominator
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=06:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=/hpc/group/romerolab/imt11/projects/Mapperatorinator
WORK=/work/imt11/Mapperatorinator
RUN_KIND=${CAMPAIGN_RUN_KIND:?Set CAMPAIGN_RUN_KIND to single_full, five_smoke, or five_full}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the pushed commit under test}
COMMIT=$(git -C "$REPO" rev-parse HEAD)
SHORT_COMMIT=$(git -C "$REPO" rev-parse --short HEAD)
RUN_ROOT="$WORK/runs/inference-denominator-${RUN_KIND}-${SLURM_JOB_ID:-manual}-${SHORT_COMMIT}"
TELEMETRY="$RUN_ROOT/nvidia-smi.csv"

case "$RUN_KIND" in
  single_full|five_smoke|five_full) ;;
  *)
    echo "Unsupported CAMPAIGN_RUN_KIND=$RUN_KIND" >&2
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

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" "$WORK/torch_extensions" "$RUN_ROOT"
export PATH="$ENV/bin:$PATH"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export TMPDIR="$WORK/tmp"
export TOKENIZERS_PARALLELISM=false
export TORCH_EXTENSIONS_DIR="$WORK/torch_extensions"

cd "$REPO"

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "run_kind=$RUN_KIND"
echo "commit=$COMMIT"
echo "branch=$(git branch --show-current)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "torch_extensions_dir=$TORCH_EXTENSIONS_DIR"
echo "torch_extensions_policy=shared_prebuilt"
echo "torch_extensions_entries=$(find "$TORCH_EXTENSIONS_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
echo "inductor_cache_dir=${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/torch/inductor}"
echo "triton_cache_dir=${TRITON_CACHE_DIR:-unset}"
nvidia-smi -L
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu --format=csv -l 1 > "$TELEMETRY" &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null || true' EXIT

for profiler in nsys ncu dcgmi; do
  if command -v "$profiler" >/dev/null 2>&1; then
    echo "$profiler=$(command -v "$profiler")"
  else
    echo "$profiler=unavailable"
  fi
done

python - <<'PY'
import torch
import transformers

print("python_torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_device", torch.cuda.get_device_name() if torch.cuda.is_available() else None)
print("transformers", transformers.__version__)
PY

python -m py_compile \
  config.py \
  inference.py \
  osuT5/osuT5/inference/optimized/adapter.py \
  osuT5/osuT5/inference/optimized/contracts.py \
  osuT5/osuT5/inference/optimized/exactness.py \
  osuT5/osuT5/inference/optimized/metrics.py \
  utils/profile_inference_suite.py \
  utils/summarize_inference_profile.py

COMMON_OVERRIDES=(
  device=cuda
  precision=fp32
  attn_implementation=sdpa
  inference_engine=v32
  optimized_inference_mode=single
  use_server=false
  parallel=false
  cfg_scale=1.0
  num_beams=1
  seed=12345
  profile_inference=true
  profile_sync_cuda=true
  profile_record_token_ids=true
  inference_generation_compile=true
  inference_active_prefix_decode_loop=true
  inference_active_prefix_decode_bucket_size=64
  inference_active_prefix_decode_cuda_graph=true
  inference_active_prefix_decode_cuda_graph_warmup=0
  inference_active_prefix_decode_cuda_graph_min_decode_steps=1
  inference_stateful_monotonic_logits_processor=true
  inference_q1_bmm_cross_attention=true
  inference_decode_session_runtime=true
  inference_decode_session_cuda_graph=true
  inference_native_decode_kernels=true
  inference_native_q1_self_attention=true
  inference_native_q1_rope_cache_self_attention=true
)

if [[ "$RUN_KIND" == single_full ]]; then
  OUT_DIR="$RUN_ROOT/profile"
  mkdir -p "$OUT_DIR"
  python inference.py --config-name profile_salvalai \
    audio_path="$WORK/data/salvalai.mp3" \
    output_path="$OUT_DIR" \
    "${COMMON_OVERRIDES[@]}"
  PROFILE=$(find "$OUT_DIR" -name '*.profile.json' -type f | sort | head -n 1)
  test -n "$PROFILE"
  python utils/summarize_inference_profile.py "$PROFILE" --limit 20 | tee "$RUN_ROOT/summary.txt"
  echo "profile=$PROFILE"
else
  if [[ "$RUN_KIND" == five_smoke ]]; then
    CONFIG_NAME=profile_salvalai_smoke15
  else
    CONFIG_NAME=profile_salvalai
  fi
  python utils/profile_inference_suite.py \
    --config-name "$CONFIG_NAME" \
    --run-kind serial_multi_song \
    --song-list "$WORK/data/five-song-profile/songs.yaml" \
    --repeats 2 \
    --seed-step 0 \
    --suite-id "${RUN_KIND}-${SLURM_JOB_ID:-manual}-${SHORT_COMMIT}" \
    --output-root "$RUN_ROOT" \
    "${COMMON_OVERRIDES[@]}"
  MANIFEST=$(find "$RUN_ROOT" -name suite_manifest.json -type f | sort | head -n 1)
  test -n "$MANIFEST"
  echo "manifest=$MANIFEST"
fi

echo "run_root=$RUN_ROOT"
echo "telemetry=$TELEMETRY"
