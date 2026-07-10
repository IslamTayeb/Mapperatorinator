#!/usr/bin/env bash
#SBATCH --job-name=l2-lane-capture
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
SHARED_REPO=/hpc/group/romerolab/imt11/projects/Mapperatorinator
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the isolated lane-physics DCC worktree}
WORK=/work/imt11/Mapperatorinator
EXPECTED_COMMIT=${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the pushed L2 verifier commit}
EXPECTED_BRANCH=${EXPECTED_BRANCH:-codex/optimized-lane-physics}
REVIEWED_L1_REPORT=/work/imt11/Mapperatorinator/runs/b1-lane-capture-49547823-08a59f5/b1-lane-capture.json
REVIEWED_L1_SHA256=b91383f063a48be95a1174197a8c7cc389c75615dfff9d369c6ad42447dd7cc8
REVIEWED_L1_COMPLETE_TPS=414.9046973558729
L1_GATE_REPORT=${L1_GATE_REPORT:-$REVIEWED_L1_REPORT}

if [[ ! -d "$REPO/.git" && ! -f "$REPO/.git" ]]; then
  echo "isolated lane checkout is missing at $REPO" >&2
  exit 2
fi
if [[ "$(realpath "$REPO")" == "$(realpath "$SHARED_REPO")" ]]; then
  echo "refusing to run L2 lane scout from the shared DCC source checkout" >&2
  exit 2
fi
if [[ ! -f "$L1_GATE_REPORT" ]]; then
  echo "reviewed L1 report is missing at $L1_GATE_REPORT" >&2
  exit 2
fi
if [[ "$(realpath "$L1_GATE_REPORT")" != "$(realpath "$REVIEWED_L1_REPORT")" ]]; then
  echo "L2 must use the reviewed L1 report $REVIEWED_L1_REPORT" >&2
  exit 2
fi
if [[ "$(sha256sum "$L1_GATE_REPORT" | awk '{print $1}')" != "$REVIEWED_L1_SHA256" ]]; then
  echo "reviewed L1 report SHA-256 mismatch" >&2
  exit 2
fi

COMMIT=$(git -C "$REPO" rev-parse HEAD)
BRANCH=$(git -C "$REPO" branch --show-current)
SHORT_COMMIT=$(git -C "$REPO" rev-parse --short HEAD)
RUN_ROOT="$WORK/runs/l2-lane-capture-${SLURM_JOB_ID:-manual}-${SHORT_COMMIT}"
REPORT="$RUN_ROOT/l2-lane-capture.json"
TELEMETRY="$RUN_ROOT/nvidia-smi.csv"
CACHE_STATE="$RUN_ROOT/cache-state.txt"

if [[ "$COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "DCC lane checkout is $COMMIT, expected $EXPECTED_COMMIT" >&2
  exit 2
fi
if [[ "$BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "DCC lane checkout branch is $BRANCH, expected $EXPECTED_BRANCH" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "DCC lane checkout is dirty; refusing to benchmark" >&2
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

cache_entries() {
  local path=$1
  find "$path" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l
}

record_cache_state() {
  local phase=$1
  {
    echo "phase=$phase"
    echo "torch_extensions_dir=$TORCH_EXTENSIONS_DIR"
    echo "torch_extensions_entries=$(cache_entries "$TORCH_EXTENSIONS_DIR")"
    echo "torchinductor_cache_dir=$TORCHINDUCTOR_CACHE_DIR"
    echo "torchinductor_cache_entries=$(cache_entries "$TORCHINDUCTOR_CACHE_DIR")"
    echo "triton_cache_dir=$TRITON_CACHE_DIR"
    echo "triton_cache_entries=$(cache_entries "$TRITON_CACHE_DIR")"
    echo "cuda_cache_path=$CUDA_CACHE_PATH"
    echo "cuda_cache_entries=$(cache_entries "$CUDA_CACHE_PATH")"
    echo "cublas_workspace_config=${CUBLAS_WORKSPACE_CONFIG:-unset}"
  } | tee -a "$CACHE_STATE"
}

cd "$REPO"

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "partition=${SLURM_JOB_PARTITION:-unknown}"
echo "commit=$COMMIT"
echo "branch=$BRANCH"
echo "repo=$REPO"
echo "reviewed_l1_report=$L1_GATE_REPORT"
echo "reviewed_l1_sha256=$REVIEWED_L1_SHA256"
echo "reviewed_l1_complete_tps=$REVIEWED_L1_COMPLETE_TPS"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "run_root=$RUN_ROOT"
echo "report=$REPORT"
echo "telemetry=$TELEMETRY"
record_cache_state before

for profiler in nsys ncu dcgmi; do
  if command -v "$profiler" >/dev/null 2>&1; then
    echo "$profiler=$(command -v "$profiler")"
  else
    echo "$profiler=unavailable"
  fi
done

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
  osuT5/osuT5/inference/optimized/batch/lane_one_token.py \
  osuT5/osuT5/inference/optimized/benchmark/batch_physics.py \
  utils/verify_optimized_b1_lane_capture.py \
  utils/verify_optimized_l2_lane_capture.py

set +e
python utils/verify_optimized_l2_lane_capture.py \
  --config-name profile_salvalai_smoke15 \
  --sequence-index 9 \
  --previous-gate-report "$L1_GATE_REPORT" \
  --expected-b1-complete-tps "$REVIEWED_L1_COMPLETE_TPS" \
  --report-path "$REPORT" \
  --row-seed 12345 \
  --row-seed 23456 \
  --capture-warmup-repeats 1 \
  --timing-repeats 50 \
  --active-prefix-bucket-size 64 \
  --q1-bmm-cross-attention \
  --native-q1-self-attention \
  --native-q1-rope-cache-self-attention \
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
  inference_active_prefix_decode_cuda_graph=true \
  inference_active_prefix_decode_cuda_graph_warmup=0 \
  inference_active_prefix_decode_cuda_graph_min_decode_steps=1 \
  inference_stateful_monotonic_logits_processor=true \
  inference_q1_bmm_cross_attention=true \
  inference_decode_session_runtime=true \
  inference_decode_session_cuda_graph=true \
  inference_native_decode_kernels=true \
  inference_native_q1_self_attention=true \
  inference_native_q1_rope_cache_self_attention=true
STATUS=$?
set -e

record_cache_state after
if [[ -f "$REPORT" ]]; then
  python - "$REPORT" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
print("pass", report.get("pass"))
print("exactness_pass", report.get("exactness_pass"))
print("resource_ownership_pass", report.get("resource_ownership_pass"))
print("capture_pass", report.get("capture_pass"))
print("performance_gate_pass", report.get("performance_gate_pass"))
print("same_job_serial_compare_pass", report.get("same_job_serial_compare_pass"))
print("required_five_percent_tokens_per_second", report.get("required_five_percent_tokens_per_second"))
performance = report.get("performance") or {}
print("worst_order_tokens_per_second", performance.get("worst_order_tokens_per_second"))
print("best_order_tokens_per_second", performance.get("best_order_tokens_per_second"))
print("reciprocal_order_spread_fraction", performance.get("reciprocal_order_spread_fraction"))
serial = report.get("serial_baseline") or {}
print("same_job_serial_complete", (serial.get("complete_timing") or {}).get("wall_tokens_per_second"))
for order, values in sorted((report.get("orders") or {}).items()):
    print("order", order, "exact", values.get("exactness_pass"), "complete_tps", (values.get("complete_timing") or {}).get("wall_tokens_per_second"), "peak_allocated", (values.get("complete_timing") or {}).get("peak_allocated_bytes"))
PY
fi

echo "report=$REPORT"
echo "telemetry=$TELEMETRY"
echo "cache_state=$CACHE_STATE"
echo "run_root=$RUN_ROOT"
exit "$STATUS"
