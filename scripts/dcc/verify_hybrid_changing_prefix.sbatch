#!/usr/bin/env bash
#SBATCH --job-name=hybrid-prefix
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
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the isolated hybrid DCC worktree}
WORK=/work/imt11/Mapperatorinator
EXPECTED_COMMIT=${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the pushed changing-prefix verifier commit}
EXPECTED_BRANCH=${EXPECTED_BRANCH:-codex/optimized-hybrid-l2-verifier}
FIXED_REPORT=/work/imt11/Mapperatorinator/runs/hybrid-l2-49552768-682abdc/hybrid-l2.json
FIXED_SHA=d7f47e5c174a8f3504401ec0d428b0070f4724e95971efc150c3d13007f343ef

if [[ ! -d "$REPO/.git" && ! -f "$REPO/.git" ]]; then
  echo "isolated changing-prefix checkout is missing at $REPO" >&2
  exit 2
fi
if [[ "$(realpath "$REPO")" == "$(realpath "$SHARED_REPO")" ]]; then
  echo "refusing to run changing-prefix scout from shared DCC checkout" >&2
  exit 2
fi
if [[ ! -f "$FIXED_REPORT" ]]; then
  echo "fixed hybrid report is missing at $FIXED_REPORT" >&2
  exit 2
fi
if [[ "$(sha256sum "$FIXED_REPORT" | awk '{print $1}')" != "$FIXED_SHA" ]]; then
  echo "fixed hybrid report SHA-256 mismatch" >&2
  exit 2
fi

COMMIT=$(git -C "$REPO" rev-parse HEAD)
BRANCH=$(git -C "$REPO" branch --show-current)
SHORT_COMMIT=$(git -C "$REPO" rev-parse --short HEAD)
if [[ "$COMMIT" != "$EXPECTED_COMMIT" || "$BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "checkout is $BRANCH@$COMMIT, expected $EXPECTED_BRANCH@$EXPECTED_COMMIT" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "changing-prefix checkout is dirty; refusing benchmark" >&2
  git -C "$REPO" status --short >&2
  exit 2
fi

RUN_ROOT="$WORK/runs/hybrid-changing-prefix-${SLURM_JOB_ID:-manual}-${SHORT_COMMIT}"
REPORT="$RUN_ROOT/hybrid-changing-prefix.json"
TELEMETRY="$RUN_ROOT/nvidia-smi.csv"
mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" "$WORK/cache/torch_inductor" \
  "$WORK/cache/triton" "$WORK/cache/cuda" "$WORK/torch_extensions" "$RUN_ROOT"
export PATH="$ENV/bin:$PATH"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export TORCHINDUCTOR_CACHE_DIR="$WORK/cache/torch_inductor"
export TRITON_CACHE_DIR="$WORK/cache/triton"
export CUDA_CACHE_PATH="$WORK/cache/cuda"
export TORCH_EXTENSIONS_DIR="$WORK/torch_extensions"
export TMPDIR="$WORK/tmp"
export TOKENIZERS_PARALLELISM=false

cd "$REPO"
echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "partition=${SLURM_JOB_PARTITION:-unknown}"
echo "commit=$COMMIT"
echo "branch=$BRANCH"
echo "repo=$REPO"
echo "fixed_report=$FIXED_REPORT"
echo "fixed_sha=$FIXED_SHA"
echo "run_root=$RUN_ROOT"
echo "torch_extensions_dir=$TORCH_EXTENSIONS_DIR"
nvidia-smi -L
nvidia-smi \
  --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv -l 1 > "$TELEMETRY" &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null || true' EXIT

python -m py_compile \
  osuT5/osuT5/inference/optimized/batch/hybrid_changing_prefix.py \
  utils/verify_optimized_hybrid_changing_prefix.py

set +e
python utils/verify_optimized_hybrid_changing_prefix.py \
  --config-name profile_salvalai_smoke15 \
  --sequence-index 9 \
  --fixed-hybrid-report "$FIXED_REPORT" \
  --report-path "$REPORT" \
  --row-seed 12345 \
  --row-seed 23456 \
  --capture-warmup-repeats 1 \
  --comparison-top-k 20 \
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

if [[ -f "$REPORT" ]]; then
  python - "$REPORT" <<'PY'
import json
import sys
from osuT5.osuT5.inference.optimized.batch.hybrid_changing_prefix import validate_changing_prefix_report

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
validate_changing_prefix_report(report)
print("report_self_validation=PASS")
print("pass", report["pass"])
for phase_name in ("phase_a", "phase_b"):
    phase = report.get(phase_name)
    if phase:
        print(phase_name, phase["exactness_pass"], phase["performance"])
PY
  sha256sum "$REPORT"
fi
exit "$STATUS"
