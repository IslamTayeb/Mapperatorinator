#!/usr/bin/env bash
#SBATCH --job-name=opt-single-short-loop
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:30:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the isolated migration worktree}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the pushed migration commit}
EXPECTED_BRANCH=${EXPECTED_BRANCH:-codex/optimized-single-architecture-migration}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8}
WORK=/work/imt11/Mapperatorinator
AUDIO="$WORK/data/salvalai.mp3"
SHORT_COMMIT=$(git -C "$REPO" rev-parse --short HEAD)
RUN_ROOT="$WORK/runs/optimized-single-short-loop-${MAX_NEW_TOKENS}-${SLURM_JOB_ID:-manual}-${SHORT_COMMIT}"
REPORT="$RUN_ROOT/direct-loop.json"

if [[ ! -d "$REPO/.git" && ! -f "$REPO/.git" ]]; then
  echo "missing isolated migration worktree at $REPO" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
  echo "migration worktree commit differs from EXPECTED_COMMIT" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" branch --show-current)" != "$EXPECTED_BRANCH" ]]; then
  echo "migration worktree branch differs from EXPECTED_BRANCH" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "migration worktree is dirty" >&2
  git -C "$REPO" status --short >&2
  exit 2
fi
if [[ ! "$MAX_NEW_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_NEW_TOKENS must be a positive integer" >&2
  exit 2
fi

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" \
  "$WORK/cache/torch_inductor" "$WORK/cache/triton" "$WORK/cache/cuda" \
  "$WORK/torch_extensions" "$RUN_ROOT"
export PATH="$ENV/bin:$PATH"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$WORK/cache/torch_inductor"
export TRITON_CACHE_DIR="$WORK/cache/triton"
export CUDA_CACHE_PATH="$WORK/cache/cuda"
export TORCH_EXTENSIONS_DIR="$WORK/torch_extensions"
export TMPDIR="$WORK/tmp"
export TOKENIZERS_PARALLELISM=false

cd "$REPO"
echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "commit=$EXPECTED_COMMIT"
echo "max_new_tokens=$MAX_NEW_TOKENS"
echo "run_root=$RUN_ROOT"
nvidia-smi -L
nvidia-smi \
  --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv -l 1 > "$RUN_ROOT/nvidia-smi.csv" &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null || true' EXIT

python utils/verify_direct_decode_loop.py \
  --config-name profile_salvalai_smoke15 \
  --sequence-index 9 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --candidate-decode-session \
  --candidate-active-prefix-decode \
  --candidate-active-prefix-decode-bucket-size 64 \
  --candidate-cuda-graph-forward \
  --candidate-cuda-graph-warmup 0 \
  --candidate-q1-bmm-cross-attention \
  --candidate-native-q1-self-attention \
  --candidate-native-q1-rope-cache-self-attention \
  --report-path "$REPORT" \
  audio_path="$AUDIO" \
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
  inference_generation_compile=true \
  profile_record_token_ids=true \
  seed=12345

python - "$REPORT" "$MAX_NEW_TOKENS" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
expected_steps = int(sys.argv[2])
for key in ("pass", "token_match", "rng_match", "logits_pass", "logits_allclose", "logits_topk_match"):
    if report.get(key) is not True:
        raise SystemExit(f"short-loop gate failed {key}")
if report.get("logits_step_count") != expected_steps:
    raise SystemExit("short-loop captured step count changed")
if report.get("reference_generated_token_ids") != report.get("candidate_generated_token_ids"):
    raise SystemExit("short-loop token transcript changed")
graph = report.get("candidate_cuda_graph_diagnostics") or {}
if int(graph.get("capture_count", 0)) <= 0 or int(graph.get("decode_replays", 0)) <= 0:
    raise SystemExit("short-loop did not capture and replay a CUDA graph")
print("optimized_single_short_loop_gate=PASS")
print(f"steps={expected_steps}")
print(f"capture_count={graph['capture_count']}")
print(f"decode_replays={graph['decode_replays']}")
PY

sha256sum "$REPORT"
