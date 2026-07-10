#!/usr/bin/env bash
#SBATCH --job-name=opt-single-one-token
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
WORK=/work/imt11/Mapperatorinator
AUDIO="$WORK/data/salvalai.mp3"
SHORT_COMMIT=$(git -C "$REPO" rev-parse --short HEAD)
RUN_ROOT="$WORK/runs/optimized-single-one-token-${SLURM_JOB_ID:-manual}-${SHORT_COMMIT}"
REPORT="$RUN_ROOT/one-token.json"

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
if [[ ! -f "$AUDIO" ]]; then
  echo "missing SALVALAI audio at $AUDIO" >&2
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
echo "branch=$EXPECTED_BRANCH"
echo "run_root=$RUN_ROOT"
echo "torch_extensions_dir=$TORCH_EXTENSIONS_DIR"
find "$TORCH_EXTENSIONS_DIR" -mindepth 1 -maxdepth 1 -printf '%f\n' \
  | sort > "$RUN_ROOT/extension-cache-before.txt"
nvidia-smi -L
nvidia-smi \
  --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv -l 1 > "$RUN_ROOT/nvidia-smi.csv" &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null || true' EXIT

python utils/verify_one_token_decode.py \
  --config-name profile_salvalai_smoke15 \
  --sequence-index 9 \
  --candidate-active-prefix-decode-length 128 \
  --candidate-q1-bmm-cross-attention \
  --candidate-native-q1-self-attention \
  --candidate-native-q1-rope-cache-self-attention \
  --candidate-decode-session \
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
  seed=12345

python - "$REPORT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
if report.get("pass") is not True:
    raise SystemExit("one-token migration report did not pass")
if report.get("allclose") is not True:
    raise SystemExit("raw logits did not pass")
if report.get("topk_match") is not True:
    raise SystemExit("top-k did not pass")
metadata = report.get("metadata", {})
required = {
    "candidate_active_prefix_decode_length": 128,
    "candidate_q1_bmm_cross_attention": True,
    "candidate_native_q1_self_attention": True,
    "candidate_native_q1_rope_cache_self_attention": True,
    "candidate_decode_session": True,
}
for key, expected in required.items():
    if metadata.get(key) != expected:
        raise SystemExit(f"metadata mismatch for {key}")
print("optimized_single_one_token_gate=PASS")
print(f"max_abs={report['max_abs']}")
print(f"topk_match={str(report['topk_match']).lower()}")
PY

find "$TORCH_EXTENSIONS_DIR" -mindepth 1 -maxdepth 1 -printf '%f\n' \
  | sort > "$RUN_ROOT/extension-cache-after.txt"
sha256sum "$REPORT"
