#!/usr/bin/env bash
#SBATCH --job-name=opt-single-session
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:45:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the isolated migration worktree}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the pushed migration commit}
EXPECTED_BRANCH=${EXPECTED_BRANCH:-codex/optimized-single-architecture-migration}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
SEQUENCE_INDICES=${SEQUENCE_INDICES:-6,7,8,9}
WORK=/work/imt11/Mapperatorinator
AUDIO="$WORK/data/salvalai.mp3"
SHORT_COMMIT=$(git -C "$REPO" rev-parse --short HEAD)
RUN_ROOT="$WORK/runs/optimized-single-session-${MAX_NEW_TOKENS}-${SLURM_JOB_ID:-manual}-${SHORT_COMMIT}"
REPORT="$RUN_ROOT/persistent-session.json"

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
echo "sequence_indices=$SEQUENCE_INDICES"
echo "max_new_tokens=$MAX_NEW_TOKENS"
echo "run_root=$RUN_ROOT"
nvidia-smi -L
nvidia-smi \
  --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv -l 1 > "$RUN_ROOT/nvidia-smi.csv" &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null || true' EXIT

python utils/verify_persistent_decode_session.py \
  --config-name profile_salvalai_smoke15 \
  --sequence-indices "$SEQUENCE_INDICES" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --active-prefix-bucket-size 64 \
  --cuda-graph-warmup 0 \
  --q1-bmm-cross-attention \
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

python - "$REPORT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
if report.get("pass") is not True:
    raise SystemExit("persistent production-session report did not pass")
if report.get("rng_state_match") is not True:
    raise SystemExit("persistent production-session RNG changed")
if report.get("production_session_cache_identity_reused") is not True:
    raise SystemExit("production cache identity was not reused")
if report.get("graph_capture_count_equals_unique_prefix_count") is not True:
    raise SystemExit("graph capture count exceeded unique prefix shapes")
if report.get("stable_encoder_output_shape") != [1, 1024, 768]:
    raise SystemExit("stable encoder output contract changed")
windows = report.get("windows", [])
if len(windows) != 4 or not all(window.get("pass") is True for window in windows):
    raise SystemExit("four-window exactness gate failed")
print("optimized_single_production_session_gate=PASS")
print(f"windows={len(windows)}")
print(f"graphs={report['graph_summary']['graphs']}")
print(f"decode_replays={report['graph_summary']['decode_replays']}")
PY

sha256sum "$REPORT"
