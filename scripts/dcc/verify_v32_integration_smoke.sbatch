#!/usr/bin/env bash
#SBATCH --job-name=v32-integration-smoke
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
WORK=/work/imt11/Mapperatorinator
CONTROL_REPO=${CONTROL_REPO:?Set CONTROL_REPO to the clean current-main worktree}
CANDIDATE_REPO=${CANDIDATE_REPO:?Set CANDIDATE_REPO to the clean integration worktree}
CONTROL_COMMIT=${CONTROL_COMMIT:?Set CONTROL_COMMIT}
CANDIDATE_COMMIT=${CANDIDATE_COMMIT:?Set CANDIDATE_COMMIT}
AUDIO="$WORK/data/salvalai.mp3"
RUN_ROOT="$WORK/runs/v32-integration-smoke-${SLURM_JOB_ID:-manual}"

for repo in "$CONTROL_REPO" "$CANDIDATE_REPO"; do
  if [[ ! -d "$repo/.git" && ! -f "$repo/.git" ]]; then
    echo "missing integration worktree at $repo" >&2
    exit 2
  fi
  if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
    echo "integration worktree is dirty at $repo" >&2
    git -C "$repo" status --short >&2
    exit 2
  fi
done
if [[ "$(git -C "$CONTROL_REPO" rev-parse HEAD)" != "$CONTROL_COMMIT" ]]; then
  echo "control commit changed" >&2
  exit 2
fi
if [[ "$(git -C "$CANDIDATE_REPO" rev-parse HEAD)" != "$CANDIDATE_COMMIT" ]]; then
  echo "candidate commit changed" >&2
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

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "control_commit=$CONTROL_COMMIT"
echo "candidate_commit=$CANDIDATE_COMMIT"
echo "run_root=$RUN_ROOT"
nvidia-smi -L
nvidia-smi \
  --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv -l 1 > "$RUN_ROOT/nvidia-smi.csv" &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null || true' EXIT

COMMON_OVERRIDES=(
  audio_path="$AUDIO"
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

run_profile() {
  local repo=$1
  local label=$2
  local output="$RUN_ROOT/$label"
  mkdir -p "$output"
  (
    cd "$repo"
    python inference.py --config-name profile_salvalai_smoke15 \
      output_path="$output" \
      "${COMMON_OVERRIDES[@]}"
  )
  local profile
  profile=$(find "$output" -name '*.profile.json' -type f | sort | head -n 1)
  if [[ -z "$profile" ]]; then
    echo "missing profile for $label" >&2
    exit 2
  fi
  echo "$profile" > "$RUN_ROOT/$label.profile-path.txt"
  echo "$label=$profile"
}

# Reciprocal launch order keeps cache/node drift visible instead of assigning it
# to one commit by construction.
run_profile "$CONTROL_REPO" control_first
run_profile "$CANDIDATE_REPO" candidate_second
run_profile "$CANDIDATE_REPO" candidate_first
run_profile "$CONTROL_REPO" control_second

CONTROL_FIRST=$(<"$RUN_ROOT/control_first.profile-path.txt")
CANDIDATE_SECOND=$(<"$RUN_ROOT/candidate_second.profile-path.txt")
CANDIDATE_FIRST=$(<"$RUN_ROOT/candidate_first.profile-path.txt")
CONTROL_SECOND=$(<"$RUN_ROOT/control_second.profile-path.txt")

set +e
(
  cd "$CANDIDATE_REPO"
  python utils/summarize_inference_profile.py \
    --compare "$CONTROL_FIRST" "$CANDIDATE_SECOND" \
    --strict-full-song \
    --json-output "$RUN_ROOT/control-first-compare.json"
)
CONTROL_FIRST_STATUS=$?
(
  cd "$CANDIDATE_REPO"
  python utils/summarize_inference_profile.py \
    --compare "$CONTROL_SECOND" "$CANDIDATE_FIRST" \
    --strict-full-song \
    --json-output "$RUN_ROOT/candidate-first-compare.json"
)
CANDIDATE_FIRST_STATUS=$?
set -e

find "$RUN_ROOT" -type f \
  \( -name '*.profile.json' -o -name '*-compare.json' \) -print0 \
  | sort -z | xargs -0 sha256sum
echo "control_first_compare_status=$CONTROL_FIRST_STATUS"
echo "candidate_first_compare_status=$CANDIDATE_FIRST_STATUS"
if (( CONTROL_FIRST_STATUS != 0 || CANDIDATE_FIRST_STATUS != 0 )); then
  echo "legacy V32 reciprocal strict smoke gate failed" >&2
  exit 1
fi
