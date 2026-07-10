#!/usr/bin/env bash
#SBATCH --job-name=static-server-regression
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=02:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
WORK=/work/imt11/Mapperatorinator
CONTROL_REPO=${CONTROL_REPO:?Set CONTROL_REPO to the clean pre-migration worktree}
CANDIDATE_REPO=${CANDIDATE_REPO:?Set CANDIDATE_REPO to the clean migration worktree}
CONTROL_COMMIT=${CONTROL_COMMIT:?Set CONTROL_COMMIT}
CANDIDATE_COMMIT=${CANDIDATE_COMMIT:?Set CANDIDATE_COMMIT}
SONG_LIST=${SONG_LIST:-$WORK/data/five-song-profile/songs.yaml}
RUN_LABEL=${RUN_LABEL:-smoke15}
RUN_ROOT="$WORK/runs/static-server-regression-${RUN_LABEL}-${SLURM_JOB_ID:-manual}"

for repo in "$CONTROL_REPO" "$CANDIDATE_REPO"; do
  if [[ ! -d "$repo/.git" && ! -f "$repo/.git" ]]; then
    echo "missing regression worktree at $repo" >&2
    exit 2
  fi
  if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
    echo "regression worktree is dirty at $repo" >&2
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
if [[ ! -f "$SONG_LIST" ]]; then
  echo "missing five-song list at $SONG_LIST" >&2
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
echo "song_list=$SONG_LIST"
echo "run_root=$RUN_ROOT"
nvidia-smi -L
find "$TORCH_EXTENSIONS_DIR" -mindepth 1 -maxdepth 1 -printf '%f\n' \
  | sort > "$RUN_ROOT/extension-cache-before.txt"
nvidia-smi \
  --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv -l 1 > "$RUN_ROOT/nvidia-smi.csv" &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null || true' EXIT

COMMON_OVERRIDES=(
  device=cuda
  precision=fp32
  attn_implementation=sdpa
  inference_engine=v32
  optimized_inference_mode=single
  use_server=true
  parallel=false
  max_batch_size=5
  server_batch_timeout=0.2
  cfg_scale=1.0
  num_beams=1
  seed=12345
  profile_inference=true
  profile_sync_cuda=true
  profile_record_token_ids=true
  inference_generation_compile=false
  inference_active_prefix_decode_loop=false
  inference_active_prefix_decode_bucket_size=128
  inference_active_prefix_decode_cuda_graph=false
  inference_active_prefix_decode_cuda_graph_warmup=0
  inference_active_prefix_decode_cuda_graph_min_decode_steps=1
  inference_stateful_monotonic_logits_processor=false
  inference_q1_bmm_cross_attention=false
  inference_decode_session_runtime=false
  inference_decode_session_cuda_graph=false
  inference_native_decode_kernels=false
  inference_native_q1_self_attention=false
  inference_native_q1_rope_cache_self_attention=false
)

run_suite() {
  local repo=$1
  local label=$2
  local output="$RUN_ROOT/$label"
  mkdir -p "$output"
  (
    cd "$repo"
    python utils/profile_static_server_batch.py \
      --config-name profile_salvalai_smoke15 \
      --song-list "$SONG_LIST" \
      --suite-id "${label}-${SLURM_JOB_ID:-manual}" \
      --output-root "$output" \
      --repeats 1 \
      --max-workers 5 \
      "${COMMON_OVERRIDES[@]}"
  )
  local manifest
  manifest=$(find "$output" -name static_server_batch_manifest.json -type f | sort | head -n 1)
  if [[ -z "$manifest" ]]; then
    echo "missing static-server manifest for $label" >&2
    exit 2
  fi
  echo "$manifest" > "$RUN_ROOT/$label.manifest-path.txt"
}

run_suite "$CONTROL_REPO" control
run_suite "$CANDIDATE_REPO" candidate

CONTROL_MANIFEST=$(<"$RUN_ROOT/control.manifest-path.txt")
CANDIDATE_MANIFEST=$(<"$RUN_ROOT/candidate.manifest-path.txt")
(
  cd "$CANDIDATE_REPO"
  python utils/summarize_inference_profile.py \
    --compare-static-server "$CONTROL_MANIFEST" "$CANDIDATE_MANIFEST" \
    --strict \
    --json-output "$RUN_ROOT/compare-static-server.json"
)

python - "$CONTROL_MANIFEST" "$CANDIDATE_MANIFEST" <<'PY'
import json
import sys

for name, path in zip(("control", "candidate"), sys.argv[1:]):
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    aggregate = manifest.get("aggregate", {})
    if aggregate.get("result_class") != "static_server_batch":
        raise SystemExit(f"{name}: real static batching was not observed")
    if aggregate.get("server_batch_observed") is not True:
        raise SystemExit(f"{name}: server_batch_observed changed")
    if aggregate.get("token_equivalence_status") != "not_checked_shared_server_rng":
        raise SystemExit(f"{name}: shared server RNG label changed")
print("static_server_regression_gate=PASS")
PY

find "$TORCH_EXTENSIONS_DIR" -mindepth 1 -maxdepth 1 -printf '%f\n' \
  | sort > "$RUN_ROOT/extension-cache-after.txt"
cmp "$RUN_ROOT/extension-cache-before.txt" "$RUN_ROOT/extension-cache-after.txt"
find "$RUN_ROOT" -type f \
  \( -name static_server_batch_manifest.json -o -name 'compare-static-server.json' \) \
  -print0 | sort -z | xargs -0 sha256sum
