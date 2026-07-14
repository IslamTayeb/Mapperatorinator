#!/usr/bin/env bash
#SBATCH --job-name=k4-mask-reuse-recip
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=01:30:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

ENV=${MAPPERATORINATOR_ENV:-/hpc/group/romerolab/imt11/envs/mapperatorinator}
PYTHON=${MAPPERATORINATOR_PYTHON:-$ENV/bin/python}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH}
REMOTE=${MAPPERATORINATOR_REMOTE:-islamtayeb}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
RUN_ROOT="$WORK/runs/k4-mask-reuse-reciprocal-${SLURM_JOB_ID:-manual}"
EXTENSIONS="$WORK/torch_extensions/k4-mask-reuse-$COMMIT"

if [[ ! -d "$REPO/.git" && ! -f "$REPO/.git" ]]; then
  echo "missing scout worktree at $REPO" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "scout worktree is dirty" >&2
  git -C "$REPO" status --short >&2
  exit 2
fi
if [[ "$(git -C "$REPO" rev-parse HEAD)" != "$COMMIT" ]]; then
  echo "scout worktree differs from MAPPERATORINATOR_COMMIT" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" branch --show-current)" != "$BRANCH" ]]; then
  echo "scout worktree differs from MAPPERATORINATOR_BRANCH" >&2
  exit 2
fi
REMOTE_REF="refs/remotes/$REMOTE/$BRANCH"
if ! git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF"; then
  echo "missing fetched remote ref $REMOTE_REF" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" != "$COMMIT" ]]; then
  echo "remote ref differs from MAPPERATORINATOR_COMMIT" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" || ! -f "$AUDIO" ]]; then
  echo "missing pinned Python or SALVALAI audio" >&2
  exit 2
fi
export PATH="$(dirname "$PYTHON"):$PATH"
for tool in ffmpeg ffprobe; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing required audio tool: $tool" >&2
    exit 2
  fi
done
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ "$GPU_NAME" != "NVIDIA GeForce RTX 2080 Ti" ]]; then
  echo "K4 mask reuse reciprocal requires RTX 2080 Ti, got $GPU_NAME" >&2
  exit 2
fi

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" "$EXTENSIONS" "$RUN_ROOT"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$WORK/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export TORCH_EXTENSIONS_DIR="$EXTENSIONS"

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "repo=$REPO"
echo "commit=$COMMIT"
echo "branch=$BRANCH"
echo "remote_ref=$REMOTE_REF"
echo "audio=$AUDIO"
echo "run_root=$RUN_ROOT"
nvidia-smi -L
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job -o "$SLURM_JOB_ID"
fi
nvidia-smi \
  --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv -l 1 > "$RUN_ROOT/nvidia-smi.csv" &
SMI_PID=$!
finalize() {
  kill "$SMI_PID" 2>/dev/null || true
  if [[ -d "$RUN_ROOT" ]]; then
    find "$RUN_ROOT" -type f ! -name sha256sums.txt -print0 \
      | sort -z | xargs -0 sha256sum > "$RUN_ROOT/sha256sums.txt"
  fi
}
trap finalize EXIT

COMMON_OVERRIDES=(
  audio_path="$AUDIO"
  device=cuda
  precision=fp32
  attn_implementation=sdpa
  inference_engine=optimized
  use_server=false
  parallel=false
  cfg_scale=1.0
  num_beams=1
  seed=12345
  super_timing=false
  generate_positions=false
  profile_inference=true
  profile_detail_ranges=false
  profile_cuda_capture=false
  profile_pass_kind=untraced_control
)

run_profile() {
  local role=$1
  local runner=$2
  local mask_role=$3
  local output="$RUN_ROOT/$role"
  local init_json="$RUN_ROOT/$role.initialization.json"
  local compiler_cache="$RUN_ROOT/compiler-cache/$role"
  if [[ -e "$output" || -e "$init_json" || -e "$compiler_cache" ]]; then
    echo "run path already exists for $role" >&2
    exit 2
  fi
  mkdir -p "$output" "$compiler_cache/torch_inductor" \
    "$compiler_cache/triton" "$compiler_cache/cuda"
  (
    cd "$REPO"
    export TORCHINDUCTOR_CACHE_DIR="$compiler_cache/torch_inductor"
    export TRITON_CACHE_DIR="$compiler_cache/triton"
    export CUDA_CACHE_PATH="$compiler_cache/cuda"
    "$PYTHON" "$runner" \
      --config-name profile_salvalai \
      --output-init-json "$init_json" \
      output_path="$output" \
      "${COMMON_OVERRIDES[@]}"
  )
  if [[ ! -s "$init_json" ]]; then
    echo "missing mixed-weight initialization evidence for $role" >&2
    exit 2
  fi
  local profile
  profile=$(find "$output" -name '*.profile.json' -type f | sort | head -n 1)
  if [[ -z "$profile" ]]; then
    echo "missing inference profile for $role" >&2
    exit 2
  fi
  printf '%s\n' "$profile" > "$RUN_ROOT/$role.profile-path.txt"
  (
    cd "$REPO"
    "$PYTHON" utils/validate_weight_only_full_song_profile.py \
      --profile "$profile" --role candidate \
      > "$RUN_ROOT/$role.weight-validation.json"
    "$PYTHON" utils/validate_k4_profile_contract.py \
      --profile "$profile" --role candidate --block-size 4 \
      > "$RUN_ROOT/$role.k4-validation.json"
    "$PYTHON" utils/validate_k4_mask_reuse_profile.py \
      --profile "$profile" --role "$mask_role" \
      > "$RUN_ROOT/$role.mask-validation.json"
  )
}

run_profile baseline_first \
  utils/run_k4_approximate_weight_only.py control
run_profile candidate_first \
  utils/run_k4_mask_reuse_approximate_weight_only.py candidate
run_profile candidate_second \
  utils/run_k4_mask_reuse_approximate_weight_only.py candidate
run_profile baseline_second \
  utils/run_k4_approximate_weight_only.py control

BASELINE_FIRST=$(<"$RUN_ROOT/baseline_first.profile-path.txt")
CANDIDATE_FIRST=$(<"$RUN_ROOT/candidate_first.profile-path.txt")
CANDIDATE_SECOND=$(<"$RUN_ROOT/candidate_second.profile-path.txt")
BASELINE_SECOND=$(<"$RUN_ROOT/baseline_second.profile-path.txt")

set +e
(
  cd "$REPO"
  "$PYTHON" utils/summarize_k4_mask_reuse_reciprocal.py \
    --baseline-first-profile "$BASELINE_FIRST" \
    --candidate-first-profile "$CANDIDATE_FIRST" \
    --candidate-second-profile "$CANDIDATE_SECOND" \
    --baseline-second-profile "$BASELINE_SECOND" \
    --json-output "$RUN_ROOT/analysis.json" \
    --text-output "$RUN_ROOT/analysis.txt"
)
STATUS=$?
set -e
printf '%s\n' "$STATUS" > "$RUN_ROOT/analysis.exit-code.txt"
if [[ "$STATUS" != 0 && "$STATUS" != 3 ]]; then
  echo "K4 mask reuse analysis failed before a valid decision" >&2
  exit "$STATUS"
fi
for artifact in \
  "$RUN_ROOT/analysis.json" \
  "$RUN_ROOT/analysis.txt" \
  "$RUN_ROOT/baseline_first.mask-validation.json" \
  "$RUN_ROOT/candidate_first.mask-validation.json" \
  "$RUN_ROOT/candidate_second.mask-validation.json" \
  "$RUN_ROOT/baseline_second.mask-validation.json"; do
  if [[ ! -s "$artifact" ]]; then
    echo "missing required K4 mask reuse artifact: $artifact" >&2
    exit 2
  fi
done
cat "$RUN_ROOT/analysis.txt"
exit "$STATUS"
