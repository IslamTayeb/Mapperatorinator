#!/usr/bin/env bash
#SBATCH --job-name=k1-input-arena
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=01:15:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

die() {
  echo "k1-shared-static-input-arena: $*" >&2
  exit 2
}

require_nonempty() {
  [[ -s "$1" ]] || die "required artifact is missing or empty: $1"
}

SLURM_JOB_ID=${SLURM_JOB_ID:?Run this wrapper inside a Slurm allocation}
PYTHON=${MAPPERATORINATOR_PYTHON:-/hpc/group/romerolab/imt11/envs/mapperatorinator/bin/python}
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean candidate worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH}
REMOTE_REF=${MAPPERATORINATOR_REMOTE_REF:?Set MAPPERATORINATOR_REMOTE_REF}
PRECISION=${MAPPERATORINATOR_PRECISION:?Set MAPPERATORINATOR_PRECISION to fp32 or fp16}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
ALLOW_PARALLEL_RECIPROCAL=${MAPPERATORINATOR_ALLOW_PARALLEL_RECIPROCAL:-0}
REGRESSION_TOLERANCE_PCT=${MAPPERATORINATOR_REGRESSION_TOLERANCE_PCT:-1}
EXPECTED_GPU=${MAPPERATORINATOR_EXPECTED_GPU:-2080 Ti}
RUN_ROOT="$WORK/runs/k1-shared-static-input-arena-${PRECISION}-${SLURM_JOB_ID}"
MIN_AVAILABLE_KIB=$((20 * 1024 * 1024))

export NVIDIA_TF32_OVERRIDE=0

case "$PRECISION" in
  fp32|fp16) ;;
  *) die "MAPPERATORINATOR_PRECISION must be fp32 or fp16" ;;
esac
[[ "$ALLOW_PARALLEL_RECIPROCAL" == 0 || "$ALLOW_PARALLEL_RECIPROCAL" == 1 ]] \
  || die "MAPPERATORINATOR_ALLOW_PARALLEL_RECIPROCAL must be 0 or 1"
[[ "$REGRESSION_TOLERANCE_PCT" =~ ^[0-9]+([.][0-9]+)?$ ]] \
  || die "MAPPERATORINATOR_REGRESSION_TOLERANCE_PCT must be non-negative"

[[ -x "$PYTHON" ]] || die "missing executable Python at $PYTHON"
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || die "missing worktree at $REPO"
[[ -f "$AUDIO" ]] || die "missing SALVALAI audio at $AUDIO"
[[ ! -e "$RUN_ROOT" ]] || die "refusing to overwrite $RUN_ROOT"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || {
  git -C "$REPO" status --short >&2
  die "candidate worktree is dirty"
}
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$COMMIT" ]] \
  || die "worktree HEAD does not match COMMIT"
[[ "$(git -C "$REPO" branch --show-current)" == "$BRANCH" ]] \
  || die "worktree branch does not match BRANCH"
git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF" \
  || die "missing pushed remote ref $REMOTE_REF"
[[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" == "$COMMIT" ]] \
  || die "pushed remote ref does not match COMMIT"

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" \
  "$WORK/cache/torch_inductor" "$WORK/cache/triton" "$WORK/cache/cuda" \
  "$WORK/torch_extensions" "$RUN_ROOT"
export PATH="$(dirname "$PYTHON"):$PATH"
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

for tool in ffmpeg ffprobe nvidia-smi scontrol squeue sha256sum date df; do
  command -v "$tool" >/dev/null 2>&1 || die "missing executable: $tool"
done
AVAILABLE_KIB=$(df -Pk "$WORK" | awk 'NR == 2 {print $4}')
[[ "$AVAILABLE_KIB" =~ ^[0-9]+$ ]] || die "could not inspect storage"
(( AVAILABLE_KIB >= MIN_AVAILABLE_KIB )) || die "less than 20 GiB is free"

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ ${#GPU_NAMES[@]} -eq 1 ]] || die "expected exactly one visible GPU"
[[ "${GPU_NAMES[0]}" == *"$EXPECTED_GPU"* ]] \
  || die "expected $EXPECTED_GPU, got ${GPU_NAMES[0]}"
mapfile -t OTHER_GPU_JOBS < <(
  squeue -h -u "$USER" -t RUNNING,PENDING -o '%A|%b|%j' \
    | awk -F'|' -v current="$SLURM_JOB_ID" \
        '$1 != current && tolower($2) ~ /gpu/ { print }'
)
if [[ ${#OTHER_GPU_JOBS[@]} -ne 0 && "$ALLOW_PARALLEL_RECIPROCAL" != 1 ]]; then
  printf '%s\n' "${OTHER_GPU_JOBS[@]}" >&2
  die "another GPU job is running or pending"
fi
if [[ ${#OTHER_GPU_JOBS[@]} -ne 0 ]]; then
  printf '%s\n' "${OTHER_GPU_JOBS[@]}" >&2
  echo "parallel_reciprocal_opt_in=1" >&2
fi

{
  echo "host=$(hostname)"
  echo "job_id=$SLURM_JOB_ID"
  echo "repo=$REPO"
  echo "commit=$COMMIT"
  echo "branch=$BRANCH"
  echo "remote_ref=$REMOTE_REF"
  echo "precision=$PRECISION"
  echo "audio=$AUDIO"
  echo "audio_sha256=$(sha256sum "$AUDIO" | awk '{print $1}')"
  echo "run_root=$RUN_ROOT"
  echo "parallel_reciprocal_opt_in=$ALLOW_PARALLEL_RECIPROCAL"
  echo "regression_tolerance_pct=$REGRESSION_TOLERANCE_PCT"
  echo "NVIDIA_TF32_OVERRIDE=$NVIDIA_TF32_OVERRIDE"
  git -C "$REPO" status --short --branch
  "$PYTHON" -c 'import torch, transformers; print(f"torch={torch.__version__}"); print(f"torch_cuda={torch.version.cuda}"); print(f"transformers={transformers.__version__}")'
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv,noheader
  scontrol show job -o "$SLURM_JOB_ID"
  squeue -u "$USER" -o '%.18i %.40j %.10T %.24b'
  df -h "$WORK"
} | tee "$RUN_ROOT/preflight.txt"

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

cd "$REPO"
COMMON_OVERRIDES=(
  audio_path="$AUDIO"
  device=cuda
  precision="$PRECISION"
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

capture_paths() {
  local run_dir=$1
  mapfile -t profiles < <(find "$run_dir/output" -type f -name '*.profile.json' | sort)
  [[ ${#profiles[@]} -eq 1 ]] || die "expected one profile in $run_dir"
  local profile=${profiles[0]}
  local result=${profile%.profile.json}
  require_nonempty "$profile"
  require_nonempty "$result"
  printf '%s\n' "$profile" > "$run_dir/profile-path.txt"
  printf '%s\n' "$result" > "$run_dir/result-path.txt"
}

run_profile() {
  local mode=$1
  local label=$2
  local run_dir="$RUN_ROOT/$label"
  mkdir -p "$run_dir/output"
  local command
  if [[ "$mode" == baseline ]]; then
    command=("$PYTHON" inference.py)
  elif [[ "$mode" == candidate ]]; then
    command=("$PYTHON" -m utils.run_k1_shared_static_input_arena)
  else
    die "unknown run mode $mode"
  fi
  command+=(
    --config-name profile_salvalai
    output_path="$run_dir/output"
    "${COMMON_OVERRIDES[@]}"
  )
  printf '%q ' "${command[@]}" > "$run_dir/command.txt"
  printf '\n' >> "$run_dir/command.txt"
  local started_ns finished_ns elapsed_ns exit_code
  started_ns=$(date +%s%N)
  set +e
  "${command[@]}" > "$run_dir/stdout.txt" 2> "$run_dir/stderr.txt"
  exit_code=$?
  set -e
  finished_ns=$(date +%s%N)
  elapsed_ns=$((finished_ns - started_ns))
  printf '{"started_unix_ns":%s,"finished_unix_ns":%s,"elapsed_ns":%s,"exit_code":%s}\n' \
    "$started_ns" "$finished_ns" "$elapsed_ns" "$exit_code" \
    > "$run_dir/process-wall.json"
  [[ $exit_code -eq 0 ]] || die "$label failed"
  capture_paths "$run_dir"
}

run_profile baseline baseline_first
run_profile candidate candidate_second
run_profile candidate candidate_first
run_profile baseline baseline_second

BASELINE_FIRST=$(<"$RUN_ROOT/baseline_first/profile-path.txt")
CANDIDATE_SECOND=$(<"$RUN_ROOT/candidate_second/profile-path.txt")
CANDIDATE_FIRST=$(<"$RUN_ROOT/candidate_first/profile-path.txt")
BASELINE_SECOND=$(<"$RUN_ROOT/baseline_second/profile-path.txt")

"$PYTHON" -m utils.summarize_inference_profile \
  --reciprocal \
    "$BASELINE_FIRST" "$CANDIDATE_SECOND" \
    "$CANDIDATE_FIRST" "$BASELINE_SECOND" \
  --labels main_generation,timing_context \
  --regression-tolerance-pct "$REGRESSION_TOLERANCE_PCT" \
  --json-output "$RUN_ROOT/reciprocal.json"
require_nonempty "$RUN_ROOT/reciprocal.json"

"$PYTHON" -m utils.validate_k1_shared_static_input_arena_reciprocal \
  --reciprocal "$RUN_ROOT/reciprocal.json" \
  --baseline-first "$BASELINE_FIRST" \
  --candidate-second "$CANDIDATE_SECOND" \
  --candidate-first "$CANDIDATE_FIRST" \
  --baseline-second "$BASELINE_SECOND" \
  --precision "$PRECISION" \
  --json-output "$RUN_ROOT/decision.json" \
  --text-output "$RUN_ROOT/decision.txt"
require_nonempty "$RUN_ROOT/decision.json"
require_nonempty "$RUN_ROOT/decision.txt"
cat "$RUN_ROOT/decision.txt"
