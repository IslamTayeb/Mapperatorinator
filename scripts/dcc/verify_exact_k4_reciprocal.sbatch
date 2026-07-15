#!/usr/bin/env bash
#SBATCH --job-name=strict-exact-k4
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=04:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

die() {
  echo "strict-exact-k4: $*" >&2
  exit 2
}

require_nonempty() {
  [[ -s "$1" ]] || die "required artifact is missing or empty: $1"
}

SLURM_JOB_ID=${SLURM_JOB_ID:?Run this wrapper inside a Slurm allocation}
PYTHON=${MAPPERATORINATOR_PYTHON:-/hpc/group/romerolab/imt11/envs/mapperatorinator/bin/python}
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean exact-K4 worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT to the pushed exact-K4 commit}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH to the exact-K4 branch}
REMOTE_REF=${MAPPERATORINATOR_REMOTE_REF:?Set MAPPERATORINATOR_REMOTE_REF to the pushed remote ref}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
EXPECTED_GPU=${MAPPERATORINATOR_EXPECTED_GPU:-2080 Ti}
EXACT_K4_SCOPE=${EXACT_K4_SCOPE:-smoke}
MINIMUM_SPEEDUP_PCT=${MINIMUM_SPEEDUP_PCT:-0}
FULL_SAVING_SECONDS=${FULL_SAVING_SECONDS:-5.0}
MIN_AVAILABLE_KIB=$((20 * 1024 * 1024))

case "$EXACT_K4_SCOPE" in
  smoke)
    PROFILE_CONFIG=profile_salvalai_smoke15
    ;;
  full)
    PROFILE_CONFIG=profile_salvalai
    ;;
  *)
    die "EXACT_K4_SCOPE must be smoke or full"
    ;;
esac
RUN_ROOT="$WORK/runs/strict-exact-k4-${EXACT_K4_SCOPE}-${SLURM_JOB_ID}"

# This must precede the first Python process and CUDA context.
export NVIDIA_TF32_OVERRIDE=0

[[ -x "$PYTHON" ]] || die "missing executable Python at $PYTHON"
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || die "missing worktree at $REPO"
[[ -f "$AUDIO" ]] || die "missing input audio at $AUDIO"
[[ ! -e "$RUN_ROOT" ]] || die "refusing to overwrite existing run root $RUN_ROOT"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || {
  git -C "$REPO" status --short >&2
  die "exact-K4 worktree is dirty"
}
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$COMMIT" ]] \
  || die "worktree HEAD does not match MAPPERATORINATOR_COMMIT"
[[ "$(git -C "$REPO" branch --show-current)" == "$BRANCH" ]] \
  || die "worktree is not on MAPPERATORINATOR_BRANCH=$BRANCH"
git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF" \
  || die "missing pushed remote ref $REMOTE_REF"
[[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" == "$COMMIT" ]] \
  || die "pushed remote ref differs from MAPPERATORINATOR_COMMIT"

export PATH="$(dirname "$PYTHON"):$PATH"
mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" \
  "$WORK/cache/torch_inductor" "$WORK/cache/triton" "$WORK/cache/cuda" \
  "$WORK/torch_extensions" "$RUN_ROOT"
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
  command -v "$tool" >/dev/null 2>&1 || die "missing required executable: $tool"
done
AVAILABLE_KIB=$(df -Pk "$WORK" | awk 'NR == 2 {print $4}')
[[ "$AVAILABLE_KIB" =~ ^[0-9]+$ ]] || die "could not determine storage for $WORK"
(( AVAILABLE_KIB >= MIN_AVAILABLE_KIB )) \
  || die "run-root filesystem has only $AVAILABLE_KIB KiB free"

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ ${#GPU_NAMES[@]} -eq 1 ]] || die "expected one visible GPU, got ${#GPU_NAMES[@]}"
[[ "${GPU_NAMES[0]}" == *"$EXPECTED_GPU"* ]] \
  || die "expected $EXPECTED_GPU, got ${GPU_NAMES[0]}"
mapfile -t OTHER_GPU_JOBS < <(
  squeue -h -u "$USER" -t RUNNING,PENDING -o '%A|%b|%j' \
    | awk -F'|' -v current="$SLURM_JOB_ID" \
        '$1 != current && tolower($2) ~ /gpu/ { print }'
)
if [[ ${#OTHER_GPU_JOBS[@]} -ne 0 ]]; then
  printf '%s\n' "${OTHER_GPU_JOBS[@]}" >&2
  die "another GPU job is running or pending for $USER"
fi

{
  echo "host=$(hostname)"
  echo "job_id=$SLURM_JOB_ID"
  echo "commit=$COMMIT"
  echo "branch=$BRANCH"
  echo "remote_ref=$REMOTE_REF"
  echo "repo=$REPO"
  echo "python=$PYTHON"
  echo "audio=$AUDIO"
  echo "audio_sha256=$(sha256sum "$AUDIO" | awk '{print $1}')"
  echo "scope=$EXACT_K4_SCOPE"
  echo "profile_config=$PROFILE_CONFIG"
  echo "minimum_speedup_pct=$MINIMUM_SPEEDUP_PCT"
  echo "full_saving_seconds=$FULL_SAVING_SECONDS"
  echo "run_root=$RUN_ROOT"
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
  precision=fp32
  attn_implementation=sdpa
  inference_engine=optimized
  use_server=false
  parallel=false
  cfg_scale=1.0
  num_beams=1
  seed=12345
  profile_inference=true
  profile_detail_ranges=false
  profile_cuda_capture=false
)

capture_paths() {
  local run_dir=$1
  mapfile -t profiles < <(find "$run_dir/output" -type f -name '*.profile.json' | sort)
  [[ ${#profiles[@]} -eq 1 ]] \
    || die "expected one profile JSON in $run_dir/output, got ${#profiles[@]}"
  local profile=${profiles[0]}
  local result=${profile%.profile.json}
  require_nonempty "$profile"
  require_nonempty "$result"
  printf '%s\n' "$profile" > "$run_dir/profile-path.txt"
  printf '%s\n' "$result" > "$run_dir/result-path.txt"
}

run_fresh_process() {
  local run_name=$1
  local runner=$2
  local pass_kind=$3
  local run_dir="$RUN_ROOT/$run_name"
  mkdir -p "$run_dir/output"
  local command=(
    "$PYTHON" "$runner" --config-name "$PROFILE_CONFIG"
    output_path="$run_dir/output"
    profile_pass_kind="$pass_kind"
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
  [[ $exit_code -eq 0 ]] || die "$run_name failed with exit code $exit_code"
  capture_paths "$run_dir"
}

# Reciprocal order controls thermal/order bias; every run is a fresh process.
run_fresh_process baseline_first inference.py untraced_control
run_fresh_process candidate_second utils/run_exact_k4_candidate.py untraced_control
run_fresh_process candidate_first utils/run_exact_k4_candidate.py untraced_control
run_fresh_process baseline_second inference.py untraced_control

# Cache and RNG hashing is deliberately outside authoritative model timing.
run_fresh_process exactness_baseline inference.py exactness_audit
run_fresh_process exactness_candidate utils/run_exact_k4_candidate.py exactness_audit

BASELINE_FIRST=$(<"$RUN_ROOT/baseline_first/profile-path.txt")
CANDIDATE_SECOND=$(<"$RUN_ROOT/candidate_second/profile-path.txt")
CANDIDATE_FIRST=$(<"$RUN_ROOT/candidate_first/profile-path.txt")
BASELINE_SECOND=$(<"$RUN_ROOT/baseline_second/profile-path.txt")
EXACTNESS_BASELINE=$(<"$RUN_ROOT/exactness_baseline/profile-path.txt")
EXACTNESS_CANDIDATE=$(<"$RUN_ROOT/exactness_candidate/profile-path.txt")

"$PYTHON" utils/validate_exact_k4_reciprocal.py \
  --scope "$EXACT_K4_SCOPE" \
  --baseline-first "$BASELINE_FIRST" \
  --candidate-second "$CANDIDATE_SECOND" \
  --candidate-first "$CANDIDATE_FIRST" \
  --baseline-second "$BASELINE_SECOND" \
  --exactness-baseline "$EXACTNESS_BASELINE" \
  --exactness-candidate "$EXACTNESS_CANDIDATE" \
  --expected-commit "$COMMIT" \
  --expected-branch "$BRANCH" \
  --minimum-speedup-pct "$MINIMUM_SPEEDUP_PCT" \
  --full-saving-seconds "$FULL_SAVING_SECONDS" \
  --json-output "$RUN_ROOT/analysis.json" \
  --text-output "$RUN_ROOT/analysis.txt"

require_nonempty "$RUN_ROOT/analysis.json"
require_nonempty "$RUN_ROOT/analysis.txt"
echo "strict_exact_k4_reciprocal=PASS"
echo "artifacts=$RUN_ROOT"
