#!/usr/bin/env bash
#SBATCH --job-name=fp32-candidate-gate
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
  echo "strict-fp32-candidate: $*" >&2
  exit 2
}

require_nonempty() {
  [[ -s "$1" ]] || die "required artifact is missing or empty: $1"
}

SLURM_JOB_ID=${SLURM_JOB_ID:?Run this wrapper inside a Slurm allocation}
PYTHON=${MAPPERATORINATOR_PYTHON:-/hpc/group/romerolab/imt11/envs/mapperatorinator/bin/python}
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean candidate worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT to the pushed candidate commit}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH to the candidate branch}
REMOTE_REF=${MAPPERATORINATOR_REMOTE_REF:?Set MAPPERATORINATOR_REMOTE_REF to the pushed candidate remote ref}
BASELINE_RUN_ROOT=${MAPPERATORINATOR_BASELINE_RUN_ROOT:?Set the authoritative baseline run root}
BASELINE_ANALYSIS=${MAPPERATORINATOR_BASELINE_ANALYSIS:-$BASELINE_RUN_ROOT/analysis.json}
CANDIDATE_MODE=${MAPPERATORINATOR_CANDIDATE_MODE:-initial}
INITIAL_ANALYSIS=${MAPPERATORINATOR_INITIAL_ANALYSIS:-}
DEFAULT_DISPATCH_DELTA_SPEC="$REPO/scripts/dcc/strict_fp32_split_kv_dispatch_delta.json"
DISPATCH_DELTA_SPEC=${MAPPERATORINATOR_DISPATCH_DELTA_SPEC:-$DEFAULT_DISPATCH_DELTA_SPEC}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
EXPECTED_GPU=${MAPPERATORINATOR_EXPECTED_GPU:-2080 Ti}
RUN_LABEL=${RUN_LABEL:-strict-fp32-candidate-$CANDIDATE_MODE}
RUN_ROOT="$WORK/runs/${RUN_LABEL}-${SLURM_JOB_ID}"
MIN_AVAILABLE_KIB=$((20 * 1024 * 1024))

# Set before the first Python process and CUDA context.
export NVIDIA_TF32_OVERRIDE=0

case "$CANDIDATE_MODE" in
  initial)
    [[ -z "$INITIAL_ANALYSIS" ]] \
      || die "initial mode must not set MAPPERATORINATOR_INITIAL_ANALYSIS"
    RUN_INDICES=(01)
    ;;
  confirm)
    [[ -n "$INITIAL_ANALYSIS" ]] \
      || die "confirm mode requires MAPPERATORINATOR_INITIAL_ANALYSIS"
    require_nonempty "$INITIAL_ANALYSIS"
    RUN_INDICES=(01 02 03 04 05)
    ;;
  *)
    die "MAPPERATORINATOR_CANDIDATE_MODE must be initial or confirm"
    ;;
esac

[[ -x "$PYTHON" ]] || die "missing executable Python at $PYTHON"
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || die "missing worktree at $REPO"
[[ -f "$AUDIO" ]] || die "missing input audio at $AUDIO"
require_nonempty "$BASELINE_ANALYSIS"
[[ -d "$BASELINE_RUN_ROOT" ]] || die "missing baseline run root $BASELINE_RUN_ROOT"
[[ ! -e "$RUN_ROOT" ]] || die "refusing to overwrite existing run root $RUN_ROOT"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || {
  git -C "$REPO" status --short >&2
  die "candidate worktree is dirty at $REPO"
}
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$COMMIT" ]] \
  || die "worktree HEAD does not match MAPPERATORINATOR_COMMIT"
[[ "$(git -C "$REPO" branch --show-current)" == "$BRANCH" ]] \
  || die "worktree is not on MAPPERATORINATOR_BRANCH=$BRANCH"
git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF" \
  || die "missing pushed remote ref $REMOTE_REF"
[[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" == "$COMMIT" ]] \
  || die "pushed remote ref $REMOTE_REF does not match MAPPERATORINATOR_COMMIT"
if [[ -n "$DISPATCH_DELTA_SPEC" ]]; then
  require_nonempty "$DISPATCH_DELTA_SPEC"
  command -v realpath >/dev/null 2>&1 || die "missing required executable: realpath"
  SPEC_REAL=$(realpath "$DISPATCH_DELTA_SPEC")
  case "$SPEC_REAL" in
    "$REPO"/*) ;;
    *) die "dispatch delta spec must be tracked inside the candidate worktree" ;;
  esac
  SPEC_RELATIVE=${SPEC_REAL#"$REPO"/}
  git -C "$REPO" ls-files --error-unmatch -- "$SPEC_RELATIVE" >/dev/null \
    || die "dispatch delta spec is not tracked: $SPEC_RELATIVE"
  EXPECTED_SPEC_SHA=$(git -C "$REPO" show "$COMMIT:$SPEC_RELATIVE" | sha256sum | awk '{print $1}')
  ACTUAL_SPEC_SHA=$(sha256sum "$SPEC_REAL" | awk '{print $1}')
  [[ "$EXPECTED_SPEC_SHA" == "$ACTUAL_SPEC_SHA" ]] \
    || die "dispatch delta spec differs from pushed candidate commit"
  DISPATCH_DELTA_SPEC=$SPEC_REAL
fi

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

for tool in ffmpeg ffprobe nvidia-smi scontrol squeue sha256sum date df realpath; do
  command -v "$tool" >/dev/null 2>&1 || die "missing required executable: $tool"
done

AVAILABLE_KIB=$(df -Pk "$WORK" | awk 'NR == 2 {print $4}')
[[ "$AVAILABLE_KIB" =~ ^[0-9]+$ ]] \
  || die "could not determine available storage for $WORK"
(( AVAILABLE_KIB >= MIN_AVAILABLE_KIB )) \
  || die "run-root filesystem has only $AVAILABLE_KIB KiB free; require at least $MIN_AVAILABLE_KIB KiB"

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
  echo "candidate_mode=$CANDIDATE_MODE"
  echo "commit=$COMMIT"
  echo "branch=$BRANCH"
  echo "remote_ref=$REMOTE_REF"
  echo "repo=$REPO"
  echo "python=$PYTHON"
  echo "audio=$AUDIO"
  echo "audio_sha256=$(sha256sum "$AUDIO" | awk '{print $1}')"
  echo "baseline_run_root=$BASELINE_RUN_ROOT"
  echo "baseline_analysis=$BASELINE_ANALYSIS"
  echo "baseline_analysis_sha256=$(sha256sum "$BASELINE_ANALYSIS" | awk '{print $1}')"
  if [[ -n "$INITIAL_ANALYSIS" ]]; then
    echo "initial_analysis=$INITIAL_ANALYSIS"
    echo "initial_analysis_sha256=$(sha256sum "$INITIAL_ANALYSIS" | awk '{print $1}')"
  fi
  if [[ -n "$DISPATCH_DELTA_SPEC" ]]; then
    echo "dispatch_delta_spec=$DISPATCH_DELTA_SPEC"
    echo "dispatch_delta_spec_sha256=$(sha256sum "$DISPATCH_DELTA_SPEC" | awk '{print $1}')"
  fi
  echo "run_root=$RUN_ROOT"
  echo "run_root_available_kib=$AVAILABLE_KIB"
  echo "run_root_minimum_available_kib=$MIN_AVAILABLE_KIB"
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
      | sort -z \
      | xargs -0 sha256sum > "$RUN_ROOT/sha256sums.txt"
  fi
}
trap finalize EXIT

cd "$REPO"

INFERENCE_COMMAND=()
build_inference_command() {
  local pass_kind=$1
  local output_dir=$2
  INFERENCE_COMMAND=(
    "$PYTHON" inference.py --config-name profile_salvalai
    audio_path="$AUDIO"
    output_path="$output_dir"
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
    profile_pass_kind="$pass_kind"
  )
}

capture_run_paths() {
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
  local pass_kind=$2
  local run_dir="$RUN_ROOT/$run_name"
  mkdir -p "$run_dir/output"
  build_inference_command "$pass_kind" "$run_dir/output"
  printf '%q ' "${INFERENCE_COMMAND[@]}" > "$run_dir/command.txt"
  printf '\n' >> "$run_dir/command.txt"
  local started_ns finished_ns elapsed_ns exit_code
  started_ns=$(date +%s%N)
  set +e
  "${INFERENCE_COMMAND[@]}" > "$run_dir/stdout.txt" 2> "$run_dir/stderr.txt"
  exit_code=$?
  set -e
  finished_ns=$(date +%s%N)
  [[ "$started_ns" =~ ^[0-9]+$ && "$finished_ns" =~ ^[0-9]+$ ]] \
    || die "date did not return integer nanosecond timestamps"
  elapsed_ns=$((finished_ns - started_ns))
  printf '{"started_unix_ns":%s,"finished_unix_ns":%s,"elapsed_ns":%s,"exit_code":%s}\n' \
    "$started_ns" "$finished_ns" "$elapsed_ns" "$exit_code" \
    > "$run_dir/process-wall.json"
  [[ $exit_code -eq 0 ]] || die "$run_name failed with exit code $exit_code"
  capture_run_paths "$run_dir"
}

# Shell execution is deliberately serial: every process exits before the next starts.
for run_index in "${RUN_INDICES[@]}"; do
  run_fresh_process "candidate-$run_index" untraced_control
done
run_fresh_process exactness-audit exactness_audit

ANALYZER_COMMAND=(
  "$PYTHON" utils/analyze_strict_fp32_candidate.py
  --baseline-run-root "$BASELINE_RUN_ROOT"
  --baseline-analysis "$BASELINE_ANALYSIS"
  --candidate-run-root "$RUN_ROOT"
  --candidate-commit "$COMMIT"
  --candidate-branch "$BRANCH"
  --mode "$CANDIDATE_MODE"
  --output "$RUN_ROOT/analysis.json"
  --text-output "$RUN_ROOT/analysis.txt"
)
if [[ -n "$INITIAL_ANALYSIS" ]]; then
  ANALYZER_COMMAND+=(--initial-analysis "$INITIAL_ANALYSIS")
fi
if [[ -n "$DISPATCH_DELTA_SPEC" ]]; then
  cp "$DISPATCH_DELTA_SPEC" "$RUN_ROOT/dispatch-delta-spec.json"
  ANALYZER_COMMAND+=(--dispatch-delta-spec "$RUN_ROOT/dispatch-delta-spec.json")
fi
set +e
"${ANALYZER_COMMAND[@]}"
ANALYZER_EXIT=$?
set -e
require_nonempty "$RUN_ROOT/analysis.json"
require_nonempty "$RUN_ROOT/analysis.txt"
if [[ $ANALYZER_EXIT -ne 0 ]]; then
  cat "$RUN_ROOT/analysis.txt" >&2
  exit "$ANALYZER_EXIT"
fi
echo "strict_fp32_candidate_$CANDIDATE_MODE=PASS"
