#!/usr/bin/env bash
#SBATCH --job-name=fp16-split-kv
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=02:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

die() {
  echo "fp16-split-kv-reciprocal: $*" >&2
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
BASELINE_RUN_ROOT=${MAPPERATORINATOR_BASELINE_RUN_ROOT:?Set the accepted FP16 run root}
BASELINE_ANALYSIS=${MAPPERATORINATOR_BASELINE_ANALYSIS:-$BASELINE_RUN_ROOT/analysis.json}
MODE=${MAPPERATORINATOR_CANDIDATE_MODE:-initial}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
EXPECTED_GPU=${MAPPERATORINATOR_EXPECTED_GPU:-2080 Ti}
EXPECTED_PRESET=candidate-fp16-all-fused-split-kv-v3
RUN_LABEL=${RUN_LABEL:-fp16-split-kv-$MODE}
RUN_ROOT="$WORK/runs/${RUN_LABEL}-${SLURM_JOB_ID}"
MIN_AVAILABLE_KIB=$((20 * 1024 * 1024))

export NVIDIA_TF32_OVERRIDE=0

case "$MODE" in
  initial) RUN_INDICES=(01) ;;
  confirm) RUN_INDICES=(01 02 03 04 05) ;;
  *) die "MAPPERATORINATOR_CANDIDATE_MODE must be initial or confirm" ;;
esac

[[ -x "$PYTHON" ]] || die "missing executable Python at $PYTHON"
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || die "missing worktree at $REPO"
[[ -f "$AUDIO" ]] || die "missing SALVALAI audio at $AUDIO"
[[ -d "$BASELINE_RUN_ROOT" ]] || die "missing baseline run root"
require_nonempty "$BASELINE_ANALYSIS"
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
if [[ ${#OTHER_GPU_JOBS[@]} -ne 0 ]]; then
  printf '%s\n' "${OTHER_GPU_JOBS[@]}" >&2
  die "another GPU job is running or pending"
fi

{
  echo "host=$(hostname)"
  echo "job_id=$SLURM_JOB_ID"
  echo "mode=$MODE"
  echo "commit=$COMMIT"
  echo "branch=$BRANCH"
  echo "remote_ref=$REMOTE_REF"
  echo "repo=$REPO"
  echo "audio=$AUDIO"
  echo "audio_sha256=$(sha256sum "$AUDIO" | awk '{print $1}')"
  echo "baseline_run_root=$BASELINE_RUN_ROOT"
  echo "baseline_analysis=$BASELINE_ANALYSIS"
  echo "baseline_analysis_sha256=$(sha256sum "$BASELINE_ANALYSIS" | awk '{print $1}')"
  echo "expected_preset=$EXPECTED_PRESET"
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
INFERENCE_COMMAND=()
build_inference_command() {
  local pass_kind=$1
  local output_dir=$2
  INFERENCE_COMMAND=(
    "$PYTHON" inference.py --config-name profile_salvalai
    audio_path="$AUDIO"
    output_path="$output_dir"
    device=cuda
    precision=fp16
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
  elapsed_ns=$((finished_ns - started_ns))
  printf '{"started_unix_ns":%s,"finished_unix_ns":%s,"elapsed_ns":%s,"exit_code":%s}\n' \
    "$started_ns" "$finished_ns" "$elapsed_ns" "$exit_code" \
    > "$run_dir/process-wall.json"
  [[ $exit_code -eq 0 ]] || die "$run_name failed"
  capture_paths "$run_dir"
}

for run_index in "${RUN_INDICES[@]}"; do
  run_fresh_process "run-$run_index" untraced_control
done
run_fresh_process exactness-audit-01 exactness_audit
run_fresh_process exactness-audit-02 exactness_audit

"$PYTHON" utils/analyze_fp16_fresh_baseline.py \
  --run-root "$RUN_ROOT" \
  --expected-commit "$COMMIT" \
  --expected-branch "$BRANCH" \
  --expected-preset-version "$EXPECTED_PRESET" \
  --expected-split-kv \
  --run-count "${#RUN_INDICES[@]}" \
  --output "$RUN_ROOT/candidate-analysis.json" \
  --text-output "$RUN_ROOT/candidate-analysis.txt"
require_nonempty "$RUN_ROOT/candidate-analysis.json"
require_nonempty "$RUN_ROOT/candidate-analysis.txt"

set +e
"$PYTHON" utils/analyze_fp16_split_kv_reciprocal.py \
  --baseline-run-root "$BASELINE_RUN_ROOT" \
  --baseline-analysis "$BASELINE_ANALYSIS" \
  --candidate-analysis "$RUN_ROOT/candidate-analysis.json" \
  --mode "$MODE" \
  --output "$RUN_ROOT/analysis.json" \
  --text-output "$RUN_ROOT/analysis.txt"
ANALYZER_EXIT=$?
set -e
require_nonempty "$RUN_ROOT/analysis.json"
require_nonempty "$RUN_ROOT/analysis.txt"
cat "$RUN_ROOT/analysis.txt"
[[ $ANALYZER_EXIT -eq 0 ]] || exit "$ANALYZER_EXIT"
echo "fp16_split_kv_reciprocal_$MODE=PASS"
