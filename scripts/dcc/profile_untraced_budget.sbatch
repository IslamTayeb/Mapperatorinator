#!/usr/bin/env bash
#SBATCH --job-name=untraced-budget
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=01:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

die() {
  echo "untraced-budget: $*" >&2
  exit 2
}

SLURM_JOB_ID=${SLURM_JOB_ID:?Run this wrapper inside a Slurm allocation}
PYTHON=${MAPPERATORINATOR_PYTHON:-/hpc/group/romerolab/imt11/envs/mapperatorinator/bin/python}
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean budget worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT to the pushed budget commit}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH to the budget worktree branch}
REMOTE_REF=${MAPPERATORINATOR_REMOTE_REF:?Set MAPPERATORINATOR_REMOTE_REF to the pushed remote ref}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
EXPECTED_GPU=${MAPPERATORINATOR_EXPECTED_GPU:-2080 Ti}
EXPECTED_MAIN_TOKENS=${MAPPERATORINATOR_EXPECTED_MAIN_TOKENS:-8294}
EXPECTED_TIMING_TOKENS=${MAPPERATORINATOR_EXPECTED_TIMING_TOKENS:-821}
RUN_LABEL=${RUN_LABEL:-strict-fp32-untraced-budget}
RUN_ROOT="$WORK/runs/${RUN_LABEL}-${SLURM_JOB_ID}"
MIN_AVAILABLE_KIB=$((20 * 1024 * 1024))

# This must exist before the first Python process or CUDA context.
export NVIDIA_TF32_OVERRIDE=0

[[ -x "$PYTHON" ]] || die "missing Python: $PYTHON"
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || die "missing worktree: $REPO"
[[ -f "$AUDIO" ]] || die "missing audio: $AUDIO"
[[ ! -e "$RUN_ROOT" ]] || die "refusing to overwrite $RUN_ROOT"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || die "worktree is dirty"
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$COMMIT" ]] || die "HEAD does not match commit"
[[ "$(git -C "$REPO" branch --show-current)" == "$BRANCH" ]] || die "worktree branch mismatch"
git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF" \
  || die "missing pushed remote ref $REMOTE_REF"
[[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" == "$COMMIT" ]] || die "remote ref does not match commit"

export PATH="$(dirname "$PYTHON"):$PATH"
for tool in ffmpeg ffprobe nvidia-smi scontrol squeue sha256sum date df; do
  command -v "$tool" >/dev/null 2>&1 || die "missing executable: $tool"
done
mapfile -t GPU_ROWS < <(
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
)
[[ ${#GPU_ROWS[@]} -eq 1 ]] || die "expected one visible GPU"
[[ "${GPU_ROWS[0]}" == *"$EXPECTED_GPU"*", 7.5" ]] \
  || die "expected $EXPECTED_GPU/SM75, got ${GPU_ROWS[0]}"

mapfile -t OTHER_GPU_JOBS < <(
  squeue -h -u "$USER" -t RUNNING,PENDING -o '%A|%b|%j' \
    | awk -F'|' -v current="$SLURM_JOB_ID" \
        '$1 != current && tolower($2) ~ /gpu/ { print }'
)
if [[ ${#OTHER_GPU_JOBS[@]} -ne 0 ]]; then
  printf '%s\n' "${OTHER_GPU_JOBS[@]}" >&2
  die "another GPU job is running or pending for $USER"
fi

mkdir -p "$WORK/logs" "$WORK/cache" "$WORK/tmp" "$RUN_ROOT"
AVAILABLE_KIB=$(df -Pk "$WORK" | awk 'NR == 2 {print $4}')
[[ "$AVAILABLE_KIB" =~ ^[0-9]+$ ]] \
  || die "could not determine available storage for $WORK"
(( AVAILABLE_KIB >= MIN_AVAILABLE_KIB )) \
  || die "run-root filesystem has only $AVAILABLE_KIB KiB free"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_EXTENSIONS_DIR="$WORK/torch_extensions"
export TMPDIR="$WORK/tmp"
export TOKENIZERS_PARALLELISM=false

finalize() {
  if [[ -d "$RUN_ROOT" ]]; then
    find "$RUN_ROOT" -type f ! -name sha256sums.txt -print0 \
      | sort -z \
      | xargs -0 -r sha256sum > "$RUN_ROOT/sha256sums.txt"
  fi
}
trap finalize EXIT

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
  echo "run_root=$RUN_ROOT"
  echo "run_root_available_kib=$AVAILABLE_KIB"
  echo "NVIDIA_TF32_OVERRIDE=$NVIDIA_TF32_OVERRIDE"
  git -C "$REPO" status --short --branch
  "$PYTHON" -c 'import torch,transformers; print(f"torch={torch.__version__}"); print(f"cuda={torch.version.cuda}"); print(f"transformers={transformers.__version__}")'
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv,noheader
  scontrol show job -o "$SLURM_JOB_ID"
  squeue -u "$USER" -o '%.18i %.40j %.10T %.24b'
  df -h "$WORK"
} | tee "$RUN_ROOT/preflight.txt"

cd "$REPO"
run_profile() {
  local pass_kind=$1
  local run_dir="$RUN_ROOT/$pass_kind"
  mkdir -p "$run_dir/output"
  local command=(
    "$PYTHON" inference.py --config-name profile_salvalai
    audio_path="$AUDIO"
    output_path="$run_dir/output"
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
  [[ $exit_code -eq 0 ]] || die "$pass_kind failed with exit code $exit_code"
  mapfile -t profiles < <(find "$run_dir/output" -type f -name '*.profile.json' | sort)
  [[ ${#profiles[@]} -eq 1 ]] || die "expected one profile for $pass_kind"
  [[ -s "${profiles[0]}" ]] || die "empty profile for $pass_kind"
  local result=${profiles[0]%.profile.json}
  [[ -s "$result" ]] || die "missing result for $pass_kind"
  printf '%s\n' "${profiles[0]}" > "$run_dir/profile-path.txt"
  printf '%s\n' "$result" > "$run_dir/result-path.txt"
}

run_profile untraced_control
run_profile untraced_budget

CONTROL_PROFILE=$(<"$RUN_ROOT/untraced_control/profile-path.txt")
BUDGET_PROFILE=$(<"$RUN_ROOT/untraced_budget/profile-path.txt")
"$PYTHON" utils/nsight_agent_profile.py transparency \
  --control "$CONTROL_PROFILE" \
  --traced "$BUDGET_PROFILE" \
  --output "$RUN_ROOT/transparency.json"
[[ -s "$RUN_ROOT/transparency.json" ]] || die "missing transparency report"

"$PYTHON" utils/summarize_untraced_budget.py \
  --profile "$BUDGET_PROFILE" \
  --control-profile "$CONTROL_PROFILE" \
  --output "$RUN_ROOT/budget-summary.json" \
  --text-output "$RUN_ROOT/budget-summary.txt"
[[ -s "$RUN_ROOT/budget-summary.json" ]] || die "missing budget summary"
[[ -s "$RUN_ROOT/budget-summary.txt" ]] || die "missing budget text summary"

"$PYTHON" - \
  "$RUN_ROOT/budget-summary.json" \
  "$RUN_ROOT/transparency.json" \
  "$EXPECTED_MAIN_TOKENS" \
  "$EXPECTED_TIMING_TOKENS" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
transparency = json.load(open(sys.argv[2], encoding="utf-8"))
expected_tokens = int(sys.argv[3])
expected_timing_tokens = int(sys.argv[4])
if not summary.get("pass"):
    raise SystemExit("budget reconciliation failed")
if not transparency.get("pass"):
    raise SystemExit("budget instrumentation changed same-precision output")
main = summary.get("by_label", {}).get("main_generation")
if not isinstance(main, dict):
    raise SystemExit("budget summary is missing main_generation")
if int(main.get("generated_tokens", -1)) != expected_tokens:
    raise SystemExit(
        f"expected {expected_tokens} fixed-work main tokens, "
        f"got {main.get('generated_tokens')}"
    )
if float(main.get("reconciliation_error_fraction", 1.0)) > 0.02:
    raise SystemExit("main-generation budget did not reconcile within 2%")
timing = summary.get("by_label", {}).get("timing_context")
if not isinstance(timing, dict):
    raise SystemExit("budget summary is missing timing_context")
if int(timing.get("generated_tokens", -1)) != expected_timing_tokens:
    raise SystemExit(
        f"expected {expected_timing_tokens} fixed-work timing tokens, "
        f"got {timing.get('generated_tokens')}"
    )
if float(timing.get("reconciliation_error_fraction", 1.0)) > 0.02:
    raise SystemExit("timing-generation budget did not reconcile within 2%")
PY

echo "PASS_UNTRACED_BUDGET run_root=$RUN_ROOT"
