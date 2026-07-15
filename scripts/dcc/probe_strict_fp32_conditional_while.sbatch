#!/usr/bin/env bash
#SBATCH --job-name=fp32-conditional-while
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:30:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

die() {
  echo "fp32-conditional-while: SETUP_FAILURE: $*" >&2
  exit 1
}

require_nonempty() {
  [[ -s "$1" ]] || die "required artifact is missing or empty: $1"
}

SLURM_JOB_ID=${SLURM_JOB_ID:?Run this wrapper inside a Slurm allocation}
PYTHON=${MAPPERATORINATOR_PYTHON:-/hpc/group/romerolab/imt11/envs/mapperatorinator/bin/python}
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean conditional-probe worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT to the pushed conditional-probe commit}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH to the conditional-probe branch}
REMOTE_REF=${MAPPERATORINATOR_REMOTE_REF:?Set MAPPERATORINATOR_REMOTE_REF to the pushed remote ref}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
EXPECTED_ACCOUNT=${MAPPERATORINATOR_EXPECTED_ACCOUNT:-romerolab}
EXPECTED_PARTITION=${MAPPERATORINATOR_EXPECTED_PARTITION:-gpu-common}
EXPECTED_GPU=${MAPPERATORINATOR_EXPECTED_GPU:-2080 Ti}
RUN_LABEL=${RUN_LABEL:-strict-fp32-conditional-while}
RUN_ROOT="$WORK/runs/${RUN_LABEL}-${SLURM_JOB_ID}"
MIN_AVAILABLE_KIB=$((20 * 1024 * 1024))

# This environment override must exist before the first Python/CUDA process.
export NVIDIA_TF32_OVERRIDE=0

[[ -x "$PYTHON" ]] || die "missing executable Python at $PYTHON"
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || die "missing worktree at $REPO"
[[ ! -e "$RUN_ROOT" ]] || die "refusing to overwrite existing run root $RUN_ROOT"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || {
  git -C "$REPO" status --short >&2
  die "conditional-probe worktree is dirty at $REPO"
}
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$COMMIT" ]] \
  || die "worktree HEAD does not match MAPPERATORINATOR_COMMIT"
[[ "$(git -C "$REPO" branch --show-current)" == "$BRANCH" ]] \
  || die "worktree is not on MAPPERATORINATOR_BRANCH=$BRANCH"
git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF" \
  || die "missing pushed remote ref $REMOTE_REF"
[[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" == "$COMMIT" ]] \
  || die "pushed remote ref $REMOTE_REF does not match MAPPERATORINATOR_COMMIT"

for tool in git nvidia-smi scontrol squeue sha256sum date df awk find sort xargs hostname tee; do
  command -v "$tool" >/dev/null 2>&1 || die "missing required executable: $tool"
done

JOB_RECORD=$(scontrol show job -o "$SLURM_JOB_ID")
[[ "$JOB_RECORD" == *"Account=$EXPECTED_ACCOUNT"* ]] \
  || die "job is not charged to account $EXPECTED_ACCOUNT"
[[ "$JOB_RECORD" == *"Partition=$EXPECTED_PARTITION"* ]] \
  || die "job is not running in partition $EXPECTED_PARTITION"
[[ "${SLURM_JOB_PARTITION:-}" == "$EXPECTED_PARTITION" ]] \
  || die "SLURM_JOB_PARTITION does not equal $EXPECTED_PARTITION"

AVAILABLE_KIB=$(df -Pk "$WORK" | awk 'NR == 2 {print $4}')
[[ "$AVAILABLE_KIB" =~ ^[0-9]+$ ]] \
  || die "could not determine available storage for $WORK"
(( AVAILABLE_KIB >= MIN_AVAILABLE_KIB )) \
  || die "run filesystem has only $AVAILABLE_KIB KiB free; require $MIN_AVAILABLE_KIB KiB"

mapfile -t GPU_ROWS < <(
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits
)
[[ ${#GPU_ROWS[@]} -eq 1 ]] || die "expected one visible GPU, got ${#GPU_ROWS[@]}"
[[ "${GPU_ROWS[0]}" == *"$EXPECTED_GPU"* ]] \
  || die "expected $EXPECTED_GPU, got ${GPU_ROWS[0]}"
[[ "${GPU_ROWS[0]}" == *"7.5"* ]] \
  || die "expected SM75 compute capability, got ${GPU_ROWS[0]}"

mapfile -t OTHER_GPU_JOBS < <(
  squeue -h -u "$USER" -t RUNNING,PENDING -o '%A|%b|%j' \
    | awk -F'|' -v current="$SLURM_JOB_ID" \
        '$1 != current && tolower($2) ~ /gpu/ { print }'
)
if [[ ${#OTHER_GPU_JOBS[@]} -ne 0 ]]; then
  printf '%s\n' "${OTHER_GPU_JOBS[@]}" >&2
  die "another GPU job is running or pending for $USER"
fi

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" \
  "$WORK/cache/cuda" "$WORK/torch_extensions" "$RUN_ROOT"
export PATH="$(dirname "$PYTHON"):$PATH"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_CACHE_PATH="$WORK/cache/cuda"
export TORCH_EXTENSIONS_DIR="$WORK/torch_extensions"
export TMPDIR="$WORK/tmp"
export TOKENIZERS_PARALLELISM=false

finalize() {
  if [[ -d "$RUN_ROOT" ]]; then
    find "$RUN_ROOT" -type f ! -name sha256sums.txt -print0 \
      | sort -z \
      | xargs -0 sha256sum > "$RUN_ROOT/sha256sums.txt"
  fi
}
trap finalize EXIT

{
  echo "host=$(hostname)"
  echo "job_id=$SLURM_JOB_ID"
  echo "account=$EXPECTED_ACCOUNT"
  echo "partition=$EXPECTED_PARTITION"
  echo "commit=$COMMIT"
  echo "branch=$BRANCH"
  echo "remote_ref=$REMOTE_REF"
  echo "repo=$REPO"
  echo "python=$PYTHON"
  echo "run_root=$RUN_ROOT"
  echo "run_root_available_kib=$AVAILABLE_KIB"
  echo "run_root_minimum_available_kib=$MIN_AVAILABLE_KIB"
  echo "NVIDIA_TF32_OVERRIDE=$NVIDIA_TF32_OVERRIDE"
  git -C "$REPO" status --short --branch
  "$PYTHON" -c 'from pathlib import Path; import torch, transformers; from torch.utils.cpp_extension import CUDA_HOME; assert CUDA_HOME and Path(CUDA_HOME, "bin", "nvcc").is_file(); import cuda.bindings.runtime; print(f"torch={torch.__version__}"); print(f"torch_cuda={torch.version.cuda}"); print(f"transformers={transformers.__version__}"); print(f"cuda_home={CUDA_HOME}")'
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv,noheader
  printf '%s\n' "$JOB_RECORD"
  squeue -u "$USER" -o '%.18i %.40j %.10T %.24b'
  df -h "$WORK"
} > "$RUN_ROOT/preflight.txt" 2>&1 || die "environment preflight failed"

cd "$REPO"
PROBE_RESULT="$RUN_ROOT/probe-result.json"
PROBE_STDOUT="$RUN_ROOT/probe-stdout.txt"
PROBE_STDERR="$RUN_ROOT/probe-stderr.txt"
ANALYSIS_JSON="$RUN_ROOT/analysis.json"
ANALYSIS_TEXT="$RUN_ROOT/analysis.txt"
PROBE_COMMAND=(
  "$PYTHON" utils/probe_conditional_while_multinomial.py
  --output "$PROBE_RESULT"
  --seed 12345
  --iterations 8
)
printf '%q ' "${PROBE_COMMAND[@]}" > "$RUN_ROOT/command.txt"
printf '\n' >> "$RUN_ROOT/command.txt"

STARTED_NS=$(date +%s%N)
set +e
"${PROBE_COMMAND[@]}" > "$PROBE_STDOUT" 2> "$PROBE_STDERR"
PROBE_EXIT_CODE=$?
set -e
FINISHED_NS=$(date +%s%N)
printf '{"started_unix_ns":%s,"finished_unix_ns":%s,"elapsed_ns":%s,"exit_code":%s}\n' \
  "$STARTED_NS" "$FINISHED_NS" "$((FINISHED_NS - STARTED_NS))" "$PROBE_EXIT_CODE" \
  > "$RUN_ROOT/process-wall.json"

case "$PROBE_EXIT_CODE" in
  0|2) ;;
  *) die "probe process failed before producing a classified result (exit $PROBE_EXIT_CODE)" ;;
esac
require_nonempty "$PROBE_RESULT"

set +e
"$PYTHON" utils/analyze_conditional_while_multinomial_probe.py \
  --input "$PROBE_RESULT" \
  --probe-exit-code "$PROBE_EXIT_CODE" \
  --output "$ANALYSIS_JSON" \
  --text-output "$ANALYSIS_TEXT" \
  > "$RUN_ROOT/analyzer-stdout.txt" \
  2> "$RUN_ROOT/analyzer-stderr.txt"
ANALYZER_EXIT_CODE=$?
set -e
require_nonempty "$ANALYSIS_JSON"
require_nonempty "$ANALYSIS_TEXT"

CLASSIFICATION=$(awk -F= '$1 == "classification" { print $2 }' "$ANALYSIS_TEXT")
if [[ $ANALYZER_EXIT_CODE -eq 3 ]]; then
  echo "result_classification=$CLASSIFICATION" | tee "$RUN_ROOT/decision.txt"
  echo "fp32_conditional_while=FAIL_INVALID_ORDINARY_GRAPH_CONTROL" >&2
  exit 3
fi
[[ $ANALYZER_EXIT_CODE -eq 0 ]] \
  || die "probe result was malformed or internally inconsistent"
case "$CLASSIFICATION" in
  EXACT_CONDITIONAL_WHILE_FEASIBLE|VALID_NEGATIVE_CONDITIONAL_WHILE_RNG) ;;
  *) die "analyzer returned unexpected classification $CLASSIFICATION" ;;
esac

echo "result_classification=$CLASSIFICATION" | tee "$RUN_ROOT/decision.txt"
mapfile -t UNEXPECTED_ARTIFACTS < <(
  find "$RUN_ROOT" -type f ! -name '*.json' ! -name '*.txt' -print
)
if [[ ${#UNEXPECTED_ARTIFACTS[@]} -ne 0 ]]; then
  printf '%s\n' "${UNEXPECTED_ARTIFACTS[@]}" >&2
  die "run root contains artifacts other than JSON or text"
fi
echo "fp32_conditional_while=COMPLETE"
