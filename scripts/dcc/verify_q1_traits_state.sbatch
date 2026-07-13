#!/usr/bin/env bash
#SBATCH --job-name=q1-traits-state
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:30:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

ENV=${MAPPERATORINATOR_ENV:-/hpc/group/romerolab/imt11/envs/mapperatorinator}
PYTHON=${MAPPERATORINATOR_PYTHON:-$ENV/bin/python}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
BASELINE_REPO=${BASELINE_REPO:?Set BASELINE_REPO}
BASELINE_BRANCH=${BASELINE_BRANCH:?Set BASELINE_BRANCH}
BASELINE_COMMIT=${BASELINE_COMMIT:?Set BASELINE_COMMIT}
FP32_CANDIDATE_REPO=${FP32_CANDIDATE_REPO:?Set FP32_CANDIDATE_REPO}
FP32_CANDIDATE_BRANCH=${FP32_CANDIDATE_BRANCH:?Set FP32_CANDIDATE_BRANCH}
FP32_CANDIDATE_COMMIT=${FP32_CANDIDATE_COMMIT:?Set FP32_CANDIDATE_COMMIT}
FP16_CANDIDATE_REPO=${FP16_CANDIDATE_REPO:?Set FP16_CANDIDATE_REPO}
FP16_CANDIDATE_BRANCH=${FP16_CANDIDATE_BRANCH:?Set FP16_CANDIDATE_BRANCH}
FP16_CANDIDATE_COMMIT=${FP16_CANDIDATE_COMMIT:?Set FP16_CANDIDATE_COMMIT}
RUN_ROOT="$WORK/runs/q1-traits-state-${SLURM_JOB_ID:-manual}"

validate_repo() {
  local repo=$1
  local branch=$2
  local commit=$3
  test "$(git -C "$repo" branch --show-current)" = "$branch"
  test "$(git -C "$repo" rev-parse HEAD)" = "$commit"
  test -z "$(git -C "$repo" status --porcelain)"
}

validate_repo "$BASELINE_REPO" "$BASELINE_BRANCH" "$BASELINE_COMMIT"
validate_repo "$FP32_CANDIDATE_REPO" "$FP32_CANDIDATE_BRANCH" "$FP32_CANDIDATE_COMMIT"
validate_repo "$FP16_CANDIDATE_REPO" "$FP16_CANDIDATE_BRANCH" "$FP16_CANDIDATE_COMMIT"
test -x "$PYTHON"

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache/cuda" "$WORK/torch_extensions" "$RUN_ROOT"
export PATH="$(dirname "$PYTHON"):$PATH"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_CACHE_PATH="$WORK/cache/cuda"
export TORCH_EXTENSIONS_DIR="$WORK/torch_extensions"
export TMPDIR="$WORK/tmp"

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "baseline_commit=$BASELINE_COMMIT"
echo "fp32_candidate_commit=$FP32_CANDIDATE_COMMIT"
echo "fp16_candidate_commit=$FP16_CANDIDATE_COMMIT"
echo "run_root=$RUN_ROOT"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job -o "$SLURM_JOB_ID"
fi
nvidia-smi -L
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ "$GPU_NAME" != "NVIDIA GeForce RTX 2080 Ti" ]]; then
  echo "q1 traits verifier requires RTX 2080 Ti, got $GPU_NAME" >&2
  exit 2
fi

VERIFIER="$FP16_CANDIDATE_REPO/utils/verify_q1_kernel_refactor.py"
test -f "$VERIFIER"

capture() {
  local repo=$1
  local precision=$2
  local label=$3
  "$PYTHON" "$VERIFIER" capture \
    --repo-root "$repo" \
    --precision "$precision" \
    --output "$RUN_ROOT/$label.pt"
}

compare() {
  local precision=$1
  "$PYTHON" "$VERIFIER" compare \
    --reference "$RUN_ROOT/${precision}_baseline.pt" \
    --candidate "$RUN_ROOT/${precision}_candidate.pt" \
    --json-output "$RUN_ROOT/${precision}_compare.json"
}

capture "$BASELINE_REPO" fp32 fp32_baseline
capture "$FP32_CANDIDATE_REPO" fp32 fp32_candidate
compare fp32

capture "$BASELINE_REPO" fp16 fp16_baseline
capture "$FP16_CANDIDATE_REPO" fp16 fp16_candidate
compare fp16

echo "Q1_TRAITS_STATE_PASS"
