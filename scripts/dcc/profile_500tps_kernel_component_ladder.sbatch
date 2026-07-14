#!/usr/bin/env bash
#SBATCH --job-name=500tps-kernel-ladder
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:45:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
ROOT=${MAPPERATORINATOR_WORKTREE_ROOT:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees}
SUITE_ROOT="$WORK/runs/500tps-kernel-component-ladder-${SLURM_JOB_ID:-manual}"

CROSS_REPO="$ROOT/500tps-cross-attention-scout"
CROSS_COMMIT=1c8417d313ce9237f7648aca434f42b6796f2a7d
CROSS_BRANCH=codex/500tps-cross-attention-scout

CONDITIONAL_REPO="$ROOT/500tps-conditional-while-scout"
CONDITIONAL_COMMIT=5fafbaf59b0a0e64bc12da83d5f340dd32aa7b79
CONDITIONAL_BRANCH=codex/500tps-conditional-while-scout

FP16_SPLIT_REPO="$ROOT/500tps-fp16-split-kv"
FP16_SPLIT_COMMIT=7b26d7fed1c0f15120c8b8f53dc638c949415ef8
FP16_SPLIT_BRANCH=codex/500tps-fp16-split-kv

INT8_MLP_REPO="$ROOT/500tps-int8-mlp-scout"
INT8_MLP_COMMIT=907652706d402a90723724cd52f2d2ca721b635f
INT8_MLP_BRANCH=codex/500tps-int8-mlp-scout-dcc
INT8_MLP_REMOTE_BRANCH=codex/500tps-int8-mlp-scout

fail() {
  echo "$*" >&2
  exit 2
}

validate_checkout() {
  local name=$1
  local repo=$2
  local commit=$3
  local branch=$4
  local remote_branch=${5:-$branch}

  [[ -d "$repo/.git" || -f "$repo/.git" ]] || fail "$name worktree is missing: $repo"
  [[ -z "$(git -C "$repo" status --porcelain)" ]] || fail "$name worktree is dirty"
  [[ "$(git -C "$repo" rev-parse HEAD)" == "$commit" ]] || fail "$name commit mismatch"
  [[ "$(git -C "$repo" branch --show-current)" == "$branch" ]] || fail "$name branch mismatch"
  git -C "$repo" show-ref --verify --quiet "refs/remotes/origin/$remote_branch" \
    || fail "$name is missing origin/$remote_branch"
  [[ "$(git -C "$repo" rev-parse "origin/$remote_branch")" == "$commit" ]] \
    || fail "$name pushed ref mismatch"
}

require_artifacts() {
  local name=$1
  shift
  local artifact
  for artifact in "$@"; do
    [[ -s "$artifact" ]] || fail "$name did not produce required artifact: $artifact"
  done
}

run_gate() {
  local name=$1
  local repo=$2
  local commit=$3
  local branch=$4
  local script=$5
  shift 5

  echo "gate.$name.started=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$SUITE_ROOT/status.txt"
  set +e
  MAPPERATORINATOR_REPO="$repo" \
  MAPPERATORINATOR_COMMIT="$commit" \
  MAPPERATORINATOR_BRANCH="$branch" \
  MAPPERATORINATOR_REMOTE=origin \
  MAPPERATORINATOR_WORK="$WORK" \
    bash "$repo/$script"
  local status=$?
  set -e
  require_artifacts "$name" "$@"
  echo "gate.$name.exit_code=$status" | tee -a "$SUITE_ROOT/status.txt"
  echo "gate.$name.finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$SUITE_ROOT/status.txt"
}

mkdir -p "$WORK/logs"
mkdir "$SUITE_ROOT"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
[[ "$GPU_NAME" == "NVIDIA GeForce RTX 2080 Ti" ]] \
  || fail "kernel ladder requires RTX 2080 Ti, got $GPU_NAME"

validate_checkout cross "$CROSS_REPO" "$CROSS_COMMIT" "$CROSS_BRANCH"
validate_checkout conditional "$CONDITIONAL_REPO" "$CONDITIONAL_COMMIT" "$CONDITIONAL_BRANCH"
validate_checkout fp16_split "$FP16_SPLIT_REPO" "$FP16_SPLIT_COMMIT" "$FP16_SPLIT_BRANCH"
validate_checkout \
  int8_mlp "$INT8_MLP_REPO" "$INT8_MLP_COMMIT" "$INT8_MLP_BRANCH" "$INT8_MLP_REMOTE_BRANCH"

{
  echo "host=$(hostname)"
  echo "job_id=${SLURM_JOB_ID:-manual}"
  echo "gpu=$GPU_NAME"
  echo "suite_root=$SUITE_ROOT"
  echo "cross=$CROSS_COMMIT"
  echo "conditional=$CONDITIONAL_COMMIT"
  echo "fp16_split=$FP16_SPLIT_COMMIT"
  echo "int8_mlp=$INT8_MLP_COMMIT"
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol show job -o "$SLURM_JOB_ID"
  fi
} | tee "$SUITE_ROOT/preflight.txt"

run_gate \
  cross "$CROSS_REPO" "$CROSS_COMMIT" "$CROSS_BRANCH" \
  scripts/dcc/profile_q1_cross_attention_component.sbatch \
  "$WORK/runs/q1-cross-component-${SLURM_JOB_ID:-manual}/component.json" \
  "$WORK/runs/q1-cross-component-${SLURM_JOB_ID:-manual}/component.txt"

run_gate \
  conditional "$CONDITIONAL_REPO" "$CONDITIONAL_COMMIT" "$CONDITIONAL_BRANCH" \
  scripts/dcc/probe_conditional_while.sbatch \
  "$WORK/runs/conditional-while-probe-${SLURM_JOB_ID:-manual}/decision.txt" \
  "$WORK/runs/conditional-while-probe-${SLURM_JOB_ID:-manual}/capability/probe.json" \
  "$WORK/runs/conditional-while-probe-${SLURM_JOB_ID:-manual}/real-prefix/real-prefix.json"

run_gate \
  fp16_split "$FP16_SPLIT_REPO" "$FP16_SPLIT_COMMIT" "$FP16_SPLIT_BRANCH" \
  scripts/dcc/profile_fp16_split_kv_q1_component.sbatch \
  "$WORK/runs/fp16-split-kv-q1-component-${SLURM_JOB_ID:-manual}/component.json"

run_gate \
  int8_mlp "$INT8_MLP_REPO" "$INT8_MLP_COMMIT" "$INT8_MLP_BRANCH" \
  scripts/dcc/profile_int8_mlp_component.sbatch \
  "$WORK/runs/int8-mlp-component-${SLURM_JOB_ID:-manual}/component.json" \
  "$WORK/runs/int8-mlp-component-${SLURM_JOB_ID:-manual}/component.txt"

echo "ladder.complete=true" | tee -a "$SUITE_ROOT/status.txt"
find "$SUITE_ROOT" -type f ! -name sha256sums.txt -print0 \
  | sort -z | xargs -0 sha256sum > "$SUITE_ROOT/sha256sums.txt"
