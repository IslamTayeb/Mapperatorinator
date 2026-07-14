#!/usr/bin/env bash
#SBATCH --job-name=500tps-kernel-ladder
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=02:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
ROOT=${MAPPERATORINATOR_WORKTREE_ROOT:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees}
SUITE_ROOT="$WORK/runs/500tps-kernel-component-ladder-${SLURM_JOB_ID:-manual}"
LADDER_REPO=${MAPPERATORINATOR_LADDER_REPO:?Set MAPPERATORINATOR_LADDER_REPO to the clean DCC ladder worktree}
LADDER_COMMIT=${MAPPERATORINATOR_LADDER_COMMIT:?Set MAPPERATORINATOR_LADDER_COMMIT to the exact pushed ladder commit}
LADDER_BRANCH=${MAPPERATORINATOR_LADDER_BRANCH:-codex/500tps-kernel-component-ladder-dcc}
LADDER_REMOTE_BRANCH=codex/500tps-kernel-component-ladder
ENV=${MAPPERATORINATOR_ENV:-/hpc/group/romerolab/imt11/envs/mapperatorinator}
PYTHON=${MAPPERATORINATOR_PYTHON:-$ENV/bin/python}

CROSS_REPO="$ROOT/500tps-cross-attention-scout"
CROSS_COMMIT=3e6a4dfb934aa03f5fcec639a0b4b8559f8e143c
CROSS_BRANCH=codex/500tps-cross-attention-scout

CONDITIONAL_REPO="$ROOT/500tps-conditional-while-scout"
CONDITIONAL_COMMIT=5fafbaf59b0a0e64bc12da83d5f340dd32aa7b79
CONDITIONAL_BRANCH=codex/500tps-conditional-while-scout

FP16_SPLIT_REPO="$ROOT/500tps-fp16-split-kv"
FP16_SPLIT_COMMIT=5f384760dff8520bc1ad0e11414f994d3024578b
FP16_SPLIT_BRANCH=codex/500tps-fp16-split-kv

INT8_MLP_REPO="$ROOT/500tps-int8-mlp-scout"
INT8_MLP_COMMIT=907652706d402a90723724cd52f2d2ca721b635f
INT8_MLP_BRANCH=codex/500tps-int8-mlp-scout-dcc
INT8_MLP_REMOTE_BRANCH=codex/500tps-int8-mlp-scout

VOCAB_REPO="$ROOT/500tps-vocab-sampling-scout"
VOCAB_COMMIT=f13e55e1bf1ad32629f2981738acb65aab09a45f
VOCAB_BRANCH=codex/500tps-vocab-sampling-scout

SHARED_ROPE_REPO="$ROOT/500tps-shared-rope-scout"
SHARED_ROPE_COMMIT=d0de363f6df44195f2b3b73c0797dd57bcab9440
SHARED_ROPE_BRANCH=codex/500tps-shared-rope-scout-dcc
SHARED_ROPE_REMOTE_BRANCH=codex/500tps-shared-rope-scout

PREFILL_REPO="$ROOT/500tps-prefill-graph-scout"
PREFILL_COMMIT=c67211b010183101de048853433e8380b02f180d
PREFILL_BRANCH=codex/500tps-prefill-graph-scout-dcc
PREFILL_REMOTE_BRANCH=codex/500tps-prefill-graph-scout

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
    sha256sum "$artifact" >> "$SUITE_ROOT/child-artifacts.sha256"
  done
}

classify_gate() {
  local name=$1
  local status=$2
  local report=$3
  local decision="$SUITE_ROOT/$name-decision.json"
  local outcome
  outcome=$("$PYTHON" "$LADDER_REPO/utils/classify_500tps_kernel_component_gate.py" \
    --name "$name" \
    --status "$status" \
    --report "$report" \
    --output "$decision")
  [[ -s "$decision" ]] || fail "$name classifier did not produce $decision"
  sha256sum "$decision" >> "$SUITE_ROOT/child-artifacts.sha256"
  echo "gate.$name.outcome=$outcome" | tee -a "$SUITE_ROOT/status.txt"
}

run_gate() {
  local name=$1
  local repo=$2
  local commit=$3
  local branch=$4
  local remote_ref=$5
  local timeout_limit=$6
  local script=$7
  local report=$8
  shift 8

  echo "gate.$name.started=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$SUITE_ROOT/status.txt"
  set +e
  MAPPERATORINATOR_REPO="$repo" \
  MAPPERATORINATOR_COMMIT="$commit" \
  MAPPERATORINATOR_BRANCH="$branch" \
  MAPPERATORINATOR_REMOTE=origin \
  MAPPERATORINATOR_REMOTE_REF="$remote_ref" \
  MAPPERATORINATOR_WORK="$WORK" \
    timeout --foreground --kill-after=30s "$timeout_limit" bash "$repo/$script"
  local status=$?
  set -e
  echo "gate.$name.child_exit_code=$status" | tee -a "$SUITE_ROOT/status.txt"
  if [[ "$status" -eq 124 ]]; then
    fail "$name exceeded its bounded wall time of $timeout_limit"
  fi
  require_artifacts "$name" "$report" "$@"
  classify_gate "$name" "$status" "$report"
  echo "gate.$name.finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$SUITE_ROOT/status.txt"
}

mkdir -p "$WORK/logs"
mkdir "$SUITE_ROOT"
: > "$SUITE_ROOT/child-artifacts.sha256"

[[ -x "$PYTHON" ]] || fail "missing environment Python: $PYTHON"
command -v timeout >/dev/null 2>&1 || fail "missing GNU timeout"
validate_checkout \
  ladder "$LADDER_REPO" "$LADDER_COMMIT" \
  "$LADDER_BRANCH" "$LADDER_REMOTE_BRANCH"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
[[ "$GPU_NAME" == "NVIDIA GeForce RTX 2080 Ti" ]] \
  || fail "kernel ladder requires RTX 2080 Ti, got $GPU_NAME"

validate_checkout cross "$CROSS_REPO" "$CROSS_COMMIT" "$CROSS_BRANCH"
validate_checkout conditional "$CONDITIONAL_REPO" "$CONDITIONAL_COMMIT" "$CONDITIONAL_BRANCH"
validate_checkout fp16_split "$FP16_SPLIT_REPO" "$FP16_SPLIT_COMMIT" "$FP16_SPLIT_BRANCH"
validate_checkout \
  int8_mlp "$INT8_MLP_REPO" "$INT8_MLP_COMMIT" "$INT8_MLP_BRANCH" "$INT8_MLP_REMOTE_BRANCH"
validate_checkout vocab "$VOCAB_REPO" "$VOCAB_COMMIT" "$VOCAB_BRANCH"
validate_checkout \
  shared_rope "$SHARED_ROPE_REPO" "$SHARED_ROPE_COMMIT" \
  "$SHARED_ROPE_BRANCH" "$SHARED_ROPE_REMOTE_BRANCH"
validate_checkout \
  prefill "$PREFILL_REPO" "$PREFILL_COMMIT" \
  "$PREFILL_BRANCH" "$PREFILL_REMOTE_BRANCH"

{
  echo "host=$(hostname)"
  echo "job_id=${SLURM_JOB_ID:-manual}"
  echo "gpu=$GPU_NAME"
  echo "suite_root=$SUITE_ROOT"
  echo "ladder_repo=$LADDER_REPO"
  echo "ladder_commit=$LADDER_COMMIT"
  echo "ladder_branch=$LADDER_BRANCH"
  echo "cross=$CROSS_COMMIT"
  echo "conditional=$CONDITIONAL_COMMIT"
  echo "fp16_split=$FP16_SPLIT_COMMIT"
  echo "int8_mlp=$INT8_MLP_COMMIT"
  echo "vocab=$VOCAB_COMMIT"
  echo "shared_rope=$SHARED_ROPE_COMMIT"
  echo "prefill=$PREFILL_COMMIT"
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol show job -o "$SLURM_JOB_ID"
  fi
} | tee "$SUITE_ROOT/preflight.txt"

run_gate \
  cross "$CROSS_REPO" "$CROSS_COMMIT" "$CROSS_BRANCH" \
  "origin/$CROSS_BRANCH" \
  25m \
  scripts/dcc/profile_q1_cross_attention_component.sbatch \
  "$WORK/runs/q1-cross-component-${SLURM_JOB_ID:-manual}/component.json" \
  "$WORK/runs/q1-cross-component-${SLURM_JOB_ID:-manual}/component.txt"

run_gate \
  conditional "$CONDITIONAL_REPO" "$CONDITIONAL_COMMIT" "$CONDITIONAL_BRANCH" \
  "origin/$CONDITIONAL_BRANCH" \
  25m \
  scripts/dcc/probe_conditional_while.sbatch \
  "$WORK/runs/conditional-while-probe-${SLURM_JOB_ID:-manual}/real-prefix/real-prefix.json" \
  "$WORK/runs/conditional-while-probe-${SLURM_JOB_ID:-manual}/capability/probe.json" \
  "$WORK/runs/conditional-while-probe-${SLURM_JOB_ID:-manual}/decision.txt" \
  "$WORK/runs/conditional-while-probe-${SLURM_JOB_ID:-manual}/real-prefix/real-prefix.txt"

run_gate \
  fp16_split "$FP16_SPLIT_REPO" "$FP16_SPLIT_COMMIT" "$FP16_SPLIT_BRANCH" \
  "origin/$FP16_SPLIT_BRANCH" \
  25m \
  scripts/dcc/profile_fp16_split_kv_q1_component.sbatch \
  "$WORK/runs/fp16-split-kv-q1-component-${SLURM_JOB_ID:-manual}/component.json"

run_gate \
  int8_mlp "$INT8_MLP_REPO" "$INT8_MLP_COMMIT" "$INT8_MLP_BRANCH" \
  "origin/$INT8_MLP_REMOTE_BRANCH" \
  20m \
  scripts/dcc/profile_int8_mlp_component.sbatch \
  "$WORK/runs/int8-mlp-component-${SLURM_JOB_ID:-manual}/component.json" \
  "$WORK/runs/int8-mlp-component-${SLURM_JOB_ID:-manual}/component.txt"

run_gate \
  vocab "$VOCAB_REPO" "$VOCAB_COMMIT" "$VOCAB_BRANCH" \
  "origin/$VOCAB_BRANCH" \
  20m \
  scripts/dcc/profile_k4_vocab_sampling_scout.sbatch \
  "$WORK/runs/k4-vocab-sampling-${SLURM_JOB_ID:-manual}/scout.json" \
  "$WORK/runs/k4-vocab-sampling-${SLURM_JOB_ID:-manual}/initialization.json" \
  "$WORK/runs/k4-vocab-sampling-${SLURM_JOB_ID:-manual}/sha256sums.txt"

run_gate \
  shared_rope "$SHARED_ROPE_REPO" "$SHARED_ROPE_COMMIT" "$SHARED_ROPE_BRANCH" \
  "origin/$SHARED_ROPE_REMOTE_BRANCH" \
  25m \
  scripts/dcc/profile_shared_rope_scout.sbatch \
  "$WORK/runs/shared-rope-component-${SLURM_JOB_ID:-manual}/shared-rope.json" \
  "$WORK/runs/shared-rope-component-${SLURM_JOB_ID:-manual}/shared-rope.txt" \
  "$WORK/runs/shared-rope-component-${SLURM_JOB_ID:-manual}/decision-class.txt" \
  "$WORK/runs/shared-rope-component-${SLURM_JOB_ID:-manual}/profile-exit-code.txt"

run_gate \
  prefill "$PREFILL_REPO" "$PREFILL_COMMIT" "$PREFILL_BRANCH" \
  "origin/$PREFILL_REMOTE_BRANCH" \
  35m \
  scripts/dcc/profile_prefill_graph_scout.sbatch \
  "$WORK/runs/prefill-graph-scout-${SLURM_JOB_ID:-manual}/analysis.json" \
  "$WORK/runs/prefill-graph-scout-${SLURM_JOB_ID:-manual}/analysis.txt" \
  "$WORK/runs/prefill-graph-scout-${SLURM_JOB_ID:-manual}/scout/prefill-graph-raw.json" \
  "$WORK/runs/prefill-graph-scout-${SLURM_JOB_ID:-manual}/sha256sums.txt"

echo "ladder.complete=true" | tee -a "$SUITE_ROOT/status.txt"
find "$SUITE_ROOT" -type f ! -name sha256sums.txt -print0 \
  | sort -z | xargs -0 sha256sum > "$SUITE_ROOT/sha256sums.txt"
