#!/usr/bin/env bash
#SBATCH --job-name=500tps-kernel-ladder
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
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
CROSS_COMMIT=1c8417d313ce9237f7648aca434f42b6796f2a7d
CROSS_BRANCH=codex/500tps-cross-attention-scout

CONDITIONAL_REPO="$ROOT/500tps-conditional-while-scout"
CONDITIONAL_COMMIT=5fafbaf59b0a0e64bc12da83d5f340dd32aa7b79
CONDITIONAL_BRANCH=codex/500tps-conditional-while-scout

FP16_SPLIT_REPO="$ROOT/500tps-fp16-split-kv"
FP16_SPLIT_COMMIT=e3279d3836d1ff6cc4f5b32f67f45afec8755902
FP16_SPLIT_BRANCH=codex/500tps-fp16-split-kv

INT8_MLP_REPO="$ROOT/500tps-int8-mlp-scout"
INT8_MLP_COMMIT=907652706d402a90723724cd52f2d2ca721b635f
INT8_MLP_BRANCH=codex/500tps-int8-mlp-scout-dcc
INT8_MLP_REMOTE_BRANCH=codex/500tps-int8-mlp-scout

VOCAB_REPO="$ROOT/500tps-vocab-sampling-scout"
VOCAB_COMMIT=f13e55e1bf1ad32629f2981738acb65aab09a45f
VOCAB_BRANCH=codex/500tps-vocab-sampling-scout

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

classify_gate() {
  local name=$1
  local status=$2
  local report=$3

  "$PYTHON" - "$name" "$status" "$report" <<'PY'
import json
import sys
from pathlib import Path

name, raw_status, raw_report = sys.argv[1:]
status = int(raw_status)
report = json.loads(Path(raw_report).read_text(encoding="utf-8"))

def reject(message: str) -> None:
    raise SystemExit(f"{name} gate classification failed: {message}")

if name == "cross":
    fp32_checks = [
        bool(entry.get("checks_pass"))
        for entry in report.get("variants", {}).values()
        if entry.get("kv_storage_dtype") == "torch.float32"
    ]
    promoted = bool(report.get("summary", {}).get("any_fp32_promotion_pass"))
    if not fp32_checks or not all(fp32_checks):
        reject("FP32 correctness checks failed or are missing")
    if status == 0 and promoted:
        pass
    elif status == 3 and not promoted:
        pass
    else:
        reject(f"unexpected exit/promotion pair status={status}, promoted={promoted}")
elif name == "conditional":
    if status != 0:
        reject(f"unexpected exit status {status}")
    if report.get("decision", {}).get("pass") is not True:
        reject("CUDA conditional capability decision did not pass")
elif name == "fp16_split":
    summary = report.get("summary", {})
    invariants = summary.get("invariants_pass")
    sizing = summary.get("sizing_pass")
    if invariants is not True:
        reject("FP16 cache, memory, or drift invariants failed")
    if status == 0 and sizing is True:
        pass
    elif status == 3 and sizing is False:
        pass
    else:
        reject(f"unexpected exit/sizing pair status={status}, sizing={sizing}")
elif name == "int8_mlp":
    summary = report.get("summary", {})
    invariants = summary.get("invariants_pass")
    sizing = summary.get("sizing_pass")
    if invariants is not True:
        reject("INT8 invariants failed")
    if status == 0 and sizing is True:
        pass
    elif status == 3 and sizing is False:
        pass
    else:
        reject(f"unexpected exit/sizing pair status={status}, sizing={sizing}")
elif name == "vocab":
    if status != 0:
        reject(f"unexpected exit status {status}")
    ceiling = report.get("fixed_work_ceiling", {})
    clears = ceiling.get("main_ceiling_clears_threshold")
    if not isinstance(clears, bool):
        reject("fixed-work ceiling decision must be a bool")
    expected = (
        "retain_for_candidate_kernel" if clears else "stop_below_main_component_gate"
    )
    if report.get("decision") != expected:
        reject("vocabulary decision is inconsistent with its fixed-work ceiling")
else:
    reject("unknown gate")
PY
}

run_gate() {
  local name=$1
  local repo=$2
  local commit=$3
  local branch=$4
  local script=$5
  local report=$6
  shift 6

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
  require_artifacts "$name" "$report" "$@"
  classify_gate "$name" "$status" "$report"
  echo "gate.$name.exit_code=$status" | tee -a "$SUITE_ROOT/status.txt"
  echo "gate.$name.finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$SUITE_ROOT/status.txt"
}

mkdir -p "$WORK/logs"
mkdir "$SUITE_ROOT"

[[ -x "$PYTHON" ]] || fail "missing environment Python: $PYTHON"
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
  "$WORK/runs/conditional-while-probe-${SLURM_JOB_ID:-manual}/capability/probe.json" \
  "$WORK/runs/conditional-while-probe-${SLURM_JOB_ID:-manual}/decision.txt" \
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

run_gate \
  vocab "$VOCAB_REPO" "$VOCAB_COMMIT" "$VOCAB_BRANCH" \
  scripts/dcc/profile_k4_vocab_sampling_scout.sbatch \
  "$WORK/runs/k4-vocab-sampling-${SLURM_JOB_ID:-manual}/scout.json" \
  "$WORK/runs/k4-vocab-sampling-${SLURM_JOB_ID:-manual}/initialization.json" \
  "$WORK/runs/k4-vocab-sampling-${SLURM_JOB_ID:-manual}/sha256sums.txt"

echo "ladder.complete=true" | tee -a "$SUITE_ROOT/status.txt"
find "$SUITE_ROOT" -type f ! -name sha256sums.txt -print0 \
  | sort -z | xargs -0 sha256sum > "$SUITE_ROOT/sha256sums.txt"
