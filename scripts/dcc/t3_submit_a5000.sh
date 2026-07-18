#!/usr/bin/env bash
# Submit T3 A5000 + 2080 cells (≤2 concurrent GPU). No push to PR #120.
set -euo pipefail
REPO=${T3_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/t3-compile-then-capture}
SBATCH="$REPO/scripts/dcc/t3_compile.sbatch"

submit_one() {
  local variant=$1 gres=$2 gpu_substr=$3
  sbatch --gres="$gres" \
    --export=ALL,VARIANT="$variant",PRECISION=fp16,EXPECTED_GPU_SUBSTR="$gpu_substr",T3_REPO="$REPO" \
    --job-name="t3-${variant}-${gpu_substr// /}" \
    "$SBATCH"
}

# Pair 1: A5000 baseline + compile (2 GPUs). Hold 2080 until A5000 pair finishes if queue busy.
submit_one baseline gpu:a5000:1 A5000
submit_one compile gpu:a5000:1 A5000
