#!/usr/bin/env bash
# T3 FULL-STEP RESEAL under relaxed gates (coherent + T5 KS).
# Usage:
#   ./t3_submit_reseal.sh a5000     # baseline+compile on A5000 (≤2 GPU)
#   ./t3_submit_reseal.sh 2080      # baseline+compile on 2080 Ti (≤2 GPU)
#   ./t3_submit_reseal.sh greedy    # optional greedy audit (not a promote gate)
# No push to PR #120. Unique TMPDIR set inside sbatch.
set -euo pipefail
REPO=${T3_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/t3-compile-then-capture}
SBATCH="$REPO/scripts/dcc/t3_compile.sbatch"
GREEDY="$REPO/scripts/dcc/t3_greedy_match.sbatch"
MODE=${1:-a5000}

echo "repo=$REPO"
echo "commit=$(git -C "$REPO" rev-parse HEAD)"
echo "mode=$MODE"
echo "live queue (imt11):"
squeue -u "${USER:-imt11}" -o "%.18i %.9P %.30j %.8u %.2t %.10M %R" || true

submit_pair() {
  local gres=$1 substr=$2 tag=$3
  sbatch --gres="$gres" \
    --export=ALL,VARIANT=baseline,PRECISION=fp16,EXPECTED_GPU_SUBSTR="$substr",T3_REPO="$REPO" \
    --job-name="t3-reseal-base-${tag}" \
    "$SBATCH"
  sbatch --gres="$gres" \
    --export=ALL,VARIANT=compile,PRECISION=fp16,EXPECTED_GPU_SUBSTR="$substr",T3_REPO="$REPO" \
    --job-name="t3-reseal-comp-${tag}" \
    "$SBATCH"
}

case "$MODE" in
  a5000|A5000)
    submit_pair gpu:a5000:1 A5000 a5000
    echo "Submitted A5000 baseline+compile (≤2 GPU). Next: $0 2080 after A5000 finishes."
    ;;
  2080|2080ti|2080Ti)
    submit_pair gpu:2080:1 "2080" 2080
    echo "Submitted 2080 Ti baseline+compile (≤2 GPU)."
    ;;
  greedy)
    sbatch --gres=gpu:a5000:1 \
      --export=ALL,T3_REPO="$REPO" \
      --job-name="t3-reseal-greedy-audit" \
      "$GREEDY"
    echo "Submitted greedy audit (report-only under relaxation)."
    ;;
  *)
    echo "usage: $0 [a5000|2080|greedy]" >&2
    exit 2
    ;;
esac
