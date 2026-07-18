#!/usr/bin/env bash
# Harvest 4: owned sub-op compile A5000 scout (≤2 concurrent GPU).
# Usage:
#   ./t3_submit_h4_a5000.sh           # baseline + compile (2 GPUs)
#   ./t3_submit_h4_a5000.sh greedy    # greedy match only (1 GPU)
# No push to PR #120. Unique TMPDIR set inside sbatch.
set -euo pipefail
REPO=${T3_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/t3-compile-then-capture}
SUBOPS=${MAPPERATORINATOR_COMPILE_SUBOPS:-proj_out,ffn}
SBATCH="$REPO/scripts/dcc/t3_compile.sbatch"
GREEDY="$REPO/scripts/dcc/t3_greedy_match.sbatch"
MODE=${1:-pair}

echo "repo=$REPO"
echo "commit=$(git -C "$REPO" rev-parse HEAD)"
echo "subops=$SUBOPS"
echo "mode=$MODE"
echo "live queue (imt11):"
squeue -u "${USER:-imt11}" -o "%.18i %.9P %.30j %.8u %.2t %.10M %R" || true

submit_compile() {
  local variant=$1
  sbatch --gres=gpu:a5000:1 \
    --export=ALL,VARIANT="$variant",PRECISION=fp16,EXPECTED_GPU_SUBSTR=A5000,T3_REPO="$REPO",MAPPERATORINATOR_COMPILE_SUBOPS="$SUBOPS" \
    --job-name="t3-h4-${variant}" \
    "$SBATCH"
}

submit_greedy() {
  sbatch --gres=gpu:a5000:1 \
    --export=ALL,T3_REPO="$REPO",MAPPERATORINATOR_COMPILE_SUBOPS="$SUBOPS" \
    --job-name="t3-h4-greedy" \
    "$GREEDY"
}

case "$MODE" in
  pair|"" )
    submit_compile baseline
    submit_compile compile
    echo "Submitted baseline+compile (≤2 GPU). Re-run with: $0 greedy"
    ;;
  greedy)
    submit_greedy
    echo "Submitted greedy match."
    ;;
  *)
    echo "usage: $0 [pair|greedy]" >&2
    exit 2
    ;;
esac
