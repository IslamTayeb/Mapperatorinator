#!/usr/bin/env bash
# Harvest 4: owned sub-op compile A5000 scout (≤2 concurrent GPU).
# Submit baseline + compile first; greedy after one finishes if queue busy.
# No push to PR #120. Unique TMPDIR set inside sbatch.
set -euo pipefail
REPO=${T3_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/t3-compile-then-capture}
SUBOPS=${MAPPERATORINATOR_COMPILE_SUBOPS:-proj_out,ffn}
SBATCH="$REPO/scripts/dcc/t3_compile.sbatch"
GREEDY="$REPO/scripts/dcc/t3_greedy_match.sbatch"

echo "repo=$REPO"
echo "commit=$(git -C "$REPO" rev-parse HEAD)"
echo "subops=$SUBOPS"
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

# Pair: baseline + compile (2 GPUs). Hold greedy until instructed / next submit.
submit_compile baseline
submit_compile compile
echo "Submitted baseline+compile. When ≤1 GPU in use, run: $0 greedy"
if [[ "${1:-}" == "greedy" ]]; then
  submit_greedy
fi
