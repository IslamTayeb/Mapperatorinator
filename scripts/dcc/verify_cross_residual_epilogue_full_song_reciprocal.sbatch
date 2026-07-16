#!/usr/bin/env bash
#SBATCH --job-name=cross-resid-epilogue-full
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=02:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err
# Keep clear of compile-cross remeasure / confirmation pins.
#SBATCH --exclude=dcc-core-gpu-ferc-s-h36-5,dcc-core-gpu-ferc-s-h36-9

set -euo pipefail

WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}

# Isolated node-local TMPDIR when available; else work scratch with unique prefix.
if [[ -d /tmp && -w /tmp ]]; then
  export MAPPERATORINATOR_JOB_TMPDIR="/tmp/imt11/cross-residual-epilogue-${SLURM_JOB_ID}"
else
  export MAPPERATORINATOR_JOB_TMPDIR="$WORK/tmp/cross-residual-epilogue-${SLURM_JOB_ID}"
fi
mkdir -p "$MAPPERATORINATOR_JOB_TMPDIR"

export BASELINE_RUNNER=utils/run_k4_shared_rope_fp16_cross_shared_arena.py
export CANDIDATE_RUNNER=utils/run_k4_shared_rope_fp16_cross_shared_arena_layout_fused.py
export REQUIRE_EXACT_TIMING=true
export REQUIRE_K4_CANDIDATE=true
export REQUIRE_SHARED_ROPE_INCREMENTAL=false
export REQUIRE_K1_REMAINDER_INCREMENTAL=false
export REQUIRE_INT8_MLP_INCREMENTAL=false
export REQUIRE_CROSS_INCREMENTAL=false
export REQUIRE_SHARED_ARENA_INCREMENTAL=false
export REQUIRE_COMPILED_CROSS_INCREMENTAL=false
export REQUIRE_CROSS_OUT_LAYOUT_FUSED_INCREMENTAL=true
export CROSS_CANDIDATE_MODE=fp16_packed_projections
export RUN_LABEL=${RUN_LABEL:-cross-residual-epilogue-full-song}

source "${CANDIDATE_REPO:?Set CANDIDATE_REPO}/scripts/dcc/verify_weight_only_full_song_reciprocal.sbatch"
