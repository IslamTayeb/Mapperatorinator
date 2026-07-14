#!/usr/bin/env bash
#SBATCH --job-name=rope-fp16-timing
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=02:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

export CONTROL_RUNNER=utils/run_k4_shared_rope_stage_aware_control.py
export CANDIDATE_RUNNER=utils/run_k4_shared_rope_stage_precision_hybrid.py
export REQUIRE_SHARED_ROPE=true
export FIXED_TIMING_TOKENS=821
export FIXED_MAIN_TOKENS=8294
export RUN_LABEL=${RUN_LABEL:-k4-split-weight-shared-rope-fp16-timing-reciprocal}

source "${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO}/scripts/dcc/verify_stage_precision_hybrid_full_song_reciprocal.sbatch"
