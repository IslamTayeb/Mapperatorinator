#!/usr/bin/env bash
#SBATCH --job-name=k4-rope-fp16-cross
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=02:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

export BASELINE_RUNNER=utils/run_k4_shared_rope_approximate_weight_only.py
export CANDIDATE_RUNNER=utils/run_k4_shared_rope_fp16_cross.py
export REQUIRE_EXACT_TIMING=true
export REQUIRE_K4_CANDIDATE=true
export REQUIRE_SHARED_ROPE_INCREMENTAL=false
export REQUIRE_CROSS_INCREMENTAL=true
export CROSS_CANDIDATE_MODE=fp16_packed_projections
export RUN_LABEL=k4-shared-rope-fp16-cross-reciprocal

source "${CANDIDATE_REPO:?Set CANDIDATE_REPO}/scripts/dcc/verify_weight_only_full_song_reciprocal.sbatch"
