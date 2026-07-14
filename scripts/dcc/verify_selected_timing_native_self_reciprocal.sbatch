#!/usr/bin/env bash
#SBATCH --job-name=selected-timing-self
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:45:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

export BASELINE_RUNNER=utils/run_k4_shared_rope_fp16_cross.py
export CANDIDATE_RUNNER=utils/run_selected_timing_native_self.py
export REQUIRE_EXACT_TIMING=false
export REQUIRE_K4_CANDIDATE=true
export REQUIRE_TIMING_NATIVE_SELF_INCREMENTAL=true
export CROSS_CANDIDATE_MODE=fp16_packed_projections
export FIXED_TIMING_TOKENS=${FIXED_TIMING_TOKENS:-821}
export RUN_LABEL=selected-timing-native-self-reciprocal

source "${CANDIDATE_REPO:?Set CANDIDATE_REPO}/scripts/dcc/verify_weight_only_full_song_reciprocal.sbatch"
