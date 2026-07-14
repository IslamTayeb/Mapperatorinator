#!/usr/bin/env bash
#SBATCH --job-name=k1-int8-fp16-timing
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=02:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

export CONTROL_RUNNER=utils/run_k4_shared_rope_k1_int8_stage_aware_control.py
export CANDIDATE_RUNNER=utils/run_k4_shared_rope_k1_int8_stage_precision_hybrid.py
export REQUIRE_SHARED_ROPE=true
export REQUIRE_K1_INT8_STAGE_COMPOSITION=true
export STAGE_COMPOSITION_VERSION=k4-split-kv-mixed-weight-shared-rope-k1-remainder-int8-mlp-stage-precision-v1
export FIXED_TIMING_TOKENS=821
export FIXED_MAIN_TOKENS=8294
export RUN_LABEL=${RUN_LABEL:-k1-int8-fp16-timing-full-song-reciprocal}

source "${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO}/scripts/dcc/verify_stage_precision_hybrid_full_song_reciprocal.sbatch"
