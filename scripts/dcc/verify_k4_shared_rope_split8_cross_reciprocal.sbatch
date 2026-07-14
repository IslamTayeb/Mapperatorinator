#!/usr/bin/env bash

set -euo pipefail

export BASELINE_RUNNER=utils/run_k4_shared_rope_approximate_weight_only.py
export CANDIDATE_RUNNER=utils/run_k4_shared_rope_split8_cross.py
export REQUIRE_EXACT_TIMING=true
export REQUIRE_K4_CANDIDATE=true
export REQUIRE_SHARED_ROPE_INCREMENTAL=false
export REQUIRE_CROSS_INCREMENTAL=true
export CROSS_CANDIDATE_MODE=split8_attention
export RUN_LABEL=k4-shared-rope-split8-cross-reciprocal

source "${CANDIDATE_REPO:?Set CANDIDATE_REPO}/scripts/dcc/verify_weight_only_full_song_reciprocal.sbatch"
