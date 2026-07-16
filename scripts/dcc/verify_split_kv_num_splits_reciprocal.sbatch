#!/usr/bin/env bash
#SBATCH --job-name=split-kv-num-splits-16
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:35:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err
# Keep clear of confirmation pin even if that job finished.
#SBATCH --exclude=dcc-core-gpu-ferc-s-h36-9

set -euo pipefail

REPO=${CANDIDATE_REPO:?Set CANDIDATE_REPO to the clean pushed worktree}
COMMIT=${CANDIDATE_COMMIT:?Set CANDIDATE_COMMIT}
BRANCH=${CANDIDATE_BRANCH:?Set CANDIDATE_BRANCH}
REMOTE=${CANDIDATE_REMOTE:-origin}
REMOTE_BRANCH=${CANDIDATE_REMOTE_BRANCH:-$BRANCH}
SPLITS=${MAPPERATORINATOR_SPLIT_KV_SPLITS:?Set MAPPERATORINATOR_SPLIT_KV_SPLITS}

export BASELINE_REPO="$REPO"
export CANDIDATE_REPO="$REPO"
export BASELINE_COMMIT="$COMMIT"
export CANDIDATE_COMMIT="$COMMIT"
export BASELINE_BRANCH="$BRANCH"
export CANDIDATE_BRANCH="$BRANCH"
export CANDIDATE_REMOTE="$REMOTE"
export CANDIDATE_REMOTE_BRANCH="$REMOTE_BRANCH"
export BASELINE_RUNNER=utils/run_k4_shared_rope_fp16_cross_shared_arena.py
export CANDIDATE_RUNNER=utils/run_k4_shared_rope_fp16_cross_shared_arena_split_kv_num_splits.py
export REQUIRE_EXACT_TIMING=true
export REQUIRE_K4_CANDIDATE=true
export REQUIRE_SPLIT_KV_NUM_SPLITS_INCREMENTAL=true
export CROSS_CANDIDATE_MODE=fp16_packed_projections
export MAPPERATORINATOR_ALLOW_PARALLEL=1
export MAPPERATORINATOR_SPLIT_KV_SPLITS="$SPLITS"
export RUN_LABEL="split-kv-num-splits-${SPLITS}-reseal"

source "$REPO/scripts/dcc/verify_weight_only_full_song_reciprocal.sbatch"
