#!/usr/bin/env bash
#SBATCH --job-name=k4-recip-smoke
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=03:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

export CANDIDATE_BLOCK_SIZE=4
export CANDIDATE_RUNNER=utils/run_k4_candidate.py
export RUN_COUNTER_CONTROL=true
export RUN_FULL=false
export RUN_TAG=k4-reciprocal-smoke

source "${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO}/scripts/dcc/verify_k8_reciprocal.sbatch"
