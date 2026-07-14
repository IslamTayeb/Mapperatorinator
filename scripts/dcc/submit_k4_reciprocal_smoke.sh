#!/usr/bin/env bash
set -euo pipefail

export MAPPERATORINATOR_VERIFY_SCRIPT=scripts/dcc/verify_k4_reciprocal_smoke.sbatch
exec "${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO}/scripts/dcc/submit_k8_reciprocal.sh"
