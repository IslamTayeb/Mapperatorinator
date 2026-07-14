#!/usr/bin/env bash
#SBATCH --job-name=int8-mlp-compile
#SBATCH --partition=common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

ENV=${MAPPERATORINATOR_ENV:-/hpc/group/romerolab/imt11/envs/mapperatorinator}
PYTHON=${MAPPERATORINATOR_PYTHON:-$ENV/bin/python}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean scout worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT to the pushed scout commit}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH to the scout branch}
RUN_ROOT="$WORK/runs/int8-mlp-compile-${SLURM_JOB_ID:-manual}"

if [[ ! -d "$REPO/.git" && ! -f "$REPO/.git" ]]; then
  echo "missing scout worktree at $REPO" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "scout worktree is dirty" >&2
  git -C "$REPO" status --short >&2
  exit 2
fi
if [[ "$(git -C "$REPO" rev-parse HEAD)" != "$COMMIT" ]]; then
  echo "scout worktree differs from MAPPERATORINATOR_COMMIT" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" branch --show-current)" != "$BRANCH" ]]; then
  echo "scout worktree differs from MAPPERATORINATOR_BRANCH" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "missing Python at $PYTHON" >&2
  exit 2
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CPU compile smoke received CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/torch_extensions_int8_mlp" "$RUN_ROOT"
export PATH="$(dirname "$PYTHON"):$PATH"
export TORCH_EXTENSIONS_DIR="$WORK/torch_extensions_int8_mlp"
export TORCH_CUDA_ARCH_LIST=7.5
export MAX_JOBS=${MAX_JOBS:-4}
export TMPDIR="$WORK/tmp"

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "commit=$COMMIT"
echo "branch=$BRANCH"
echo "repo=$REPO"
echo "python=$PYTHON"
echo "torch_extensions=$TORCH_EXTENSIONS_DIR"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job -o "$SLURM_JOB_ID"
fi

cd "$REPO"
"$PYTHON" -m pytest -q \
  tests/test_int8_mlp_scout.py \
  tests/test_profile_int8_mlp_component.py
"$PYTHON" utils/compile_int8_mlp_extension.py \
  --json-path "$RUN_ROOT/compile.json" \
  --text-path "$RUN_ROOT/compile.txt"
sha256sum "$RUN_ROOT/compile.json" "$RUN_ROOT/compile.txt"
