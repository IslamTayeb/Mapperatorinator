#!/usr/bin/env bash
#SBATCH --job-name=elemwise-comp
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:20:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

die() { echo "elemwise-comp: $*" >&2; exit 2; }

SLURM_JOB_ID=${SLURM_JOB_ID:?}
PYTHON=${MAPPERATORINATOR_PYTHON:-/hpc/group/romerolab/imt11/envs/mapperatorinator/bin/python}
REPO=${MAPPERATORINATOR_REPO:?}
COMMIT=${MAPPERATORINATOR_COMMIT:?}
BRANCH=${MAPPERATORINATOR_BRANCH:?}
REMOTE_REF=${MAPPERATORINATOR_REMOTE_REF:?}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
RUN_LABEL=${RUN_LABEL:-elemwise-fusion-component}
RUN_ROOT="$WORK/runs/${RUN_LABEL}-${SLURM_JOB_ID}"
JOB_TMP="${SLURM_TMPDIR:-/tmp}/imt11-elemwise-comp-${SLURM_JOB_ID}"
JOB_EXT="$WORK/tmp/torch_extensions-elemwise-comp-${SLURM_JOB_ID}"

[[ -x "$PYTHON" ]] || die "missing python"
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || die "missing worktree"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || die "dirty worktree"
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$COMMIT" ]] || die "HEAD mismatch"
[[ "$(git -C "$REPO" branch --show-current)" == "$BRANCH" ]] || die "branch mismatch"
git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF" || die "missing remote ref"
[[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" == "$COMMIT" ]] || die "remote mismatch"
[[ ! -e "$RUN_ROOT" ]] || die "refusing to overwrite $RUN_ROOT"

mkdir -p "$WORK/logs" "$WORK/tmp" "$JOB_TMP" "$JOB_EXT" "$RUN_ROOT"
export PATH="$(dirname "$PYTHON"):$PATH"
export TORCH_EXTENSIONS_DIR="$JOB_EXT"
export TMPDIR="$JOB_TMP"
export NVIDIA_TF32_OVERRIDE=0
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

{
  echo "host=$(hostname)"
  echo "job_id=$SLURM_JOB_ID"
  echo "commit=$COMMIT"
  echo "branch=$BRANCH"
  nvidia-smi --query-gpu=name,uuid --format=csv,noheader
} | tee "$RUN_ROOT/preflight.txt"

cd "$REPO"
"$PYTHON" utils/profile_elementwise_fusion_component.py \
  --output "$RUN_ROOT/component.json" \
  > "$RUN_ROOT/stdout.txt" 2> "$RUN_ROOT/stderr.txt"

"$PYTHON" - "$RUN_ROOT/component.json" <<'PY'
import json, pathlib, sys
p = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert p["pass"] is True
for name, check in p["checks"].items():
    assert check.get("exact") is True, name
print(
    "component_gate=PASS",
    "saving_ms_per_token=",
    p["timing_ms"]["projected_saving_ms_per_token"],
    "clears_budget=",
    p["timing_ms"]["clears_budget"],
)
PY

echo "elemwise-comp=PASS"
